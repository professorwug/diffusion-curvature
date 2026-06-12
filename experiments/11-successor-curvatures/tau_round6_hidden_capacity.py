"""Round 6: try doubling FB hidden_dim to see if capacity is the
remaining bottleneck for d=2.

Round 4b plateaued at ρ_avg = 0.645 at d=2 with hidden=256, no single τ
exceeding 0.65. The remaining failure mode is ε=0.20 where the envelope
caps at 0.41. This round tests whether more FB capacity (hidden=512)
plus the round-4 winning softmax τ=0.05 can lift ε=0.20 past 0.5 and
the overall mean past 0.7.

15 manifolds × 4 noise × 2 τ values (0.05 and 1.0 for reference) on
d=2. ~60 FB trains × ~70 s = ~70 min.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pandas as pd
import scipy.stats

EXP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_ROOT))

from diffusion_curvature.successor import SuccessorEntropyCurvature
from diffusion_curvature.successor.measures import compute_successor_measures
from diffusion_curvature.successor.train import FBTrainer
from diffusion_curvature.tau_datasets import TauColosseum

OUT_DIR = EXP_ROOT / "processed_data" / "agg_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV = OUT_DIR / "round6_hidden512.csv"

DEVICE = os.environ.get("DEVICE", "cuda:0")

DIM = 2
NOISES = [0.01, 0.05, 0.10, 0.20]
NUM_MANIFOLDS = 15
SOFTMAX_TAUS = [0.05, 1.0]

FB_CFG = dict(
    z_dim=8, hidden_dim=512, gamma=0.9,
    n_epochs=300, cosine_lr=True, batch_size=1024, lr=1e-4,
)
DATASET_CFG = dict(
    n_samples=2000, n_trajectories=500, traj_length=50, knn=10,
)


def pearson(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return float("nan")
    r, _ = scipy.stats.pearsonr(a[m], b[m])
    return float(r) if np.isfinite(r) else float("nan")


def existing_keys() -> set[tuple[float, int, float]]:
    if not CSV.exists() or CSV.stat().st_size == 0:
        return set()
    df = pd.read_csv(CSV, usecols=["noise", "instance", "softmax_tau"])
    return set(zip(df["noise"].astype(float),
                    df["instance"].astype(int),
                    df["softmax_tau"].astype(float)))


def append_rows(rows: list[dict]) -> None:
    if not rows:
        return
    header = not CSV.exists() or CSV.stat().st_size == 0
    pd.DataFrame(rows).to_csv(CSV, mode="a", index=False, header=header)


def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"FB cfg: {FB_CFG}")
    print(f"Sweep: d={DIM}, ε ∈ {NOISES}, τ ∈ {SOFTMAX_TAUS}, "
          f"{NUM_MANIFOLDS} manifolds")
    done = existing_keys()
    if done:
        print(f"Resume: {len(done)} (noise, instance, τ) triples saved.")

    np.random.seed(2024)

    for noise in NOISES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cc = TauColosseum(
                intrinsic_dims=[DIM], codimensions=[1],
                noise_levels=[noise], num_manifolds_per_dim=NUM_MANIFOLDS,
                seed=2024,
                save_directory=f"/tmp/.tau-r6-d{DIM}-n{noise:g}",
                **DATASET_CFG,
            )
        print(f"\n--- d={DIM} ε={noise:g} ({len(cc)} manifolds) ---", flush=True)
        for i in range(len(cc)):
            need = [t for t in SOFTMAX_TAUS
                    if (float(noise), i, float(t)) not in done]
            if not need:
                continue
            inst = cc.get_item(i)
            ks_true = float(np.asarray(inst["ks"]).mean())
            t0 = time.time()
            trainer = FBTrainer(
                obs_dim=inst["X"].shape[1], **FB_CFG,
                device=DEVICE, seed=11 + i,
            )
            trainer.fit(inst["trajectories"].astype(np.float32))
            M, F_emb, B_emb = compute_successor_measures(
                trainer.F_net, trainer.B_net,
                inst["X"], inst["X"], device=DEVICE,
            )
            rows: list[dict] = []
            for sm_tau in need:
                k = np.asarray(
                    SuccessorEntropyCurvature(tau=sm_tau).fit_transform(
                        M=M, F_embeddings=F_emb, B_embeddings=B_emb,
                    ),
                    dtype=float,
                )
                k = k[np.isfinite(k)]
                ks_hat = float(k.mean()) if k.size else float("nan")
                rows.append({
                    "dim": DIM, "noise": noise, "instance": i,
                    "softmax_tau": sm_tau,
                    "ks_true": ks_true, "ks_hat": ks_hat,
                })
            append_rows(rows)
            print(
                f"  inst {i+1:2d}/{len(cc)}  ks_true={ks_true:+7.3f}  "
                + "  ".join(f"τ={r['softmax_tau']:.2g}:{r['ks_hat']:+7.3f}"
                            for r in rows)
                + f"   [{time.time()-t0:.1f}s]",
                flush=True,
            )

    df = pd.read_csv(CSV)
    print(f"\n========== Pearson r per (softmax τ × ε) at d={DIM}, hidden=512 ==========")
    print(f"{'τ':>6}  " + "  ".join(f"ε={n:>5g}" for n in NOISES) + "    mean")
    for sm_tau in SOFTMAX_TAUS:
        rs = []
        for noise in NOISES:
            cell = df[(df.dim == DIM) & np.isclose(df.noise, noise)
                       & np.isclose(df.softmax_tau, sm_tau)]
            r = pearson(cell.ks_hat, cell.ks_true)
            rs.append(r)
        print(f"{sm_tau:>6.2g}  " + "  ".join(f"{r:+7.3f}" for r in rs)
              + f"    {np.nanmean(rs):+7.3f}")


if __name__ == "__main__":
    main()
