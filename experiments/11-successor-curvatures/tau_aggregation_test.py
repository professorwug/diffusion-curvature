"""Local sanity check on three Successor-Entropy aggregations for the
trajectory-sampled Colosseum.

For each (dim, noise) cell we draw a small TauColosseum (n_manifolds=20),
train FB once per manifold with the v3 hyperparameters (matching the
production sweep), then read the successor curvature `k_SR(b)` at every
corpus point. Three scalar aggregations per manifold are compared:

    visited  — mean of k_SR over the unique trajectory-visited indices
                (this is what `benchmark_all_on_tau_colosseum.py` currently
                uses and what shipped in the tau half of the merged figure).
    corpus   — mean of k_SR over the FULL n-point corpus (matches the iid
                aggregation in `benchmark_all_on_iid.py`).
    idx0     — k_SR(b_0) at the first corpus point only — a pointwise
                reading with no aggregation.

For each aggregation we compute Pearson r across the 20 manifolds in the
cell against the manifold's mean true scalar curvature.

Output: pretty-printed table to stdout, plus
`processed_data/tau_aggregation_test.csv` with the per-manifold ks_hat.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")

import numpy as np
import pandas as pd
import scipy.stats

EXP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_ROOT))

from diffusion_curvature.successor import SuccessorEntropyCurvature
from diffusion_curvature.successor.measures import compute_successor_measures
from diffusion_curvature.successor.train import FBTrainer
from diffusion_curvature.tau_datasets import TauColosseum

# ---------------------------------------------------------------------------
# Sweep config (small; intended to run in a few minutes locally)
# ---------------------------------------------------------------------------

DIMS = [2, 3]
NOISES = [0.05, 0.20]
NUM_MANIFOLDS = 15

DATASET_KW = dict(
    n_samples=600,
    n_trajectories=120,
    traj_length=40,
    knn=10,
)

FB_KW = dict(
    z_dim=16, hidden_dim=256, gamma=0.9,
    # Halved from production for the local-CPU test; the v2 reference torus
    # validation showed FB converges well under 300 epochs with cosine LR.
    n_epochs=300, cosine_lr=True, batch_size=512,
)

DEVICE = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "-1" else "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pearson(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return float("nan")
    r, _ = scipy.stats.pearsonr(a[m], b[m])
    return float(r) if np.isfinite(r) else float("nan")


def visited_subset(inst) -> np.ndarray:
    return np.unique(np.asarray(inst["trajectories_idx"]).ravel())


def aggregations_for(M, F_emb, B_emb, visited) -> dict[str, float]:
    k = np.asarray(
        SuccessorEntropyCurvature().fit_transform(
            M=M, F_embeddings=F_emb, B_embeddings=B_emb,
        ),
        dtype=float,
    )
    out: dict[str, float] = {}

    # 1. visited-subset mean (current production behaviour)
    k_vis = k[visited]
    k_vis = k_vis[np.isfinite(k_vis)]
    out["visited"] = float(k_vis.mean()) if k_vis.size else float("nan")

    # 2. full-corpus mean (matches the iid aggregation)
    k_all = k[np.isfinite(k)]
    out["corpus"] = float(k_all.mean()) if k_all.size else float("nan")

    # 3. point at idx=0 (no aggregation)
    out["idx0"] = float(k[0]) if np.isfinite(k[0]) else float("nan")

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATASET_KW}")
    print(f"FB:      {FB_KW}")
    print(f"Sweep:   dims={DIMS}, noises={NOISES}, n_manifolds={NUM_MANIFOLDS}")
    print()

    rows = []
    for d in DIMS:
        for noise in NOISES:
            t0 = time.time()
            print(f"=== d={d}  ε={noise:.2g} ===", flush=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cc = TauColosseum(
                    intrinsic_dims=[d],
                    codimensions=[1],
                    noise_levels=[noise],
                    num_manifolds_per_dim=NUM_MANIFOLDS,
                    seed=1234,
                    save_directory=f"/tmp/.tau-agg-test-d{d}-n{noise:g}",
                    **DATASET_KW,
                )
            for i in range(len(cc)):
                inst = cc.get_item(i)
                visited = visited_subset(inst)
                ks_true_scalar = float(np.asarray(inst["ks"]).mean())

                trainer = FBTrainer(
                    obs_dim=inst["X"].shape[1],
                    **FB_KW,
                    device=DEVICE,
                    seed=42 + i,
                )
                trainer.fit(inst["trajectories"].astype(np.float32))
                M, F_emb, B_emb = compute_successor_measures(
                    trainer.F_net, trainer.B_net,
                    inst["X"], inst["X"],
                    device=DEVICE,
                )
                aggs = aggregations_for(M, F_emb, B_emb, visited)
                rows.append({
                    "dim": d, "noise": noise, "instance": i,
                    "ks_true": ks_true_scalar,
                    "n_visited": int(visited.size),
                    **aggs,
                })
                print(
                    f"  inst {i+1:2d}/{len(cc)}: "
                    f"ks_true={ks_true_scalar:+7.3f}  "
                    f"n_vis={visited.size:4d}/{inst['X'].shape[0]}  "
                    f"visited={aggs['visited']:+7.3f}  "
                    f"corpus={aggs['corpus']:+7.3f}  "
                    f"idx0={aggs['idx0']:+7.3f}",
                    flush=True,
                )
            print(f"  cell elapsed: {time.time()-t0:.1f}s")

    df = pd.DataFrame(rows)
    out_csv = EXP_ROOT / "processed_data" / "tau_aggregation_test.csv"
    out_csv.parent.mkdir(exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    print("\n========== Pearson r per (dim, ε, aggregation) ==========")
    print(f"{'dim':>4} {'ε':>6} | {'visited':>10} {'corpus':>10} {'idx0':>10} | n_manifolds")
    for d in DIMS:
        for noise in NOISES:
            cell = df[(df.dim == d) & np.isclose(df.noise, noise)]
            r_vis = pearson(cell.visited, cell.ks_true)
            r_cor = pearson(cell.corpus,  cell.ks_true)
            r_id0 = pearson(cell.idx0,    cell.ks_true)
            print(f"{d:>4} {noise:>6.2g} | {r_vis:>10.3f} {r_cor:>10.3f} {r_id0:>10.3f} | {len(cell)}")


if __name__ == "__main__":
    main()
