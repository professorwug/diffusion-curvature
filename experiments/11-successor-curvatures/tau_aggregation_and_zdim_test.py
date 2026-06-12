"""Local 2-manifold sweep over (z_dim, aggregation) for Successor Entropy.

The earlier `tau_fb_hparam_sweep.py` showed that lowering z_dim from 16
to 8 lifts Pearson r at d=2/ε=0.05 from -0.17 to +0.37. The user's other
hypothesis is that aggregating k_SR over the *local kNN-neighbours of
the target point* (rather than over the full corpus) might recover
performance lost by the global-mean reduction.

This script combines both knobs:

    z_dim ∈ {16, 8, 4}
    aggregation ∈ {corpus, local-knn-k}     where k ∈ {10, 25}

Sweep: TauColosseum at d=2, codim=1, ε ∈ {0.05, 0.10}, 20 manifolds per
cell. For each manifold we train one FB per z_dim and read off all four
aggregation variants (since the FB is independent of the aggregation).

Outputs:
    processed_data/agg_test/per_instance.csv         — per-row ks_hat
    processed_data/agg_test/summary_pearson.csv      — Pearson r per
                                                       (z_dim, aggregation,
                                                        noise) cell
    stdout: pretty-printed Pearson summary table
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from itertools import product
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.neighbors import NearestNeighbors

EXP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_ROOT))

from diffusion_curvature.successor import SuccessorEntropyCurvature
from diffusion_curvature.successor.measures import compute_successor_measures
from diffusion_curvature.successor.train import FBTrainer
from diffusion_curvature.tau_datasets import TauColosseum

# Fixed across the sweep — match the production τ-Colosseum configuration
# (n_samples / n_trajectories / traj_length all match the iid v3 sweep so
# FB sees the same data budget).
DATASET_KW = dict(
    n_samples=2000,
    # Down from 500 → 200 to keep per-FB train under ~30s on a contented
    # GPU. The earlier hparam sweep at n_traj=120 still surfaced the
    # z_dim effect cleanly, so 200 is safely above the danger zone.
    n_trajectories=200,
    traj_length=50,
    knn=10,
)

DIM = 2
NOISES = [0.05, 0.10]
NUM_MANIFOLDS = 20

# FB hparam grid — z_dim only. Other hparams are pinned at the v3 defaults
# (gamma=0.9, hidden=256, cosine_lr=True, n_epochs=600 for tractable CPU
# wall time; the previous tau_fb_hparam_sweep.py at n_epochs=300 already
# revealed the z_dim effect).
Z_DIMS = [16, 8, 4]
FB_BASE = dict(
    # n_epochs halved (600 → 300) — the prior local sweep showed FB
    # converging well within 300 epochs at z_dim=8 with cosine LR.
    hidden_dim=256, gamma=0.9, n_epochs=300,
    cosine_lr=True, batch_size=1024, lr=1e-4,
)

INCR_CSV_NAME = "per_instance.csv"  # incremental output; columns match `rows`

# Aggregation grid — applied to a single SuccessorEntropyCurvature output.
AGGREGATIONS = [
    ("corpus",        None),   # mean over the full n=2000 corpus
    ("local-knn-10",  10),     # mean over the 10 nearest neighbours of idx 0
    ("local-knn-25",  25),     # ... 25 nearest
    ("idx0",          1),      # k_SR at idx 0 only (no aggregation)
]

# Default to CPU — the local GPUs are shared with ollama and we hit silent
# kills under contention. Override via DEVICE=cuda:N env if the GPUs are
# free.
DEVICE = os.environ.get("DEVICE", "cpu")


def pearson(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return float("nan")
    r, _ = scipy.stats.pearsonr(a[m], b[m])
    return float(r) if np.isfinite(r) else float("nan")


def aggregate(k: np.ndarray, X: np.ndarray, target: int, agg_name: str,
              k_neighbours: int | None) -> float:
    """Reduce a per-corpus curvature vector to a scalar per the aggregation."""
    if agg_name == "corpus":
        finite = k[np.isfinite(k)]
        return float(finite.mean()) if finite.size else float("nan")
    if agg_name == "idx0":
        v = k[target]
        return float(v) if np.isfinite(v) else float("nan")
    # local-knn-K: mean of k_SR over the K nearest neighbours of `target`,
    # where neighbours are computed in the ambient corpus (X).
    nn = NearestNeighbors(n_neighbors=min(k_neighbours, X.shape[0])).fit(X)
    _, idx = nn.kneighbors(X[target:target + 1])
    nb = idx[0]
    vals = k[nb]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def _existing_keys(csv_path: Path) -> set[tuple[float, int, int]]:
    """Return (noise, instance, z_dim) triples already present in the CSV."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    df = pd.read_csv(csv_path, usecols=["noise", "instance", "z_dim"])
    return set(zip(df["noise"].astype(float),
                    df["instance"].astype(int),
                    df["z_dim"].astype(int)))


def _append_rows(csv_path: Path, rows: list[dict]) -> None:
    """Append rows to the incremental CSV; write header on first call."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not csv_path.exists() or csv_path.stat().st_size == 0
    df.to_csv(csv_path, mode="a", index=False, header=header)


def main() -> None:
    out_dir = EXP_ROOT / "processed_data" / "agg_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    incr_csv = out_dir / INCR_CSV_NAME
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATASET_KW}")
    print(f"FB base: {FB_BASE}")
    print(f"Sweep:   d={DIM}, noises={NOISES}, n_manifolds={NUM_MANIFOLDS}")
    print(f"         z_dims={Z_DIMS}, aggregations={[a for a, _ in AGGREGATIONS]}")
    print(f"Output → {out_dir}")
    done = _existing_keys(incr_csv)
    if done:
        print(f"Resume: {len(done)} (noise, instance, z_dim) triples "
              f"already in {incr_csv.name}; skipping.")
    print()

    rows: list[dict] = []
    for noise in NOISES:
        cache = f"/tmp/.tau-aggtest-d{DIM}-n{noise:g}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cc = TauColosseum(
                intrinsic_dims=[DIM], codimensions=[1],
                noise_levels=[noise], num_manifolds_per_dim=NUM_MANIFOLDS,
                seed=2024, save_directory=cache,
                **DATASET_KW,
            )
        print(f"=== d={DIM}  ε={noise:.2g}  built {len(cc)} manifolds ===",
              flush=True)
        for i in range(len(cc)):
            inst = cc.get_item(i)
            visited = np.unique(np.asarray(inst["trajectories_idx"]).ravel())
            target = int(visited[0]) if visited.size else 0
            ks_true = float(np.asarray(inst["ks"]).mean())
            for z_dim in Z_DIMS:
                if (float(noise), i, z_dim) in done:
                    continue
                t0 = time.time()
                trainer = FBTrainer(
                    obs_dim=inst["X"].shape[1],
                    z_dim=z_dim,
                    **FB_BASE,
                    device=DEVICE,
                    seed=11 + i,
                )
                trainer.fit(inst["trajectories"].astype(np.float32))
                M, F_emb, B_emb = compute_successor_measures(
                    trainer.F_net, trainer.B_net,
                    inst["X"], inst["X"], device=DEVICE,
                )
                k_per_pt = np.asarray(
                    SuccessorEntropyCurvature().fit_transform(
                        M=M, F_embeddings=F_emb, B_embeddings=B_emb,
                    ),
                    dtype=float,
                )
                new_rows: list[dict] = []
                for agg_name, agg_k in AGGREGATIONS:
                    ks_hat = aggregate(k_per_pt, inst["X"], target,
                                        agg_name, agg_k)
                    new_rows.append({
                        "dim": DIM, "noise": noise, "instance": i,
                        "z_dim": z_dim, "aggregation": agg_name,
                        "ks_hat": ks_hat, "ks_true": ks_true,
                        "n_visited": int(visited.size),
                    })
                # Persist immediately so a crash mid-run doesn't lose work.
                _append_rows(incr_csv, new_rows)
                rows.extend(new_rows)
                done.add((float(noise), i, z_dim))
                print(
                    f"  inst {i+1:2d}/{len(cc)}  z={z_dim:2d}  "
                    f"ks_true={ks_true:+7.3f}  "
                    + "  ".join(
                        f"{a}={new_rows[ai]['ks_hat']:+7.3f}"
                        for ai, (a, _) in enumerate(AGGREGATIONS)
                    )
                    + f"   [{time.time()-t0:.1f}s]",
                    flush=True,
                )

    # Re-load the full incremental CSV so the summary covers any rows
    # already on disk from prior runs.
    df = pd.read_csv(incr_csv)
    print(f"\nIncremental CSV has {len(df):,} rows total ({incr_csv.name}).")

    # Pearson r per (z_dim, aggregation, noise) cell.
    summary_rows = []
    for noise, z_dim, (agg_name, _) in product(NOISES, Z_DIMS, AGGREGATIONS):
        cell = df[(df.noise == noise) & (df.z_dim == z_dim)
                   & (df.aggregation == agg_name)]
        r = pearson(cell.ks_hat, cell.ks_true)
        summary_rows.append({
            "z_dim": z_dim, "aggregation": agg_name, "noise": noise,
            "n_manifolds": len(cell),
            "pearson_r": r,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary_pearson.csv", index=False)
    print(f"Wrote {out_dir / 'summary_pearson.csv'}")

    print("\n========== Pearson r per (z_dim × aggregation × ε) ==========")
    print(f"{'z_dim':>5}  {'aggregation':>14}  {'ε=0.05':>9}  {'ε=0.10':>9}")
    for z_dim in Z_DIMS:
        for agg_name, _ in AGGREGATIONS:
            row = []
            for noise in NOISES:
                cell = summary[(summary.z_dim == z_dim)
                                & (summary.aggregation == agg_name)
                                & (summary.noise == noise)]
                r = float(cell.pearson_r.iloc[0])
                row.append(f"{r:+8.3f}")
            print(f"{z_dim:>5}  {agg_name:>14}  {row[0]:>9}  {row[1]:>9}")


if __name__ == "__main__":
    main()
