"""Sweep: Diffusion Laziness vs Successor Entropy on τ-Saddle-Sphere and τ-Colosseum.

For each (n_samples, n_trajectories, traj_length) config and each dataset
instance, evaluates both methods on the *visited* corpus subset (the
concatenation of trajectory nodes, deduplicated). Writes per-instance
summaries to `processed_data/metrics.csv`.

Each row records the per-instance scalar summary of the method output
(`ks_hat_mean`, `ks_hat_std`) and the per-instance ground truth
(`ks_true_scalar` for Colosseum, sphere/saddle `label` for SadSpheres).
Config-level metrics (Spearman for Colosseum, AUC for SadSpheres) are
computed at plot time by aggregating across instances.

Run via Snakefile (uses `test_run` flag there) or directly:
    pixi run python main.py --test-run
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Keep JAX from pre-allocating all VRAM (it shares the card with torch).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")

import numpy as np
import pandas as pd
import pygsp

from diffusion_curvature.diffusion_laziness import DiffusionLaziness
from diffusion_curvature.successor import (
    SuccessorEntropyCurvature,
    SuccessorORC,
)
from diffusion_curvature.successor.measures import compute_successor_measures
from diffusion_curvature.successor.train import FBTrainer
from diffusion_curvature.tau_datasets import TauColosseum, TauSadSpheres


def _pick_torch_device(requested: str | None) -> str:
    """Resolve `requested` into a concrete torch device string."""
    import torch
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        # Use cuda:1 by default so JAX (laziness) keeps cuda:0.
        return "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0"
    return "cpu"


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepConfig:
    n_samples_list: tuple[int, ...]
    n_trajectories_list: tuple[int, ...]
    traj_length: int
    num_sadsphere_instances: int
    num_colosseum_per_dim: int
    colosseum_dims: tuple[int, ...]
    # Laziness grid: list of (name, graph_construction, t) tuples
    # graph_construction ∈ {"knn", "adaptive"}
    laziness_variants: tuple[tuple[str, str, int], ...]
    fb_epochs: int
    fb_z_dim: int
    fb_hidden: int
    fb_gamma: float
    orc_top_n: int = 150
    orc_k_neighbors: int = 1
    orc_n_projections: int = 32


_LAZINESS_GRID = (
    # (variant_name, graph_construction, t)
    ("laziness_knn_t3",        "knn",      3),
    ("laziness_knn_t5",        "knn",      5),
    ("laziness_knn_t10",       "knn",      10),
    ("laziness_adaptive_t5",   "adaptive", 5),
)

PROD = SweepConfig(
    n_samples_list=(500, 2000),
    n_trajectories_list=(100, 500),
    traj_length=50,
    num_sadsphere_instances=10,
    num_colosseum_per_dim=5,
    colosseum_dims=(2,),
    laziness_variants=_LAZINESS_GRID,
    fb_epochs=150,
    fb_z_dim=2,
    fb_hidden=256,
    fb_gamma=0.5,
)

TEST = SweepConfig(
    n_samples_list=(300, 600),
    n_trajectories_list=(30, 80),
    traj_length=20,
    num_sadsphere_instances=3,
    num_colosseum_per_dim=3,
    colosseum_dims=(2,),
    laziness_variants=(("laziness_knn_t3", "knn", 3),),
    fb_epochs=30,
    fb_z_dim=2,
    fb_hidden=64,
    fb_gamma=0.5,
)

# Method registry: maps method_name -> is_signed. Plot scripts use this
# to decide between AUC (ranking) and sign-accuracy for each method.
SIGNED_METHODS: set[str] = set()  # populated at import time from laziness grid
UNSIGNED_METHODS: set[str] = {"successor_entropy", "successor_orc"}
for name, _, _ in _LAZINESS_GRID:
    UNSIGNED_METHODS.add(name)


# ---------------------------------------------------------------------------
# Method evaluation on one dataset instance
# ---------------------------------------------------------------------------


def _visited_subset(inst: dict[str, Any]) -> np.ndarray:
    return np.unique(np.asarray(inst["trajectories_idx"]).ravel())


def _build_graph(
    X: np.ndarray, construction: str, knn: int,
) -> pygsp.graphs.Graph:
    """Build a PyGSP graph via the requested construction."""
    if construction == "knn":
        return pygsp.graphs.NNGraph(X, k=knn)
    if construction == "adaptive":
        import graphtools as gt
        return gt.Graph(X, knn=knn, use_pygsp=True, decay=40)
    raise ValueError(f"Unknown graph construction: {construction!r}")


def run_laziness(
    inst: dict[str, Any],
    t: int,
    construction: str = "knn",
    knn: int = 10,
) -> np.ndarray:
    """Negated entropic diffusion laziness on the visit-deduplicated corpus.

    `construction` chooses the PyGSP graph builder:
    - "knn":      unweighted k-nearest-neighbors (pygsp.NNGraph)
    - "adaptive": graphtools adaptive Gaussian kernel (decay=40)
    """
    visited = _visited_subset(inst)
    X_vis = inst["X"][visited]
    if X_vis.shape[0] <= knn + 1:
        return np.full(visited.shape[0], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        G_vis = _build_graph(X_vis, construction, knn)
        dl = DiffusionLaziness(laziness_method="Entropic")
        laz = np.asarray(dl.fit_transform(G_vis, ts=t)).squeeze()
    return -laz


def train_fb_once(
    inst: dict[str, Any],
    cfg: SweepConfig,
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train FB once on the instance's trajectories; return (M, F, B) over the full corpus."""
    traj_shape = inst["trajectories"].shape
    batch_size = min(512, max(8, traj_shape[0] * traj_shape[1] // 4))
    trainer = FBTrainer(
        obs_dim=inst["X"].shape[1],
        z_dim=cfg.fb_z_dim,
        hidden_dim=cfg.fb_hidden,
        gamma=cfg.fb_gamma,
        n_epochs=cfg.fb_epochs,
        batch_size=batch_size,
        device=device,
        seed=seed,
    )
    trainer.fit(inst["trajectories"])
    M, F_emb, B_emb = compute_successor_measures(
        trainer.F_net, trainer.B_net, inst["X"], inst["X"], device=device,
    )
    return M, F_emb, B_emb


def run_successor_entropy(
    inst: dict[str, Any],
    M: np.ndarray,
    F_emb: np.ndarray,
    B_emb: np.ndarray,
) -> np.ndarray:
    """Entropy curvature from precomputed successor measures; restricted to visited."""
    visited = _visited_subset(inst)
    sec = SuccessorEntropyCurvature()
    k_hat_full = sec.fit_transform(M=M, F_embeddings=F_emb, B_embeddings=B_emb)
    return np.asarray(k_hat_full)[visited]


def run_successor_orc(
    inst: dict[str, Any],
    cfg: SweepConfig,
    M: np.ndarray,
    F_emb: np.ndarray,
    B_emb: np.ndarray,
) -> np.ndarray:
    """ORC (B-ground) from precomputed successor measures; restricted to visited."""
    visited = _visited_subset(inst)
    orc = SuccessorORC(
        ground="B",
        k_neighbors=cfg.orc_k_neighbors,
        top_n=min(cfg.orc_top_n, M.shape[0] - 1),
        n_projections=cfg.orc_n_projections,
        n_jobs=4,
    )
    k_hat_full = orc.fit_transform(M=M, F_embeddings=F_emb, B_embeddings=B_emb)
    return np.asarray(k_hat_full)[visited]


# ---------------------------------------------------------------------------
# Per-instance summary helpers
# ---------------------------------------------------------------------------


def _summarize_hat(ks_hat: np.ndarray) -> tuple[float, float]:
    v = np.asarray(ks_hat, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std())


# ---------------------------------------------------------------------------
# Sweep execution
# ---------------------------------------------------------------------------


def sweep_dataset(
    dataset_name: str,
    cfg: SweepConfig,
    seed_base: int,
    device: str = "cpu",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for n_samples in cfg.n_samples_list:
        for n_traj in cfg.n_trajectories_list:
            if dataset_name == "sadspheres":
                ds: Any = TauSadSpheres(
                    n_samples=n_samples,
                    n_trajectories=n_traj,
                    traj_length=cfg.traj_length,
                    dimension=2,
                    num_pointclouds=cfg.num_sadsphere_instances,
                    knn=min(10, n_samples - 2),
                    seed=seed_base,
                    save_directory=f"/tmp/.tau-ss-{n_samples}-{n_traj}",
                )
            elif dataset_name == "colosseum":
                ds = TauColosseum(
                    n_samples=n_samples,
                    n_trajectories=n_traj,
                    traj_length=cfg.traj_length,
                    intrinsic_dims=list(cfg.colosseum_dims),
                    codimensions=[1],
                    noise_levels=[0.0],
                    num_manifolds_per_dim=cfg.num_colosseum_per_dim,
                    knn=min(10, n_samples - 2),
                    seed=seed_base,
                    save_directory=f"/tmp/.tau-cc-{n_samples}-{n_traj}",
                )
            else:
                raise ValueError(dataset_name)

            for i in range(len(ds)):
                inst = ds.get_item(i)
                ks_true = inst["ks"][_visited_subset(inst)]
                knn_eval = min(10, len(ks_true) - 2) if len(ks_true) > 12 else 5

                # Train FB once; both successor methods reuse the result.
                try:
                    M, F_emb, B_emb = train_fb_once(
                        inst, cfg, seed=seed_base + i, device=device,
                    )
                    fb_ok = True
                except Exception as e:
                    print(f"[err] FB train n_s={n_samples} n_t={n_traj} inst={i}: {e}")
                    M = F_emb = B_emb = None
                    fb_ok = False

                method_runners: list[tuple[str, Any]] = [
                    (name, (lambda c=construction, tt=t: run_laziness(inst, tt, construction=c, knn=knn_eval)))
                    for name, construction, t in cfg.laziness_variants
                ]
                method_runners += [
                    ("successor_entropy", lambda: run_successor_entropy(inst, M, F_emb, B_emb) if fb_ok else np.full_like(ks_true, np.nan, dtype=float)),
                    ("successor_orc", lambda: run_successor_orc(inst, cfg, M, F_emb, B_emb) if fb_ok else np.full_like(ks_true, np.nan, dtype=float)),
                ]

                for method_name, runner in method_runners:
                    try:
                        ks_hat = runner()
                    except Exception as e:
                        print(f"[err] {dataset_name} {method_name} n_s={n_samples} n_t={n_traj} inst={i}: {e}")
                        ks_hat = np.full_like(ks_true, np.nan, dtype=float)

                    ks_hat_mean, ks_hat_std = _summarize_hat(ks_hat)
                    ks_true_scalar = float(np.mean(ks_true))
                    row: dict[str, Any] = dict(
                        dataset=dataset_name,
                        method=method_name,
                        n_samples=n_samples,
                        n_trajectories=n_traj,
                        instance=i,
                        name=ds.names[i],
                        ks_hat_mean=ks_hat_mean,
                        ks_hat_std=ks_hat_std,
                        ks_true_scalar=ks_true_scalar,
                    )
                    if dataset_name == "sadspheres":
                        row["label"] = 1 if ks_true_scalar > 0 else 0
                    rows.append(row)

            print(f"[done] {dataset_name}  n_s={n_samples}  n_t={n_traj}  "
                  f"instances={len(ds)}")

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-run", action="store_true")
    parser.add_argument("--out", default="processed_data/metrics.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="auto",
        help="Torch device for FB training. 'auto' (default) → cuda:1 if two GPUs, else cuda:0, else cpu.",
    )
    args = parser.parse_args()

    cfg = TEST if args.test_run else PROD
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    device = _pick_torch_device(args.device)
    print(f"Torch device: {device}")
    print(f"Config: {cfg}")

    dfs = []
    for ds in ("sadspheres", "colosseum"):
        dfs.append(sweep_dataset(ds, cfg, seed_base=args.seed, device=device))
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows → {out}")


if __name__ == "__main__":
    main()
