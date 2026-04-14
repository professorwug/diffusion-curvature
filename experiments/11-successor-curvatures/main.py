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
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pygsp

from diffusion_curvature.diffusion_laziness import DiffusionLaziness
from diffusion_curvature.successor import SuccessorEntropyCurvature
from diffusion_curvature.tau_datasets import TauColosseum, TauSadSpheres


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
    laziness_t: int
    fb_epochs: int
    fb_z_dim: int
    fb_hidden: int
    fb_gamma: float


PROD = SweepConfig(
    n_samples_list=(500, 2000),
    n_trajectories_list=(50, 200, 500),
    traj_length=50,
    num_sadsphere_instances=20,
    num_colosseum_per_dim=10,
    colosseum_dims=(2,),
    laziness_t=5,
    fb_epochs=200,
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
    laziness_t=3,
    fb_epochs=30,
    fb_z_dim=2,
    fb_hidden=64,
    fb_gamma=0.5,
)


# ---------------------------------------------------------------------------
# Method evaluation on one dataset instance
# ---------------------------------------------------------------------------


def _visited_subset(inst: dict[str, Any]) -> np.ndarray:
    return np.unique(np.asarray(inst["trajectories_idx"]).ravel())


def run_laziness(inst: dict[str, Any], t: int, knn: int = 10) -> np.ndarray:
    """Negated entropic diffusion laziness on the visit-deduplicated corpus."""
    visited = _visited_subset(inst)
    X_vis = inst["X"][visited]
    if X_vis.shape[0] <= knn + 1:
        return np.full(visited.shape[0], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        G_vis = pygsp.graphs.NNGraph(X_vis, k=knn)
        dl = DiffusionLaziness(laziness_method="Entropic")
        laz = np.asarray(dl.fit_transform(G_vis, ts=t)).squeeze()
    return -laz


def run_successor_entropy(
    inst: dict[str, Any], cfg: SweepConfig, seed: int = 42,
) -> np.ndarray:
    """Successor entropy curvature on the full corpus; restricted to visited."""
    visited = _visited_subset(inst)
    traj_shape = inst["trajectories"].shape
    batch_size = min(512, max(8, traj_shape[0] * traj_shape[1] // 4))
    sec = SuccessorEntropyCurvature(
        z_dim=cfg.fb_z_dim,
        hidden_dim=cfg.fb_hidden,
        gamma=cfg.fb_gamma,
        n_epochs=cfg.fb_epochs,
        batch_size=batch_size,
        seed=seed,
    )
    k_hat_full = sec.fit_transform(X=inst["X"], trajectories=inst["trajectories"])
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

                for method_name in ("laziness", "successor_entropy"):
                    try:
                        if method_name == "laziness":
                            ks_hat = run_laziness(inst, cfg.laziness_t, knn=knn_eval)
                        else:
                            ks_hat = run_successor_entropy(inst, cfg, seed=seed_base + i)
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
    args = parser.parse_args()

    cfg = TEST if args.test_run else PROD
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Config: {cfg}")
    dfs = []
    for ds in ("sadspheres", "colosseum"):
        dfs.append(sweep_dataset(ds, cfg, seed_base=args.seed))
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows → {out}")


if __name__ == "__main__":
    main()
