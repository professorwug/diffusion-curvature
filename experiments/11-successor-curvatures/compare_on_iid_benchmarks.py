"""Evaluate Laziness, Successor Entropy, Successor ORC on the original
(iid-sampled) SadSpheres and Curvature Colosseum benchmarks.

Question: does Successor ORC's advantage on τ-datasets hold up on
the well-sampled benchmarks? If so, it's a promising new method.

Metrics:
- SadSpheres → (a) per-point sign accuracy (library convention, mainly useful
  for signed methods; unsigned methods land near 50%), and
  (b) sphere-vs-saddle AUC across instances on mean(ks_hat).
- Colosseum → Spearman across manifolds of mean(ks_hat) vs scalar curvature.

Successor methods sample trajectories from the graph internally so they can
consume iid point clouds with the same sklearn-style API.

Output: `processed_data/iid_metrics.csv`.
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")

import numpy as np
import pandas as pd
import pygsp
import scipy.stats
from sklearn.metrics import roc_auc_score

from diffusion_curvature.diffusion_laziness import DiffusionLaziness
from diffusion_curvature.random_surfaces import samples_from_random_surface
from diffusion_curvature.sadspheres import SadSpheres
from diffusion_curvature.successor import SuccessorEntropyCurvature, SuccessorORC
from diffusion_curvature.successor.measures import compute_successor_measures
from diffusion_curvature.successor.train import FBTrainer
from diffusion_curvature.trajectory_utils import subsample_trajectories


def _pick_torch_device(requested: str) -> str:
    import torch
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0"
    return "cpu"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IIDBenchConfig:
    # SadSpheres
    ss_num_pointclouds: int
    ss_num_points: int
    ss_dimension: int
    # Colosseum
    cc_num_manifolds_per_dim: int
    cc_num_samples: int
    cc_intrinsic_dims: tuple[int, ...]
    cc_codimensions: tuple[int, ...]
    cc_noise_levels: tuple[float, ...]
    # Shared graph construction
    knn: int
    # Laziness variants: list of (name, construction, t)
    laziness_variants: tuple[tuple[str, str, int], ...]
    # FB trainer
    fb_epochs: int
    fb_z_dim: int
    fb_hidden: int
    fb_gamma: float
    n_trajectories: int
    traj_length: int
    # SuccessorORC
    orc_top_n: int
    orc_k_neighbors: int
    orc_n_projections: int


_LAZINESS_GRID = (
    ("laziness_knn_t3",      "knn",      3),
    ("laziness_knn_t5",      "knn",      5),
    ("laziness_knn_t10",     "knn",      10),
    ("laziness_adaptive_t5", "adaptive", 5),
)

PROD = IIDBenchConfig(
    ss_num_pointclouds=10,
    ss_num_points=500,
    ss_dimension=2,
    cc_num_manifolds_per_dim=5,
    cc_num_samples=2000,
    cc_intrinsic_dims=(2,),
    cc_codimensions=(1,),
    cc_noise_levels=(0.0,),
    knn=10,
    laziness_variants=_LAZINESS_GRID,
    fb_epochs=150,
    fb_z_dim=2,
    fb_hidden=256,
    fb_gamma=0.5,
    n_trajectories=200,
    traj_length=50,
    orc_top_n=150,
    orc_k_neighbors=1,
    orc_n_projections=32,
)

TEST = IIDBenchConfig(
    ss_num_pointclouds=2,
    ss_num_points=200,
    ss_dimension=2,
    cc_num_manifolds_per_dim=2,
    cc_num_samples=300,
    cc_intrinsic_dims=(2,),
    cc_codimensions=(1,),
    cc_noise_levels=(0.0,),
    knn=8,
    laziness_variants=(("laziness_knn_t3", "knn", 3),),
    fb_epochs=30,
    fb_z_dim=2,
    fb_hidden=64,
    fb_gamma=0.5,
    n_trajectories=30,
    traj_length=20,
    orc_top_n=40,
    orc_k_neighbors=1,
    orc_n_projections=16,
)


# ---------------------------------------------------------------------------
# Graph + method runners
# ---------------------------------------------------------------------------


def _build_graph(X: np.ndarray, construction: str, knn: int) -> pygsp.graphs.Graph:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if construction == "knn":
            return pygsp.graphs.NNGraph(X, k=knn)
        if construction == "adaptive":
            import graphtools as gt
            return gt.Graph(X, knn=knn, use_pygsp=True, decay=40)
    raise ValueError(construction)


def run_laziness(X: np.ndarray, construction: str, t: int, knn: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        G = _build_graph(X, construction, knn)
        dl = DiffusionLaziness(laziness_method="Entropic")
        laz = np.asarray(dl.fit_transform(G, ts=t)).squeeze()
    return -laz


def train_fb_once(
    X: np.ndarray, cfg: IIDBenchConfig, seed: int, device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build kNN graph, sample trajectories, train FB, return (M, F, B)."""
    G = _build_graph(X, "knn", cfg.knn)
    traj_idx = subsample_trajectories(
        G, n_trajectories=cfg.n_trajectories, length=cfg.traj_length, rng=seed,
    )
    traj = X[traj_idx].astype(np.float32)

    trainer = FBTrainer(
        obs_dim=X.shape[1],
        z_dim=cfg.fb_z_dim,
        hidden_dim=cfg.fb_hidden,
        gamma=cfg.fb_gamma,
        n_epochs=cfg.fb_epochs,
        batch_size=min(512, max(8, traj.shape[0] * traj.shape[1] // 4)),
        device=device,
        seed=seed,
    )
    trainer.fit(traj)
    M, F_emb, B_emb = compute_successor_measures(
        trainer.F_net, trainer.B_net, X, X, device=device,
    )
    return M, F_emb, B_emb


def run_successor_entropy(M: np.ndarray, F_emb: np.ndarray, B_emb: np.ndarray) -> np.ndarray:
    sec = SuccessorEntropyCurvature()
    return sec.fit_transform(M=M, F_embeddings=F_emb, B_embeddings=B_emb)


def run_successor_orc(
    cfg: IIDBenchConfig, M: np.ndarray, F_emb: np.ndarray, B_emb: np.ndarray,
) -> np.ndarray:
    orc = SuccessorORC(
        ground="B",
        k_neighbors=cfg.orc_k_neighbors,
        top_n=min(cfg.orc_top_n, M.shape[0] - 1),
        n_projections=cfg.orc_n_projections,
        n_jobs=4,
    )
    return orc.fit_transform(M=M, F_embeddings=F_emb, B_embeddings=B_emb)


# ---------------------------------------------------------------------------
# Evaluation loops
# ---------------------------------------------------------------------------


def _evaluate_instance(
    X: np.ndarray, cfg: IIDBenchConfig, seed: int, device: str,
) -> dict[str, np.ndarray]:
    results: dict[str, np.ndarray] = {}
    n = X.shape[0]

    for name, construction, t in cfg.laziness_variants:
        try:
            results[name] = np.asarray(run_laziness(X, construction, t, cfg.knn))
        except Exception as e:
            print(f"[err] {name}: {e}")
            results[name] = np.full(n, np.nan)

    try:
        M, F_emb, B_emb = train_fb_once(X, cfg, seed=seed, device=device)
        results["successor_entropy"] = np.asarray(run_successor_entropy(M, F_emb, B_emb))
        results["successor_orc"] = np.asarray(run_successor_orc(cfg, M, F_emb, B_emb))
    except Exception as e:
        print(f"[err] successor stack: {e}")
        results["successor_entropy"] = np.full(n, np.nan)
        results["successor_orc"] = np.full(n, np.nan)

    return results


def evaluate_sadspheres(cfg: IIDBenchConfig, device: str, seed: int = 42) -> pd.DataFrame:
    ss = SadSpheres(
        dimension=[cfg.ss_dimension],
        num_pointclouds=cfg.ss_num_pointclouds,
        num_points=cfg.ss_num_points,
        include_planes=False,
    )
    rows: list[dict[str, Any]] = []
    for i in range(len(ss)):
        X = np.asarray(ss.DS[i].obj["X"], dtype=np.float32)
        ks_true = np.asarray(ss.DS[i].obj["ks"]).astype(float)
        if np.isscalar(ks_true) or ks_true.ndim == 0:
            ks_true = np.full(X.shape[0], float(ks_true))
        label = 1 if float(np.mean(ks_true)) > 0 else 0
        name = ss.names[i]
        print(f"  [SS {i+1}/{len(ss)}] {name} (n={X.shape[0]})")

        preds = _evaluate_instance(X, cfg, seed=seed + i, device=device)
        for method, k_hat in preds.items():
            k = np.asarray(k_hat, dtype=float)
            finite = k[np.isfinite(k)]
            k_mean = float(np.mean(finite)) if finite.size else float("nan")
            # per-point sign accuracy (library convention); excludes ks_true=0 pts
            mask = (ks_true != 0) & np.isfinite(k)
            if mask.any():
                sign_acc = float(np.mean(np.sign(k[mask]) == np.sign(ks_true[mask])))
            else:
                sign_acc = float("nan")
            rows.append(dict(
                dataset="sadspheres",
                method=method,
                instance=i,
                name=name,
                label=label,
                ks_true_scalar=float(np.mean(ks_true)),
                ks_hat_mean=k_mean,
                sign_acc=sign_acc,
            ))
    return pd.DataFrame(rows)


def evaluate_colosseum(cfg: IIDBenchConfig, device: str, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    inst_idx = 0
    for d in cfg.cc_intrinsic_dims:
        for c in cfg.cc_codimensions:
            for noise in cfg.cc_noise_levels:
                N = d + c
                for m_i in range(cfg.cc_num_manifolds_per_dim):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        X, ks = samples_from_random_surface(
                            cfg.cc_num_samples, d, N, degree=2, noise_level=noise,
                        )
                    X = np.asarray(X, dtype=np.float32)
                    ks_true_scalar = float(np.mean(np.asarray(ks, dtype=float)))
                    name = f"d{d}-c{c}-n{noise}-m{m_i}"
                    print(f"  [CC {inst_idx+1}] {name} (n={X.shape[0]}, ks={ks_true_scalar:+.3f})")

                    preds = _evaluate_instance(X, cfg, seed=seed + inst_idx, device=device)
                    for method, k_hat in preds.items():
                        k = np.asarray(k_hat, dtype=float)
                        finite = k[np.isfinite(k)]
                        k_mean = float(np.mean(finite)) if finite.size else float("nan")
                        rows.append(dict(
                            dataset="colosseum",
                            method=method,
                            instance=inst_idx,
                            name=name,
                            d=d, c=c, noise=noise,
                            ks_true_scalar=ks_true_scalar,
                            ks_hat_mean=k_mean,
                        ))
                    inst_idx += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation (config-level summary)
# ---------------------------------------------------------------------------


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Compute dataset-level metrics per method."""
    summary_rows: list[dict[str, Any]] = []
    for method, g in df[df.dataset == "sadspheres"].groupby("method"):
        # sphere-vs-saddle AUC on mean(ks_hat)
        try:
            auc = float(roc_auc_score(g["label"].values, g["ks_hat_mean"].values))
        except (ValueError, KeyError):
            auc = float("nan")
        sign_acc = float(g["sign_acc"].mean()) if "sign_acc" in g else float("nan")
        summary_rows.append(dict(
            dataset="sadspheres",
            method=method,
            auc=auc,
            sign_acc=sign_acc,
            n_instances=len(g),
        ))
    for method, g in df[df.dataset == "colosseum"].groupby("method"):
        try:
            rho, _ = scipy.stats.spearmanr(g["ks_hat_mean"].values, g["ks_true_scalar"].values)
            rho = float(rho) if np.isfinite(rho) else float("nan")
        except Exception:
            rho = float("nan")
        summary_rows.append(dict(
            dataset="colosseum",
            method=method,
            spearman=rho,
            n_instances=len(g),
        ))
    return pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-run", action="store_true")
    p.add_argument("--out", default="processed_data/iid_metrics.csv")
    p.add_argument("--summary-out", default="processed_data/iid_summary.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    cfg = TEST if args.test_run else PROD
    device = _pick_torch_device(args.device)
    print(f"Torch device: {device}")
    print(f"Config: {cfg}")

    print("\n=== SadSpheres ===")
    df_ss = evaluate_sadspheres(cfg, device, seed=args.seed)
    print(f"\n=== Colosseum ===")
    df_cc = evaluate_colosseum(cfg, device, seed=args.seed)

    df = pd.concat([df_ss, df_cc], ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} rows → {out}")

    summary = summarize(df)
    sout = Path(args.summary_out)
    summary.to_csv(sout, index=False)
    print(f"\nSummary:\n{summary.to_string(index=False)}")
    print(f"Wrote {sout}")


if __name__ == "__main__":
    main()
