"""Benchmark all curvature methods on the full Curvature Colosseum + SadSpheres.

Per-instance: computes a scalar curvature estimate per method (at `idx=0`
where applicable, mean-over-node-0's-incident-edges for edge methods).
Rows are appended to `processed_data/iid_metrics.csv` with incremental
checkpointing: if you kill the script and restart, completed
`(dataset, instance, method)` triples are skipped.

Methods (13):
    1.  DC2 (signed, entropic, subtraction, curvature-agnostic kernel)
    2.  DC (Wasserstein + Ollivier, fixed kernel)
    3.  DC (Entropic + Subtraction, fixed kernel)
    4.  Hickock (scalar_curvature_est, rmax=2, idx=0)
    5.  FRC (weighted Forman-Ricci on adaptive graph)
    6.  FRC (unweighted Forman-Ricci on kNN graph)
    7.  ORC (Ollivier-Ricci on kNN graph via GraphRicciCurvature)
    8.  SPC-hop (Steinerberger, hop distances)
    9.  SPC-diffusion-t5 (Steinerberger, diffusion distances at t=5)
    10. Laziness (Entropic, kNN, t=5)  [unsigned]
    11. Laziness (Entropic, adaptive, t=5)  [unsigned]
    12. Successor Entropy  [unsigned]
    13. Successor ORC (B)  [unsigned]

Datasets:
    - SadSpheres: dims=[2,3,4,5,6], 20 pointclouds × 3 shapes × 5 dims = 300 instances
    - Colosseum:  dims=[2,3,4,5], codims=[1,2,3,4], 6 noise levels, 50 manifolds = 4800 instances

Runtime: multi-hour to overnight. Full checkpointing on each row.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")

import networkx as nx
import numpy as np
import pandas as pd
import pygsp
from sklearn.neighbors import NearestNeighbors

from diffusion_curvature.baselines.steinerberger import SteinerbergerCurvature
from diffusion_curvature.benchmarking.hickok import scalar_curvature_est
from diffusion_curvature.colosseum import CurvatureColosseum
from diffusion_curvature.core import (
    DiffusionCurvature,
    DiffusionCurvature2,
    get_fixed_graph,
)
from diffusion_curvature.diffusion_laziness import DiffusionLaziness
from diffusion_curvature.kernels import tune_curvature_agnostic_kernel
from diffusion_curvature.sadspheres import SadSpheres
from diffusion_curvature.successor import SuccessorEntropyCurvature, SuccessorORC
from diffusion_curvature.successor.measures import compute_successor_measures
from diffusion_curvature.successor.train import FBTrainer
from diffusion_curvature.trajectory_utils import subsample_trajectories


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SADSPHERES_KW = dict(
    dimension=[2, 3, 4, 5, 6],
    num_pointclouds=20,
    num_points=2000,
    noise_level=0,
    include_planes=True,
)

COLOSSEUM_KW = dict(
    intrinsic_dims=[2, 3, 4, 5],
    codimensions=[1, 2, 3, 4],
    noise_levels=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
    num_manifolds_per_dim=50,
    # Uniform corpus size regardless of dim — keeps O(n^k) methods tractable
    n_samples_rule=lambda d: 3000,
)

# SPC is O(n^3) (Floyd-Warshall + eigendecomposition). At n=3000, d=5,6 cells
# overshoot any realistic SLURM budget — d=4 fits within ~3-5h though, so we
# cap at 4 to gain one new dim of SPC coverage while keeping cells tractable.
SPC_MAX_DIM = 4

FB_KW = dict(
    # v2-equivalent settings (match `regen_successor_v2.py`) with epochs lifted
    # past 1000 to be on the safe side of FB convergence.
    z_dim=16, hidden_dim=256, gamma=0.9,
    n_epochs=1200, cosine_lr=True, batch_size=1024,
)
N_TRAJECTORIES = 500
TRAJ_LENGTH = 50
KNN = 10
ORC_TOP_N = 150
DC2_K = 40
DC2_ALPHA = 0
DC2_TS = list(range(1, 80))
DC_SIGNED_T = 25
DC_SIGNED_SIGMA = 0.2
DC_SIGNED_ALPHA = 1
LAZINESS_T = 5


# ---------------------------------------------------------------------------
# Device pickers
# ---------------------------------------------------------------------------


def _pick_torch_device(req: str) -> str:
    import torch
    if req and req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0"
    return "cpu"


# ---------------------------------------------------------------------------
# Method implementations (all return a scalar)
# ---------------------------------------------------------------------------


def _nx_knn_graph(X: np.ndarray, k: int, weighted: bool = False) -> nx.Graph:
    n = X.shape[0]
    k_eff = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean").fit(X)
    dists, indices = nn.kneighbors(X)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j_rank, j in enumerate(indices[i, 1:], 1):
            w = float(dists[i, j_rank])
            if weighted:
                G.add_edge(i, int(j), weight=w if w > 0 else 1e-6)
            else:
                G.add_edge(i, int(j))
    return G


def _graphtools_graph(X: np.ndarray, knn: int, decay: float = 40):
    import graphtools as gt
    return gt.Graph(X, knn=knn, use_pygsp=True, decay=decay)


def _edge_agg_at_node0(G: nx.Graph, attr: str = "curvature") -> float:
    vals = [G[0][j].get(attr, np.nan) for j in G.neighbors(0)]
    finite = [v for v in vals if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


# --- DC2 (signed, entropic, curvature-agnostic kernel) ---

def run_dc2(X: np.ndarray, dim: int) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph_former, _ = tune_curvature_agnostic_kernel(
            X, DC2_K, tolerance=3, max_iterations=100, alpha=DC2_ALPHA,
        )
        DC = DiffusionCurvature2(
            diffusion_type="diffusion matrix",
            laziness_method="Entropic",
            comparison_method="Subtraction",
            graph_former=graph_former,
        )
        k = DC.fit(X, dim, ts=DC2_TS, idx=0)
    return float(np.asarray(k).item())


# --- DC (Wasserstein + Ollivier, fixed kernel) ---

def _dc_fixed(X: np.ndarray, dim: int, laziness: str, comparison: str) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph_former = partial(get_fixed_graph, sigma=DC_SIGNED_SIGMA, alpha=DC_SIGNED_ALPHA)
        G = graph_former(X)
        DC = DiffusionCurvature(
            laziness_method=laziness,
            flattening_method="Fixed",
            comparison_method=comparison,
            graph_former=graph_former,
            points_per_cluster=None,
            comparison_space_size_factor=1,
        )
        ks = DC.curvature(G, dim=dim, t=DC_SIGNED_T)
    return float(np.asarray(ks)[0])


def run_dc_wasserstein_ollivier(X, dim): return _dc_fixed(X, dim, "Wasserstein", "Ollivier")
def run_dc_entropic_subtraction(X, dim): return _dc_fixed(X, dim, "Entropic", "Subtraction")


# --- Hickock ---

def run_hickock(X: np.ndarray, dim: int) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        SC = scalar_curvature_est(n=dim, X=X, verbose=False)
        ks_est = SC.estimate(rmax=2, indices=[0])
    return float(ks_est[0])


# --- Forman-Ricci ---

def _frc(X: np.ndarray, weighted: bool, adaptive: bool) -> float:
    from GraphRicciCurvature.FormanRicci import FormanRicci
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if adaptive:
            Ggt = _graphtools_graph(X, knn=KNN)
            W = Ggt.K.toarray() if hasattr(Ggt.K, "toarray") else np.asarray(Ggt.K)
            G = nx.Graph()
            n = W.shape[0]
            G.add_nodes_from(range(n))
            for i in range(n):
                for j in range(i + 1, n):
                    if W[i, j] > 1e-8:
                        G.add_edge(i, j, weight=float(W[i, j]))
        else:
            G = _nx_knn_graph(X, KNN, weighted=False)
        frc = FormanRicci(G, weight="weight") if weighted else FormanRicci(G)
        frc.compute_ricci_curvature()
    # node 0's edges carry formanCurvature attribute
    vals = [frc.G[0][j].get("formanCurvature", np.nan) for j in frc.G.neighbors(0)]
    finite = [v for v in vals if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def run_frc_weighted_adaptive(X, dim): return _frc(X, weighted=True, adaptive=True)
def run_frc_unweighted_knn(X, dim): return _frc(X, weighted=False, adaptive=False)


# --- Ollivier-Ricci ---

def run_orc(X: np.ndarray, dim: int) -> float:
    from GraphRicciCurvature.OllivierRicci import OllivierRicci
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        G = _nx_knn_graph(X, KNN, weighted=True)
        orc = OllivierRicci(G, alpha=0.5, verbose="ERROR")
        orc.compute_ricci_curvature()
    vals = [orc.G[0][j].get("ricciCurvature", np.nan) for j in orc.G.neighbors(0)]
    finite = [v for v in vals if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


# --- Steinerberger ---

def run_spc_hop(X: np.ndarray, dim: int) -> float:
    if dim > SPC_MAX_DIM:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k = SteinerbergerCurvature(distance="shortest-path", knn=KNN).fit_transform(X=X)
    return float(np.asarray(k)[0])


def run_spc_diffusion_t5(X: np.ndarray, dim: int) -> float:
    if dim > SPC_MAX_DIM:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k = SteinerbergerCurvature(distance="diffusion", knn=KNN, t=5).fit_transform(X=X)
    return float(np.asarray(k)[0])


# --- Laziness (unsigned) ---

def _laziness(X: np.ndarray, construction: str, t: int) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if construction == "knn":
            G = pygsp.graphs.NNGraph(X, k=KNN)
        elif construction == "adaptive":
            G = _graphtools_graph(X, knn=KNN)
        else:
            raise ValueError(construction)
        dl = DiffusionLaziness(laziness_method="Entropic")
        laz = np.asarray(dl.fit_transform(G, ts=t)).squeeze()
    return float(-laz[0])  # negated entropy, at idx=0


def run_laziness_knn_t5(X, dim): return _laziness(X, "knn", LAZINESS_T)
def run_laziness_adaptive_t5(X, dim): return _laziness(X, "adaptive", LAZINESS_T)


# --- Successor methods (share FB training) ---

def train_fb_once(X: np.ndarray, seed: int, device: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        G = pygsp.graphs.NNGraph(X, k=KNN)
    traj_idx = subsample_trajectories(
        G, n_trajectories=N_TRAJECTORIES, length=TRAJ_LENGTH, rng=seed,
    )
    traj = X[traj_idx].astype(np.float32)
    trainer = FBTrainer(
        obs_dim=X.shape[1],
        **FB_KW,
        device=device,
        seed=seed,
    )
    trainer.fit(traj)
    M, F_emb, B_emb = compute_successor_measures(
        trainer.F_net, trainer.B_net, X, X, device=device,
    )
    return M, F_emb, B_emb


def run_successor_entropy(M, F_emb, B_emb) -> float:
    """Mean (over the corpus) successor-entropy curvature, NaN-filtered."""
    k = SuccessorEntropyCurvature().fit_transform(M=M, F_embeddings=F_emb, B_embeddings=B_emb)
    k = np.asarray(k, dtype=float)
    k = k[np.isfinite(k)]
    return float(k.mean()) if k.size else float("nan")


def run_successor_orc(M, F_emb, B_emb) -> float:
    """Mean (over the corpus) successor-ORC curvature, NaN-filtered."""
    orc = SuccessorORC(
        ground="B",
        k_neighbors=1,
        top_n=min(ORC_TOP_N, M.shape[0] - 1),
        n_projections=32,
        n_jobs=4,
    )
    k = orc.fit_transform(M=M, F_embeddings=F_emb, B_embeddings=B_emb)
    k = np.asarray(k, dtype=float)
    k = k[np.isfinite(k)]
    return float(k.mean()) if k.size else float("nan")


# Non-successor method registry
NON_SUCCESSOR_METHODS: list[tuple[str, Callable[[np.ndarray, int], float]]] = [
    ("dc2",                        run_dc2),
    ("dc_wasserstein_ollivier",    run_dc_wasserstein_ollivier),
    ("dc_entropic_subtraction",    run_dc_entropic_subtraction),
    ("hickock",                    run_hickock),
    ("frc_weighted_adaptive",      run_frc_weighted_adaptive),
    ("frc_unweighted_knn",         run_frc_unweighted_knn),
    ("orc",                        run_orc),
    ("spc_hop",                    run_spc_hop),
    ("spc_diffusion_t5",           run_spc_diffusion_t5),
    ("laziness_knn_t5",            run_laziness_knn_t5),
    ("laziness_adaptive_t5",       run_laziness_adaptive_t5),
]


ALL_METHOD_NAMES = [m[0] for m in NON_SUCCESSOR_METHODS] + ["successor_entropy", "successor_orc"]


# ---------------------------------------------------------------------------
# Checkpointed row writer
# ---------------------------------------------------------------------------


_ROW_SCHEMA = [
    "dataset", "instance", "method", "ks_hat", "elapsed_s", "fb_elapsed_s", "err",
    "name", "dim", "codim", "noise", "m", "shape", "ks_true",
]


class IncrementalWriter:
    """Append-one-row-at-a-time CSV writer with a fixed schema."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = path.exists() and path.stat().st_size > 0

    def write(self, row: dict[str, Any]) -> None:
        # Normalize to fixed column list so all rows align in the CSV.
        normalized = {c: row.get(c, "") for c in _ROW_SCHEMA}
        df = pd.DataFrame([normalized], columns=_ROW_SCHEMA)
        df.to_csv(
            self.path, mode="a", index=False,
            header=not self._header_written,
        )
        self._header_written = True


_EXPECTED_COLS = {"dataset", "instance", "method", "ks_hat", "elapsed_s"}


def already_done(csv_path: Path) -> set[tuple[str, int, str]]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    if not _EXPECTED_COLS.issubset(set(header)):
        raise ValueError(
            f"{csv_path} has columns {header} which do not match the expected "
            f"benchmark schema. Move or delete it before resuming."
        )
    df = pd.read_csv(csv_path, usecols=["dataset", "instance", "method"])
    return set(zip(df["dataset"], df["instance"].astype(int), df["method"]))


# ---------------------------------------------------------------------------
# Instance evaluation
# ---------------------------------------------------------------------------


def evaluate_instance(
    dataset: str,
    idx: int,
    X: np.ndarray,
    dim_for_method: int,
    meta: dict[str, Any],
    done: set[tuple[str, int, str]],
    writer: IncrementalWriter,
    device: str,
    seed: int,
) -> None:
    """Run all methods on one instance; write one row per method."""
    # --- Run non-successor methods one by one ---
    for name, fn in NON_SUCCESSOR_METHODS:
        key = (dataset, idx, name)
        if key in done:
            continue
        t0 = time.time()
        try:
            val = fn(X, dim_for_method)
        except Exception as e:
            val = float("nan")
            err = str(e)[:200]
            print(f"  [err] {dataset}[{idx}] {name}: {err}")
        else:
            err = ""
        row = {
            "dataset": dataset,
            "instance": idx,
            "method": name,
            "ks_hat": val,
            "elapsed_s": round(time.time() - t0, 2),
            "err": err,
            **meta,
        }
        writer.write(row)

    # --- Successor stack (shared FB training) ---
    if any((dataset, idx, s) not in done for s in ("successor_entropy", "successor_orc")):
        t0 = time.time()
        try:
            M, F_emb, B_emb = train_fb_once(X, seed=seed + idx, device=device)
            fb_elapsed = time.time() - t0
            for s_name, s_fn in (
                ("successor_entropy", run_successor_entropy),
                ("successor_orc",     run_successor_orc),
            ):
                key = (dataset, idx, s_name)
                if key in done:
                    continue
                t1 = time.time()
                try:
                    val = s_fn(M, F_emb, B_emb)
                    err = ""
                except Exception as e:
                    val = float("nan")
                    err = str(e)[:200]
                    print(f"  [err] {dataset}[{idx}] {s_name}: {err}")
                writer.write({
                    "dataset": dataset, "instance": idx, "method": s_name,
                    "ks_hat": val, "elapsed_s": round(time.time() - t1, 2),
                    "fb_elapsed_s": round(fb_elapsed, 2), "err": err, **meta,
                })
        except Exception as e:
            err = str(e)[:200]
            print(f"  [err] {dataset}[{idx}] FB train: {err}")
            for s_name in ("successor_entropy", "successor_orc"):
                if (dataset, idx, s_name) in done:
                    continue
                writer.write({
                    "dataset": dataset, "instance": idx, "method": s_name,
                    "ks_hat": float("nan"), "elapsed_s": 0.0,
                    "fb_elapsed_s": 0.0, "err": err, **meta,
                })


# ---------------------------------------------------------------------------
# Datasets → instance iterators
# ---------------------------------------------------------------------------


def iter_sadspheres_instances(ss: SadSpheres):
    for i in range(len(ss)):
        obj = ss.DS[i].obj
        X = np.asarray(obj["X"], dtype=np.float32)
        ks = obj["ks"]
        ks_scalar = float(np.mean(np.asarray(ks))) if not np.isscalar(ks) else float(ks)
        d = int(obj.get("d", 0))
        name = ss.names[i]
        meta = dict(
            name=name,
            dim=d,
            shape=name.split("-", 1)[1] if "-" in name else name,
            ks_true=ks_scalar,
        )
        # dim passed to methods = intrinsic dim of the manifold
        yield i, X, d, meta


def iter_colosseum_instances(cc: CurvatureColosseum):
    for i in range(len(cc)):
        obj = cc.DS[i].obj
        X = np.asarray(obj["X"], dtype=np.float32)
        d = int(obj["d"])
        meta = dict(
            name=cc.names[i],
            dim=d, codim=int(obj["c"]), noise=float(obj["noise"]), m=int(obj["m"]),
            ks_true=float(obj["ks"]),
        )
        yield i, X, d, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="processed_data/iid_metrics.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--datasets", default="sadspheres,colosseum",
                   help="Comma-separated: sadspheres,colosseum")
    p.add_argument("--test-run", action="store_true",
                   help="Tiny sub-config for pipeline verification")
    p.add_argument("--dim", type=int, default=None,
                   help="Restrict Colosseum sweep to one intrinsic dim "
                   "(SadSpheres unaffected).")
    p.add_argument("--noise", type=float, default=None,
                   help="Restrict Colosseum sweep to one noise level.")
    p.add_argument("--codim", type=int, default=None,
                   help="Restrict Colosseum sweep to one codimension.")
    p.add_argument("--num-manifolds", type=int, default=None,
                   help="Override num_manifolds_per_dim for the cell.")
    args = p.parse_args()

    device = _pick_torch_device(args.device)
    print(f"Torch device: {device}")
    print(f"Methods: {ALL_METHOD_NAMES}")

    out = Path(args.out)
    done = already_done(out)
    if done:
        print(f"Resuming: {len(done)} (dataset, instance, method) rows already present")
    writer = IncrementalWriter(out)

    # Graceful shutdown: flush-and-exit on SIGTERM/SIGINT (writer already flushes per-row)
    def _graceful(signum, frame):
        print(f"\nReceived signal {signum}, exiting cleanly.")
        sys.exit(0)
    signal.signal(signal.SIGTERM, _graceful)
    signal.signal(signal.SIGINT, _graceful)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    if "sadspheres" in datasets:
        print("\n=== SadSpheres ===")
        if args.test_run:
            ss_kw = dict(dimension=[2], num_pointclouds=2, num_points=200,
                         noise_level=0, include_planes=True)
        else:
            ss_kw = SADSPHERES_KW
        print(f"  params: {ss_kw}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ss = SadSpheres(**ss_kw)
        for i, X, dim, meta in iter_sadspheres_instances(ss):
            print(f"[SS {i+1}/{len(ss)}] {meta['name']} dim={dim} n={X.shape[0]}")
            evaluate_instance(
                "sadspheres", i, X, dim, meta, done, writer, device, args.seed,
            )

    if "colosseum" in datasets:
        print("\n=== Colosseum ===")
        if args.test_run:
            cc_kw = dict(intrinsic_dims=[2], codimensions=[1],
                         noise_levels=[0.0], num_manifolds_per_dim=2)
        else:
            cc_kw = dict(COLOSSEUM_KW)
            if args.dim is not None:
                cc_kw["intrinsic_dims"] = [args.dim]
            if args.noise is not None:
                cc_kw["noise_levels"] = [args.noise]
            if args.codim is not None:
                cc_kw["codimensions"] = [args.codim]
            if args.num_manifolds is not None:
                cc_kw["num_manifolds_per_dim"] = args.num_manifolds
        print(f"  params: {cc_kw}")
        cc = CurvatureColosseum(**cc_kw)
        for i, X, dim, meta in iter_colosseum_instances(cc):
            print(f"[CC {i+1}/{len(cc)}] {meta['name']} n={X.shape[0]} ks={meta['ks_true']:+.3f}")
            evaluate_instance(
                "colosseum", i, X, dim, meta, done, writer, device, args.seed,
            )

    print(f"\nDone. Wrote to {out}")


if __name__ == "__main__":
    main()
