"""Benchmark all curvature methods on the *trajectory-sampled* Curvature
Colosseum (TauColosseum).

This is the trajectory-sampled analog to `benchmark_all_on_iid.py`. Instead
of feeding each method the full uniformly-sampled point cloud `X`, we feed
it `X[visited]` — the unique nodes touched by `n_trajectories` random walks
on a kNN graph over `X`. Successor methods train FB on the trajectories
themselves.

Sweep grid:
    intrinsic_dims = [2, 3, 4, 5, 6]
    codimensions   = [1]
    noise_levels   = [0.01, 0.05, 0.10, 0.20]
    num_manifolds  = 50  (per (dim, noise) cell → 1000 instances total)

Sharding: each worker processes instances where `idx % num_workers ==
worker_id`, writing to `processed_data/tau_metrics_w{K}.csv`. After all
workers finish, merge with `--cmd merge`.

Methods are imported from `benchmark_all_on_iid.py` so the implementations
match the iid benchmark exactly.

Usage:
    # Single worker (full sweep, slow)
    python benchmark_all_on_tau_colosseum.py --cmd run

    # Sharded across 8 workers (recommended)
    for K in 0 1 2 3 4 5 6 7; do
        python benchmark_all_on_tau_colosseum.py --cmd run \\
            --worker-id $K --num-workers 8 --device cuda:$((K % 2)) &
    done; wait
    python benchmark_all_on_tau_colosseum.py --cmd merge --num-workers 8

    # Test-run sanity check (~30s)
    python benchmark_all_on_tau_colosseum.py --cmd run --test-run
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")

import numpy as np
import pandas as pd

# Reuse all the iid method implementations + FB training logic.
import benchmark_all_on_iid as biv1
from benchmark_all_on_iid import (
    ALL_METHOD_NAMES,
    FB_KW,
    NON_SUCCESSOR_METHODS,
    IncrementalWriter,
    _pick_torch_device,
    already_done,
    run_successor_entropy,
    run_successor_orc,
)

from diffusion_curvature.successor.measures import compute_successor_measures
from diffusion_curvature.successor.train import FBTrainer
from diffusion_curvature.tau_datasets import TauColosseum


# ---------------------------------------------------------------------------
# Sweep config (production)
# ---------------------------------------------------------------------------

PROD_KW = dict(
    # Matched to the iid sweep's FB trajectory configuration
    # (`benchmark_all_on_iid.py` N_TRAJECTORIES=500, TRAJ_LENGTH=50, n=3000),
    # to give tau-side FB the same training signal as iid-side FB.
    n_samples=3000,
    n_trajectories=500,
    traj_length=50,
    intrinsic_dims=[2, 3, 4, 5, 6],
    codimensions=[1],
    noise_levels=[0.01, 0.05, 0.10, 0.20],
    num_manifolds_per_dim=50,
    knn=10,
)

TEST_KW = dict(
    n_samples=400,
    n_trajectories=40,
    traj_length=20,
    intrinsic_dims=[2, 3],
    codimensions=[1],
    noise_levels=[0.01, 0.20],
    num_manifolds_per_dim=2,
    knn=10,
)

DATASET_SEED = 17  # fixed across workers so they see the same instance order


# ---------------------------------------------------------------------------
# FB training adapted to TauColosseum trajectories
# ---------------------------------------------------------------------------


def train_fb_on_inst(inst: dict[str, Any], seed: int, device: str):
    """Train FB on the instance's pre-sampled trajectories, then compute
    successor measures over the full corpus."""
    traj = inst["trajectories"].astype(np.float32)
    trainer = FBTrainer(
        obs_dim=inst["X"].shape[1],
        **FB_KW,
        device=device,
        seed=seed,
    )
    trainer.fit(traj)
    M, F_emb, B_emb = compute_successor_measures(
        trainer.F_net, trainer.B_net, inst["X"], inst["X"], device=device,
    )
    return M, F_emb, B_emb


# ---------------------------------------------------------------------------
# Per-instance evaluation: trajectory-sampled
# ---------------------------------------------------------------------------


def _visited_subset(inst: dict[str, Any]) -> np.ndarray:
    return np.unique(np.asarray(inst["trajectories_idx"]).ravel())


def evaluate_instance_tau(
    idx: int,
    inst: dict[str, Any],
    meta: dict[str, Any],
    done: set[tuple[str, int, str]],
    writer: IncrementalWriter,
    device: str,
    seed: int,
) -> None:
    """Run all methods on one TauColosseum instance, writing one row per method."""
    visited = _visited_subset(inst)
    X_vis = inst["X"][visited].astype(np.float32)
    dim_for_method = int(meta["dim"])

    if X_vis.shape[0] < 12:
        # Too few visited nodes to do anything meaningful.
        for name in ALL_METHOD_NAMES:
            key = ("tau_colosseum", idx, name)
            if key in done:
                continue
            writer.write({
                "dataset": "tau_colosseum", "instance": idx, "method": name,
                "ks_hat": float("nan"), "elapsed_s": 0.0,
                "err": f"too few visited nodes ({X_vis.shape[0]})",
                **meta,
            })
        return

    # --- Non-successor methods on the visited point cloud ---
    for name, fn in NON_SUCCESSOR_METHODS:
        key = ("tau_colosseum", idx, name)
        if key in done:
            continue
        t0 = time.time()
        try:
            val = fn(X_vis, dim_for_method)
            err = ""
        except Exception as e:
            val = float("nan")
            err = str(e)[:200]
            print(f"  [err] tau_colosseum[{idx}] {name}: {err}")
        writer.write({
            "dataset": "tau_colosseum", "instance": idx, "method": name,
            "ks_hat": val, "elapsed_s": round(time.time() - t0, 2),
            "err": err, **meta,
        })

    # --- Successor methods on the full corpus, then read off at visited[0] ---
    needs = [s for s in ("successor_entropy", "successor_orc")
             if ("tau_colosseum", idx, s) not in done]
    if not needs:
        return
    t0 = time.time()
    try:
        M, F_emb, B_emb = train_fb_on_inst(inst, seed=seed + idx, device=device)
        fb_elapsed = time.time() - t0
    except Exception as e:
        err = str(e)[:200]
        print(f"  [err] tau_colosseum[{idx}] FB train: {err}")
        for s_name in needs:
            writer.write({
                "dataset": "tau_colosseum", "instance": idx, "method": s_name,
                "ks_hat": float("nan"), "elapsed_s": 0.0,
                "fb_elapsed_s": 0.0, "err": err, **meta,
            })
        return

    # Match the v2 IID aggregation: take the mean curvature over the visited
    # subset (with NaN filtering). This yields a single instance-level scalar
    # that lines up with `ks_true` for cell-level Pearson aggregation.
    runners = {
        "successor_entropy": lambda: _eval_successor_entropy_visited(M, F_emb, B_emb, visited),
        "successor_orc": lambda: _eval_successor_orc_visited(M, F_emb, B_emb, visited),
    }
    for s_name in needs:
        t1 = time.time()
        try:
            val = runners[s_name]()
            err = ""
        except Exception as e:
            val = float("nan")
            err = str(e)[:200]
            print(f"  [err] tau_colosseum[{idx}] {s_name}: {err}")
        writer.write({
            "dataset": "tau_colosseum", "instance": idx, "method": s_name,
            "ks_hat": val, "elapsed_s": round(time.time() - t1, 2),
            "fb_elapsed_s": round(fb_elapsed, 2), "err": err, **meta,
        })


def _eval_successor_entropy_visited(M, F_emb, B_emb, visited: np.ndarray) -> float:
    """Successor entropy: mean over the visited corpus subset, NaN-filtered."""
    from diffusion_curvature.successor import SuccessorEntropyCurvature
    k = SuccessorEntropyCurvature().fit_transform(M=M, F_embeddings=F_emb, B_embeddings=B_emb)
    k = np.asarray(k, dtype=float)[visited]
    k = k[np.isfinite(k)]
    return float(k.mean()) if k.size else float("nan")


def _eval_successor_orc_visited(M, F_emb, B_emb, visited: np.ndarray) -> float:
    """Successor ORC (B-ground): mean over the visited corpus subset."""
    from diffusion_curvature.successor import SuccessorORC
    orc = SuccessorORC(
        ground="B",
        k_neighbors=1,
        top_n=min(biv1.ORC_TOP_N, M.shape[0] - 1),
        n_projections=32,
        n_jobs=4,
    )
    k = orc.fit_transform(M=M, F_embeddings=F_emb, B_embeddings=B_emb)
    k = np.asarray(k, dtype=float)[visited]
    k = k[np.isfinite(k)]
    return float(k.mean()) if k.size else float("nan")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def iter_tau_colosseum(cc: TauColosseum):
    for i in range(len(cc)):
        inst = cc.get_item(i)
        meta = dict(
            name=cc.names[i],
            dim=int(inst["d"]),
            codim=int(inst["c"]),
            noise=float(inst["noise"]),
            m=int(inst["instance"]),
            ks_true=float(np.asarray(inst["ks"]).mean()),
            n_visited=int(_visited_subset(inst).shape[0]),
        )
        yield i, inst, meta


def _instance_filter(args: argparse.Namespace, meta: dict[str, Any]) -> bool:
    """True if this instance should be processed under the current sharding/cell flags."""
    if args.dim is not None and meta["dim"] != args.dim:
        return False
    if args.noise is not None and not np.isclose(meta["noise"], args.noise):
        return False
    return True


def cmd_run(args: argparse.Namespace) -> None:
    device = _pick_torch_device(args.device)
    print(f"Torch device: {device}")
    if args.dim is not None or args.noise is not None:
        print(f"Cell mode: dim={args.dim} noise={args.noise}")
    else:
        print(f"Worker {args.worker_id}/{args.num_workers}")
    print(f"Methods: {ALL_METHOD_NAMES}")

    out_path = Path(args.out)
    print(f"Output: {out_path}")

    done = already_done(out_path)
    if done:
        print(f"Resuming: {len(done)} (dataset, instance, method) rows present")
    writer = IncrementalWriter(out_path)

    def _graceful(signum, frame):
        print(f"\nReceived signal {signum}, exiting cleanly.")
        sys.exit(0)
    signal.signal(signal.SIGTERM, _graceful)
    signal.signal(signal.SIGINT, _graceful)

    kw = TEST_KW if args.test_run else PROD_KW
    save_dir = f"/tmp/.tau-cc-bench-{'test' if args.test_run else 'prod'}-{DATASET_SEED}"
    print(f"Building TauColosseum: {kw}")
    print(f"  cache: {save_dir}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cc = TauColosseum(seed=DATASET_SEED, save_directory=save_dir, **kw)
    print(f"  built {len(cc)} instances")

    n_processed = 0
    t_start = time.time()
    for i, inst, meta in iter_tau_colosseum(cc):
        # Two filtering modes: (a) cell mode via --dim/--noise, (b) shard mode
        # via --worker-id/--num-workers. Cell mode takes precedence.
        if args.dim is not None or args.noise is not None:
            if not _instance_filter(args, meta):
                continue
        else:
            if i % args.num_workers != args.worker_id:
                continue
        print(f"[{i+1}/{len(cc)}] {meta['name']} "
              f"d={meta['dim']} ε={meta['noise']:.2g} "
              f"n_vis={meta['n_visited']} ks_true={meta['ks_true']:+.3f}")
        evaluate_instance_tau(
            i, inst, meta, done, writer, device, seed=args.seed,
        )
        n_processed += 1
        if n_processed % 5 == 0:
            elapsed = time.time() - t_start
            rate = n_processed / max(elapsed, 1.0)
            print(f"  {n_processed} instances done, {rate*60:.1f} inst/min")

    print(f"\nDone. Wrote to {out_path}")


def cmd_merge(args: argparse.Namespace) -> None:
    shards: list[pd.DataFrame] = []
    if args.merge_inputs:
        # Explicit list of shard CSVs (used by Snakemake).
        for path_str in args.merge_inputs:
            p = Path(path_str)
            if p.exists() and p.stat().st_size > 0:
                shards.append(pd.read_csv(p))
            else:
                print(f"  skipping missing/empty shard: {p}")
    else:
        # Worker-template fallback (local sharded run).
        for k in range(args.num_workers):
            p = Path(args.out_template.format(wid=k))
            if p.exists() and p.stat().st_size > 0:
                shards.append(pd.read_csv(p))
            else:
                print(f"  skipping missing/empty shard: {p}")
    if not shards:
        raise SystemExit("No shards found.")
    merged = pd.concat(shards, ignore_index=True)
    merged = merged.drop_duplicates(subset=["dataset", "instance", "method"], keep="last")
    out = Path(args.merged_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"Merged {len(shards)} shards → {out}  ({len(merged)} rows)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cmd", choices=("run", "merge"), default="run")
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=1)
    # Cell mode: process only instances matching --dim and/or --noise.
    p.add_argument("--dim", type=int, default=None,
                   help="Restrict to one intrinsic dim (cell-mode).")
    p.add_argument("--noise", type=float, default=None,
                   help="Restrict to one noise level (cell-mode).")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-run", action="store_true")
    p.add_argument("--out", default="processed_data/tau_metrics_w0.csv",
                   help="Output CSV for `run`. In Snakemake-cell mode, set "
                   "this to the per-cell output path.")
    p.add_argument("--out-template",
                   default="processed_data/tau_metrics_w{wid}.csv",
                   help="Used by `merge` to find local sharded outputs when "
                   "--merge-inputs is not supplied.")
    p.add_argument("--merged-out",
                   default="processed_data/tau_metrics.csv")
    p.add_argument("--merge-inputs", nargs="*", default=None,
                   help="Explicit list of shard CSVs to merge (Snakemake).")
    args = p.parse_args()

    if args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "merge":
        cmd_merge(args)


if __name__ == "__main__":
    main()
