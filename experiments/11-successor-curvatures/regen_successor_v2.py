"""Re-generate the `successor_entropy` and `successor_orc` rows of
`processed_data/iid_metrics.csv` using the torus-verified v2 FB recipe.

v2 defaults:
  FB: z_dim=16, hidden_dim=256, gamma=0.9, n_epochs=600, cosine_lr=True, batch_size=1024
  N_TRAJECTORIES=500, TRAJ_LENGTH=50

One FB training per (dataset, instance) is shared between SE and SORC.
Output aggregation: per-instance scalar is mean-over-all-nodes of the
per-point curvature estimate (the prior benchmark used `k[0]` which is noisy;
v2 sweep showed mean-over-visited is the correct aggregator).

Shard-parallel by index:  --worker-id K --num-workers W  picks instances where
`global_idx % W == K`. Each worker can target a different GPU via --device.

Writes to `processed_data/iid_metrics_successor_v2_w{K}.csv`. Merge with
`--merge` after workers complete.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")

import numpy as np
import pandas as pd
import pygsp

from diffusion_curvature.colosseum import CurvatureColosseum
from diffusion_curvature.sadspheres import SadSpheres
from diffusion_curvature.successor import SuccessorEntropyCurvature, SuccessorORC
from diffusion_curvature.successor.measures import compute_successor_measures
from diffusion_curvature.successor.train import FBTrainer
from diffusion_curvature.trajectory_utils import subsample_trajectories


# ---------------------------------------------------------------------------
# v2 config
# ---------------------------------------------------------------------------

FB_KW = dict(
    z_dim=16, hidden_dim=256, gamma=0.9, n_epochs=600,
    cosine_lr=True, batch_size=1024,
)
N_TRAJECTORIES = 500
TRAJ_LENGTH = 50
KNN = 10
ORC_TOP_N = 150
ORC_N_PROJ = 32

# Must match benchmark_all_on_iid.py configs exactly for instance indices to align.
SADSPHERES_KW = dict(
    dimension=[2, 3, 4, 5, 6],
    num_pointclouds=20,
    num_points=2000,
    noise_level=0,
    include_planes=True,
)
# Must match the benchmark_all_on_iid_v2 config for instance indices to align.
COLOSSEUM_KW = dict(
    intrinsic_dims=[2, 3, 4, 5, 6],
    codimensions=[1],
    noise_levels=[0.01, 0.05, 0.1, 0.2],
    num_manifolds_per_dim=50,
    n_samples_rule=lambda d: 3000,
)


_ROW_SCHEMA = [
    "dataset", "instance", "method", "ks_hat", "elapsed_s", "fb_elapsed_s",
    "err", "name", "dim", "codim", "noise", "m", "shape", "ks_true",
]


def _pick_torch_device(req: str) -> str:
    import torch
    if req and req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0"
    return "cpu"


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
    trainer.fit(traj, log_every=None)
    return compute_successor_measures(
        trainer.F_net, trainer.B_net, X, X, device=device,
    )


def run_successor_entropy(M, F_emb, B_emb) -> float:
    k = SuccessorEntropyCurvature().fit_transform(M=M, F_embeddings=F_emb, B_embeddings=B_emb)
    k = np.asarray(k, dtype=float)
    k = k[np.isfinite(k)]
    return float(k.mean()) if k.size else float("nan")


def run_successor_orc(M, F_emb, B_emb) -> float:
    orc = SuccessorORC(
        ground="B",
        k_neighbors=1,
        top_n=min(ORC_TOP_N, M.shape[0] - 1),
        n_projections=ORC_N_PROJ,
        n_jobs=4,
    ).fit_transform(M=M, F_embeddings=F_emb, B_embeddings=B_emb)
    orc = np.asarray(orc, dtype=float)
    orc = orc[np.isfinite(orc)]
    return float(orc.mean()) if orc.size else float("nan")


def _write_row(csv_path: Path, row: dict, header_written: bool) -> bool:
    complete = {k: row.get(k, "") for k in _ROW_SCHEMA}
    pd.DataFrame([complete]).to_csv(csv_path, mode="a", index=False, header=not header_written)
    return True


def _existing_pairs(reference_csv: Path) -> set[tuple[str, int]]:
    if not reference_csv.exists():
        raise FileNotFoundError(f"Reference CSV not found: {reference_csv}")
    df = pd.read_csv(reference_csv, usecols=["dataset", "instance", "method"])
    se = df[df.method == "successor_entropy"]
    return set(zip(se["dataset"], se["instance"].astype(int)))


def iterate_sadspheres_instances():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ss = SadSpheres(**SADSPHERES_KW)
    for i in range(len(ss)):
        obj = ss.DS[i].obj
        X = np.asarray(obj["X"], dtype=np.float32)
        ks = obj["ks"]
        ks_scalar = float(np.mean(np.asarray(ks))) if not np.isscalar(ks) else float(ks)
        d = int(obj.get("d", 0))
        name = ss.names[i]
        meta = dict(
            name=name, dim=d,
            shape=name.split("-", 1)[1] if "-" in name else name,
            ks_true=ks_scalar,
        )
        yield i, X, meta


def iterate_colosseum_instances():
    cc = CurvatureColosseum(**COLOSSEUM_KW)
    for i in range(len(cc)):
        obj = cc.DS[i].obj
        X = np.asarray(obj["X"], dtype=np.float32)
        d = int(obj["d"])
        meta = dict(
            name=cc.names[i],
            dim=d, codim=int(obj["c"]), noise=float(obj["noise"]), m=int(obj["m"]),
            ks_true=float(obj["ks"]),
        )
        yield i, X, meta


def run_worker(args: argparse.Namespace) -> None:
    device = _pick_torch_device(args.device)
    print(f"[worker {args.worker_id}/{args.num_workers}] device={device}", flush=True)
    print(f"FB: {FB_KW}", flush=True)

    pairs = _existing_pairs(Path(args.reference))
    print(f"  reference has {len(pairs)} (dataset, instance) pairs with SE rows", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header_written = out.exists() and out.stat().st_size > 0
    if header_written:
        done = _existing_pairs(out) if out.exists() and out.stat().st_size > 0 else set()
        # This worker's already-done (dataset, instance) pairs (for resume):
        done_pairs = done
    else:
        done_pairs = set()

    global_idx = -1
    rows_written = 0
    datasets_to_run = [d.strip() for d in args.datasets.split(",") if d.strip()]

    iterators = []
    if "sadspheres" in datasets_to_run:
        iterators.append(("sadspheres", iterate_sadspheres_instances))
    if "colosseum" in datasets_to_run:
        iterators.append(("colosseum", iterate_colosseum_instances))

    for dataset, iter_fn in iterators:
        print(f"\n=== [{dataset}] ===", flush=True)
        for i, X, meta in iter_fn():
            if (dataset, i) not in pairs:
                continue
            global_idx += 1
            if global_idx % args.num_workers != args.worker_id:
                continue
            if (dataset, i) in done_pairs:
                continue
            fb_t = time.time()
            try:
                M, F_emb, B_emb = train_fb_once(
                    X, seed=args.seed + i, device=device,
                )
                fb_elapsed = time.time() - fb_t
                for name, fn in (
                    ("successor_entropy", run_successor_entropy),
                    ("successor_orc",     run_successor_orc),
                ):
                    t1 = time.time()
                    try:
                        val = fn(M, F_emb, B_emb)
                        err = ""
                    except Exception as e:
                        val = float("nan")
                        err = str(e)[:200]
                        print(f"  [err] {dataset}[{i}] {name}: {err}", flush=True)
                    header_written = _write_row(out, {
                        "dataset": dataset, "instance": i, "method": name,
                        "ks_hat": val, "elapsed_s": round(time.time() - t1, 2),
                        "fb_elapsed_s": round(fb_elapsed, 2), "err": err,
                        **meta,
                    }, header_written)
                    rows_written += 1
                print(f"  [{dataset} {i}] {meta.get('name', '')}  "
                      f"fb={fb_elapsed:.1f}s  SE+SORC done", flush=True)
            except Exception as e:
                err = str(e)[:200]
                print(f"  [err] {dataset}[{i}] FB train: {err}", flush=True)
                for name in ("successor_entropy", "successor_orc"):
                    header_written = _write_row(out, {
                        "dataset": dataset, "instance": i, "method": name,
                        "ks_hat": float("nan"), "elapsed_s": 0.0,
                        "fb_elapsed_s": 0.0, "err": err, **meta,
                    }, header_written)
                    rows_written += 1

    print(f"\n[worker {args.worker_id}] wrote {rows_written} rows → {out}", flush=True)


def run_merge(args: argparse.Namespace) -> None:
    frames = []
    for path in args.merge_inputs:
        p = Path(path)
        if not p.exists():
            print(f"  [skip] missing: {p}")
            continue
        frames.append(pd.read_csv(p))
    if not frames:
        print("No inputs to merge.")
        return
    new_rows = pd.concat(frames, ignore_index=True)
    new_rows = new_rows.drop_duplicates(["dataset", "instance", "method"], keep="last")
    print(f"Combined {len(new_rows)} v2 rows from {len(frames)} workers.")

    ref = pd.read_csv(args.reference)
    keep = ref[~ref.method.isin(["successor_entropy", "successor_orc"])]
    merged = pd.concat([keep, new_rows], ignore_index=True)
    merged = merged[_ROW_SCHEMA] if set(_ROW_SCHEMA).issubset(merged.columns) else merged
    out = Path(args.merge_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"Wrote merged CSV → {out} ({len(merged)} total rows)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    work = sub.add_parser("run", help="Run a worker shard.")
    work.add_argument("--worker-id", type=int, required=True)
    work.add_argument("--num-workers", type=int, default=2)
    work.add_argument("--device", default="auto")
    work.add_argument("--seed", type=int, default=42)
    work.add_argument("--datasets", default="sadspheres,colosseum")
    work.add_argument("--reference", default="processed_data/iid_metrics.csv")
    work.add_argument("--out",
                      default="processed_data/iid_metrics_successor_v2_w{wid}.csv")

    merge = sub.add_parser("merge", help="Merge worker CSVs with reference.")
    merge.add_argument("--reference", default="processed_data/iid_metrics.csv")
    merge.add_argument("--merge-inputs", nargs="+", required=True)
    merge.add_argument("--merge-out",
                       default="processed_data/iid_metrics_v2.csv")

    args = p.parse_args()
    if args.cmd == "run":
        args.out = args.out.format(wid=args.worker_id)
        run_worker(args)
    else:
        run_merge(args)


if __name__ == "__main__":
    main()
