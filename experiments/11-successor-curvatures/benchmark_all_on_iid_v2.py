"""Extended Colosseum benchmark: dims 2..6, noise {0.01, 0.05, 0.1, 0.2}, codim=1.

Reuses the method implementations from `benchmark_all_on_iid.py` (11 methods,
FRC variants excluded) and overrides:
  - COLOSSEUM_KW (dims 2..6, 4 noise levels)
  - FB_KW (v2: z_dim=16, γ=0.9, 600 epochs, cosine LR, batch=1024)
  - N_TRAJECTORIES=500 (was 200)

Successor rows aggregate mean-over-all-nodes (v2 convention), unlike the prior
script's `k[0]`.

Usage:
  python benchmark_all_on_iid_v2.py run --worker-id K --num-workers 8 --device cuda:X
  python benchmark_all_on_iid_v2.py merge --merge-inputs <shard CSVs>
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")

import numpy as np
import pandas as pd

# Inherit method implementations + helpers from the v1 script.
import benchmark_all_on_iid as bv1


# Override configs for v2 run.
COLOSSEUM_KW_V2 = dict(
    intrinsic_dims=[2, 3, 4, 5, 6],
    codimensions=[1],
    noise_levels=[0.01, 0.05, 0.1, 0.2],
    num_manifolds_per_dim=50,
    n_samples_rule=lambda d: 3000,
)

FB_KW_V2 = dict(
    z_dim=16, hidden_dim=256, gamma=0.9, n_epochs=600,
    cosine_lr=True, batch_size=1024,
)
N_TRAJECTORIES_V2 = 500
TRAJ_LENGTH_V2 = 50

# Mutate the v1 module so its non-FB methods pick up v2 values where relevant.
# NB: FB_KW must NOT include batch_size, because bv1.train_fb_once passes it
# separately. We override train_fb_once entirely to use v2 settings cleanly.
bv1.FB_KW = dict(z_dim=16, hidden_dim=256, gamma=0.9, n_epochs=600, cosine_lr=True)
bv1.N_TRAJECTORIES = N_TRAJECTORIES_V2
bv1.TRAJ_LENGTH = TRAJ_LENGTH_V2


def _train_fb_once_v2(X, seed, device):
    """Replace bv1.train_fb_once with v2 recipe (fixed batch_size=1024)."""
    import pygsp as _pygsp
    from diffusion_curvature.trajectory_utils import subsample_trajectories as _st
    from diffusion_curvature.successor.measures import compute_successor_measures as _csm
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        G = _pygsp.graphs.NNGraph(X, k=bv1.KNN)
    traj_idx = _st(G, n_trajectories=N_TRAJECTORIES_V2, length=TRAJ_LENGTH_V2, rng=seed)
    traj = X[traj_idx].astype(np.float32)
    trainer = FBTrainer(
        obs_dim=X.shape[1],
        z_dim=16, hidden_dim=256, gamma=0.9, n_epochs=600,
        cosine_lr=True, batch_size=1024,
        device=device, seed=seed,
    )
    trainer.fit(traj, log_every=None)
    return _csm(trainer.F_net, trainer.B_net, X, X, device=device)


bv1.train_fb_once = _train_fb_once_v2

_ROW_SCHEMA = [
    "dataset", "instance", "method", "ks_hat", "elapsed_s", "fb_elapsed_s",
    "err", "name", "dim", "codim", "noise", "m", "shape", "ks_true",
]

# Non-successor methods, FRC excluded.
NON_SUCCESSOR_METHODS = [
    (name, fn) for name, fn in bv1.NON_SUCCESSOR_METHODS
    if not name.startswith("frc_")
]


def _mean_finite_curvature(k) -> float:
    arr = np.asarray(k, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def run_successor_entropy_mean(M, F_emb, B_emb) -> float:
    k = bv1.SuccessorEntropyCurvature().fit_transform(
        M=M, F_embeddings=F_emb, B_embeddings=B_emb
    )
    return _mean_finite_curvature(k)


def run_successor_orc_mean(M, F_emb, B_emb) -> float:
    k = bv1.SuccessorORC(
        ground="B", k_neighbors=1,
        top_n=min(bv1.ORC_TOP_N, M.shape[0] - 1),
        n_projections=32, n_jobs=4,
    ).fit_transform(M=M, F_embeddings=F_emb, B_embeddings=B_emb)
    return _mean_finite_curvature(k)


def _existing_done(csv_path: Path) -> set[tuple[str, int, str]]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    df = pd.read_csv(csv_path, usecols=["dataset", "instance", "method"])
    return set(zip(df["dataset"], df["instance"].astype(int), df["method"]))


def _write_row(csv_path: Path, row: dict, header_written: bool) -> bool:
    complete = {k: row.get(k, "") for k in _ROW_SCHEMA}
    pd.DataFrame([complete]).to_csv(
        csv_path, mode="a", index=False, header=not header_written,
    )
    return True


def _pick_torch_device(req: str) -> str:
    import torch
    if req and req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0"
    return "cpu"


def run_worker(args: argparse.Namespace) -> None:
    device = _pick_torch_device(args.device)
    print(f"[worker {args.worker_id}/{args.num_workers}] device={device}", flush=True)
    print(f"FB v2: {FB_KW_V2} | n_traj={N_TRAJECTORIES_V2} T={TRAJ_LENGTH_V2}", flush=True)

    done = _existing_done(Path(args.reference))
    own = _existing_done(Path(args.out))
    done |= own
    print(f"  resume: {len(done)} done triples "
          f"(ref={len(done) - len(own)}, own_shard={len(own)})", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header_written = out.exists() and out.stat().st_size > 0

    print("Building CurvatureColosseum (may take a bit)…", flush=True)
    cc = bv1.CurvatureColosseum(**COLOSSEUM_KW_V2)
    total = len(cc)
    assigned = sum(1 for i in range(total) if i % args.num_workers == args.worker_id)
    print(f"  {total} total instances, worker claims {assigned}", flush=True)

    t0 = time.time()
    processed = 0
    for local_idx, i in enumerate(
        (j for j in range(total) if j % args.num_workers == args.worker_id), start=1,
    ):
        obj = cc.DS[i].obj
        X = np.asarray(obj["X"], dtype=np.float32)
        d = int(obj["d"])
        meta = dict(
            name=cc.names[i],
            dim=d, codim=int(obj["c"]),
            noise=float(obj["noise"]), m=int(obj["m"]),
            ks_true=float(obj["ks"]),
        )
        dataset = "colosseum"

        # Non-successor methods
        non_succ_t = time.time()
        for name, fn in NON_SUCCESSOR_METHODS:
            if (dataset, i, name) in done:
                continue
            t_start = time.time()
            try:
                val = fn(X, d)
                err = ""
            except Exception as e:
                val = float("nan")
                err = str(e)[:200]
                print(f"  [err] {dataset}[{i}] {name}: {err}", flush=True)
            header_written = _write_row(out, {
                "dataset": dataset, "instance": i, "method": name,
                "ks_hat": val, "elapsed_s": round(time.time() - t_start, 2),
                "err": err, **meta,
            }, header_written)
        non_succ_elapsed = time.time() - non_succ_t

        # Successor stack (shared FB)
        need_se = (dataset, i, "successor_entropy") not in done
        need_so = (dataset, i, "successor_orc") not in done
        fb_elapsed = 0.0
        if need_se or need_so:
            fb_t = time.time()
            try:
                M, F_emb, B_emb = bv1.train_fb_once(X, seed=args.seed + i, device=device)
                fb_elapsed = time.time() - fb_t
                for s_name, s_fn in (
                    ("successor_entropy", run_successor_entropy_mean),
                    ("successor_orc",     run_successor_orc_mean),
                ):
                    if (dataset, i, s_name) in done:
                        continue
                    t_start = time.time()
                    try:
                        val = s_fn(M, F_emb, B_emb)
                        err = ""
                    except Exception as e:
                        val = float("nan")
                        err = str(e)[:200]
                        print(f"  [err] {dataset}[{i}] {s_name}: {err}", flush=True)
                    header_written = _write_row(out, {
                        "dataset": dataset, "instance": i, "method": s_name,
                        "ks_hat": val,
                        "elapsed_s": round(time.time() - t_start, 2),
                        "fb_elapsed_s": round(fb_elapsed, 2),
                        "err": err, **meta,
                    }, header_written)
            except Exception as e:
                err = str(e)[:200]
                print(f"  [err] {dataset}[{i}] FB train: {err}", flush=True)
                for s_name in ("successor_entropy", "successor_orc"):
                    if (dataset, i, s_name) in done:
                        continue
                    header_written = _write_row(out, {
                        "dataset": dataset, "instance": i, "method": s_name,
                        "ks_hat": float("nan"), "elapsed_s": 0.0,
                        "fb_elapsed_s": 0.0, "err": err, **meta,
                    }, header_written)

        processed += 1
        elapsed_total = time.time() - t0
        rate_per_min = processed / max(elapsed_total / 60.0, 1e-9)
        eta_min = (assigned - local_idx) / max(rate_per_min, 1e-9)
        print(f"  [w{args.worker_id}] {local_idx:3d}/{assigned} "
              f"inst={i:4d} d={d} ε={meta['noise']:.2f} ks={meta['ks_true']:+.2f}  "
              f"non_succ={non_succ_elapsed:.1f}s fb={fb_elapsed:.1f}s  "
              f"rate={rate_per_min:.2f}/min  eta={eta_min:.1f}min", flush=True)

    print(f"\n[worker {args.worker_id}] finished in {time.time() - t0:.1f}s → {out}",
          flush=True)


def run_merge(args: argparse.Namespace) -> None:
    frames = []
    ref = Path(args.reference)
    if ref.exists():
        frames.append(pd.read_csv(ref).assign(_source="ref"))
    for p in args.merge_inputs:
        pth = Path(p)
        if not pth.exists():
            print(f"  [skip] missing: {pth}")
            continue
        frames.append(pd.read_csv(pth).assign(_source=f"shard:{pth.name}"))
    if not frames:
        print("Nothing to merge."); return
    df = pd.concat(frames, ignore_index=True)
    df["_rank"] = (df["_source"] != "ref").astype(int)
    df = df.sort_values(["dataset", "instance", "method", "_rank"])
    df = df.drop_duplicates(["dataset", "instance", "method"], keep="last")
    df = df.drop(columns=["_rank", "_source"])
    cols = [c for c in _ROW_SCHEMA if c in df.columns] + \
           [c for c in df.columns if c not in _ROW_SCHEMA]
    df = df[cols]
    out = Path(args.merge_out or args.reference)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote merged CSV → {out} ({len(df)} total rows)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("run", help="Run one worker shard")
    w.add_argument("--worker-id", type=int, required=True)
    w.add_argument("--num-workers", type=int, default=8)
    w.add_argument("--device", default="auto")
    w.add_argument("--seed", type=int, default=42)
    w.add_argument("--reference", default="processed_data/iid_metrics_v2.csv")
    w.add_argument("--out",
                   default="processed_data/iid_metrics_v2_bench_w{wid}.csv")

    m = sub.add_parser("merge", help="Merge shard CSVs into reference")
    m.add_argument("--reference", default="processed_data/iid_metrics_v2.csv")
    m.add_argument("--merge-inputs", nargs="+", required=True)
    m.add_argument("--merge-out", default=None)

    args = p.parse_args()
    if args.cmd == "run":
        args.out = args.out.format(wid=args.worker_id)
        run_worker(args)
    else:
        run_merge(args)


if __name__ == "__main__":
    main()
