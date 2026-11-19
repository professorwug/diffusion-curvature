"""The gESS profile area on the Curvature Colosseum, iid and trajectory-sampled.

One FB net is trained per manifold and regime, as in `benchmark_all_on_iid_v2.py`
(the iid recipe: 600 epochs, every node read) and `benchmark_all_on_tau_colosseum.py`
(the trajectory recipe: 1200 epochs, visited nodes read). From each net this
script stores the manifold's profile: the mean over its points of the log
geometric effective sample size at every kernel scale, and the median share of
the ceiling at every scale. `derive` then sets ONE convergence scale for each
cell of the battery (a dimension and a noise level, the 50 manifolds a Pearson
compares) and writes the area under it as a method row, beside the paired
successor entropy of the same nets.

The battery here is seeded per instance (`BATTERY_SEED + idx`), which the
earlier scripts' was not: both regimes see the same manifold at an index, and
any later method can be paired with these rows. The stored `iid_metrics_v2.csv`
and `tau_metrics.csv` were made on other draws of the battery, so a comparison
with them is between cells, not between manifolds.

One net is too small for a card, which would idle under it; `run` therefore
trains serially on CPU. `fleet` trains every pending manifold of a regime and
dimension at once on one card (`FleetFBTrainer`), the same fits up to
floating-point order at about a seventieth of the cost, and is the form for
iteration. Both write shards of one schema, and each resumes over the other's.

Usage:
    python benchmark_profile_area.py fleet --task-id K --num-tasks N [--fleet 200] [--max-minutes 50]
    python benchmark_profile_area.py run --worker-id K --num-workers N [--max-minutes 80]
    python benchmark_profile_area.py run --test-run
    python benchmark_profile_area.py derive
"""

from __future__ import annotations

import argparse
import glob
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

DIMS = [2, 3, 4, 5, 6]
NOISES = [0.01, 0.05, 0.10, 0.20]
N_MANIFOLDS = 50
N_SAMPLES = 3000
N_TRAJECTORIES = 500
TRAJ_LENGTH = 50
KNN = 10
BATTERY_SEED = 20261118
FB_SEED = 42
FB_BASE = dict(z_dim=16, hidden_dim=256, gamma=0.9, cosine_lr=True, batch_size=1024)
EPOCHS = {"colosseum": 600, "tau_colosseum": 1200}
OUT_DIR = Path(__file__).parent / "processed_data"
SHARD = "profile_area_w{k}.csv"
MERGED = "profile_area_profiles.csv"
METRICS = "profile_area_metrics.csv"
SUMMARY = "profile_area_pearson_summary.csv"


def cells(test_run: bool = False):
    """Every (idx, dim, noise, m) of the battery, in a fixed order."""
    dims, noises, n = (DIMS[:2], [NOISES[0], NOISES[-1]], 2) if test_run else (DIMS, NOISES, N_MANIFOLDS)
    return [(idx, d, e, m) for idx, (d, e, m) in enumerate((d, e, m) for d in dims for e in noises for m in range(n))]


def make_instance(idx: int, d: int, noise: float, n_samples: int):
    """The manifold at `idx`: a random quadric surface of dimension d in d + 1, its samples and its scalar curvature at the origin."""
    from diffusion_curvature.random_surfaces import samples_from_random_surface
    np.random.seed(BATTERY_SEED + idx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X, k = samples_from_random_surface(n_samples, d, d + 1, degree=2, noise_level=noise)
    return np.asarray(X, dtype=np.float32), float(np.mean(np.asarray(k, dtype=float)))


def walks(dataset: str, X: np.ndarray, idx: int, sizes: dict) -> np.ndarray:
    """The regime's random walks over the kNN graph of X, as node indices."""
    import pygsp
    from diffusion_curvature.trajectory_utils import subsample_trajectories
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        G = pygsp.graphs.NNGraph(X, k=KNN)
    walk_seed = FB_SEED + idx + (0 if dataset == "colosseum" else 10_000_019)
    return subsample_trajectories(G, n_trajectories=sizes["n_traj"], length=sizes["traj_len"], rng=walk_seed)


def read_measures(dataset: str, M, F_emb, B_emb, traj_idx, device: str) -> dict:
    """From one net's successor measures: the paired successor entropy and the profile, over the regime's rows."""
    from diffusion_curvature.successor import SuccessorEntropyCurvature
    from diffusion_curvature.successor.measures import PROFILE_KAPPAS, gess_profile
    rows = np.arange(len(M)) if dataset == "colosseum" else np.unique(np.asarray(traj_idx).ravel())
    H = np.asarray(SuccessorEntropyCurvature().fit_transform(M=M, F_embeddings=F_emb, B_embeddings=B_emb), dtype=float)[rows]
    Gp, top = gess_profile(M[rows], B_emb, tau=1.0, device=device)
    share = Gp / np.maximum(top[:, None], 1e-9)
    out = dict(n_rows=int(len(rows)), successor_entropy=float(H[np.isfinite(H)].mean()), logess=float(top.mean()))
    out.update({f"g{j}": float(Gp[:, j].mean()) for j in range(len(PROFILE_KAPPAS))})
    out.update({f"s{j}": float(np.median(share[:, j])) for j in range(len(PROFILE_KAPPAS))})
    return out


def read_instance(dataset: str, X: np.ndarray, idx: int, device: str, sizes: dict):
    """Train the regime's FB net on walks over the kNN graph of X and read its successor measures: the paired successor entropy and the profile."""
    from diffusion_curvature.successor.measures import compute_successor_measures
    from diffusion_curvature.successor.train import FBTrainer
    traj_idx = walks(dataset, X, idx, sizes)
    t0 = time.time()
    trainer = FBTrainer(obs_dim=X.shape[1], **FB_BASE, n_epochs=sizes["epochs"][dataset], device=device, seed=FB_SEED + idx)
    trainer.fit(X[traj_idx].astype(np.float32), log_every=None)
    M, F_emb, B_emb = compute_successor_measures(trainer.F_net, trainer.B_net, X, X, device=device)
    return dict(fb_elapsed_s=round(time.time() - t0, 1), **read_measures(dataset, M, F_emb, B_emb, traj_idx, device))


def prepare(job):
    """One (dataset, cell) made ready for a fleet, in a worker process: the manifold, its curvature and its walks."""
    dataset, (idx, d, noise, m), sizes = job
    X, ks = make_instance(idx, d, noise, sizes["n"])
    return dataset, (idx, d, noise, m), X, ks, walks(dataset, X, idx, sizes)


def done_keys(out_dir: Path) -> set:
    keys = set()
    for f in glob.glob(str(out_dir / SHARD.format(k="*"))) + glob.glob(str(out_dir / MERGED)):
        if Path(f).stat().st_size:
            d = pd.read_csv(f, usecols=["dataset", "instance"])
            keys |= set(zip(d.dataset, d.instance.astype(int)))
    return keys


def cmd_run(args) -> None:
    import torch
    torch.set_num_threads(args.threads)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    shard = out_dir / (("test_" if args.test_run else "") + SHARD.format(k=args.worker_id))
    sizes = dict(n=400, n_traj=40, traj_len=20, epochs={k: 30 for k in EPOCHS}) if args.test_run else dict(n=N_SAMPLES, n_traj=N_TRAJECTORIES, traj_len=TRAJ_LENGTH, epochs=EPOCHS)
    done = set() if args.test_run else done_keys(out_dir)
    mine = [c for c in cells(args.test_run) if c[0] % args.num_workers == args.worker_id]
    t0, took, n_done, stop = time.time(), {ds: [] for ds in EPOCHS}, 0, False
    print(f"[w{args.worker_id}/{args.num_workers}] {len(mine)} instances, {len(done)} (dataset, instance) pairs already stored, device {args.device}, {args.threads} threads", flush=True)
    for idx, d, noise, m in mine:
        todo = [ds for ds in args.datasets if (ds, idx) not in done]
        if not todo or stop:
            continue
        X, ks = make_instance(idx, d, noise, sizes["n"])
        for ds in todo:
            # the wall: a regime's fit is not begun when its like has shown it would pass the limit
            if args.max_minutes and took[ds] and (time.time() - t0 + 1.25 * max(took[ds])) / 60 > args.max_minutes:
                print(f"[w{args.worker_id}] stopping before {ds}[{idx}]: it would pass {args.max_minutes} minutes. Resubmit to resume.", flush=True)
                stop = True
                break
            t2 = time.time()
            try:
                row, err = read_instance(ds, X, idx, args.device, sizes), ""
            except Exception as e:   # one bad manifold must not cost the shard
                row, err = {}, str(e)[:200]
                print(f"  [err] {ds}[{idx}]: {err}", flush=True)
            took[ds].append(time.time() - t2)
            row = dict(dataset=ds, instance=idx, name=f"d{d}-c1-n{noise}-m{m}", dim=d, codim=1, noise=noise, m=m, ks_true=ks, elapsed_s=round(took[ds][-1], 1), err=err, **row)
            pd.DataFrame([row]).to_csv(shard, mode="a", index=False, header=not (shard.exists() and shard.stat().st_size))
            n_done += 1
            print(f"  [w{args.worker_id}] {ds}[{idx}] d={d} noise={noise} ks={ks:+.2f} {took[ds][-1]:.0f}s  ({n_done} done, {(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"[w{args.worker_id}] finished {n_done} fits in {(time.time() - t0) / 60:.1f} min -> {shard}", flush=True)


def cmd_fleet(args) -> None:
    """The same fits as `run`, a fleet of them to a card: every pending manifold of a (regime, dimension) group trains at once.

    A group shares its ambient dimension and its transition count, which is what a fleet needs. The manifolds and
    walks of the next fleet are prepared by a pool of CPU workers while the card trains the present one, so the card
    waits on the CPU only for the first.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    import torch
    from diffusion_curvature.successor.ensemble import FleetFBTrainer
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    shard = out_dir / (("test_" if args.test_run else "") + SHARD.format(k=f"g{args.task_id}"))
    sizes = dict(n=400, n_traj=40, traj_len=20, epochs={k: 30 for k in EPOCHS}) if args.test_run else dict(n=N_SAMPLES, n_traj=N_TRAJECTORIES, traj_len=TRAJ_LENGTH, epochs=EPOCHS)
    done = set() if args.test_run else done_keys(out_dir)
    dims = sorted({c[1] for c in cells(args.test_run)})
    groups = [(ds, d) for ds in args.datasets for d in dims]
    mine = [g for i, g in enumerate(groups) if i % args.num_tasks == args.task_id]
    fleets = [(ds, d, todo[i:i + args.fleet]) for ds, d in mine
              for todo in [[c for c in cells(args.test_run) if c[1] == d and (ds, c[0]) not in done]] for i in range(0, len(todo), args.fleet)]
    print(f"[g{args.task_id}/{args.num_tasks}] groups {mine}: {sum(len(f[2]) for f in fleets)} fits in {len(fleets)} fleets of up to {args.fleet}, device {args.device}, {args.prep_workers} CPU workers", flush=True)
    t0, per_epoch, n_done = time.time(), [], 0
    with ProcessPoolExecutor(args.prep_workers, mp_context=mp.get_context("spawn")) as pool:
        submit = lambda f: [pool.submit(prepare, (f[0], c, sizes)) for c in f[2]]
        ahead = submit(fleets[0]) if fleets else []
        for j, (ds, d, cs) in enumerate(fleets):
            epochs = sizes["epochs"][ds]
            # the wall: a fleet is not begun when the last one's pace says it would pass the limit
            if args.max_minutes and per_epoch and (time.time() - t0 + 1.25 * per_epoch[-1] * epochs * len(cs)) / 60 > args.max_minutes:
                print(f"[g{args.task_id}] stopping before fleet {j} ({ds}, d={d}): it would pass {args.max_minutes} minutes. Resubmit to resume.", flush=True)
                break
            t1 = time.time(); ready = [f.result() for f in ahead]; prep_s = time.time() - t1
            ahead = submit(fleets[j + 1]) if j + 1 < len(fleets) else []
            t2 = time.time()
            trainer = FleetFBTrainer(obs_dim=d + 1, seeds=[FB_SEED + c[0] for c in cs], **FB_BASE, n_epochs=epochs, device=args.device)
            trainer.fit([X[tr].astype(np.float32) for _, _, X, _, tr in ready])
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            fit_s = time.time() - t2; per_epoch.append(fit_s / epochs / len(cs)); t3 = time.time()
            rows = []
            for k, (_, (idx, _, noise, m), X, ks, tr) in enumerate(ready):
                try:
                    row, err = read_measures(ds, *trainer.measures(k, X, X), tr, args.device), ""
                except Exception as e:   # one bad manifold must not cost the fleet
                    row, err = {}, str(e)[:200]
                    print(f"  [err] {ds}[{idx}]: {err}", flush=True)
                rows.append(dict(dataset=ds, instance=idx, name=f"d{d}-c1-n{noise}-m{m}", dim=d, codim=1, noise=noise, m=m, ks_true=ks,
                                 elapsed_s=round(fit_s / len(cs), 1), err=err, fleet=len(cs), fb_elapsed_s=round(fit_s / len(cs), 1), **row))
            pd.DataFrame(rows).to_csv(shard, mode="a", index=False, header=not (shard.exists() and shard.stat().st_size))
            n_done += len(rows); del trainer
            print(f"  [g{args.task_id}] fleet {j} {ds} d={d}: {len(cs)} nets x {epochs} epochs, wait on prep {prep_s:.0f}s, fit {fit_s:.0f}s ({per_epoch[-1] * 1000:.1f} ms per net-epoch), "
                  f"read {time.time() - t3:.0f}s  ({n_done} done, {(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"[g{args.task_id}] finished {n_done} fits in {(time.time() - t0) / 60:.1f} min -> {shard}", flush=True)


def derive(P: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """From the stored profiles: the method rows (curvature convention, minus the reading) and each method's Pearson and Spearman with the true curvature in every cell."""
    from scipy.stats import pearsonr, spearmanr
    from diffusion_curvature.successor.measures import PROFILE_KAPPAS as K
    g = [f"g{j}" for j in range(len(K))]; s = [f"s{j}" for j in range(len(K))]
    P = P[P.err.fillna("") == ""].drop_duplicates(["dataset", "instance"]).reset_index(drop=True)
    first = lambda med: int(np.argmax(med >= 0.95)) if (med >= 0.95).any() else len(K) - 1
    rows = []
    for (ds, d, e), c in P.groupby(["dataset", "dim", "noise"]):
        Gm, Sm = c[g].to_numpy(), c[s].to_numpy()
        jc = max(first(np.median(Sm, 0)), 1)                       # one ceiling for the cell: the population a Pearson compares
        ja = int(np.argmin(np.abs(np.median(Sm, 0) - 0.5)))        # the cell's anchored scale, at half the ceiling
        own = np.array([Gm[i, :max(first(Sm[i]), 1)].mean() for i in range(len(c))])
        reads = dict(successor_area=-Gm[:, :jc].mean(1), successor_area_own=-own, successor_gess_anchor=-Gm[:, ja], successor_area_full=-Gm.mean(1),
                     successor_logess=-c.logess.to_numpy(), successor_entropy_refit=c.successor_entropy.to_numpy())
        for name, v in reads.items():
            rows.append(c[["dataset", "instance", "name", "dim", "codim", "noise", "m", "ks_true"]].assign(method=name, ks_hat=v, kappa_cell=float(K[jc]), kappa_anchor=float(K[ja])))
    Mx = pd.concat(rows, ignore_index=True)
    S = Mx.groupby(["dataset", "method", "dim", "noise"]).apply(lambda q: pd.Series(dict(pearson=pearsonr(q.ks_hat, q.ks_true)[0], spearman=spearmanr(q.ks_hat, q.ks_true)[0], n=len(q))), include_groups=False).reset_index()
    return Mx, S


def cmd_derive(args) -> None:
    out_dir = Path(args.out_dir); pre = "test_" if args.test_run else ""
    files = sorted(glob.glob(str(out_dir / (pre + SHARD.format(k="*"))))) + ([str(out_dir / MERGED)] if (out_dir / MERGED).exists() and not args.test_run else [])
    P = pd.concat([pd.read_csv(f) for f in files if Path(f).stat().st_size], ignore_index=True).drop_duplicates(["dataset", "instance"])
    if not args.test_run:
        P.to_csv(out_dir / MERGED, index=False)
    Mx, S = derive(P)
    Mx.to_csv(out_dir / (pre + METRICS), index=False); S.to_csv(out_dir / (pre + SUMMARY), index=False)
    print(f"{len(P)} profiles from {len(files)} files; {P.groupby('dataset').instance.nunique().to_dict()} instances")
    print(S.pivot_table(index=["dataset", "method"], columns="dim", values="pearson").round(2).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--worker-id", type=int, default=0); r.add_argument("--num-workers", type=int, default=1)
    r.add_argument("--device", default="cpu"); r.add_argument("--threads", type=int, default=4)
    r.add_argument("--datasets", nargs="+", default=list(EPOCHS), choices=list(EPOCHS))
    r.add_argument("--max-minutes", type=float, default=0.0, help="stop before an instance that would pass this many minutes; 0 for no limit")
    r.add_argument("--out-dir", default=str(OUT_DIR)); r.add_argument("--test-run", action="store_true")
    f = sub.add_parser("fleet")
    f.add_argument("--task-id", type=int, default=0); f.add_argument("--num-tasks", type=int, default=1)
    f.add_argument("--device", default="cuda"); f.add_argument("--fleet", type=int, default=200, help="nets trained at once; about 50 MiB of card memory each")
    f.add_argument("--prep-workers", type=int, default=8)
    f.add_argument("--datasets", nargs="+", default=list(EPOCHS), choices=list(EPOCHS))
    f.add_argument("--max-minutes", type=float, default=0.0, help="stop before a fleet that would pass this many minutes; 0 for no limit")
    f.add_argument("--out-dir", default=str(OUT_DIR)); f.add_argument("--test-run", action="store_true")
    d = sub.add_parser("derive"); d.add_argument("--out-dir", default=str(OUT_DIR)); d.add_argument("--test-run", action="store_true")
    args = ap.parse_args()
    dict(run=cmd_run, fleet=cmd_fleet, derive=cmd_derive)[args.cmd](args)


if __name__ == "__main__":
    main()
