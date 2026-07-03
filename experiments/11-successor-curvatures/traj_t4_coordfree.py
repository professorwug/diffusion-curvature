"""T4 — coordinate-free trajectory channels: empirical chain vs successor kernel.

Estimators see ONLY the observed transitions (no coordinates; coordinates are
used solely to pick the evaluation anchor, which is benchmark infrastructure).

Operators:
  emp : symmetrized transition-count chain P_hat (rownorm; 1e-9 floor)
  fbP : normalized successor kernel — diag-zeroed rownorm(relu(K_gamma=0.8))
        (the user's proposal: denoised multi-step occupancy as the operator)

Metrics:
  surp : surprisal geodesics — sparse graph of observed transitions with
         weights -log P_hat, lazily Dijkstra'd (chain-native metric)
  fbD  : gamma=0.98 FB potential distances + kNN Dijkstra graph (validated)

Channels per (instance d>=3, nt in {10,25,50,100}), paired walks with T1/T3:
  frac_emp     : escape fraction on P_hat, edges = observed out-transitions
  frac_fb      : escape fraction on fbP, edges = fbD-nearest
  ent_emp_t{2,4,8} : entropy of P_hat^t rows at anchor pool (graph-neighbor pool)
  ent_fb       : successor entropy (H of fbP rows)
  kappa_surp   : Diffusion ORC, measures P_hat^8, ground metric = surprisal
                 geodesics (kappa_fbD lives in traj_t3.csv for comparison)

Pre-registered: crossover — emp wins at moderate coverage, fbP at nt=10.

Usage:
  pixi run python traj_t4_coordfree.py run --worker-id K --num-workers W --device cuda:X
  pixi run python traj_t4_coordfree.py summarize
"""

from __future__ import annotations

import argparse
import itertools
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pygsp
import scipy.sparse as sp
import torch

from diffusion_curvature.successor.ensemble import EnsembleFBTrainer
from diffusion_curvature.trajectory_utils import subsample_trajectories
from diffusion_curvature.wasserstein_signed import WassersteinSignedCurvature

from benchmark_kmetric_colosseum import knn_distance_graph, potential_distances
from diffusing_fraction_colosseum import edge_fractions

warnings.filterwarnings("ignore")

BATTERY_PATH = Path("processed_data/signed_battery.joblib")
M_MAX = 15
NTS = (10, 25, 50, 100)
DIMS = (3, 4, 5, 6)
TRAJ_LEN = 50
KNN = 10
T_FRAC = 16
TS_ENT = (2, 4, 8)
T_ORC = 8
K_EDGE = 16
N_LOCAL = 5
MIN_V = 60
SEEDS = (7, 8, 9, 10)
Z_DIM = 64
N_EPOCHS = 300
GAMMA_OP, GAMMA_DIST = 0.8, 0.98

OUT_TPL = "processed_data/traj_t4_w{wid}.csv"
OUT_MERGED = Path("processed_data/traj_t4.csv")


def empirical_chain(traj_local: np.ndarray, nV: int):
    """Symmetrized transition counts -> dense P_hat + sparse surprisal graph."""
    a = traj_local[:, :-1].ravel()
    b = traj_local[:, 1:].ravel()
    C = sp.coo_matrix((np.ones(len(a)), (a, b)), shape=(nV, nV)).tocsr()
    C = C + C.T
    Cd = np.asarray(C.todense(), dtype=np.float64)
    P_hat = (Cd + 1e-9) / (Cd + 1e-9).sum(axis=1, keepdims=True)
    # surprisal weights only on observed edges
    rows, cols = C.nonzero()
    probs = Cd[rows, cols] / np.maximum(Cd.sum(axis=1)[rows], 1)
    w = -np.log(np.clip(probs, 1e-12, 1 - 1e-12))
    G_surp = sp.csr_matrix((w, (rows, cols)), shape=(nV, nV))
    out_edges = {int(r): [] for r in range(nV)}
    for r, c in zip(rows, cols):
        out_edges[int(r)].append(int(c))
    return P_hat, G_surp, out_edges


def entropy_rows_t(P: torch.Tensor, anchors, ts) -> dict[int, float]:
    out = {}
    with torch.no_grad():
        rows = torch.zeros((len(anchors), P.shape[0]), device=P.device)
        for q, a in enumerate(anchors):
            rows[q, a] = 1.0
        for step in range(1, max(ts) + 1):
            rows = rows @ P
            if step in ts:
                pr = rows.clamp_min(1e-12)
                out[step] = float((-(pr * pr.log()).sum(dim=1)).mean())
    return out


def coordfree_channels(traj_local, nV, i0, K_op, K_dist, device):
    out: dict[str, float] = {}
    P_hat_np, G_surp, out_edges = empirical_chain(traj_local, nV)
    P_hat = torch.as_tensor(P_hat_np, dtype=torch.float32, device=device)

    # anchor pool: i0 + its most-visited graph neighbors
    nbrs = sorted(out_edges.get(i0, []),
                  key=lambda j: -P_hat_np[i0, j])[:N_LOCAL - 1]
    anchors = [i0] + nbrs if nbrs else [i0]

    # frac_emp: observed out-transitions as probe edges
    edges = [(a, j) for a in anchors
             for j in sorted(out_edges.get(a, []),
                             key=lambda jj: -P_hat_np[a, jj])[:K_EDGE]]
    if edges:
        fr = edge_fractions(P_hat, edges, (T_FRAC,))
        out["frac_emp"] = float(fr[T_FRAC].mean())

    # ent_emp
    for t, v in entropy_rows_t(P_hat, anchors, TS_ENT).items():
        out[f"ent_emp_t{t}"] = v

    # kappa_surp: P_hat^8 measures, surprisal-geodesic ground metric
    est = WassersteinSignedCurvature(t=T_ORC, n_pairs=8, seed=0,
                                     compute_midpoint=False)
    try:
        est.fit(G=type("G", (), {"W": P_hat_np})(), D_graph=G_surp, idx=[i0])
        out["kappa_surp"] = float(est.orc_[0])
    except Exception:
        out["kappa_surp"] = np.nan

    # fbP operator channels (per-seed kernels passed in)
    fb_frac, fb_ent = [], []
    for s in range(K_op.shape[0]):
        Kk = K_op[s].copy()
        np.fill_diagonal(Kk, 0.0)
        Pfb_np = (Kk + 1e-12) / (Kk + 1e-12).sum(axis=1, keepdims=True)
        Pfb = torch.as_tensor(Pfb_np, dtype=torch.float32, device=device)
        D_pot = potential_distances(K_dist[s])
        order = np.argsort(D_pot[anchors], axis=1)
        edges_fb = [(a, int(j)) for q, a in enumerate(anchors)
                    for j in order[q, 1:min(K_EDGE, nV - 2) + 1]]
        fr = edge_fractions(Pfb, edges_fb, (T_FRAC,))
        fb_frac.append(float(fr[T_FRAC].mean()))
        pr = Pfb.clamp_min(1e-12)
        fb_ent.append(float((-(pr[anchors] * pr[anchors].log())
                             .sum(dim=1)).mean()))
    out["frac_fb"] = float(np.mean(fb_frac)) if fb_frac else np.nan
    out["ent_fb"] = float(np.mean(fb_ent)) if fb_ent else np.nan
    return out


def run_worker(args) -> None:
    instances = joblib.load(BATTERY_PATH)
    cc = [(i, inst) for i, inst in enumerate(instances)
          if inst["dataset"] == "colosseum" and inst["m"] < M_MAX
          and inst["dim"] in DIMS]
    units = [(i, inst, nt) for (i, inst), nt in itertools.product(cc, NTS)]
    mine = [u for k, u in enumerate(units) if k % args.num_workers == args.worker_id]
    out = Path(OUT_TPL.format(wid=args.worker_id))
    done = set()
    if out.exists() and out.stat().st_size > 0:
        prev = pd.read_csv(out, usecols=["instance", "n_traj"])
        done = set(map(tuple, prev.values))
    header = out.exists() and out.stat().st_size > 0
    print(f"[w{args.worker_id}] {len(mine)} units", flush=True)

    t0 = time.time()
    for prog, (i, inst, nt) in enumerate(mine, 1):
        if (i, nt) in done:
            continue
        X = np.asarray(inst["X"], dtype=np.float64)
        err = ""
        ch: dict[str, float] = {}
        coverage = np.nan
        try:
            G = pygsp.graphs.NNGraph(X, k=KNN)
            traj_idx = subsample_trajectories(
                G, n_trajectories=nt, length=TRAJ_LEN, rng=1000 + nt)
            V, traj_local = np.unique(traj_idx, return_inverse=True)
            traj_local = traj_local.reshape(traj_idx.shape)
            nV = len(V)
            coverage = nV / X.shape[0]
            X_V = X[V]
            i0 = int(np.argmin(np.linalg.norm(X_V - X[0], axis=1)))
            if nV >= MIN_V:
                ens_op = EnsembleFBTrainer(
                    obs_dim=X.shape[1], seeds=SEEDS, z_dim=Z_DIM,
                    gamma=GAMMA_OP, n_epochs=N_EPOCHS, device=args.device)
                ens_op.fit(X[traj_idx].astype(np.float32))
                K_op = np.maximum(ens_op.raw_kernels(
                    X_V.astype(np.float32)), 0.0)
                ens_d = EnsembleFBTrainer(
                    obs_dim=X.shape[1], seeds=SEEDS, z_dim=Z_DIM,
                    gamma=GAMMA_DIST, n_epochs=N_EPOCHS, device=args.device)
                ens_d.fit(X[traj_idx].astype(np.float32))
                K_dist = np.maximum(ens_d.raw_kernels(
                    X_V.astype(np.float32)), 0.0)
                ch = coordfree_channels(traj_local, nV, i0, K_op, K_dist,
                                        args.device)
        except Exception as e:
            err = str(e)[:200]
            print(f"  [err] inst={i} nt={nt}: {err}", flush=True)
        pd.DataFrame([dict(
            instance=i, n_traj=nt, err=err, dim=inst["dim"],
            noise=inst["noise"], ks_true=inst["ks_true"],
            coverage=round(coverage, 3) if np.isfinite(coverage) else np.nan,
            **ch)]).to_csv(out, mode="a", index=False, header=not header)
        header = True
        if prog % 10 == 0:
            rate = prog / max((time.time() - t0) / 60, 1e-9)
            print(f"  [w{args.worker_id}] {prog}/{len(mine)} "
                  f"eta={(len(mine)-prog)/max(rate,1e-9):.0f}min", flush=True)
    print(f"[w{args.worker_id}] done in {(time.time()-t0)/60:.1f} min", flush=True)


def run_summarize() -> None:
    from scipy.stats import pearsonr
    df = pd.concat([pd.read_csv(p) for p in
                    sorted(Path("processed_data").glob("traj_t4_w*.csv"))],
                   ignore_index=True).drop_duplicates(
        ["instance", "n_traj"], keep="last")
    df.to_csv(OUT_MERGED, index=False)
    print(f"wrote {OUT_MERGED} ({len(df)} rows)\n")
    channels = [("frac_emp", -1), ("frac_fb", -1), ("kappa_surp", +1),
                ("ent_emp_t2", -1), ("ent_emp_t4", -1), ("ent_emp_t8", -1),
                ("ent_fb", -1)]
    for ch, orient in channels:
        if ch not in df.columns:
            continue
        print(f"=== {ch}: Pearson per (n_traj, dim) ===")
        rows = []
        for nt in NTS:
            row = {"n_traj": nt}
            for d in DIMS:
                g = df[(df.n_traj == nt) & (df.dim == d)]
                v = orient * pd.to_numeric(g[ch], errors="coerce")
                m = np.isfinite(v)
                row[f"d{d}"] = (round(pearsonr(v[m], g.ks_true[m])[0], 2)
                                if m.sum() > 5 and v[m].std() > 0 else np.nan)
            rows.append(row)
        print(pd.DataFrame(rows).set_index("n_traj").to_string())
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("run")
    w.add_argument("--worker-id", type=int, default=0)
    w.add_argument("--num-workers", type=int, default=2)
    w.add_argument("--device", default="cuda:0")
    sub.add_parser("summarize")
    args = ap.parse_args()
    if args.cmd == "run":
        run_worker(args)
    else:
        run_summarize()


if __name__ == "__main__":
    main()
