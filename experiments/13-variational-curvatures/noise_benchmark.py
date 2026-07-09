"""N-suite — realistic-noise benchmark: TD-InfoNCE vs successor entropy.

72 units = {necklace b=0.7 k=2, necklace b=0.5 k=3} x d in {3,4,5}
         x walk seed {0,1} x noise setting {clean, iso05, iso15, hd64,
         hetero, ar1}.
Continuous BM trajectories (nt=40, T=2000, dt=2e-3 normalized); channels see
only noisy observations; analytic ks fields; 48 stratified eval targets from
an independent run.

Noise settings:
  clean   — sigma = 0
  iso05/15— isotropic Gaussian, independent per observation
  hd64    — random smooth lift chi -> R^64 (tanh MLP, unit-rescaled) + 0.1
            isotropic noise: the learned-embedding analog
  hetero  — sigma(x) = 0.2*(1+u1)/2, curvature-independent direction
  ar1     — temporally correlated noise (rho=0.8, marginal sigma=0.15)

Channels:
  sent_bin — successor entropy of the empirical chain over 1500 Voronoi bins
  sent_knn — successor entropy on the kNN graph of 2500 visited points
  td       — TD-InfoNCE critic (z=128, 300 epochs, 2 net seeds, pooled DV)

Army pattern: run --worker-id K --num-workers W --device cuda:X (shard-CSV
resume) | summarize.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent / "../11-successor-curvatures"))
from benchmark_kmetric_colosseum import affinity_from_D
from diffusion_curvature.continuous_walks import brownian_walks, chi_embed
from diffusion_curvature.menagerie import WarpedProduct, necklace_profile
from diffusion_curvature.variational import TDInfoNCE

warnings.filterwarnings("ignore")

NT, T = 40, 2000
DT_NORM = 2e-3
SPREAD_T = 0.09
N_EVAL = 48
POOL_STATES = 32
NET_SEEDS = (0, 1)
BIN_NC = 1500
KNN_SUB = 2500
N_SCALE = 1500

PROFILES = {"nk2": dict(b=0.7, k=2), "nk3": dict(b=0.5, k=3)}
NOISES = ("clean", "iso05", "iso15", "hd64", "hetero", "ar1")
OUT_TPL = "processed_data/noise_bench_w{wid}.csv"


def units():
    out = []
    for prof in PROFILES:
        for d in (3, 4, 5):
            for wseed in (0, 1):
                for noise in NOISES:
                    out.append(dict(profile=prof, d=d, wseed=wseed,
                                    noise=noise))
    return out


def entropy_rows(R):
    R = np.clip(R, 1e-30, 1)
    return -(R * np.log(R)).sum(1)


def resolvent_rows(P, gamma, idx=None, device="cpu"):
    n = P.shape[0]
    Pt = torch.as_tensor(P, dtype=torch.float64, device=device)
    M = (1 - gamma) * torch.linalg.inv(
        torch.eye(n, dtype=torch.float64, device=device) - gamma * Pt)
    M = np.maximum(M.cpu().numpy(), 0.0)
    M = M / np.maximum(M.sum(1, keepdims=True), 1e-30)
    return M if idx is None else M[idx]


def pooled_dv(est, member_groups, n_neg=8192, rng=None):
    rng = np.random.default_rng(rng)
    dev = est.device
    a, p = est._held_pairs
    f_ap = np.empty(len(a))
    At = torch.as_tensor(a, dtype=torch.long, device=dev)
    Pt = torch.as_tensor(p, dtype=torch.long, device=dev)
    with torch.no_grad():
        for lo in range(0, len(a), 262144):
            hi = min(lo + 262144, len(a))
            f_ap[lo:hi] = est.net.score_pairs(
                At[lo:hi], Pt[lo:hi]).cpu().numpy()
    sums = np.zeros(est.n_states)
    cnts = np.zeros(est.n_states)
    np.add.at(sums, a, f_ap)
    np.add.at(cnts, a, 1)
    neg = torch.as_tensor(rng.integers(0, est.n_states, n_neg),
                          dtype=torch.long, device=dev)
    out = np.empty(len(member_groups))
    with torch.no_grad():
        for q, members in enumerate(member_groups):
            Mt = torch.as_tensor(members, dtype=torch.long, device=dev)
            fr = est.net.score(Mt, neg)
            logZ = (torch.logsumexp(fr, dim=1) - np.log(n_neg)).mean().item()
            tot, cnt = sums[members].sum(), cnts[members].sum()
            out[q] = tot / cnt - logZ if cnt >= 8 else np.nan
    return out


def apply_noise(unit, X_true, Xt_true, u1, rng):
    """Returns (X_obs, Xt_obs) for the unit's noise setting.
    X_true: (nt, T+1, D0) walk observations; Xt_true: (n_eval, D0)."""
    noise = unit["noise"]
    nt, T1, D0 = X_true.shape
    if noise == "clean":
        return X_true.reshape(-1, D0), Xt_true
    if noise in ("iso05", "iso15"):
        sig = 0.05 if noise == "iso05" else 0.15
        return (X_true.reshape(-1, D0)
                + sig * rng.normal(size=(nt * T1, D0)),
                Xt_true + sig * rng.normal(size=Xt_true.shape))
    if noise == "hetero":
        sig = 0.2 * (1 + u1) / 2       # u1: (nt, T1) walk sphere-coord
        X = X_true + sig[..., None] * rng.normal(size=X_true.shape)
        # targets: mild constant noise at the mean level
        return (X.reshape(-1, D0),
                Xt_true + 0.1 * rng.normal(size=Xt_true.shape))
    if noise == "ar1":
        rho, sig = 0.8, 0.15
        eps = np.empty_like(X_true)
        eps[:, 0] = rng.normal(size=(nt, D0))
        for t in range(1, T1):
            eps[:, t] = (rho * eps[:, t - 1]
                         + np.sqrt(1 - rho**2) * rng.normal(size=(nt, D0)))
        return ((X_true + sig * eps).reshape(-1, D0),
                Xt_true + sig * rng.normal(size=Xt_true.shape))
    if noise == "hd64":
        useed = abs(hash((unit["profile"], unit["d"]))) % 2**31
        rl = np.random.default_rng(useed)
        W1 = rl.normal(size=(D0, 32)) / np.sqrt(D0)
        b1 = rl.normal(size=32) * 0.3
        W2 = rl.normal(size=(32, 64)) / np.sqrt(32)
        lift = lambda z: np.tanh(z @ W1 * 3.0 + b1) @ W2
        Xl = lift(X_true.reshape(-1, D0))
        sub = rl.choice(len(Xl), 2000, replace=False)
        sc = np.median(np.linalg.norm(Xl[sub][:, None] - Xl[sub][None],
                                      axis=-1))
        Xl, Xtl = Xl / sc, lift(Xt_true) / sc
        return (Xl + 0.1 * rng.normal(size=Xl.shape),
                Xtl + 0.1 * rng.normal(size=Xtl.shape))
    raise ValueError(noise)


def prepare_unit(unit, caches):
    """Deterministic data prep shared by all channel scripts: same walks,
    same noise draw, same eval targets and pooled groups per unit."""
    key = (unit["profile"], unit["d"])
    if key not in caches:
        p = PROFILES[unit["profile"]]
        f, L = necklace_profile(b=p["b"], k=p["k"])
        wp = WarpedProduct(f, L, d=unit["d"], periodic=True)
        m = wp.sample(N_SCALE, rng=42)
        s = np.median(m["D"][np.triu_indices(N_SCALE, 1)])
        caches[key] = (wp, s)
    wp, s = caches[key]
    d, wseed = unit["d"], unit["wseed"]
    gamma_c = 1 - d * DT_NORM / SPREAD_T
    walks = brownian_walks(wp, NT, T, dt=DT_NORM * s**2, rng=wseed)
    X_true = chi_embed(wp, walks["r"], walks["u"]) / s
    n_pts = NT * (T + 1)
    traj_idx = np.arange(n_pts).reshape(NT, T + 1)
    tw = brownian_walks(wp, 8, 1200, dt=DT_NORM * s**2, rng=1000 + wseed)
    tchi = (chi_embed(wp, tw["r"], tw["u"]) / s).reshape(-1,
                                                         X_true.shape[-1])
    tks = (tw["ks"] * s**2).ravel()
    torder = np.argsort(tks)
    targets = torder[(np.linspace(0.02, 0.98, N_EVAL)
                      * (len(tks) - 1)).astype(int)]
    rng = np.random.default_rng(7000 + wseed)
    X, Xt_ev = apply_noise(unit, X_true, tchi[targets],
                           walks["u"][..., 0], rng)
    kt_w = tks[targets]
    tree = cKDTree(X)
    _, grp = tree.query(Xt_ev, k=POOL_STATES)
    return dict(X=X, Xt_ev=Xt_ev, kt_w=kt_w, groups=list(grp),
                traj_idx=traj_idx, n_pts=n_pts, gamma_c=gamma_c)


def run_unit(unit, caches, device):
    prep = prepare_unit(unit, caches)
    X, Xt_ev, kt_w = prep["X"], prep["Xt_ev"], prep["kt_w"]
    groups, traj_idx = prep["groups"], prep["traj_idx"]
    n_pts, gamma_c = prep["n_pts"], prep["gamma_c"]
    wseed = unit["wseed"]

    def rw(v):
        mm = np.isfinite(v) & np.isfinite(kt_w)
        return pearsonr(v[mm], kt_w[mm])[0] if mm.sum() > 8 else np.nan

    row = dict(unit)
    row["gamma_c"] = round(gamma_c, 4)
    # sent_bin
    rngb = np.random.default_rng(10 + wseed)
    centers = rngb.choice(n_pts, BIN_NC, replace=False)
    ctree = cKDTree(X[centers])
    _, assign = ctree.query(X, k=1)
    lab = assign.reshape(NT, T + 1)
    a_, b_ = lab[:, :-1].ravel(), lab[:, 1:].ravel()
    C = sp.coo_matrix((np.ones(len(a_)), (a_, b_)),
                      shape=(BIN_NC, BIN_NC)).tocsr()
    C = np.asarray((C + C.T).todense(), dtype=np.float64)
    Pe = (C + 1e-9) / (C + 1e-9).sum(1, keepdims=True)
    _, tassign = ctree.query(Xt_ev, k=5)
    H_b = entropy_rows(resolvent_rows(Pe, gamma_c, device=device))
    row["r_sent_bin"] = rw(np.nanmean((-H_b)[tassign], axis=1))
    # sent_knn
    rngk = np.random.default_rng(20 + wseed)
    sub = rngk.choice(n_pts, KNN_SUB, replace=False)
    Dk = np.linalg.norm(X[sub][:, None] - X[sub][None], axis=-1)
    Wk = affinity_from_D(Dk, k=10)
    Pk = Wk / np.maximum(Wk.sum(1, keepdims=True), 1e-30)
    ktree = cKDTree(X[sub])
    _, kassign = ktree.query(Xt_ev, k=5)
    H_k = entropy_rows(resolvent_rows(Pk, 0.9, device=device))
    row["r_sent_knn"] = rw(np.nanmean((-H_k)[kassign], axis=1))
    # td_infonce
    t0 = time.time()
    vals = []
    for ns in NET_SEEDS:
        est = TDInfoNCE(gamma=gamma_c, z_dim=128, features="coords",
                        hidden=256, n_epochs=300, batch_size=4096,
                        n_candidates=511, lr=3e-4, holdout_frac=0.5,
                        lags_per_step=16, device=device, seed=ns).fit(
            traj_idx, n_pts, X=X)
        vals.append(pooled_dv(est, groups, rng=ns))
    row["r_td"] = rw(np.nanmean(vals, axis=0))
    row["t_td"] = round(time.time() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("run")
    w.add_argument("--worker-id", type=int, default=0)
    w.add_argument("--num-workers", type=int, default=1)
    w.add_argument("--device", default="cuda:0")
    w.add_argument("--test-run", action="store_true")
    sub.add_parser("summarize")
    args = ap.parse_args()
    if args.cmd == "summarize":
        df = pd.concat([pd.read_csv(p) for p in
                        sorted(Path("processed_data").glob(
                            "noise_bench_w*.csv"))], ignore_index=True)
        df = df.drop_duplicates(["profile", "d", "wseed", "noise"],
                                keep="last")
        df.to_csv("processed_data/noise_bench.csv", index=False)
        print(f"{len(df)}/72 units")
        pd.set_option("display.width", 200)
        print(df.groupby(["noise", "d"])[
            ["r_sent_bin", "r_sent_knn", "r_td"]].mean().round(3))
        return
    us = units()
    if args.test_run:
        us = [u for u in us if u["profile"] == "nk2" and u["d"] == 3
              and u["wseed"] == 0][:2]
    W = args.num_workers
    mine = [(i, u) for i, u in enumerate(us) if i % W == args.worker_id]
    out = Path(OUT_TPL.format(wid=args.worker_id))
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists() and out.stat().st_size > 0:
        dfd = pd.read_csv(out)
        done = set(zip(dfd.profile, dfd.d, dfd.wseed, dfd.noise))
    header = out.exists() and out.stat().st_size > 0
    caches = {}
    print(f"[w{args.worker_id}] {len(mine)} units", flush=True)
    for i, u in mine:
        if (u["profile"], u["d"], u["wseed"], u["noise"]) in done:
            continue
        try:
            row = run_unit(u, caches, args.device)
        except Exception as e:
            print(f"  [err] unit {i}: {str(e)[:150]}", flush=True)
            continue
        pd.DataFrame([row]).to_csv(out, mode="a", index=False,
                                   header=not header)
        header = True
        print("  " + " ".join(f"{k}={v:.3f}" if isinstance(v, float)
                              else f"{k}={v}" for k, v in row.items()),
              flush=True)
    print(f"[w{args.worker_id}] done", flush=True)


if __name__ == "__main__":
    main()
