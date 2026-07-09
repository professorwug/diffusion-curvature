"""C-series v2 — observation noise axis.

Same continuous BM trajectories as c_series, but every OBSERVATION is
x_obs = chi(state) + sigma * N(0, I): independent noise per time step
(consecutive observed displacements are noise-dominated at sigma >~ 0.1;
true per-step displacement ~ sqrt(d*dt) ~ 0.08). All channels consume only
the noisy coordinates; ground-truth ks remains attached to the true state.

Pre-registered: knn_graph degrades most (noise edges); bin_count degrades
once sigma approaches the bin scale; the learned critics degrade least in
relative terms (regression through function approximation averages
independent observation noise) and may overtake at sigma = 0.15.

Output: processed_data/c_noise.csv
"""
from __future__ import annotations

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
from c_series import (DT_NORM, N_ORACLE, NT, POOL_STATES, SPREAD_T, T,
                      entropy_rows, pooled_dv, resolvent_rows)
from diffusion_curvature.continuous_walks import brownian_walks, chi_embed
from diffusion_curvature.menagerie import WarpedProduct, necklace_profile
from diffusion_curvature.variational import VMI, TDInfoNCE

warnings.filterwarnings("ignore")

DEV = "cuda:1" if torch.cuda.is_available() else "cpu"
NET_SEEDS = (0, 1)
SIGMAS = (0.05, 0.15)
BIN_NC = (500, 1500)
KNN_SUB = 2500
N_EVAL = 48


def main():
    rows = []
    for d in (3, 4):
        f, L = necklace_profile(b=0.7)
        wp = WarpedProduct(f, L, d=d, periodic=True)
        m = wp.sample(N_ORACLE, rng=100 + d)
        s = np.median(m["D"][np.triu_indices(N_ORACLE, 1)])
        gamma_c = 1 - d * DT_NORM / SPREAD_T
        for wseed in (0, 1):
            walks = brownian_walks(wp, NT, T, dt=DT_NORM * s**2, rng=wseed)
            chi = chi_embed(wp, walks["r"], walks["u"]) / s
            n_pts = NT * (T + 1)
            X_true = chi.reshape(n_pts, -1)
            traj_idx = np.arange(n_pts).reshape(NT, T + 1)
            tw = brownian_walks(wp, 8, 1200, dt=DT_NORM * s**2,
                                rng=1000 + wseed)
            tchi = (chi_embed(wp, tw["r"], tw["u"]) / s).reshape(
                -1, X_true.shape[1])
            tks = (tw["ks"] * s**2).ravel()
            torder = np.argsort(tks)
            targets = torder[(np.linspace(0.02, 0.98, N_EVAL)
                              * (len(tks) - 1)).astype(int)]
            for sigma in SIGMAS:
                rngo = np.random.default_rng(500 + wseed)
                X = X_true + sigma * rngo.normal(size=X_true.shape)
                Xt_ev = (tchi[targets]
                         + sigma * rngo.normal(size=(N_EVAL,
                                                     X.shape[1])))
                kt_w = tks[targets]
                tree = cKDTree(X)
                _, grp = tree.query(Xt_ev, k=POOL_STATES)
                groups = list(grp)

                def rw(v):
                    mm = np.isfinite(v) & np.isfinite(kt_w)
                    return (pearsonr(v[mm], kt_w[mm])[0]
                            if mm.sum() > 8 else np.nan)

                res = dict(d=d, wseed=wseed, sigma=sigma)
                # bins
                for Nc in BIN_NC:
                    rngb = np.random.default_rng(10 + wseed)
                    centers = rngb.choice(n_pts, Nc, replace=False)
                    ctree = cKDTree(X[centers])
                    _, assign = ctree.query(X, k=1)
                    lab = assign.reshape(NT, T + 1)
                    a_, b_ = lab[:, :-1].ravel(), lab[:, 1:].ravel()
                    C = sp.coo_matrix((np.ones(len(a_)), (a_, b_)),
                                      shape=(Nc, Nc)).tocsr()
                    C = np.asarray((C + C.T).todense(), dtype=np.float64)
                    Pe = (C + 1e-9) / (C + 1e-9).sum(1, keepdims=True)
                    _, tassign = ctree.query(Xt_ev, k=5)
                    H_b = entropy_rows(resolvent_rows(Pe, gamma_c))
                    res[f"r_bin{Nc}"] = rw(
                        np.nanmean((-H_b)[tassign], axis=1))
                # knn graph
                rngk = np.random.default_rng(20 + wseed)
                sub = rngk.choice(n_pts, KNN_SUB, replace=False)
                Dk = np.linalg.norm(X[sub][:, None] - X[sub][None],
                                    axis=-1)
                Wk = affinity_from_D(Dk, k=10)
                Pk = Wk / np.maximum(Wk.sum(1, keepdims=True), 1e-30)
                ktree = cKDTree(X[sub])
                _, kassign = ktree.query(Xt_ev, k=5)
                H_k = entropy_rows(resolvent_rows(Pk, 0.9))
                res["r_knn"] = rw(np.nanmean((-H_k)[kassign], axis=1))
                # nystrom
                Nl = 1500
                rngn = np.random.default_rng(30 + wseed)
                lm = rngn.choice(n_pts, Nl, replace=False)
                ltree = cKDTree(X[lm])
                sig_k = np.median(ltree.query(X[lm], k=2)[0][:, 1]) * 1.5
                Xl = torch.as_tensor(X[lm], dtype=torch.float32,
                                     device=DEV)
                s_t = traj_idx[:, :-1].ravel()
                s_tp = traj_idx[:, 1:].ravel()
                A = torch.zeros((Nl, Nl), dtype=torch.float64, device=DEV)
                for lo in range(0, len(s_t), 16384):
                    hi = min(lo + 16384, len(s_t))
                    x1 = torch.as_tensor(X[s_t[lo:hi]],
                                         dtype=torch.float32, device=DEV)
                    x2 = torch.as_tensor(X[s_tp[lo:hi]],
                                         dtype=torch.float32, device=DEV)
                    k1 = torch.exp(-torch.cdist(x1, Xl)**2
                                   / (2 * sig_k**2))
                    k2 = torch.exp(-torch.cdist(x2, Xl)**2
                                   / (2 * sig_k**2))
                    k1 = k1 / k1.sum(1, keepdim=True).clamp_min(1e-30)
                    k2 = k2 / k2.sum(1, keepdim=True).clamp_min(1e-30)
                    A += (k1.T.double() @ k2.double())
                Pn = (A / A.sum(1, keepdim=True).clamp_min(
                    1e-30)).cpu().numpy()
                _, nassign = ltree.query(Xt_ev, k=5)
                H_n = entropy_rows(resolvent_rows(Pn, gamma_c))
                res["r_nystrom"] = rw(np.nanmean((-H_n)[nassign], axis=1))
                # vmi (tuned config)
                t0 = time.time()
                vals = []
                for ns in NET_SEEDS:
                    est = VMI(gamma=gamma_c, z_dim=128, features="coords",
                              hidden=256, n_epochs=150, batch_size=4096,
                              lags_per_step=16, holdout_frac=0.5, lr=3e-4,
                              device=DEV, seed=ns).fit(traj_idx, n_pts,
                                                       X=X)
                    vals.append(pooled_dv(est, groups, rng=ns))
                res["r_vmi"] = rw(np.nanmean(vals, axis=0))
                # td (tuned config)
                vals = []
                for ns in NET_SEEDS:
                    est = TDInfoNCE(gamma=gamma_c, z_dim=128,
                                    features="coords", hidden=256,
                                    n_epochs=300, batch_size=4096,
                                    n_candidates=511, lr=3e-4,
                                    holdout_frac=0.5, lags_per_step=16,
                                    device=DEV, seed=ns).fit(
                        traj_idx, n_pts, X=X)
                    vals.append(pooled_dv(est, groups, rng=ns))
                res["r_td"] = rw(np.nanmean(vals, axis=0))
                res["t"] = round(time.time() - t0, 1)
                rows.append(res)
                print(" ".join(f"{k}={v:.3f}" if isinstance(v, float)
                               else f"{k}={v}" for k, v in res.items()),
                      flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("processed_data/c_noise.csv", index=False)
    print("\n=== noise summary (mean over walk seeds) ===")
    cols = [c for c in df.columns if c.startswith("r_")]
    print(df.groupby(["d", "sigma"])[cols].mean().round(3))


if __name__ == "__main__":
    main()
