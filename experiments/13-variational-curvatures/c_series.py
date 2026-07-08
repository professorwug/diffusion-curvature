"""C-series — the continuous-trajectory race.

Brownian trajectories on periodic warped products (every visited state
unique — no tabular counts exist). All channels consume the same smooth
chi-embedding coordinates of the same trajectories; ground truth ks is
analytic at every point.

Channels:
  bin_count[Nc]   — Voronoi discretization at Nc centers -> empirical chain
                    -> resolvent entropy (counts transplanted; scale sweep)
  knn_graph       — kNN graph on subsampled visited points, graph sent
                    (T-series coord channel; ignores dynamics)
  nystrom         — kernel-smoothed one-step operator on landmarks ->
                    numerical resolvent ("learn one-step + resolvent")
  vmi_coord       — flat lag-pair InfoNCE critic (coordinate MLP), pooled
                    DV readout
  td_infonce      — bootstrapped contrastive critic (one-step pairs +
                    Bellman bootstrap), same readout
Oracle ceiling: dense iid sample kNN chain sent (gamma=0.9).

Pre-registered: bin_count dominated at realistic budgets (S2 inverts);
td_infonce >= vmi_coord everywhere; td_infonce vs nystrom is the open race.

Output: processed_data/c_series.csv
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
from diffusion_curvature.continuous_walks import brownian_walks, chi_embed
from diffusion_curvature.menagerie import WarpedProduct, necklace_profile
from diffusion_curvature.variational import VMI, TDInfoNCE

warnings.filterwarnings("ignore")

DEV = "cuda:1" if torch.cuda.is_available() else "cpu"
NT, T = 40, 2000
DT_NORM = 2e-3           # time step in unit-median-geodesic units
SPREAD_T = 0.09          # target horizon time: spread = sqrt(d*t_h) = 0.3
N_ORACLE = 5000
N_EVAL = 48
POOL_STATES = 32         # pooled readout: nearest visited states per target
NET_SEEDS = (0, 1)
BIN_NC = (500, 1500, 4000)
KNN_SUB = 2500


def entropy_rows(R):
    R = np.clip(R, 1e-30, 1)
    return -(R * np.log(R)).sum(1)


def resolvent_rows(P, gamma, idx=None, device=DEV):
    n = P.shape[0]
    Pt = torch.as_tensor(P, dtype=torch.float64, device=device)
    M = (1 - gamma) * torch.linalg.inv(
        torch.eye(n, dtype=torch.float64, device=device) - gamma * Pt)
    M = np.maximum(M.cpu().numpy(), 0.0)
    M = M / np.maximum(M.sum(1, keepdims=True), 1e-30)
    return M if idx is None else M[idx]


def pooled_dv(est, member_groups, n_neg=8192, rng=None):
    """DV readout over anchor GROUPS (continuous regime: each state visited
    once). For group B: mean f over held pairs with s in B, minus the
    group-mean log-mean-exp of f(s, negatives)."""
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
    neg = torch.as_tensor(
        rng.integers(0, est.n_states, n_neg), dtype=torch.long, device=dev)
    out = np.empty(len(member_groups))
    with torch.no_grad():
        for q, members in enumerate(member_groups):
            Mt = torch.as_tensor(members, dtype=torch.long, device=dev)
            fr = est.net.score(Mt, neg)                  # (m, n_neg)
            logZ = (torch.logsumexp(fr, dim=1)
                    - np.log(n_neg)).mean().item()
            tot, cnt = sums[members].sum(), cnts[members].sum()
            out[q] = tot / cnt - logZ if cnt >= 8 else np.nan
    return out


def main():
    rows = []
    for d in (3, 4):
        f, L = necklace_profile(b=0.7)
        wp = WarpedProduct(f, L, d=d, periodic=True)
        # oracle sample: scale, eval targets, ceiling
        m = wp.sample(N_ORACLE, rng=100 + d)
        s = np.median(m["D"][np.triu_indices(N_ORACLE, 1)])
        D_or, ks_or = m["D"] / s, m["ks_field"] * s**2
        # oracle sent (gamma=0.9, program standard)
        W = affinity_from_D(D_or, k=10)
        P_or = W / np.maximum(W.sum(1, keepdims=True), 1e-30)
        order = np.argsort(ks_or)
        eval_idx = order[(np.linspace(0.02, 0.98, N_EVAL)
                          * (N_ORACLE - 1)).astype(int)]
        H_or = entropy_rows(resolvent_rows(P_or, 0.9, eval_idx))
        kt = ks_or[eval_idx]
        # oracle sample has no u coordinates; rebuild eval-target chi from r
        # by sampling a random sphere direction consistent with distances:
        # instead, draw eval targets from a walk-independent BM run.
        gamma_c = 1 - d * DT_NORM / SPREAD_T

        def r(v):
            mm = np.isfinite(v) & np.isfinite(kt)
            return pearsonr(v[mm], kt[mm])[0] if mm.sum() > 8 else np.nan

        for wseed in (0, 1):
            walks = brownian_walks(wp, NT, T, dt=DT_NORM * s**2,
                                   rng=wseed)
            chi = chi_embed(wp, walks["r"], walks["u"]) / s
            ks_pts = walks["ks"] * s**2
            n_pts = NT * (T + 1)
            X = chi.reshape(n_pts, -1)
            ks_flat = ks_pts.ravel()
            traj_idx = np.arange(n_pts).reshape(NT, T + 1)
            tree = cKDTree(X)
            # eval targets: independent BM run, stratified by ks
            tw = brownian_walks(wp, 8, 1200, dt=DT_NORM * s**2,
                                rng=1000 + wseed)
            tchi = (chi_embed(wp, tw["r"], tw["u"]) / s).reshape(-1, X.shape[1])
            tks = (tw["ks"] * s**2).ravel()
            torder = np.argsort(tks)
            targets = torder[(np.linspace(0.02, 0.98, N_EVAL)
                              * (len(tks) - 1)).astype(int)]
            Xt_ev, kt_w = tchi[targets], tks[targets]

            def rw(v):
                mm = np.isfinite(v) & np.isfinite(kt_w)
                return pearsonr(v[mm], kt_w[mm])[0] if mm.sum() > 8 else np.nan

            _, grp = tree.query(Xt_ev, k=POOL_STATES)
            groups = list(grp)
            res = dict(d=d, wseed=wseed, gamma_c=round(gamma_c, 4),
                       r_oracle=r(-H_or))
            # ---- bin_count at three resolutions
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
                res[f"r_bin{Nc}"] = rw(np.nanmean((-H_b)[tassign], axis=1))
            # ---- knn_graph on subsampled visited points
            rngk = np.random.default_rng(20 + wseed)
            sub = rngk.choice(n_pts, KNN_SUB, replace=False)
            Dk = np.linalg.norm(X[sub][:, None] - X[sub][None], axis=-1)
            Wk = affinity_from_D(Dk, k=10)
            Pk = Wk / np.maximum(Wk.sum(1, keepdims=True), 1e-30)
            ktree = cKDTree(X[sub])
            _, kassign = ktree.query(Xt_ev, k=5)
            H_k = entropy_rows(resolvent_rows(Pk, 0.9))
            res["r_knn"] = rw(np.nanmean((-H_k)[kassign], axis=1))
            # ---- nystrom one-step operator on landmarks
            Nl = 1500
            rngn = np.random.default_rng(30 + wseed)
            lm = rngn.choice(n_pts, Nl, replace=False)
            ltree = cKDTree(X[lm])
            sig = np.median(ltree.query(X[lm], k=2)[0][:, 1]) * 1.5
            Xl = torch.as_tensor(X[lm], dtype=torch.float32, device=DEV)
            s_t = traj_idx[:, :-1].ravel()
            s_tp = traj_idx[:, 1:].ravel()
            A = torch.zeros((Nl, Nl), dtype=torch.float64, device=DEV)
            for lo in range(0, len(s_t), 16384):
                hi = min(lo + 16384, len(s_t))
                x1 = torch.as_tensor(X[s_t[lo:hi]], dtype=torch.float32,
                                     device=DEV)
                x2 = torch.as_tensor(X[s_tp[lo:hi]], dtype=torch.float32,
                                     device=DEV)
                k1 = torch.exp(-torch.cdist(x1, Xl)**2 / (2 * sig**2))
                k2 = torch.exp(-torch.cdist(x2, Xl)**2 / (2 * sig**2))
                k1 = k1 / k1.sum(1, keepdim=True).clamp_min(1e-30)
                k2 = k2 / k2.sum(1, keepdim=True).clamp_min(1e-30)
                A += (k1.T.double() @ k2.double())
            Pn = (A / A.sum(1, keepdim=True).clamp_min(1e-30)).cpu().numpy()
            _, nassign = ltree.query(Xt_ev, k=5)
            H_n = entropy_rows(resolvent_rows(Pn, gamma_c))
            res["r_nystrom"] = rw(np.nanmean((-H_n)[nassign], axis=1))
            # ---- vmi_coord (flat lag pairs, coordinate MLP)
            t0 = time.time()
            vals = []
            for ns in NET_SEEDS:
                est = VMI(gamma=gamma_c, z_dim=64, features="coords",
                          hidden=256, n_epochs=40, batch_size=4096,
                          lags_per_step=8, holdout_frac=0.5, lr=1e-3,
                          device=DEV, seed=ns).fit(traj_idx, n_pts, X=X)
                vals.append(pooled_dv(est, groups, rng=ns))
            res["r_vmi"] = rw(np.nanmean(vals, axis=0))
            res["t_vmi"] = round(time.time() - t0, 1)
            # ---- td_infonce (one-step pairs + bootstrap)
            t0 = time.time()
            vals = []
            for ns in NET_SEEDS:
                est = TDInfoNCE(gamma=gamma_c, z_dim=64, features="coords",
                                hidden=256, n_epochs=150, batch_size=4096,
                                n_candidates=511, lr=1e-3,
                                holdout_frac=0.5, lags_per_step=8,
                                device=DEV, seed=ns).fit(traj_idx, n_pts,
                                                         X=X)
                vals.append(pooled_dv(est, groups, rng=ns))
            res["r_td"] = rw(np.nanmean(vals, axis=0))
            res["t_td"] = round(time.time() - t0, 1)
            rows.append(res)
            print(" ".join(f"{k}={v:.3f}" if isinstance(v, float)
                           else f"{k}={v}" for k, v in res.items()),
                  flush=True)
    df = pd.DataFrame(rows)
    Path("processed_data").mkdir(exist_ok=True)
    df.to_csv("processed_data/c_series.csv", index=False)
    print("\n=== C-series summary (mean over walk seeds) ===")
    cols = [c for c in df.columns if c.startswith("r_")]
    print(df.groupby("d")[cols].mean().round(3))


if __name__ == "__main__":
    main()
