"""S2 — the tps frontier: VMI (pair-based, coordinate-free) vs trace-H0
(count-based, coordinate-free) on paired walks as transitions-per-state
scales. Fixed nt=40 starts, T in {70, 170, 375, 715, 1600}
(tps ~ 2..46). gamma=0.9 for both (gate-v2 winner). 5-anchor pooling.

Hypothesis (pre-registered): pair-based estimation shares statistical
strength across states and beats per-row counting at low tps.

Output: processed_data/vmi_tps_ladder.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent / "../11-successor-curvatures"))
from benchmark_kmetric_colosseum import affinity_from_D
from diffusion_curvature.menagerie import (WarpedProduct, dumbbell_profile,
                                           necklace_profile)
from diffusion_curvature.variational import vmi_ensemble

warnings.filterwarnings("ignore")

N = 1400
KNN = 10
GAMMA = 0.9
DEV = "cuda:1" if torch.cuda.is_available() else "cpu"
NT = 40
TS = (70, 170, 375, 715, 1600)
NET_SEEDS = (0, 1, 2, 3)
POOL = 5


def chain_from_D(D):
    W = affinity_from_D(D, k=KNN)
    return W / np.maximum(W.sum(1, keepdims=True), 1e-30)


def resolvent_rows(P, gamma, idx=None, device=DEV):
    n = P.shape[0]
    Pt = torch.as_tensor(P, dtype=torch.float64, device=device)
    M = (1 - gamma) * torch.linalg.inv(
        torch.eye(n, dtype=torch.float64, device=device) - gamma * Pt)
    M = np.maximum(M.cpu().numpy(), 0.0)
    M = M / np.maximum(M.sum(1, keepdims=True), 1e-30)
    return M if idx is None else M[idx]


def entropy_rows(R):
    R = np.clip(R, 1e-30, 1)
    return -(R * np.log(R)).sum(1)


def walks_on(P, nt, T, seed):
    n = P.shape[0]
    rng = np.random.default_rng(seed)
    cum = np.cumsum(P, axis=1)
    traj = np.empty((nt, T + 1), dtype=int)
    traj[:, 0] = rng.integers(0, n, nt)
    for t in range(T):
        u = rng.random(nt)
        traj[:, t + 1] = [np.searchsorted(cum[traj[w, t]], u[w])
                          for w in range(nt)]
    return traj


def make_manifold(mani, d):
    if mani == "dumbbell":
        f, L = dumbbell_profile(beta=0.8)
        return WarpedProduct(f, L, d=d)
    f, L = necklace_profile(b=0.7)
    return WarpedProduct(f, L, d=d, periodic=True)


def main():
    rows = []
    for mani, d in [("necklace", 3), ("dumbbell", 3)]:
        wp = make_manifold(mani, d)
        for wseed in (0, 1):
            m = wp.sample(N, rng=wseed)
            D, ks = m["D"], m["ks_field"]
            s = np.median(D[np.triu_indices_from(D, 1)])
            D, ks = D / s, ks * s**2
            P_full = chain_from_D(D)
            order = np.argsort(ks)
            eval_pts = order[(np.linspace(0.02, 0.98, 48)
                              * (N - 1)).astype(int)]
            kt = ks[eval_pts]

            def r(v, ref):
                mm = np.isfinite(v) & np.isfinite(ref)
                return pearsonr(v[mm], ref[mm])[0] if mm.sum() > 8 else np.nan

            for T in TS:
                traj = walks_on(P_full, NT, T, wseed)
                V_idx, tl = np.unique(traj, return_inverse=True)
                tl = tl.reshape(traj.shape)
                nV = len(V_idx)
                tps = traj.size / nV
                pool_idx = np.argsort(D[np.ix_(eval_pts, V_idx)],
                                      axis=1)[:, :POOL]
                anch_states = np.unique(pool_idx)
                loc = {st: q for q, st in enumerate(anch_states)}
                pool_q = np.vectorize(loc.get)(pool_idx)
                # exact ceiling at same pooled anchors
                H_ex = entropy_rows(
                    resolvent_rows(P_full, GAMMA, V_idx[anch_states]))
                I_exact = np.nanmean((np.log(N) - H_ex)[pool_q], axis=1)
                # counts baseline
                a, b = tl[:, :-1].ravel(), tl[:, 1:].ravel()
                C = sp.coo_matrix((np.ones(len(a)), (a, b)),
                                  shape=(nV, nV)).tocsr()
                C = np.asarray((C + C.T).todense(), dtype=np.float64)
                Pe = (C + 1e-9) / (C + 1e-9).sum(1, keepdims=True)
                H_tr = entropy_rows(resolvent_rows(Pe, GAMMA, anch_states))
                tr_field = np.nanmean((-H_tr)[pool_q], axis=1)
                # VMI
                out = vmi_ensemble(
                    tl, nV, anch_states, seeds=NET_SEEDS, gamma=GAMMA,
                    z_dim=128, n_epochs=90, lags_per_step=16,
                    holdout_frac=0.5, device=DEV)
                bound_f = np.nanmean(out.mi_bound[pool_q], axis=1)
                row = dict(mani=mani, wseed=wseed, T=T, nV=nV,
                           tps=round(tps, 1), cov=round(nV / N, 2),
                           r_exact_ks=r(I_exact, kt),
                           r_traceH0_ks=r(tr_field, kt),
                           r_traceH0_I=r(tr_field, I_exact),
                           r_vmi_ks=r(bound_f, kt),
                           r_vmi_I=r(bound_f, I_exact))
                rows.append(row)
                print(" ".join(f"{k}={v:.3f}" if isinstance(v, float)
                               else f"{k}={v}" for k, v in row.items()),
                      flush=True)
    df = pd.DataFrame(rows)
    Path("processed_data").mkdir(exist_ok=True)
    df.to_csv("processed_data/vmi_tps_ladder.csv", index=False)
    print("\n=== tps ladder (mean over manifolds+seeds) ===")
    print(df.groupby("T")[["tps", "cov", "r_exact_ks", "r_traceH0_ks",
                           "r_vmi_ks", "r_traceH0_I", "r_vmi_I"]]
          .mean().round(3))


if __name__ == "__main__":
    main()
