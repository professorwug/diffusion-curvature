"""S1 gate v2 — variance-attacked VMI: more lags, 50% holdout, z=128,
4 net seeds, 5-anchor neighborhood pooling of the readout, gamma sweep.

Output: processed_data/vmi_gate_v2.csv
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
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent / "../11-successor-curvatures"))
from benchmark_kmetric_colosseum import affinity_from_D
from diffusion_curvature.menagerie import (WarpedProduct, dumbbell_profile,
                                           necklace_profile)
from diffusion_curvature.variational import vmi_ensemble

warnings.filterwarnings("ignore")

N = 1400
KNN = 10
DEV = "cuda:1" if torch.cuda.is_available() else "cpu"
NT, T = 40, 1600
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


def pooled(field_at_states, pool_idx):
    """pool_idx: (n_eval, POOL) local state indices; field over all states."""
    return np.nanmean(field_at_states[pool_idx], axis=1)


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
            traj = walks_on(P_full, NT, T, wseed)
            V_idx, tl = np.unique(traj, return_inverse=True)
            tl = tl.reshape(traj.shape)
            nV = len(V_idx)
            # pooled anchors: POOL nearest visited states per eval point
            pool_idx = np.argsort(D[np.ix_(eval_pts, V_idx)],
                                  axis=1)[:, :POOL]
            anch_states = np.unique(pool_idx)
            loc = {st: q for q, st in enumerate(anch_states)}
            pool_q = np.vectorize(loc.get)(pool_idx)
            kt = ks[eval_pts]

            def r(v, ref):
                mm = np.isfinite(v) & np.isfinite(ref)
                return pearsonr(v[mm], ref[mm])[0] if mm.sum() > 8 else np.nan

            # counts baseline (gamma-independent build)
            a, b = tl[:, :-1].ravel(), tl[:, 1:].ravel()
            C = sp.coo_matrix((np.ones(len(a)), (a, b)),
                              shape=(nV, nV)).tocsr()
            C = np.asarray((C + C.T).todense(), dtype=np.float64)
            Pe = (C + 1e-9) / (C + 1e-9).sum(1, keepdims=True)
            for gamma in (0.9, 0.97):
                H_ex_states = entropy_rows(
                    resolvent_rows(P_full, gamma, V_idx[anch_states]))
                I_exact = pooled(np.log(N) - H_ex_states, pool_q)
                r_exact_ks = r(I_exact, kt)
                H_tr = entropy_rows(resolvent_rows(Pe, gamma, anch_states))
                tr_field = pooled(-H_tr, pool_q)
                t0 = time.time()
                out = vmi_ensemble(
                    tl, nV, anch_states, seeds=NET_SEEDS, gamma=gamma,
                    z_dim=128, n_epochs=90, lags_per_step=16,
                    holdout_frac=0.5, device=DEV)
                dt = time.time() - t0
                bound_f = pooled(out.mi_bound, pool_q)
                plug_f = pooled(out.mi_plugin, pool_q)
                row = dict(mani=mani, d=d, wseed=wseed, gamma=gamma,
                           nV=nV, fit_s=round(dt, 1),
                           r_exact_ks=r_exact_ks,
                           r_traceH0_I=r(tr_field, I_exact),
                           r_traceH0_ks=r(tr_field, kt),
                           r_bound_I=r(bound_f, I_exact),
                           r_bound_ks=r(bound_f, kt),
                           r_plugin_I=r(plug_f, I_exact),
                           r_plugin_ks=r(plug_f, kt))
                rows.append(row)
                print(" ".join(f"{k}={v:.3f}" if isinstance(v, float)
                               else f"{k}={v}" for k, v in row.items()),
                      flush=True)
    df = pd.DataFrame(rows)
    Path("processed_data").mkdir(exist_ok=True)
    df.to_csv("processed_data/vmi_gate_v2.csv", index=False)
    print("\n=== v2 summary (mean over walk seeds) ===")
    print(df.groupby(["mani", "gamma"])[
        ["r_exact_ks", "r_traceH0_ks", "r_bound_I", "r_bound_ks",
         "r_plugin_I", "r_plugin_ks"]].mean().round(3))


if __name__ == "__main__":
    main()
