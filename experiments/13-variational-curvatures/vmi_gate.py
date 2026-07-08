"""S1 — fidelity gate for the V1 InfoNCE estimator (VMI).

Necklace + dumbbell d=3, generous walk budget (nt=40, T=1600, tps ~ 21).
Score the VMI readouts against (a) the EXACT resolvent-MI field at the same
anchors and (b) ks; baselines: trace-H0 (counts) on the same walks, and the
exact ceiling. Paired walks across methods.

Gate: Pearson(vmi, I_exact) >= 0.8 and vs-ks within 0.1 of the exact ceiling
for some readout.

Output: processed_data/vmi_gate.csv
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
GAMMA = 0.97
DEV = "cuda:1" if torch.cuda.is_available() else "cpu"
NT, T = 40, 1600
NET_SEEDS = (0, 1)


def chain_from_D(D):
    W = affinity_from_D(D, k=KNN)
    return W / np.maximum(W.sum(1, keepdims=True), 1e-30)


def resolvent_rows(P, idx=None, device=DEV):
    n = P.shape[0]
    Pt = torch.as_tensor(P, dtype=torch.float64, device=device)
    M = (1 - GAMMA) * torch.linalg.inv(
        torch.eye(n, dtype=torch.float64, device=device) - GAMMA * Pt)
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
            H_exact = entropy_rows(resolvent_rows(P_full, eval_pts))
            traj = walks_on(P_full, NT, T, wseed)
            V_idx, tl = np.unique(traj, return_inverse=True)
            tl = tl.reshape(traj.shape)
            nV = len(V_idx)
            tps = traj.size / nV
            # anchors: visited state nearest each eval point (exact D)
            anch_local = np.array([np.argmin(D[e, V_idx]) for e in eval_pts])
            # exact reference at the anchor states themselves
            anch_global = V_idx[anch_local]
            H_exact_anch = entropy_rows(resolvent_rows(P_full, anch_global))
            I_exact = np.log(N) - H_exact_anch
            # baseline: trace-H0 (counts) on same walks
            a, b = tl[:, :-1].ravel(), tl[:, 1:].ravel()
            C = sp.coo_matrix((np.ones(len(a)), (a, b)),
                              shape=(nV, nV)).tocsr()
            C = np.asarray((C + C.T).todense(), dtype=np.float64)
            Pe = (C + 1e-9) / (C + 1e-9).sum(1, keepdims=True)
            H_tr = entropy_rows(resolvent_rows(Pe, anch_local))
            # VMI (tabular, seed-ensembled at estimate level)
            t0 = time.time()
            out = vmi_ensemble(tl, nV, anch_local, seeds=NET_SEEDS,
                               gamma=GAMMA, device=DEV)
            dt = time.time() - t0
            kt = ks[eval_pts]

            def r(v, ref):
                mm = np.isfinite(v) & np.isfinite(ref)
                return pearsonr(v[mm], ref[mm])[0] if mm.sum() > 8 else np.nan

            row = dict(mani=mani, d=d, wseed=wseed, nV=nV, tps=tps,
                       fit_s=dt,
                       r_exact_ks=r(-H_exact, kt),
                       r_traceH0_ks=r(-H_tr, kt),
                       r_traceH0_I=r(-H_tr, I_exact),
                       r_bound_I=r(out.mi_bound, I_exact),
                       r_plugin_I=r(out.mi_plugin, I_exact),
                       r_bound_ks=r(out.mi_bound, kt),
                       r_plugin_ks=r(out.mi_plugin, kt),
                       mean_bound=float(np.nanmean(out.mi_bound)),
                       mean_I_exact=float(np.mean(I_exact)))
            rows.append(row)
            print(" ".join(f"{k}={v:.3f}" if isinstance(v, float)
                           else f"{k}={v}" for k, v in row.items()),
                  flush=True)
    df = pd.DataFrame(rows)
    Path("processed_data").mkdir(exist_ok=True)
    df.to_csv("processed_data/vmi_gate.csv", index=False)
    print("\n=== gate summary ===")
    print(df.groupby("mani")[["r_exact_ks", "r_traceH0_ks", "r_bound_I",
                              "r_plugin_I", "r_bound_ks",
                              "r_plugin_ks"]].mean().round(3))


if __name__ == "__main__":
    main()
