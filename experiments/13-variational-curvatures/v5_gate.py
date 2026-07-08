"""S1 gate for V5 (Mohamed/BA decoder): necklace + dumbbell d=3,
nt=40, T=1600, gamma=0.9, 4 net seeds, 5-anchor pooling.

BA targets I(A;S+|s) = H(S+|s) (deterministic path-dynamics), so the
curvature orientation is -BA; reference field is the exact resolvent
entropy H at the same pooled anchors. Also reports the capacity readout
(log Z, orientation +: fewer equivalent paths = more reachable volume =
negative curvature... recorded both ways, judged empirically).

Output: processed_data/v5_gate.csv
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent / "../11-successor-curvatures"))
from benchmark_kmetric_colosseum import affinity_from_D
from diffusion_curvature.menagerie import (WarpedProduct, dumbbell_profile,
                                           necklace_profile)
from diffusion_curvature.variational import ba_ensemble

warnings.filterwarnings("ignore")

N = 1400
KNN = 10
GAMMA = 0.9
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
NT, T = 40, 1600
NET_SEEDS = (0, 1, 2, 3)
POOL = 5


def chain_from_D(D):
    W = affinity_from_D(D, k=KNN)
    return W / np.maximum(W.sum(1, keepdims=True), 1e-30)


def resolvent_entropy(P, gamma, idx, device=DEV):
    n = P.shape[0]
    Pt = torch.as_tensor(P, dtype=torch.float64, device=device)
    M = (1 - gamma) * torch.linalg.inv(
        torch.eye(n, dtype=torch.float64, device=device) - gamma * Pt)
    M = np.maximum(M.cpu().numpy(), 0.0)
    R = M[idx] / np.maximum(M[idx].sum(1, keepdims=True), 1e-30)
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
            traj = walks_on(P_full, NT, T, wseed)
            V_idx, tl = np.unique(traj, return_inverse=True)
            tl = tl.reshape(traj.shape)
            nV = len(V_idx)
            pool_idx = np.argsort(D[np.ix_(eval_pts, V_idx)],
                                  axis=1)[:, :POOL]
            anch_states = np.unique(pool_idx)
            loc = {st: q for q, st in enumerate(anch_states)}
            pool_q = np.vectorize(loc.get)(pool_idx)
            H_ex = resolvent_entropy(P_full, GAMMA, V_idx[anch_states])
            H_exact = np.nanmean(H_ex[pool_q], axis=1)

            def r(v, ref):
                mm = np.isfinite(v) & np.isfinite(ref)
                return pearsonr(v[mm], ref[mm])[0] if mm.sum() > 8 else np.nan

            t0 = time.time()
            out = ba_ensemble(tl, nV, anch_states, seeds=NET_SEEDS,
                              gamma=GAMMA, z_dim=64, n_epochs=30,
                              lags_per_step=4, device=DEV)
            dt = time.time() - t0
            ba_f = np.nanmean(out.ba[pool_q], axis=1)
            cap_f = np.nanmean(out.capacity[pool_q], axis=1)
            row = dict(mani=mani, wseed=wseed, nV=nV, fit_s=round(dt, 1),
                       r_exact_ks=r(-H_exact, kt),
                       r_ba_H=r(out_ba := ba_f, H_exact),
                       r_ba_ks=r(-out_ba, kt),
                       r_cap_H=r(cap_f, H_exact),
                       r_cap_ks=r(-cap_f, kt),
                       mean_ba=float(np.nanmean(ba_f)),
                       mean_H=float(np.mean(H_exact)))
            rows.append(row)
            print(" ".join(f"{k}={v:.3f}" if isinstance(v, float)
                           else f"{k}={v}" for k, v in row.items()),
                  flush=True)
    df = pd.DataFrame(rows)
    Path("processed_data").mkdir(exist_ok=True)
    df.to_csv("processed_data/v5_gate.csv", index=False)
    print("\n=== V5 gate summary ===")
    print(df.groupby("mani")[["r_exact_ks", "r_ba_H", "r_ba_ks",
                              "r_cap_H", "r_cap_ks"]].mean().round(3))


if __name__ == "__main__":
    main()
