"""Track 3 — trace-estimated resolvent entropy (H0) on menagerie fields.

Exact H0 (gamma-resolvent entropy) is the new best field channel. Here:
estimate it from WALKS only, two ways:
  coord      : visited-cloud kNN graph (coordinates of visited states)
  coordfree  : empirical transition chain P_hat (counts, symmetrized)
Budgets: nt=40, T in {400, 1600} (tps ~5.5 / ~21 per T5).
Field readout at visited points nearest stratified eval targets; compare to
exact-H0 field r on the same manifolds (0.93 necklace / 0.42 dumbbell d3).
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
from diffusion_curvature.menagerie import (WarpedProduct, dumbbell_profile,
                                           necklace_profile)
from benchmark_kmetric_colosseum import affinity_from_D

warnings.filterwarnings("ignore")

N = 1400
KNN = 10
GAMMA = 0.97
DEV = "cuda:0"


def resolvent_entropy(P_np, idx, device=DEV):
    n = P_np.shape[0]
    Pt = torch.as_tensor(P_np, dtype=torch.float64, device=device)
    M = (1 - GAMMA) * torch.linalg.inv(
        torch.eye(n, dtype=torch.float64, device=device) - GAMMA * Pt)
    Mn = np.maximum(M.cpu().numpy(), 0.0)
    R = Mn[idx]
    R = R / np.maximum(R.sum(1, keepdims=True), 1e-30)
    R = np.clip(R, 1e-15, 1)
    return -(R * np.log(R)).sum(1)


def walks_on(D, nt, T, seed):
    """Random walks on the exact-distance kNN chain."""
    n = D.shape[0]
    W = affinity_from_D(D, k=KNN)
    P = W / np.maximum(W.sum(1, keepdims=True), 1e-30)
    rng = np.random.default_rng(seed)
    cum = np.cumsum(P, axis=1)
    traj = np.empty((nt, T + 1), dtype=int)
    traj[:, 0] = rng.integers(0, n, nt)
    for t in range(T):
        u = rng.random(nt)
        traj[:, t + 1] = [np.searchsorted(cum[traj[w, t]], u[w])
                          for w in range(nt)]
    return traj


for mani, d in [("dumbbell", 3), ("necklace", 3)]:
    if mani == "dumbbell":
        f, L = dumbbell_profile(beta=0.8)
        wp = WarpedProduct(f, L, d=d)
    else:
        f, L = necklace_profile(b=0.7)
        wp = WarpedProduct(f, L, d=d, periodic=True)
    for seed in (0, 1):
        m = wp.sample(N, rng=seed)
        D, ks = m["D"], m["ks_field"]
        s = np.median(D[np.triu_indices_from(D, 1)])
        D, ks = D / s, ks * s**2
        # exact reference
        W = affinity_from_D(D, k=KNN)
        P_full = W / np.maximum(W.sum(1, keepdims=True), 1e-30)
        order = np.argsort(ks)
        eval_pts = order[(np.linspace(0.02, 0.98, 40) * (N - 1)).astype(int)]
        H_exact = resolvent_entropy(P_full, eval_pts)
        r_exact = pearsonr(-H_exact, ks[eval_pts])[0]
        for T in (400, 1600):
            traj = walks_on(D, 40, T, seed)
            V_idx, tl = np.unique(traj, return_inverse=True)
            tl = tl.reshape(traj.shape)
            nV = len(V_idx)
            tps = traj.size / nV
            X_like = D[np.ix_(V_idx, V_idx)]
            # anchors: visited nearest each eval target (by exact D)
            anch = np.array([np.argmin(D[e, V_idx]) for e in eval_pts])
            # coord: kNN graph of visited cloud (distance submatrix)
            Wv = affinity_from_D(X_like, k=min(KNN, nV - 2))
            Pv = Wv / np.maximum(Wv.sum(1, keepdims=True), 1e-30)
            H_coord = resolvent_entropy(Pv, anch)
            # coordfree: empirical chain
            a = tl[:, :-1].ravel(); b = tl[:, 1:].ravel()
            C = sp.coo_matrix((np.ones(len(a)), (a, b)),
                              shape=(nV, nV)).tocsr()
            C = np.asarray((C + C.T).todense(), dtype=np.float64)
            Pe = (C + 1e-9) / (C + 1e-9).sum(1, keepdims=True)
            H_free = resolvent_entropy(Pe, anch)
            kt = ks[eval_pts]
            print(f"{mani} d={d} s={seed} T={T} (cov={nV/N:.2f} tps={tps:.1f}): "
                  f"exact r={r_exact:+.2f} | coord r={pearsonr(-H_coord, kt)[0]:+.2f} "
                  f"| coordfree r={pearsonr(-H_free, kt)[0]:+.2f}", flush=True)
