"""Stage-0 probe: is the EXACT pointwise resolvent MI field a good curvature
correlate, relative to the exact resolvent entropy (sent)?

Pointwise MI (the quantity an InfoNCE critic estimates at its optimum):
    I(s) = sum_j M(s,j) * log( M(s,j) / rho(j) ),   rho = E_anchor[M(anchor,·)]
vs sent:
    H(s) = -sum_j M(s,j) log M(s,j)          (curvature orientation: -H)

If the marginal rho were uniform, I(s) = log n - H(s) exactly; with a
non-uniform marginal (kNN-chain stationary structure, density bias) the two
diverge — and MI is the density-ratio object that should be more robust.

Also probes the density-bias axis: resample the same manifold with a biased
density and compare field Pearson of both channels.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent / "../11-successor-curvatures"))
from benchmark_kmetric_colosseum import affinity_from_D
from diffusion_curvature.menagerie import (WarpedProduct, dumbbell_profile,
                                           necklace_profile)

warnings.filterwarnings("ignore")

N = 1400
KNN = 10
GAMMA = 0.97
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


def resolvent(D):
    n = D.shape[0]
    W = affinity_from_D(D, k=KNN)
    P = W / np.maximum(W.sum(1, keepdims=True), 1e-30)
    Pt = torch.as_tensor(P, dtype=torch.float64, device=DEV)
    M = (1 - GAMMA) * torch.linalg.inv(
        torch.eye(n, dtype=torch.float64, device=DEV) - GAMMA * Pt)
    M = np.maximum(M.cpu().numpy(), 0.0)
    return M / np.maximum(M.sum(1, keepdims=True), 1e-30)


def fields(M):
    R = np.clip(M, 1e-15, 1)
    H = -(R * np.log(R)).sum(1)
    rho = np.clip(M.mean(0), 1e-15, 1)          # uniform-anchor marginal
    I = (R * (np.log(R) - np.log(rho)[None, :])).sum(1)
    return H, I


for mani, d in [("necklace", 3), ("necklace", 5), ("dumbbell", 3), ("dumbbell", 4)]:
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
        order = np.argsort(ks)
        ev = order[(np.linspace(0.02, 0.98, 48) * (N - 1)).astype(int)]
        H, I = fields(resolvent(D))
        r_H = pearsonr(-H[ev], ks[ev])[0]
        r_I = pearsonr(I[ev], ks[ev])[0]
        r_HI = pearsonr(-H[ev], I[ev])[0]
        print(f"{mani} d={d} seed={seed}: sent(-H) r={r_H:+.3f} | "
              f"exact-MI r={r_I:+.3f} | corr(-H, I)={r_HI:+.3f}", flush=True)
