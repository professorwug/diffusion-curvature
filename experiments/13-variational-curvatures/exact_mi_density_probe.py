"""Stage-0b probe: density-bias stress test of exact resolvent entropy (sent)
vs exact pointwise resolvent MI.

Bias is applied along the sphere-factor direction u1 (curvature depends only
on the profile coordinate r), so density is decoupled from curvature by
construction: keep-probability prop. to exp(beta * u1).

Prediction under test: sent (-H) confounds density with volume growth; the MI
field divides the marginal out and should degrade less.
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


def sample_biased(wp: WarpedProduct, n: int, beta: float, rng):
    """Replicates WarpedProduct.sample but with density bias exp(beta*u1)
    on the sphere factor (curvature-independent direction)."""
    rng = np.random.default_rng(rng)
    n_big = 4 * n
    pdf = np.maximum(wp.fg, 0) ** (wp.d - 1)
    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]
    r = np.interp(rng.random(n_big), cdf, wp.rg)
    u = rng.normal(size=(n_big, wp.d))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    w = np.exp(beta * u[:, 0])
    keep = rng.choice(n_big, size=n, replace=False, p=w / w.sum())
    r, u = r[keep], u[keep]
    cosang = np.clip(u @ u.T, -1, 1)
    ang = np.arccos(cosang)
    r1 = np.repeat(r, n).reshape(n, n)
    D = wp.pair_distance(r1.ravel(), r1.T.ravel(), ang.ravel()).reshape(n, n)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D, wp.scalar(r)


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
    rho = np.clip(M.mean(0), 1e-15, 1)
    I = (R * (np.log(R) - np.log(rho)[None, :])).sum(1)
    return H, I


for mani, d in [("necklace", 3), ("dumbbell", 3)]:
    if mani == "dumbbell":
        f, L = dumbbell_profile(beta=0.8)
        wp = WarpedProduct(f, L, d=d)
    else:
        f, L = necklace_profile(b=0.7)
        wp = WarpedProduct(f, L, d=d, periodic=True)
    for beta in (0.0, 1.0, 2.0):
        for seed in (0, 1):
            D, ks = sample_biased(wp, N, beta, seed)
            s = np.median(D[np.triu_indices_from(D, 1)])
            D, ks = D / s, ks * s**2
            order = np.argsort(ks)
            ev = order[(np.linspace(0.02, 0.98, 48) * (N - 1)).astype(int)]
            H, I = fields(resolvent(D))
            print(f"{mani} d={d} beta={beta} seed={seed}: "
                  f"sent(-H) r={pearsonr(-H[ev], ks[ev])[0]:+.3f} | "
                  f"exact-MI r={pearsonr(I[ev], ks[ev])[0]:+.3f}", flush=True)
