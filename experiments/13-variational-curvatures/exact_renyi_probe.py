"""Stage-0c probe: does the CAPACITY-flavored target escape the density trap?

Mohamed et al.'s estimator differs from fixed-policy InfoNCE by the
maximization over the action SOURCE: empowerment = channel capacity, not MI
under the behavior policy. On deterministic chains (zettel theorem):
    E(s) = log |S_n(s)|  (reachable-set log-volume, Renyi-0 of occupancy)
vs fixed-policy MI = Shannon entropy of occupancy (Renyi-1) -- which stage-0
showed is density-fragile.

Probe: Renyi-alpha spectrum of the exact resolvent rows, alpha in
{0.25, 0.5, 1, 2} + effective-support N90 (atoms covering 90% mass, a robust
Renyi-0 proxy), field Pearson vs ks, unbiased and density-biased.

Pre-registered prediction: lower alpha (closer to support-counting) degrades
LESS under curvature-independent density bias, because support is less
density-sensitive than probability mass.
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
ALPHAS = (0.25, 0.5, 1.0, 2.0)


def sample_biased(wp, n, beta, rng):
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


def renyi(R, alpha):
    R = np.clip(R, 1e-30, 1)
    if alpha == 1.0:
        return -(R * np.log(R)).sum(1)
    return np.log((R ** alpha).sum(1)) / (1 - alpha)


def n90(R):
    S = -np.sort(-R, axis=1)
    return (np.cumsum(S, axis=1) < 0.9).sum(1) + 1


for mani, d in [("necklace", 3), ("dumbbell", 3)]:
    if mani == "dumbbell":
        f, L = dumbbell_profile(beta=0.8)
        wp = WarpedProduct(f, L, d=d)
    else:
        f, L = necklace_profile(b=0.7)
        wp = WarpedProduct(f, L, d=d, periodic=True)
    for beta in (0.0, 2.0):
        for seed in (0, 1):
            D, ks = sample_biased(wp, N, beta, seed)
            s = np.median(D[np.triu_indices_from(D, 1)])
            D, ks = D / s, ks * s**2
            order = np.argsort(ks)
            ev = order[(np.linspace(0.02, 0.98, 48) * (N - 1)).astype(int)]
            M = resolvent(D)[ev]
            kt = ks[ev]
            cols = [f"a={a}: {pearsonr(-renyi(M, a), kt)[0]:+.3f}"
                    for a in ALPHAS]
            cols.append(f"N90: {pearsonr(-np.log(n90(M)), kt)[0]:+.3f}")
            print(f"{mani} d={d} beta={beta} seed={seed}:  " + "  ".join(cols),
                  flush=True)
