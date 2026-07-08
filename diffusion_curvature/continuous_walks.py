"""Continuous Brownian trajectories on warped-product manifolds (C-series).

Simulates Brownian motion (generator Delta/2) on g = dr^2 + f(r)^2 g_{S^{d-1}}
directly in intrinsic coordinates (r, u):
    dr = (d-1)/2 * (f'(r)/f(r)) dt + dW_r          (radial 1D diffusion)
    u  : spherical BM on S^{d-1} with time increment dt/f(r)^2
        (tangent-Gaussian projection scheme, substepped when f is small)

Every visited state is unique — the tabular "counts" shortcut does not exist
here. Channels consume the smooth (non-isometric, realistic) embedding
    chi(r, u) = (a cos(2 pi r/L), a sin(2 pi r/L), f(r) u),  a = L/(2 pi)
with |d chi|^2 = (1 + f'^2) dr^2 + f^2 |du|^2 — mild radial distortion.
Ground-truth scalar curvature is analytic at every point: wp.scalar(r).

Periodic profiles only (caps are coordinate singularities; use necklaces).
"""
from __future__ import annotations

import numpy as np


def _periodic_gradient(rg: np.ndarray, fg: np.ndarray, L: float) -> np.ndarray:
    pad = 50
    fg_p = np.concatenate([fg[-pad - 1:-1], fg, fg[1:pad + 1]])
    rg_p = np.concatenate([rg[-pad - 1:-1] - L, rg, rg[1:pad + 1] + L])
    return np.gradient(fg_p, rg_p)[pad:-pad]


def brownian_walks(wp, nt: int, T: int, dt: float, rng=None,
                   max_sphere_dt: float = 0.01):
    """Simulate nt Brownian walks of T steps (time step dt, in the
    manifold's RAW length units; time ~ length^2).

    Returns dict(r=(nt,T+1), u=(nt,T+1,d), ks=(nt,T+1)).
    """
    if not wp.periodic:
        raise ValueError("continuous walks: periodic profiles only (v1)")
    rng = np.random.default_rng(rng)
    d, L = wp.d, wp.L
    fp_g = _periodic_gradient(wp.rg, wp.fg, L)
    # stationary start: r ~ f^{d-1} volume element, u uniform on S^{d-1}
    pdf = np.maximum(wp.fg, 0) ** (d - 1)
    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]
    r = np.interp(rng.random(nt), cdf, wp.rg)
    u = rng.normal(size=(nt, d))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    R = np.empty((nt, T + 1))
    U = np.empty((nt, T + 1, d))
    R[:, 0], U[:, 0] = r, u
    sq_dt = np.sqrt(dt)
    for t in range(T):
        f = np.interp(r, wp.rg, wp.fg)
        fp = np.interp(r, wp.rg, fp_g)
        r = r + 0.5 * (d - 1) * (fp / f) * dt + sq_dt * rng.normal(size=nt)
        r = np.mod(r, L)
        # sphere factor: BM with time dt/f^2, substepped for stability
        dt_u = dt / f**2
        n_sub = np.maximum(1, np.ceil(dt_u / max_sphere_dt)).astype(int)
        max_sub = int(n_sub.max())
        for j in range(max_sub):
            act = j < n_sub
            h = np.where(act, dt_u / n_sub, 0.0)
            xi = rng.normal(size=(nt, d))
            xi -= (xi * u).sum(1, keepdims=True) * u
            u = u + np.sqrt(h)[:, None] * xi
            u /= np.linalg.norm(u, axis=1, keepdims=True)
        R[:, t + 1], U[:, t + 1] = r, u
    return dict(r=R, u=U, ks=wp.scalar(R))


def chi_embed(wp, r: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Smooth (non-isometric) embedding into R^{d+2}."""
    a = wp.L / (2 * np.pi)
    f = np.interp(r, wp.rg, wp.fg)
    ang = 2 * np.pi * r / wp.L
    return np.concatenate([a * np.cos(ang)[..., None],
                           a * np.sin(ang)[..., None],
                           f[..., None] * u], axis=-1)
