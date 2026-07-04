"""The Curvature Menagerie — exact-geometry benchmark manifolds.

Design principle ("distance-matrix mode"): estimators consume distance
matrices, so a benchmark manifold needs only (i) an intrinsic sampling law
and (ii) exact geodesic distances. Euclidean embeddings are an optional
realism axis. This unlocks hyperbolic geometry and warped-product curvature
landscapes with per-point analytic ground truth.

Tier 1 (homogeneous, closed-form): sphere_sf, hyperbolic, torus_flat,
product. Tier 2 (warped products, per-point fields): WarpedProduct with
profile factories (dumbbell_profile, sphere_profile).

Every generator returns a dict:
    D         (n, n) exact geodesic distances
    X         optional isometric embedding (or None)
    ks_field  (n,) scalar curvature at each sample
    meta      dict (name, dim, params, sec ranges, ...)

Scalar-curvature conventions: S^d(r): d(d-1)/r^2; H^d(kappa): -d(d-1)*kappa;
products: sum of factor scalars; warped product over S^{d-1}:
    R(r) = 2(d-1)(-f''/f) + (d-1)(d-2)(1-f'^2)/f^2.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra


# ---------------------------------------------------------------------------
# Tier 1 — homogeneous space forms and products
# ---------------------------------------------------------------------------


def sphere_sf(n: int, d: int, r: float = 1.0, rng=None) -> dict:
    """Round sphere S^d(r), uniform sampling, exact great-circle distances."""
    rng = np.random.default_rng(rng)
    u = rng.normal(size=(n, d + 1))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    X = r * u
    cosang = np.clip(u @ u.T, -1.0, 1.0)
    D = r * np.arccos(cosang)
    ks = np.full(n, d * (d - 1) / r**2)
    return dict(D=D, X=X, ks_field=ks,
                meta=dict(name=f"S{d}(r={r:g})", dim=d, r=r,
                          sec=(1 / r**2, 1 / r**2)))


def hyperbolic(n: int, d: int, kappa: float = 1.0, R: float = 2.0,
               rng=None) -> dict:
    """Ball of geodesic radius R in H^d with curvature -kappa; hyperboloid
    model; intrinsic uniform sampling (radial density ~ sinh^{d-1});
    exact arccosh distances. No Euclidean embedding (X=None)."""
    rng = np.random.default_rng(rng)
    sk = np.sqrt(kappa)
    # radial density prop. to sinh(sk*rho)^{d-1} on [0, R]
    grid = np.linspace(1e-9, R, 4096)
    pdf = np.sinh(sk * grid) ** (d - 1)
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    rho = np.interp(rng.random(n), cdf, grid)
    u = rng.normal(size=(n, d))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    # hyperboloid coords scaled so <x,x>_M = 1/kappa
    x0 = np.cosh(sk * rho) / sk
    xi = (np.sinh(sk * rho) / sk)[:, None] * u
    mink = np.outer(x0, x0) - xi @ xi.T
    D = np.arccosh(np.clip(kappa * mink, 1.0, None)) / sk
    ks = np.full(n, -d * (d - 1) * kappa)
    return dict(D=D, X=None, ks_field=ks,
                meta=dict(name=f"H{d}(k={kappa:g})", dim=d, kappa=kappa, R=R,
                          sec=(-kappa, -kappa)))


def torus_flat(n: int, d: int, L: float = 2 * np.pi, rng=None) -> dict:
    """Flat torus T^d = [0,L)^d, minimum-image distances, K = 0."""
    rng = np.random.default_rng(rng)
    X = rng.random((n, d)) * L
    D2 = np.zeros((n, n))
    for k in range(d):
        dk = np.abs(X[:, k, None] - X[None, :, k])
        dk = np.minimum(dk, L - dk)
        D2 += dk**2
    return dict(D=np.sqrt(D2), X=None, ks_field=np.zeros(n),
                meta=dict(name=f"T{d}(L={L:g})", dim=d, L=L, sec=(0.0, 0.0)))


def product(*factors: dict) -> dict:
    """Riemannian product: d^2 = sum d_i^2; scalar = sum of scalars.
    Factors must share n."""
    n = factors[0]["D"].shape[0]
    assert all(f["D"].shape[0] == n for f in factors)
    D = np.sqrt(sum(f["D"] ** 2 for f in factors))
    ks = sum(f["ks_field"] for f in factors)
    dim = sum(f["meta"]["dim"] for f in factors)
    secs = [s for f in factors for s in f["meta"]["sec"]]
    name = " x ".join(f["meta"]["name"] for f in factors)
    Xs = [f["X"] for f in factors]
    X = np.hstack(Xs) if all(x is not None for x in Xs) else None
    return dict(D=D, X=X, ks_field=ks,
                meta=dict(name=name, dim=dim, sec=(min(secs + [0.0]),
                                                   max(secs + [0.0]))))


# ---------------------------------------------------------------------------
# Tier 2 — warped products over S^{d-1}: g = dr^2 + f(r)^2 g_{S^{d-1}}
# ---------------------------------------------------------------------------


def sphere_profile(L: float = np.pi):
    """f(r) = sin(r): recovers the unit sphere S^d — validation profile."""
    return lambda r: np.sin(np.clip(r, 0, L)), L


def dumbbell_profile(L: float = np.pi, beta: float = 0.65,
                     w: float | None = None):
    """Two bulbs joined by a neck at r = L/2. f'(0) = 1 - O(e^-12) so the
    caps close smoothly. beta controls neck depth; w its width."""
    w = w or L / 10

    def f(r):
        r = np.clip(r, 0, L)
        return (L / np.pi) * np.sin(np.pi * r / L) * (
            1 - beta * np.exp(-((r - L / 2) ** 2) / (2 * w**2)))
    return f, L


class WarpedProduct:
    """Warped product over an interval with smooth caps: topology S^d.

    Per-point analytic curvature (via dense-grid derivatives of f):
        K_rad = -f''/f ; K_tan = (1 - f'^2)/f^2
        R = 2(d-1) K_rad + (d-1)(d-2) K_tan
    Geodesic distances by symmetry reduction: every geodesic lies in a 2D
    surface of revolution (profile f), so d((r1,u1),(r2,u2)) = geodesic
    distance between (r1, 0) and (r2, ang(u1,u2)) on that 2D surface —
    computed on a fine (r, phi) grid graph, dimension-independent.
    """

    def __init__(self, profile_fn, L: float, d: int, n_grid_r: int = 500,
                 n_grid_phi: int = 300, n_knn_grid: int = 8):
        self.f, self.L, self.d = profile_fn, L, d
        rg = np.linspace(0, L, 20001)
        fg = self.f(rg)
        fp = np.gradient(fg, rg)
        fpp = np.gradient(fp, rg)
        eps = 1e-6 * fg.max()
        self.rg, self.fg = rg, fg
        with np.errstate(divide="ignore", invalid="ignore"):
            k_rad = -fpp / np.maximum(fg, eps)
            k_tan = (1 - fp**2) / np.maximum(fg, eps) ** 2
        self.k_rad_g, self.k_tan_g = k_rad, k_tan
        self.R_g = 2 * (d - 1) * k_rad + (d - 1) * (d - 2) * k_tan
        self._build_2d_solver(n_grid_r, n_grid_phi)

    def scalar(self, r: np.ndarray) -> np.ndarray:
        return np.interp(r, self.rg, self.R_g)

    def _build_2d_solver(self, nr: int, nphi: int) -> None:
        """Grid graph on the 2D surface of revolution: ds^2 = dr^2 + f^2 dphi^2.
        Margins avoid the coordinate degeneracy at the caps."""
        r0 = self.L * 2e-3
        self.grid_r = np.linspace(r0, self.L - r0, nr)
        self.grid_phi = np.linspace(0, np.pi, nphi)
        fr = np.maximum(self.f(self.grid_r), 1e-9)
        nodes = nr * nphi

        def nid(i, j):
            return i * nphi + j

        rows, cols, wts = [], [], []
        dr = self.grid_r[1] - self.grid_r[0]
        dphi = self.grid_phi[1] - self.grid_phi[0]
        for di, dj in [(1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (1, 2),
                       (2, -1), (1, -2), (3, 1), (1, 3), (3, -1), (1, -3),
                       (3, 2), (2, 3), (3, -2), (2, -3)]:
            i = np.arange(nr)
            j = np.arange(nphi)
            ii, jj = np.meshgrid(i, j, indexing="ij")
            i2, j2 = ii + di, jj + dj
            ok = (i2 >= 0) & (i2 < nr) & (j2 >= 0) & (j2 < nphi)
            a = nid(ii[ok], jj[ok])
            b = nid(i2[ok], j2[ok])
            fmid = 0.5 * (fr[ii[ok]] + fr[i2[ok]])
            w = np.sqrt((di * dr) ** 2 + (fmid * dj * dphi) ** 2)
            rows.append(a)
            cols.append(b)
            wts.append(w)
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        wts = np.concatenate(wts)
        self.G2 = sp.csr_matrix((wts, (rows, cols)), shape=(nodes, nodes))
        self.nr, self.nphi = nr, nphi
        # distance table from every (r_i, phi=0) source to all grid nodes
        src = [nid(i, 0) for i in range(nr)]
        self.table = dijkstra(self.G2, directed=False, indices=src) \
            .reshape(nr, nr, nphi)

    def pair_distance(self, r1, r2, ang):
        """Geodesic distance for arrays r1, r2, ang: grid table for long
        pairs, local metric (exact to O(K d^3)) for short ones."""
        r1 = np.asarray(r1, dtype=float)
        r2 = np.asarray(r2, dtype=float)
        ang = np.asarray(ang, dtype=float)
        i1 = np.clip(np.searchsorted(self.grid_r, r1), 1, self.nr - 1)
        i2 = np.clip(np.searchsorted(self.grid_r, r2), 1, self.nr - 1)
        ja = np.clip(np.searchsorted(self.grid_phi, ang), 1, self.nphi - 1)
        cands = []
        for a in (i1 - 1, i1):
            for b in (i2 - 1, i2):
                for c in (ja - 1, ja):
                    cands.append(self.table[a, b, c])
        d_grid = np.min(cands, axis=0)
        # short-pair refinement: quadrature of ds = sqrt(dr^2 + f^2 dphi^2)
        # along the straight coordinate path (valid path => upper bound;
        # tight for short near-geodesic segments, handles f-variation)
        Q = 9
        ts = np.linspace(0, 1, Q)[None, :]
        rpath = r1[:, None] * (1 - ts) + r2[:, None] * ts
        fpath = np.interp(rpath.ravel(), self.rg, self.fg).reshape(rpath.shape)
        fmid = 0.5 * (fpath[:, 1:] + fpath[:, :-1])
        seg = np.sqrt(((r2 - r1)[:, None] / (Q - 1)) ** 2
                      + (fmid * (ang[:, None] / (Q - 1))) ** 2)
        d_loc = seg.sum(axis=1)
        cell = 15.0 * max(self.grid_r[1] - self.grid_r[0],
                          (self.grid_phi[1] - self.grid_phi[0])
                          * float(self.fg.max()))
        return np.where(d_loc < cell, np.minimum(d_loc, d_grid + cell / 15),
                        d_grid)

    def sample(self, n: int, rng=None) -> dict:
        rng = np.random.default_rng(rng)
        pdf = np.maximum(self.fg, 0) ** (self.d - 1)
        cdf = np.cumsum(pdf)
        cdf = cdf / cdf[-1]
        r = np.interp(rng.random(n), cdf, self.rg)
        u = rng.normal(size=(n, self.d))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        cosang = np.clip(u @ u.T, -1, 1)
        ang = np.arccos(cosang)
        r1 = np.repeat(r, n).reshape(n, n)
        D = self.pair_distance(r1.ravel(), r1.T.ravel(),
                               ang.ravel()).reshape(n, n)
        D = 0.5 * (D + D.T)
        np.fill_diagonal(D, 0.0)
        ks = self.scalar(r)
        return dict(D=D, X=None, ks_field=ks, r=r,
                    meta=dict(name=f"warped(d={self.d})", dim=self.d,
                              sec=(float(np.nanmin(self.R_g[50:-50])),
                                   float(np.nanmax(self.R_g[50:-50])))))
