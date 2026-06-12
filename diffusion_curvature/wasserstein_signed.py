"""Natively signed curvature from optimal transport of diffusion-type measures.

Two estimators sharing one OT core, both signed *without* a comparison space:

- **Diffusion ORC**: ``k(i,j) = 1 - W1(mu_i, mu_j) / d(i,j)`` where
  ``mu_i = P^t delta_i`` (or a supplied successor measure), ``d`` is the graph
  geodesic distance, and the *same* geodesic metric is the W1 ground cost.
  In flat space W1 = d by translation invariance, so k = 0 natively; positive
  curvature contracts diffusions (W1 < d), negative expands them.

- **Midpoint Entropy Curvature** (Lott-Sturm-Villani): entropy along the W2
  displacement interpolation between mu_i and mu_j is K-concave -- geodesic
  bundles bulge in positive curvature, so the midpoint measure is *more*
  spread out. We push the exact OT plan to graph midpoints and estimate
  ``K = 8 [H(nu_mid) - (H(mu_i) + H(mu_j))/2] / W2^2``.

Both consume either diffusion measures (rows of P^t; no training required) or
precomputed successor measures (the repaired Successor ORC).
"""

from __future__ import annotations

import warnings

import numpy as np
import ot
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import kneighbors_graph

from .kernels import gaussian_kernel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _truncate_measure(
    mu: np.ndarray, support_mass: float, max_support: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (atom_indices, renormalized_weights) keeping `support_mass` of mu."""
    order = np.argsort(mu)[::-1]
    csum = np.cumsum(mu[order])
    n_keep = int(np.searchsorted(csum, support_mass * csum[-1]) + 1)
    n_keep = min(n_keep, max_support)
    idx = order[:n_keep]
    w = mu[idx].astype(np.float64)
    return idx, w / w.sum()


class _DijkstraCache:
    """Memoized single-source geodesic distances on a sparse distance graph."""

    def __init__(self, graph: sp.csr_matrix):
        self.graph = graph
        self.rows: dict[int, np.ndarray] = {}

    def fetch(self, sources: np.ndarray) -> None:
        new = [int(s) for s in np.unique(sources) if int(s) not in self.rows]
        if not new:
            return
        D = dijkstra(self.graph, directed=False, indices=new)
        for s, row in zip(new, D):
            self.rows[s] = row

    def __getitem__(self, s: int) -> np.ndarray:
        if s not in self.rows:
            self.fetch(np.array([s]))
        return self.rows[s]


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------


class WassersteinSignedCurvature:
    """Signed curvature via exact OT between diffusion-type measures.

    Computes, per requested point, both the Wasserstein-contraction (Diffusion
    ORC) and midpoint-entropy (LSV) curvature estimates. Access them after
    ``fit`` via ``orc_`` and ``midpoint_``.

    Parameters
    ----------
    t : int | 'auto'
        Diffusion steps for the measure family (ignored when ``M`` is given).
        With ``'auto'``, t is chosen per evaluation point as the largest t
        whose diffusion spread stays below ``spread_fraction`` of the median
        geodesic radius — guarding against mixing (in sparse/high-dim data a
        fixed t can cover the whole dataset, making all measures identical
        and biasing the curvature toward +1).
    spread_fraction : float
        Target spread as a fraction of the median geodesic distance from the
        evaluation point (only used with ``t='auto'``).
    t_max : int
        Upper bound for auto-selected t.
    knn : int
        Neighbors for both the affinity kernel and the geodesic graph.
    alpha : float
        Anisotropic density normalization for the diffusion kernel.
    n_pairs : int
        Number of partner points j per evaluation point.
    pair_radius_factor : float
        Pair separation, as a multiple of the W1 spread of mu_i.
    band : (float, float)
        Tolerance band around the target separation radius.
    support_mass : float
        Mass quantile retained when truncating measures for exact OT.
    max_support : int
        Hard cap on atoms per measure.
    smear : int | None
        Width (in diffusion steps) of the bridge used to realize OT midpoints,
        and equally of the smoothing applied to the endpoint measures before
        entropy comparison. Each coupling atom (a, b) contributes a pinned
        diffusion-bridge midpoint ``nu(z) ~ P^{2s}(a,z) P^{2s}(z,b)`` (variance
        s); endpoints are compared as ``mu P^s`` (variance t + s, matching the
        midpoint mixture's t + s) -- so flat space gives a zero entropy deficit
        natively while s washes out atom-collision artifacts. Default: t // 2.
    compute_midpoint : bool
        Disable to skip the bridge-midpoint estimator (saves the dense
        ``P^{2s}`` matrix power).
    """

    def __init__(
        self,
        t: int | str = 8,
        spread_fraction: float = 0.3,
        t_max: int = 64,
        knn: int = 10,
        alpha: float = 1.0,
        n_pairs: int = 8,
        pair_radius_factor: float = 1.0,
        band: tuple[float, float] = (0.75, 1.25),
        support_mass: float = 0.999,
        max_support: int = 384,
        smear: int | None = None,
        compute_midpoint: bool = True,
        seed: int = 42,
    ):
        self.t = t
        self.spread_fraction = spread_fraction
        self.t_max = t_max
        self.knn = knn
        self.alpha = alpha
        self.n_pairs = n_pairs
        self.pair_radius_factor = pair_radius_factor
        self.band = band
        self.support_mass = support_mass
        self.max_support = max_support
        self.smear = smear
        self.compute_midpoint = compute_midpoint
        self.seed = seed

        self.orc_: np.ndarray | None = None
        self.orc_phys_: np.ndarray | None = None
        self.midpoint_: np.ndarray | None = None
        self.spread_: np.ndarray | None = None

    # -- inputs --------------------------------------------------------------

    def _build_operators(self, G, X, M):
        """Set self.P (dense diffusion matrix), self.measures (callable row->mu),
        and self.geo (sparse distance graph for geodesics)."""
        if X is None and G is None:
            raise ValueError("Supply X (pointcloud) and/or G (pygsp graph).")

        if G is not None:
            W = G.W
            if sp.issparse(W):
                W = W.toarray()
            W = np.asarray(W, dtype=np.float64)
        else:
            W = gaussian_kernel(
                X, kernel_type="adaptive", k=self.knn,
                anisotropic_density_normalization=self.alpha,
            )
        rs = W.sum(axis=1)
        rs[rs <= 0] = 1.0
        self.P = (W / rs[:, None]).astype(np.float64)
        n = self.P.shape[0]

        # Geodesic graph: kNN distance graph on X if available, else 1/affinity.
        if X is not None:
            geo = kneighbors_graph(
                X, min(self.knn, n - 1), mode="distance", include_self=False
            )
        else:
            A = sp.csr_matrix(W)
            A.setdiag(0)
            A.eliminate_zeros()
            A.data = 1.0 / A.data
            geo = A
        self.geo = sp.csr_matrix(geo)

        if M is not None:
            M = np.asarray(M, dtype=np.float64)
            rsm = M.sum(axis=1, keepdims=True)
            rsm[rsm <= 0] = 1.0
            self._M = M / rsm
        else:
            self._M = None

    def _measure_rows(self, idxs: np.ndarray) -> np.ndarray:
        """Rows of P^t (or of the supplied measure matrix) for `idxs`."""
        if self._M is not None:
            return self._M[idxs]
        out = np.zeros((len(idxs), self.P.shape[0]))
        out[np.arange(len(idxs)), idxs] = 1.0
        t = self._t_eff
        for _ in range(t):
            out = out @ self.P
        return out

    @property
    def _t_eff(self) -> int:
        """Concrete diffusion time for the current point (auto-resolved)."""
        if isinstance(self.t, str):
            return getattr(self, "_t_auto", 1)
        return self.t

    def _auto_t_measure(self, i: int, d_i: np.ndarray, finite: np.ndarray):
        """Step the diffusion from i until spread reaches the target radius."""
        target = self.spread_fraction * float(np.median(d_i[finite]))
        mu = np.zeros(self.P.shape[0])
        mu[i] = 1.0
        t = 0
        while t < self.t_max:
            nxt = mu @ self.P
            if float((nxt[finite] * d_i[finite]).sum()) > target and t >= 1:
                break
            mu = nxt
            t += 1
        self._t_auto = max(t, 1)
        return mu

    @property
    def _smear_steps(self) -> int:
        return max(1, self._t_eff // 2) if self.smear is None else self.smear

    def _smear_rows(self, rows: np.ndarray) -> np.ndarray:
        for _ in range(self._smear_steps):
            rows = rows @ self.P
        return rows

    # -- main ----------------------------------------------------------------

    def fit(
        self,
        G=None,
        X: np.ndarray | None = None,
        M: np.ndarray | None = None,
        idx=None,
    ) -> "WassersteinSignedCurvature":
        self._build_operators(G, X, M)
        n = self.P.shape[0]
        idxs = np.arange(n) if idx is None else np.atleast_1d(np.asarray(idx, dtype=int))
        rng = np.random.default_rng(self.seed)
        cache = _DijkstraCache(self.geo)

        if self.compute_midpoint and not isinstance(self.t, str):
            self._A = np.linalg.matrix_power(self.P, 2 * self._smear_steps)
        else:
            self._A = None

        orc = np.full(len(idxs), np.nan)
        mid = np.full(len(idxs), np.nan)
        spread = np.full(len(idxs), np.nan)

        for out_i, i in enumerate(idxs):
            try:
                o, m, s = self._point_estimate(int(i), cache, rng)
            except Exception as e:  # pragma: no cover - per-point robustness
                warnings.warn(f"signed curvature failed at node {i}: {e}")
                o, m, s = np.nan, np.nan, np.nan
            orc[out_i], mid[out_i], spread[out_i] = o, m, s

        self.orc_ = orc
        self.midpoint_ = mid
        self.spread_ = spread
        with np.errstate(divide="ignore", invalid="ignore"):
            self.orc_phys_ = orc / spread**2
        return self

    def _point_estimate(self, i, cache, rng):
        n = self.P.shape[0]
        d_i = cache[i]
        finite = np.isfinite(d_i)

        if isinstance(self.t, str) and self._M is None:
            mu_i = self._auto_t_measure(i, d_i, finite)
        else:
            mu_i = self._measure_rows(np.array([i]))[0]

        spread = float((mu_i[finite] * d_i[finite]).sum())
        if spread <= 0:
            return np.nan, np.nan, np.nan
        r = self.pair_radius_factor * spread

        lo, hi = self.band[0] * r, self.band[1] * r
        candidates = np.where(finite & (d_i >= lo) & (d_i <= hi))[0]
        candidates = candidates[candidates != i]
        widen = 1.0
        while candidates.size == 0 and widen < 4.0:
            widen *= 1.5
            candidates = np.where(
                finite & (d_i >= lo / widen) & (d_i <= hi * widen)
            )[0]
            candidates = candidates[candidates != i]
        if candidates.size == 0:
            return np.nan, np.nan, np.nan
        pairs = rng.choice(
            candidates, size=min(self.n_pairs, candidates.size), replace=False
        )

        mu_js = self._measure_rows(pairs)

        # Truncate supports, then geodesics from every needed source at once.
        si, wi = _truncate_measure(mu_i, self.support_mass, self.max_support)
        supports = [
            _truncate_measure(mu_js[k], self.support_mass, self.max_support)
            for k in range(len(pairs))
        ]
        all_sources = np.unique(np.concatenate([si] + [s for s, _ in supports]))
        cache.fetch(all_sources)

        # Pre-smear endpoint entropies once (shared across pairs; midpoint only).
        if self._A is not None:
            mu_i_s = self._smear_rows(mu_i[None, :])[0]
            H_i = _entropy(mu_i_s)
            mu_js_s = self._smear_rows(mu_js)

        orc_vals, mid_vals = [], []
        for k, j in enumerate(pairs):
            sj, wj = supports[k]
            S = np.unique(np.concatenate([si, sj]))
            pos = {a: q for q, a in enumerate(S)}
            DS = np.stack([cache[int(a)][S] for a in S])
            # Symmetrize (Dijkstra is symmetric in theory; guard numeric noise)
            DS = 0.5 * (DS + DS.T)
            bad = ~np.isfinite(DS)
            if bad.any():
                DS[bad] = DS[np.isfinite(DS)].max() * 2.0

            a_full = np.zeros(len(S))
            a_full[[pos[a] for a in si]] = wi
            b_full = np.zeros(len(S))
            b_full[[pos[a] for a in sj]] = wj

            d_ij = float(d_i[j])
            if not np.isfinite(d_ij) or d_ij <= 0:
                continue

            w1 = float(ot.emd2(a_full, b_full, DS))
            orc_vals.append(1.0 - w1 / d_ij)

            # --- bridge-midpoint entropy (W2 plan) ---
            if self._A is None:
                continue
            plan = ot.emd(a_full, b_full, DS**2)
            w2sq = float((plan * DS**2).sum())
            if w2sq <= 0:
                continue
            rows_, cols_ = np.nonzero(plan)
            nu = np.zeros(n)
            A = self._A
            for a_loc, b_loc in zip(rows_, cols_):
                mass = plan[a_loc, b_loc]
                a_g, b_g = int(S[a_loc]), int(S[b_loc])
                bridge = A[a_g] * A[:, b_g]
                tot = bridge.sum()
                if tot <= 0:
                    nu[a_g] += mass
                else:
                    nu += mass * (bridge / tot)
            H_j = _entropy(mu_js_s[k])
            H_mid = _entropy(nu)
            mid_vals.append(8.0 * (H_mid - 0.5 * (H_i + H_j)) / w2sq)

        o = float(np.mean(orc_vals)) if orc_vals else np.nan
        m = float(np.mean(mid_vals)) if mid_vals else np.nan
        return o, m, spread

    def fit_transform(self, G=None, X=None, M=None, idx=None) -> np.ndarray:
        """Returns the Diffusion-ORC estimate (use MidpointEntropyCurvature for
        the midpoint variant)."""
        self.fit(G=G, X=X, M=M, idx=idx)
        return self.orc_


class DiffusionORC(WassersteinSignedCurvature):
    """Wasserstein contraction of diffusion measures: 1 - W1(mu_i, mu_j)/d(i,j)."""

    def fit_transform(self, G=None, X=None, M=None, idx=None) -> np.ndarray:
        self.fit(G=G, X=X, M=M, idx=idx)
        return self.orc_


class MidpointEntropyCurvature(WassersteinSignedCurvature):
    """LSV midpoint-entropy curvature along W2 displacement interpolations."""

    def fit_transform(self, G=None, X=None, M=None, idx=None) -> np.ndarray:
        self.fit(G=G, X=X, M=M, idx=idx)
        return self.midpoint_
