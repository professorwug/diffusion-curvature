"""Steinerberger Potential Curvature (SPC).

For an n-point dataset with pairwise distance matrix D, SPC solves
    D @ k = n * 1
for k in R^n. The entries of k are interpreted as a curvature-like
potential on the nodes: regions where the distance matrix has more
"mass" yield smaller k (more negative); regions that are isolated
yield larger k.

Two distance variants are supported:

- `'shortest-path'` — hop-count shortest paths on an unweighted kNN graph
  built from the input point cloud.
- `'diffusion'` — diffusion distances at time t, computed via the
  spectrum of the row-stochastic transition matrix of an affinity kernel
  (graphtools adaptive Gaussian by default).

Reference implementation ported from
`reason_reckon/experiments/20-curvature-as-prm/scripts/compute_alternate_curvatures.py`.
"""

from __future__ import annotations

from typing import Literal

import networkx as nx
import numpy as np
import pygsp
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

_DISTANCE = Literal["shortest-path", "diffusion"]


def _knn_graph(X: np.ndarray, k: int) -> nx.Graph:
    n = X.shape[0]
    k_eff = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nn.fit(X)
    _, idx = nn.kneighbors(X)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in idx[i, 1:]:
            G.add_edge(i, int(j))
    return G


def _hop_distances(G: nx.Graph) -> np.ndarray:
    D = np.array(nx.floyd_warshall_numpy(G), dtype=np.float64)
    if np.any(np.isinf(D)):
        finite_max = np.max(D[np.isfinite(D)]) if np.isfinite(D).any() else 1.0
        D[np.isinf(D)] = finite_max * 2
    return D


def _diffusion_distances(W: np.ndarray | sp.spmatrix, t: int) -> np.ndarray:
    """Diffusion distances at time t from an affinity matrix W."""
    if sp.issparse(W):
        W = W.toarray()
    W = np.asarray(W, dtype=np.float64)
    row_sums = W.sum(axis=1)
    row_sums = np.where(row_sums > 0, row_sums, 1e-10)
    P = W / row_sums[:, None]

    # eigendecomposition of row-stochastic P (which is generally not symmetric);
    # symmetrize for numerical stability via the P ~ D^{1/2} P_sym D^{-1/2} trick.
    # For simplicity we use eigh on the symmetrized normalization D^{-1/2} W D^{-1/2}.
    d_inv_sqrt = 1.0 / np.sqrt(row_sums)
    M = (W * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]
    M = (M + M.T) / 2
    eigvals, eigvecs = np.linalg.eigh(M)
    order = np.argsort(-eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Drop the leading trivial eigenpair (eigenvalue ~ 1)
    lam_t = np.abs(eigvals[1:]) ** t
    Psi_t = (eigvecs[:, 1:] * lam_t[None, :]) * d_inv_sqrt[:, None]
    D = squareform(pdist(Psi_t, metric="euclidean"))
    np.fill_diagonal(D, 0.0)
    return D


def _solve_spc(D: np.ndarray) -> np.ndarray:
    n = D.shape[0]
    try:
        return np.linalg.solve(D, n * np.ones(n))
    except np.linalg.LinAlgError:
        return np.full(n, np.nan)


class SteinerbergerCurvature:
    """Steinerberger Potential Curvature with pluggable ground distance.

    sklearn-style: construct with hyperparameters, then call `fit_transform`.

    Parameters
    ----------
    distance : {'shortest-path', 'diffusion'}
        Which distance matrix to feed into SPC.
    knn : int
        Nearest-neighbor count (shortest-path variant only).
    t : int
        Diffusion time (diffusion variant only).

    Examples
    --------
    >>> spc = SteinerbergerCurvature(distance="diffusion", t=3)
    >>> k = spc.fit_transform(G)  # PyGSP graph OR point cloud
    """

    def __init__(
        self,
        distance: _DISTANCE = "shortest-path",
        knn: int = 5,
        t: int = 3,
    ):
        self.distance = distance
        self.knn = knn
        self.t = t
        self.D_: np.ndarray | None = None
        self.k_: np.ndarray | None = None

    def fit_transform(
        self,
        G: pygsp.graphs.Graph | None = None,
        X: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute SPC. Supply either a PyGSP graph or a point cloud.

        - `shortest-path` distance works on either input (builds kNN if X given).
        - `diffusion` distance requires an affinity matrix; pass a PyGSP graph,
          or a point cloud X (kNN graph is built internally).
        """
        if G is None and X is None:
            raise ValueError("Must provide either G (PyGSP graph) or X (point cloud).")

        if self.distance == "shortest-path":
            if G is not None:
                # Use the graph's connectivity as unweighted
                W = G.W.toarray() if sp.issparse(G.W) else np.asarray(G.W)
                nxg = nx.from_numpy_array((W > 0).astype(int))
            else:
                nxg = _knn_graph(X, self.knn)
            self.D_ = _hop_distances(nxg)

        elif self.distance == "diffusion":
            if G is not None:
                W = G.W
            else:
                import graphtools as gt

                G_gt = gt.Graph(X, knn=self.knn, use_pygsp=True)
                W = G_gt.K
            self.D_ = _diffusion_distances(W, self.t)

        else:
            raise ValueError(
                f"distance must be one of 'shortest-path', 'diffusion'; got {self.distance!r}"
            )

        self.k_ = _solve_spc(self.D_)
        return self.k_
