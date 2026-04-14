"""Sample random-walk trajectories from a PyGSP graph via its diffusion matrix.

Used by successor-curvature methods that need trajectory data but have
only a graph to work with.
"""

from __future__ import annotations

import numpy as np
import pygsp
import scipy.sparse as sp


def _row_normalize(W: np.ndarray) -> np.ndarray:
    """Row-normalize a (possibly sparse) affinity matrix into a transition matrix."""
    if sp.issparse(W):
        W = W.toarray()
    W = np.asarray(W, dtype=np.float64)
    row_sums = W.sum(axis=1)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return W / row_sums[:, None]


def sample_random_walk(
    P: np.ndarray,
    start_idx: int,
    length: int = 100,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample a single random walk of `length` steps starting at `start_idx`.

    Args:
        P: (n, n) row-stochastic diffusion / transition matrix.
        start_idx: Node index to start from.
        length: Number of steps; returned trajectory has length + 1 nodes.
        rng: Optional numpy Generator for reproducibility.

    Returns:
        (length + 1,) array of node indices along the walk.
    """
    if rng is None:
        rng = np.random.default_rng()
    P = np.asarray(P)
    n = P.shape[0]
    traj = np.empty(length + 1, dtype=np.int64)
    traj[0] = start_idx
    for i in range(length):
        row = P[traj[i]]
        if row.sum() <= 0:
            traj[i + 1 :] = traj[i]
            break
        traj[i + 1] = rng.choice(n, p=row / row.sum())
    return traj


def subsample_trajectories(
    G: pygsp.graphs.Graph,
    n_trajectories: int = 1,
    length: int = 100,
    start_indices: np.ndarray | list[int] | None = None,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Sample `n_trajectories` random walks on graph `G` via its diffusion matrix.

    Args:
        G: PyGSP graph.
        n_trajectories: Number of walks to sample. Ignored if `start_indices` is given.
        length: Steps per walk; each row has length + 1 node indices.
        start_indices: Optional starting nodes; if None, sampled uniformly at random.
        rng: numpy Generator, int seed, or None.

    Returns:
        (n_trajectories, length + 1) int array of node indices.
    """
    if isinstance(rng, (int, np.integer)) or rng is None:
        rng = np.random.default_rng(rng)

    P = _row_normalize(G.W)
    n = P.shape[0]

    if start_indices is None:
        start_indices = rng.integers(0, n, size=n_trajectories)
    else:
        start_indices = np.asarray(start_indices, dtype=np.int64)
        n_trajectories = len(start_indices)

    trajs = np.empty((n_trajectories, length + 1), dtype=np.int64)
    for i, idx in enumerate(start_indices):
        trajs[i] = sample_random_walk(P, int(idx), length=length, rng=rng)
    return trajs
