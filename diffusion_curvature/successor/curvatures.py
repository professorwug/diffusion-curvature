"""Successor-curvature classes.

Three sklearn-style classes with interchangeable `fit_transform` signatures:

- ``SuccessorEntropyCurvature`` — entropy of the softmax of F^T B.
- ``SuccessorORC``              — Ollivier-Ricci curvature with F- or B-space ground distance.
- ``SuccessorW1``               — raw W1 distance between successor measures of adjacent obs.

Call ``fit_transform(G=..., ...)`` to auto-sample trajectories from a PyGSP graph,
or pass ``trajectories=...``, or pass precomputed ``F_embeddings=..., B_embeddings=...``
plus raw ``M=...`` to skip both training and successor-measure computation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

import numpy as np
import pygsp
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors

from ..trajectory_utils import subsample_trajectories
from .fb_modules import BackwardMap, ForwardMap
from .measures import (
    compute_successor_measures,
    softmax_measure,
    truncated_sliced_w1,
)
from .train import FBTrainer

GroundMetric = Literal["F", "B"]


# ---------------------------------------------------------------------------
# Common FB-training / measure-computation mixin
# ---------------------------------------------------------------------------


class _SuccessorCurvatureBase:
    """Shared fit/transform plumbing for successor curvatures."""

    def __init__(
        self,
        z_dim: int = 2,
        hidden_dim: int = 256,
        gamma: float = 0.5,
        tau_polyak: float = 0.01,
        ortho_coef: float = 1.0,
        lr: float = 1e-4,
        lr_min: float = 1e-6,
        cosine_lr: bool = False,
        batch_size: int = 512,
        n_epochs: int = 200,
        traj_length: int = 100,
        n_trajectories: int = 64,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau_polyak = tau_polyak
        self.ortho_coef = ortho_coef
        self.lr = lr
        self.lr_min = lr_min
        self.cosine_lr = cosine_lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.traj_length = traj_length
        self.n_trajectories = n_trajectories
        self.device = device
        self.seed = seed

        self.trainer_: FBTrainer | None = None
        self.F_: np.ndarray | None = None  # (N, z_dim) F-space embedding
        self.B_: np.ndarray | None = None  # (N, z_dim) B-space embedding
        self.M_: np.ndarray | None = None  # (N, N) successor measure (clipped)
        self.curvature_: np.ndarray | None = None

    # ------- inputs resolution ---------------------------------------------

    def _resolve_inputs(
        self,
        G: pygsp.graphs.Graph | None,
        X: np.ndarray | None,
        trajectories: np.ndarray | list[np.ndarray] | None,
        F_embeddings: np.ndarray | None,
        B_embeddings: np.ndarray | None,
        M: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (M, F, B). Train FB and compute M if not precomputed."""
        if M is not None and F_embeddings is not None and B_embeddings is not None:
            return M, F_embeddings, B_embeddings

        # Need a point cloud (corpus) and trajectories to train FB.
        if X is None and G is None:
            raise ValueError(
                "Must supply one of: (G), (X), (trajectories as (K, T, d) array)."
            )

        if X is None:
            # Use graph node signals; we need node coordinates.
            raise NotImplementedError(
                "Training from a bare PyGSP graph without node features is not yet "
                "supported; pass X (corpus point cloud) alongside G, or pass "
                "precomputed F/B embeddings."
            )

        corpus = np.asarray(X, dtype=np.float32)

        if trajectories is None:
            if G is None:
                raise ValueError(
                    "Need either `trajectories` or a PyGSP graph `G` to sample them."
                )
            node_trajs = subsample_trajectories(
                G,
                n_trajectories=self.n_trajectories,
                length=self.traj_length,
                rng=self.seed,
            )
            traj_data = corpus[node_trajs]  # (K, T, d)
        else:
            traj_data = trajectories

        trainer = FBTrainer(
            obs_dim=corpus.shape[1],
            z_dim=self.z_dim,
            hidden_dim=self.hidden_dim,
            gamma=self.gamma,
            tau_polyak=self.tau_polyak,
            ortho_coef=self.ortho_coef,
            lr=self.lr,
            lr_min=self.lr_min,
            cosine_lr=self.cosine_lr,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            device=self.device,
            seed=self.seed,
        )
        trainer.fit(traj_data)
        self.trainer_ = trainer

        M_mat, F_emb, B_emb = compute_successor_measures(
            trainer.F_net, trainer.B_net, corpus, corpus, device=self.device
        )
        return M_mat, F_emb, B_emb


# ---------------------------------------------------------------------------
# Public classes
# ---------------------------------------------------------------------------


class SuccessorEntropyCurvature(_SuccessorCurvatureBase):
    """Entropy of the softmax(F^T B) successor measure per point.

    Negated so that *concentrated* (low-entropy) measures correspond to
    positive curvature, and *diffuse* (high-entropy) measures to negative
    curvature — matching the sign conventions of the other curvature classes.
    """

    def __init__(self, tau: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau

    def fit_transform(
        self,
        G: pygsp.graphs.Graph | None = None,
        X: np.ndarray | None = None,
        trajectories: np.ndarray | list[np.ndarray] | None = None,
        F_embeddings: np.ndarray | None = None,
        B_embeddings: np.ndarray | None = None,
        M: np.ndarray | None = None,
    ) -> np.ndarray:
        M_mat, F_emb, B_emb = self._resolve_inputs(
            G, X, trajectories, F_embeddings, B_embeddings, M
        )
        self.M_ = M_mat
        self.F_ = F_emb
        self.B_ = B_emb

        P = softmax_measure(M_mat, tau=self.tau)
        P = np.clip(P, 1e-12, 1.0)
        H = -(P * np.log(P)).sum(axis=1)
        self.curvature_ = -H  # high entropy → negative curvature
        return self.curvature_


class SuccessorORC(_SuccessorCurvatureBase):
    """Ollivier-Ricci curvature with F- or B-space ground metric.

    For each point i, computes
        ORC(i) = 1 - mean_{j ∈ kNN(i)} [ W1(M(i), M(j)) / d(i, j) ]
    with d taken in F- or B-space.

    Parameters
    ----------
    ground : {'F', 'B'}
        Ground-distance space.
    k_neighbors : int
        Number of neighbors to average over.
    top_n : int
        Top-n support truncation for sliced-W1 computation.
    use_softmax : bool
        If True, softmax-normalize M before computing W1 (per the linked zettel).
    softmax_tau : float
    n_projections : int
        Sliced-W1 projection count.
    n_jobs : int
    """

    def __init__(
        self,
        ground: GroundMetric = "B",
        k_neighbors: int = 1,
        top_n: int = 200,
        use_softmax: bool = False,
        softmax_tau: float = 1.0,
        n_projections: int = 50,
        n_jobs: int = -1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ground = ground
        self.k_neighbors = k_neighbors
        self.top_n = top_n
        self.use_softmax = use_softmax
        self.softmax_tau = softmax_tau
        self.n_projections = n_projections
        self.n_jobs = n_jobs

    def fit_transform(
        self,
        G: pygsp.graphs.Graph | None = None,
        X: np.ndarray | None = None,
        trajectories: np.ndarray | list[np.ndarray] | None = None,
        F_embeddings: np.ndarray | None = None,
        B_embeddings: np.ndarray | None = None,
        M: np.ndarray | None = None,
    ) -> np.ndarray:
        M_mat, F_emb, B_emb = self._resolve_inputs(
            G, X, trajectories, F_embeddings, B_embeddings, M
        )
        self.M_ = M_mat
        self.F_ = F_emb
        self.B_ = B_emb

        coords = F_emb if self.ground == "F" else B_emb
        measures = softmax_measure(M_mat, tau=self.softmax_tau) if self.use_softmax else M_mat

        T = coords.shape[0]
        k = min(self.k_neighbors, T - 1)
        if k < 1:
            self.curvature_ = np.zeros(T)
            return self.curvature_

        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nn.fit(coords)
        dists, idx = nn.kneighbors(coords)
        neigh_d = dists[:, 1:]
        neigh_i = idx[:, 1:]

        pairs = [
            (t, int(neigh_i[t, j]), float(neigh_d[t, j]))
            for t in range(T)
            for j in range(k)
            if neigh_d[t, j] > 1e-8
        ]
        if not pairs:
            self.curvature_ = np.zeros(T)
            return self.curvature_

        def _one(t: int, tp_: int, d: float) -> tuple[int, float]:
            w = truncated_sliced_w1(
                measures[t], measures[tp_], coords,
                top_n=self.top_n, n_projections=self.n_projections, seed=self.seed,
            )
            return t, w / d

        results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_one)(t, tp_, d) for t, tp_, d in pairs
        )
        per_t: dict[int, list[float]] = defaultdict(list)
        for t, ratio in results:
            per_t[t].append(ratio)
        orc = np.zeros(T)
        for t in range(T):
            if per_t[t]:
                orc[t] = 1.0 - float(np.mean(per_t[t]))
        self.curvature_ = orc
        return orc


class SuccessorW1(_SuccessorCurvatureBase):
    """Raw mean W1 distance between a point's successor measure and its neighbors'.

    Negated so that points whose successor measure *changes a lot* across the
    neighborhood (≈ unstable region) read as negative curvature.

    Parameters mirror SuccessorORC but the final statistic is
        W1(i) = mean_{j ∈ kNN(i)} W1(M(i), M(j))
    (no division by ground distance).
    """

    def __init__(
        self,
        ground: GroundMetric = "B",
        k_neighbors: int = 1,
        top_n: int = 200,
        use_softmax: bool = False,
        softmax_tau: float = 1.0,
        n_projections: int = 50,
        n_jobs: int = -1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ground = ground
        self.k_neighbors = k_neighbors
        self.top_n = top_n
        self.use_softmax = use_softmax
        self.softmax_tau = softmax_tau
        self.n_projections = n_projections
        self.n_jobs = n_jobs

    def fit_transform(
        self,
        G: pygsp.graphs.Graph | None = None,
        X: np.ndarray | None = None,
        trajectories: np.ndarray | list[np.ndarray] | None = None,
        F_embeddings: np.ndarray | None = None,
        B_embeddings: np.ndarray | None = None,
        M: np.ndarray | None = None,
    ) -> np.ndarray:
        M_mat, F_emb, B_emb = self._resolve_inputs(
            G, X, trajectories, F_embeddings, B_embeddings, M
        )
        self.M_ = M_mat
        self.F_ = F_emb
        self.B_ = B_emb

        coords = F_emb if self.ground == "F" else B_emb
        measures = softmax_measure(M_mat, tau=self.softmax_tau) if self.use_softmax else M_mat

        T = coords.shape[0]
        k = min(self.k_neighbors, T - 1)
        if k < 1:
            self.curvature_ = np.zeros(T)
            return self.curvature_

        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nn.fit(coords)
        _, idx = nn.kneighbors(coords)
        neigh_i = idx[:, 1:]

        def _one(t: int, tp_: int) -> tuple[int, float]:
            w = truncated_sliced_w1(
                measures[t], measures[tp_], coords,
                top_n=self.top_n, n_projections=self.n_projections, seed=self.seed,
            )
            return t, w

        pairs = [(t, int(neigh_i[t, j])) for t in range(T) for j in range(k)]
        results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_one)(t, tp_) for t, tp_ in pairs
        )
        per_t: dict[int, list[float]] = defaultdict(list)
        for t, w in results:
            per_t[t].append(w)
        w1 = np.zeros(T)
        for t in range(T):
            if per_t[t]:
                w1[t] = float(np.mean(per_t[t]))
        self.curvature_ = -w1
        return self.curvature_
