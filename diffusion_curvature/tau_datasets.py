"""Trajectory-sampled (τ-) versions of the SadSpheres and Colosseum benchmarks.

Each instance of a τ-dataset is built as follows:

1. Sample a corpus of `n_samples` points from the underlying manifold (via the
   usual rejection-sampled SadSpheres / Colosseum generators).
2. Build a kNN PyGSP graph on the corpus.
3. Draw `n_trajectories` random walks of length `traj_length` on that graph via
   `trajectory_utils.subsample_trajectories`.

The instance retains both the raw corpus coordinates (for non-trajectory methods)
and the trajectory-indexed view (for successor-style methods). Per-point
ground-truth scalar curvature `ks` is inherited from the generator.

Dataset instances plug into the existing `SelfEvaluatingDataset` machinery.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pygsp

from .datasets import rejection_sample_from_saddle, sphere
from .random_surfaces import samples_from_random_surface
from .self_evaluating_datasets import SelfEvaluatingDataset
from .trajectory_utils import subsample_trajectories


def build_tau_instance(
    X: np.ndarray,
    ks: np.ndarray | float,
    n_trajectories: int,
    traj_length: int,
    knn: int,
    seed: int,
) -> dict[str, Any]:
    """Wrap a corpus (X, ks) with a kNN graph + sampled trajectories."""
    X = np.asarray(X, dtype=np.float32)
    if np.isscalar(ks):
        ks = np.full(X.shape[0], float(ks), dtype=np.float64)
    else:
        ks = np.asarray(ks, dtype=np.float64)
    G = pygsp.graphs.NNGraph(X, k=knn)
    traj_idx = subsample_trajectories(
        G, n_trajectories=n_trajectories, length=traj_length, rng=seed,
    )
    return {
        "X": X,
        "ks": ks,
        "G": G,
        "trajectories_idx": traj_idx,          # (n_traj, traj_length + 1) node indices
        "trajectories": X[traj_idx],           # (n_traj, traj_length + 1, d) coords
    }


def _as_list(x: int | Iterable[int]) -> list[int]:
    return [x] if isinstance(x, int) else list(x)


class TauSadSpheres(SelfEvaluatingDataset):
    """Trajectory-sampled SadSpheres.

    For each intrinsic dim and each of `num_pointclouds` instances, generates a
    saddle and a sphere corpus (plus optional plane), builds a kNN graph, and
    samples trajectories.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        n_trajectories: int = 200,
        traj_length: int = 50,
        dimension: int | list[int] = 2,
        num_pointclouds: int = 20,
        knn: int = 10,
        seed: int = 0,
        save_directory: str = ".tau-sad-spheres",
    ):
        rng = np.random.default_rng(seed)
        dims = _as_list(dimension)
        datalist: list[dict[str, Any]] = []
        names: list[str] = []

        for d in dims:
            for i in range(num_pointclouds):
                X_s, ks_s = rejection_sample_from_saddle(n_samples, d)
                datalist.append({
                    **build_tau_instance(X_s, ks_s, n_trajectories, traj_length, knn,
                                         int(rng.integers(0, 2**31 - 1))),
                    "d": d, "shape": "saddle", "instance": i,
                })
                names.append(f"{d}-Saddle-{i}")

                X_sp, ks_sp = sphere(n_samples, d)
                datalist.append({
                    **build_tau_instance(X_sp, float(ks_sp[0]) if np.ndim(ks_sp) > 0 else float(ks_sp),
                                         n_trajectories, traj_length, knn,
                                         int(rng.integers(0, 2**31 - 1))),
                    "d": d, "shape": "sphere", "instance": i,
                })
                names.append(f"{d}-Sphere-{i}")

        self.n_samples = n_samples
        self.n_trajectories = n_trajectories
        self.traj_length = traj_length
        self.knn = knn
        super().__init__(datalist, names, ["ks"], save_directory=save_directory)

    def get_item(self, idx: int) -> dict[str, Any]:
        return self.DS[idx].obj

    def get_truth(self, result_name: str, idx: int) -> np.ndarray:
        return self.DS[idx].obj["ks"]


class TauColosseum(SelfEvaluatingDataset):
    """Trajectory-sampled Curvature Colosseum.

    For each (intrinsic dim, codim, noise) triple and each of
    `num_manifolds_per_dim` random polynomial surfaces, samples a corpus,
    builds a kNN graph, and draws trajectories.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        n_trajectories: int = 200,
        traj_length: int = 50,
        intrinsic_dims: list[int] = (2,),
        codimensions: list[int] = (1,),
        noise_levels: list[float] = (0.0,),
        num_manifolds_per_dim: int = 10,
        knn: int = 10,
        seed: int = 0,
        save_directory: str = ".tau-colosseum",
    ):
        rng = np.random.default_rng(seed)
        datalist: list[dict[str, Any]] = []
        names: list[str] = []

        for d in intrinsic_dims:
            for c in codimensions:
                N = d + c
                for noise in noise_levels:
                    for i in range(num_manifolds_per_dim):
                        X, ks = samples_from_random_surface(
                            n_samples, d, N, degree=2, noise_level=noise,
                        )
                        datalist.append({
                            **build_tau_instance(X, ks, n_trajectories, traj_length, knn,
                                                 int(rng.integers(0, 2**31 - 1))),
                            "d": d, "c": c, "noise": float(noise), "instance": i,
                        })
                        names.append(f"d{d}-c{c}-n{noise}-m{i}")

        self.n_samples = n_samples
        self.n_trajectories = n_trajectories
        self.traj_length = traj_length
        self.knn = knn
        super().__init__(datalist, names, ["ks"], save_directory=save_directory)

    def get_item(self, idx: int) -> dict[str, Any]:
        return self.DS[idx].obj

    def get_truth(self, result_name: str, idx: int) -> np.ndarray:
        return self.DS[idx].obj["ks"]
