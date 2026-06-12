"""Self-evaluating wrapper around the Curvature Colosseum battery.

Flattens the nested ``CC[d][c][noise]`` grid into a list of per-manifold
instances compatible with ``SelfEvaluatingDataset``. Each instance carries
its coordinates, ground-truth scalar curvature (at origin), and the
`(d, c, noise, m)` tuple so metrics can be aggregated per cell.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from .random_surfaces import samples_from_random_surface
from .self_evaluating_datasets import SelfEvaluatingDataset


def build_battery(
    intrinsic_dims: list[int] = (2, 3, 4, 5),
    codimensions: list[int] = (1, 2, 3, 4),
    noise_levels: list[float] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5),
    num_manifolds_per_dim: int = 50,
    n_samples_rule=None,
    degree: int = 2,
) -> dict[Any, Any]:
    """Construct a battery dict matching ``construct_battery.create_battery``
    without requiring the ``deepdish`` / ``graphtools`` imports that file
    does at module level.

    ``n_samples_rule(d)`` overrides the heuristic if provided.
    """
    def default_rule(d: int) -> int:
        return 1000 * 2**d if d > 3 else 5000

    rule = n_samples_rule if n_samples_rule is not None else default_rule

    CC: dict[Any, Any] = {"dims": list(intrinsic_dims)}
    for d in intrinsic_dims:
        CC[d] = {"codims": list(codimensions)}
        for c in codimensions:
            CC[d][c] = {"noise_levels": list(noise_levels)}
            for noise in noise_levels:
                CC[d][c][noise] = {"Xs": [], "k": []}
                N = d + c
                n_samples = rule(d)
                for _ in range(num_manifolds_per_dim):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        X, k = samples_from_random_surface(
                            n_samples, d, N, degree=degree, noise_level=noise,
                        )
                    CC[d][c][noise]["Xs"].append(np.asarray(X, dtype=np.float32))
                    CC[d][c][noise]["k"].append(float(np.asarray(k).item()
                                                       if np.asarray(k).ndim == 0
                                                       else float(np.mean(k))))
    return CC


class CurvatureColosseum(SelfEvaluatingDataset):
    """Flattened self-evaluating dataset over the Curvature Colosseum battery."""

    def __init__(
        self,
        CC: dict[Any, Any] | None = None,
        intrinsic_dims: list[int] = (2, 3, 4, 5),
        codimensions: list[int] = (1, 2, 3, 4),
        noise_levels: list[float] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5),
        num_manifolds_per_dim: int = 50,
        save_directory: str = ".curvature-colosseum",
        **build_kwargs,
    ):
        if CC is None:
            CC = build_battery(
                intrinsic_dims=intrinsic_dims,
                codimensions=codimensions,
                noise_levels=noise_levels,
                num_manifolds_per_dim=num_manifolds_per_dim,
                **build_kwargs,
            )

        datalist: list[dict[str, Any]] = []
        names: list[str] = []
        for d in intrinsic_dims:
            for c in codimensions:
                for noise in noise_levels:
                    slot = CC[d][c][noise]
                    for m_i, (X, k) in enumerate(zip(slot["Xs"], slot["k"])):
                        datalist.append({
                            "X": np.asarray(X, dtype=np.float32),
                            "ks": float(k),
                            "d": int(d), "c": int(c),
                            "noise": float(noise), "m": m_i,
                        })
                        names.append(f"d{d}-c{c}-n{noise}-m{m_i}")

        self.CC = CC
        super().__init__(datalist, names, ["ks"], save_directory=save_directory)

    def get_item(self, idx: int) -> np.ndarray:
        return self.DS[idx].obj["X"]

    def get_truth(self, result_name: str, idx: int) -> float:
        return self.DS[idx].obj["ks"]

    def meta(self, idx: int) -> dict[str, Any]:
        o = self.DS[idx].obj
        return {"d": o["d"], "c": o["c"], "noise": o["noise"], "m": o["m"]}
