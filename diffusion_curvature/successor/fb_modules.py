"""Forward/Backward representation networks for successor features.

Adapted from reason_reckon/experiments/24-successor-representations-spectacular/
ollivier-fb/fb_modules.py (Meta AI, MIT licensed). Simplifications for our
non-RL setting:

- No action or task-embedding (z) conditioning. `ForwardMap(obs) -> (F1, F2)`.
- `BackwardMap(obs) -> B`, unchanged.
- `mlp`, `_L2`, `_nl` helpers copied verbatim so we can reuse `"ntanh"` /
  `"L2"` / etc. activation labels from the source.
"""

from __future__ import annotations

import math
from typing import Sequence, Union

import torch
import torch.nn.functional as F
from torch import nn


class _L2(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return math.sqrt(self.dim) * F.normalize(x, dim=1)


def _nl(name: str, dim: int) -> list[nn.Module]:
    match name:
        case "irelu":
            return [nn.ReLU(inplace=True)]
        case "relu":
            return [nn.ReLU()]
        case "ntanh":
            return [nn.LayerNorm(dim), nn.Tanh()]
        case "layernorm":
            return [nn.LayerNorm(dim)]
        case "tanh":
            return [nn.Tanh()]
        case "L2":
            return [_L2(dim)]
        case _:
            raise ValueError(f"Unknown non-linearity {name!r}")


def mlp(*layers: Union[int, str]) -> nn.Sequential:
    """Build a Sequential MLP. See the original fb_modules.mlp for details."""
    assert len(layers) >= 2
    assert isinstance(layers[0], int), "First input must be a dimension"
    prev_dim: int = layers[0]
    seq: list[nn.Module] = []
    for layer in layers[1:]:
        if isinstance(layer, str):
            seq.extend(_nl(layer, prev_dim))
        else:
            seq.append(nn.Linear(prev_dim, layer))
            prev_dim = layer
    return nn.Sequential(*seq)


def _xavier_init(m: nn.Module) -> None:
    """Xavier-uniform init matching reason_reckon's torus-baseline/utils_shim."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ForwardMap(nn.Module):
    """Twin-head forward map: `obs -> (F1, F2)`.

    The two heads implement the min-trick from reason_reckon's
    `compute_successor_measures`: the successor affinity is
    ``min(F1 @ B.T, F2 @ B.T)``, which mirrors clipped-double-Q in DDPG-style FB.

    Simplifications vs. reason_reckon: no `z` (task) or `action` inputs.
    """

    def __init__(
        self,
        obs_dim: int,
        z_dim: int = 100,
        hidden_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        trunk = (obs_dim, hidden_dim, "ntanh", hidden_dim, "irelu", hidden_dim, "irelu")
        self.trunk = mlp(*trunk)
        head = (hidden_dim, hidden_dim, "irelu", z_dim)
        self.F1 = mlp(*head)
        self.F2 = mlp(*head)
        self.apply(_xavier_init)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        return self.F1(h), self.F2(h)


class BackwardMap(nn.Module):
    """Backward map `obs -> B`, with optional L2 normalization (scaled by sqrt(z_dim))."""

    def __init__(
        self,
        obs_dim: int,
        z_dim: int = 100,
        hidden_dim: int = 512,
        norm_z: bool = True,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.norm_z = norm_z
        self.B = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim)
        self.apply(_xavier_init)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = self.B(obs)
        if self.norm_z:
            B = math.sqrt(self.z_dim) * F.normalize(B, dim=1)
        return B
