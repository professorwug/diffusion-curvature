"""Successor-representation curvatures.

Public API:
    - SuccessorEntropyCurvature : entropy of softmax F^T B successor measure
    - SuccessorORC              : Ollivier-Ricci curvature in F- or B-space
    - SuccessorProfileArea      : area under the gESS profile of the successor measure

Each class owns F and B networks and trains them on `fit` (when trajectories
or a graph are supplied), or accepts precomputed F/B embeddings for inference.

Adapted from reason_reckon/experiments/24-successor-representations-spectacular.
See the zettel [[20260404 Diffusion Curvature Revival with Trajectory Additions]]
for the design rationale.
"""

from .curvatures import (
    SuccessorEntropyCurvature,
    SuccessorORC,
    SuccessorProfileArea,
)
from .fb_modules import BackwardMap, ForwardMap
from .measures import (
    PROFILE_KAPPAS,
    compute_successor_measures,
    convergence_scale,
    gess_profile,
    profile_area,
    truncated_sliced_w1,
)
from .train import FBTrainer, fb_loss

__all__ = [
    "SuccessorEntropyCurvature",
    "SuccessorORC",
    "SuccessorProfileArea",
    "PROFILE_KAPPAS",
    "gess_profile",
    "convergence_scale",
    "profile_area",
    "BackwardMap",
    "ForwardMap",
    "FBTrainer",
    "fb_loss",
    "compute_successor_measures",
    "truncated_sliced_w1",
]
