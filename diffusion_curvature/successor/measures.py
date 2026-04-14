"""Successor measures and distances.

Ported from reason_reckon/experiments/24-successor-representations-spectacular/
successor-orc/successor_orc.py. Adapted so F(obs) takes only an observation
(no action, no z) — see fb_modules.ForwardMap.
"""

from __future__ import annotations

import numpy as np
import ot
import torch

from .fb_modules import BackwardMap, ForwardMap


def compute_successor_measures(
    F_net: ForwardMap,
    B_net: BackwardMap,
    obs: np.ndarray,
    corpus: np.ndarray,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute clipped successor affinities + F/B embeddings.

    Returns:
        M_pos : (T, N) = max(0, min(F1(obs), F2(obs)) @ B(corpus).T)
        F_obs : (T, z_dim) F-space representation of obs points
        B_corp: (N, z_dim) B-space representation of corpus points
    """
    F_net.eval()
    B_net.eval()
    with torch.no_grad():
        o = torch.as_tensor(obs, dtype=torch.float32, device=device)
        c = torch.as_tensor(corpus, dtype=torch.float32, device=device)
        F1, F2 = F_net(o)
        Bc = B_net(c)
        M = torch.minimum(F1 @ Bc.T, F2 @ Bc.T).cpu().numpy()
        F_obs = ((F1 + F2) / 2).cpu().numpy()
        B_corp = Bc.cpu().numpy()
    return np.maximum(M, 0.0), F_obs, B_corp


def softmax_measure(M: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Row-wise softmax with temperature `tau` applied to raw affinities M."""
    x = M / max(tau, 1e-12)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def truncated_sliced_w1(
    m1: np.ndarray,
    m2: np.ndarray,
    ground: np.ndarray,
    top_n: int = 200,
    n_projections: int = 50,
    seed: int = 42,
) -> float:
    """Sliced W1 between measures m1, m2 truncated to the union of their top-n supports.

    Args:
        m1, m2    : (N,) non-negative mass vectors over the corpus.
        ground    : (N, d) ground-metric coordinates (F- or B-space).
        top_n     : retain only the top-n support of each measure.
    """
    idx1 = np.argsort(m1)[-top_n:]
    idx2 = np.argsort(m2)[-top_n:]
    shared = np.union1d(idx1, idx2)
    w1 = m1[shared].copy()
    w2 = m2[shared].copy()
    s1, s2 = w1.sum(), w2.sum()
    if s1 == 0 or s2 == 0:
        return 0.0
    w1 /= s1
    w2 /= s2
    coords = ground[shared]
    return float(
        ot.sliced_wasserstein_distance(
            coords, coords, w1, w2, n_projections=n_projections, seed=seed
        )
    )
