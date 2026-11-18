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


# ---------------------------------------------------------------------------
# The gESS profile and its area
# ---------------------------------------------------------------------------
# Ported 2026-11-18 from reason_reckon (`successor_entropies.gess_profile` and its
# neighbours, experiments/44, notebook 06, section 2b). The evidence for the three
# choices (no hand-set scale, no crowding division, a ceiling set by the population)
# is in reason_reckon experiments/47, notebook 05, sections 9 and 10.

PROFILE_KAPPAS = 2.0 ** np.arange(-1, 13.5, 0.5)   # angular resolutions sqrt(2 / kappa) from 115 degrees down to 0.9


def gess_profile(
    M: np.ndarray,
    B_embeddings: np.ndarray,
    tau: float = 1.0,
    kappas: np.ndarray = PROFILE_KAPPAS,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """The successor measure of every row of `M`, read at every scale.

    The weights are softmax(M / tau) over the corpus. At each kernel
    concentration kappa the reading is the log geometric effective sample size
    (You and Cho 2026): minus the log of pi' K pi with K = exp(-kappa (1 - cos))
    between the unit-normalised backward embeddings.

    Returns:
        G   : (T, len(kappas)), rising with kappa from 0 towards its ceiling
        top : (T,) the ceiling, log ESS of the weights
    """
    with torch.no_grad():
        Bu = torch.nn.functional.normalize(
            torch.as_tensor(np.asarray(B_embeddings), dtype=torch.float64, device=device), dim=1)
        D = 1 - (Bu @ Bu.T).clamp(-1, 1)
        Pi = torch.softmax(
            torch.as_tensor(np.asarray(M), dtype=torch.float64, device=device) / max(tau, 1e-12), 1)
        top = -torch.log((Pi ** 2).sum(1))
        G = torch.stack(
            [-torch.log(((Pi @ torch.exp(-float(k) * D)) * Pi).sum(1)) for k in kappas], 1)
        return G.cpu().numpy(), top.cpu().numpy()


def convergence_scale(
    G: np.ndarray, top: np.ndarray, kappas: np.ndarray = PROFILE_KAPPAS, share: float = 0.95,
) -> float:
    """The population's ceiling for the area (after Limbeck et al. 2024): the
    first concentration of the grid at which the median row's profile has
    reached `share` of its ceiling. The grid's last scale when it never does."""
    med = np.median(G / np.maximum(top[:, None], 1e-9), 0)
    hit = np.where(med >= share)[0]
    return float(kappas[hit[0]] if len(hit) else kappas[-1])


def profile_area(G: np.ndarray, kappa_max: float, kappas: np.ndarray = PROFILE_KAPPAS) -> np.ndarray:
    """The mean of each row's profile over the grid's scales coarser than
    `kappa_max` (at least the coarsest one)."""
    keep = np.asarray(kappas) < kappa_max
    keep[0] = True
    return G[:, keep].mean(1)
