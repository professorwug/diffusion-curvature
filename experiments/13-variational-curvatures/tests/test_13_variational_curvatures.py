"""Exp-13 tests: VMI estimator recovers the exact resolvent-MI field on a
small known chain."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[3]))
from diffusion_curvature.variational import VMI, harvest_pairs  # noqa: E402


def _two_scale_chain(n=80, seed=0):
    """Ring chain with a heterogeneous laziness profile -> nonuniform
    resolvent-entropy field with known exact values."""
    rng = np.random.default_rng(seed)
    P = np.zeros((n, n))
    lazy = 0.1 + 0.8 * (np.sin(2 * np.pi * np.arange(n) / n) + 1) / 2
    for i in range(n):
        P[i, i] = lazy[i]
        P[i, (i - 1) % n] = (1 - lazy[i]) / 2
        P[i, (i + 1) % n] = (1 - lazy[i]) / 2
    return P, rng


def _walks(P, nt, T, rng):
    n = P.shape[0]
    cum = np.cumsum(P, axis=1)
    traj = np.empty((nt, T + 1), dtype=int)
    traj[:, 0] = rng.integers(0, n, nt)
    for t in range(T):
        u = rng.random(nt)
        traj[:, t + 1] = [np.searchsorted(cum[traj[w, t]], u[w])
                          for w in range(nt)]
    return traj


def _exact_mi(P, gamma):
    n = P.shape[0]
    M = (1 - gamma) * np.linalg.inv(np.eye(n) - gamma * P)
    M = np.clip(M, 1e-30, None)
    M = M / M.sum(1, keepdims=True)
    return np.log(n) + (M * np.log(M)).sum(1)


def test_harvest_pairs_lags_geometric():
    traj = np.arange(200)[None, :].repeat(3, axis=0)
    a, p = harvest_pairs(traj, gamma=0.9, lags_per_step=2,
                         rng=np.random.default_rng(0))
    lags = p - a
    assert (lags >= 1).all()
    assert 5 < lags.mean() < 15  # Geom(0.1) mean 10, truncation pulls down


@pytest.mark.parametrize("gamma", [0.9])
def test_vmi_recovers_exact_field(gamma):
    P, rng = _two_scale_chain()
    traj = _walks(P, nt=20, T=800, rng=rng)
    mi_exact = _exact_mi(P, gamma)
    est = VMI(gamma=gamma, z_dim=16, n_epochs=40, batch_size=2048,
              lr=1e-2, device="cpu", seed=0).fit(traj, P.shape[0])
    out = est.mi_field(np.arange(P.shape[0]))
    for name, v in [("plugin", out.mi_plugin), ("bound", out.mi_bound)]:
        m = np.isfinite(v)
        r = np.corrcoef(v[m], mi_exact[m])[0, 1]
        assert r > 0.8, f"{name} field corr {r:.2f} vs exact"
    # DV bound should actually lower-bound the (pooled) exact MI
    assert np.nanmean(out.mi_bound) <= mi_exact.mean() + 0.1


def test_ba_decoder_recovers_exact_field():
    """For deterministic path-dynamics, I(A;S+|s) = H(S+|s): the BA field
    targets the ENTROPY (negative of V1's concentration-MI, log n - H)."""
    from diffusion_curvature.variational import BADecoder
    gamma = 0.9
    P, rng = _two_scale_chain()
    traj = _walks(P, nt=20, T=800, rng=rng)
    H_exact = np.log(P.shape[0]) - _exact_mi(P, gamma)
    est = BADecoder(gamma=gamma, z_dim=16, n_epochs=20, batch_size=4096,
                    lr=1e-2, device="cpu", seed=0).fit(traj, P.shape[0])
    out = est.ba_field(np.arange(P.shape[0]))
    m = np.isfinite(out.ba)
    r = np.corrcoef(out.ba[m], H_exact[m])[0, 1]
    assert r > 0.8, f"BA field corr {r:.2f} vs exact H"
    # BA lower-bounds the pooled I(A;S+|s) = H(S+|s)
    assert np.nanmean(out.ba) <= H_exact.mean() + 0.1
