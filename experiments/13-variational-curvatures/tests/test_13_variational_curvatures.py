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


def test_brownian_walks_stationarity_and_msd():
    """BM on the necklace: stationary r-histogram matches the volume
    element f^{d-1}; short-time mean squared geodesic displacement ~ d*t."""
    from diffusion_curvature.continuous_walks import brownian_walks
    from diffusion_curvature.menagerie import WarpedProduct, necklace_profile
    f, L = necklace_profile(b=0.7)
    wp = WarpedProduct(f, L, d=3, periodic=True)
    dt = 0.004
    out = brownian_walks(wp, nt=20, T=1500, rng=0, dt=dt)
    r = out["r"][:, 300:].ravel()  # discard burn-in
    hist, edges = np.histogram(r, bins=40, range=(0, wp.L), density=True)
    mid = 0.5 * (edges[:-1] + edges[1:])
    pdf = np.interp(mid, wp.rg, np.maximum(wp.fg, 0) ** (wp.d - 1))
    pdf = pdf / np.trapz(pdf, mid)
    assert np.corrcoef(hist, pdf)[0, 1] > 0.9, "stationary r-density wrong"
    # MSD over k steps: E[d_geo^2] ~ d * (k dt) for small times
    rng = np.random.default_rng(1)
    for k in (5, 20):
        idx = rng.integers(0, 20, 200), rng.integers(300, 1400, 200)
        r1, u1 = out["r"][idx], out["u"][idx]
        r2 = out["r"][idx[0], idx[1] + k]
        u2 = out["u"][idx[0], idx[1] + k]
        ang = np.arccos(np.clip((u1 * u2).sum(1), -1, 1))
        dg = wp.pair_distance(r1, r2, ang)
        ratio = (dg**2).mean() / (wp.d * k * dt)
        assert 0.6 < ratio < 1.5, f"MSD ratio {ratio:.2f} at k={k}"


def test_td_infonce_recovers_exact_field():
    from diffusion_curvature.variational import TDInfoNCE
    gamma = 0.9
    P, rng = _two_scale_chain()
    traj = _walks(P, nt=20, T=800, rng=rng)
    mi_exact = _exact_mi(P, gamma)
    est = TDInfoNCE(gamma=gamma, z_dim=16, features="tabular", n_epochs=60,
                    batch_size=2048, n_candidates=79, lr=1e-2,
                    device="cpu", seed=0).fit(traj, P.shape[0])
    out = est.mi_field(np.arange(P.shape[0]))
    m = np.isfinite(out.mi_bound)
    r = np.corrcoef(out.mi_bound[m], mi_exact[m])[0, 1]
    assert r > 0.8, f"TD-InfoNCE DV field corr {r:.2f} vs exact"


@pytest.mark.xfail(reason="boundary result (round 6): analytic resolvent "
                   "readouts through estimated spectra are hypersensitive "
                   "to near-1 eigenvalue error (dg/dlam ~ 10); even exact "
                   "eigenvectors + 4-decimal lambdas scramble the field. "
                   "Use spectral_coords + TD readout instead.")
def test_spectral_sf_recovers_exact_field():
    """specSENT: learned eigenbasis + analytic resolvent should recover the
    exact resolvent-entropy field on the small chain."""
    from diffusion_curvature.variational import SpectralSF
    gamma = 0.9
    P, rng = _two_scale_chain()
    traj = _walks(P, nt=20, T=800, rng=rng)
    H_exact = np.log(P.shape[0]) - _exact_mi(P, gamma)
    est = SpectralSF(gamma=gamma, k_eig=16, features="tabular",
                     n_epochs=150, batch_size=2048, lr=1e-2, aug_scale=0.0,
                     device="cpu", seed=0).fit(traj, P.shape[0])
    # corpus must be OCCUPANCY-sampled: the spectral reconstruction is the
    # ratio kernel M/rho, so rho enters through the corpus atoms
    corpus = np.random.default_rng(3).choice(traj.ravel(), 2000)
    field = est.sent_field(np.arange(P.shape[0]), corpus)
    r = np.corrcoef(-field, H_exact)[0, 1]   # field is -H
    assert r > 0.8, f"specSENT field corr {r:.2f} vs exact H"


def test_spectral_coords_plus_td_recovers_exact_field():
    """Round-6 pivot: learned eigenfunctions as denoised COORDINATES for
    TD-InfoNCE (analytic spectral readouts are lambda-hypersensitive)."""
    from diffusion_curvature.variational import (SpectralSF, TDInfoNCE,
                                                 spectral_coords)
    gamma = 0.9
    P, rng = _two_scale_chain()
    traj = _walks(P, nt=20, T=800, rng=rng)
    mi_exact = _exact_mi(P, gamma)
    spec = SpectralSF(gamma=gamma, k_eig=16, features="tabular",
                      n_epochs=150, batch_size=2048, lr=1e-2, aug_scale=0.0,
                      device="cpu", seed=0).fit(traj, P.shape[0])
    Psi = spectral_coords(spec, P.shape[0])
    est = TDInfoNCE(gamma=gamma, z_dim=16, features="coords", hidden=128,
                    n_epochs=60, batch_size=2048, n_candidates=79,
                    lr=1e-3, device="cpu", seed=0).fit(
        traj, P.shape[0], X=Psi)
    out = est.mi_field(np.arange(P.shape[0]))
    m = np.isfinite(out.mi_bound)
    r = np.corrcoef(out.mi_bound[m], mi_exact[m])[0, 1]
    assert r > 0.8, f"spec+td field corr {r:.2f} vs exact"
