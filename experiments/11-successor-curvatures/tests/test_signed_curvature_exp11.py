"""Tests for diffusion_curvature.wasserstein_signed (experiment 11)."""

import numpy as np
import pytest

from diffusion_curvature.datasets import rejection_sample_from_saddle, sphere
from diffusion_curvature.wasserstein_signed import (
    DiffusionORC,
    MidpointEntropyCurvature,
    WassersteinSignedCurvature,
)


@pytest.fixture(scope="module")
def small_sphere():
    X, _ = sphere(600, d=2, seed=3)
    return np.asarray(X, dtype=np.float64)


@pytest.fixture(scope="module")
def small_saddle():
    X, ks = rejection_sample_from_saddle(600, 2)
    return np.asarray(X, dtype=np.float64), float(ks)


def test_signed_orc_sphere_positive(small_sphere):
    est = WassersteinSignedCurvature(t=4, n_pairs=6, seed=0,
                                     compute_midpoint=False)
    est.fit(X=small_sphere, idx=[0, 7, 21])
    assert est.orc_.shape == (3,)
    assert np.isfinite(est.orc_).all()
    assert np.nanmean(est.orc_) > 0
    assert np.isfinite(est.orc_phys_).all()
    assert np.nanmean(est.orc_phys_) > 0


def test_signed_orc_saddle_negative(small_saddle):
    X, ks = small_saddle
    assert ks < 0
    est = WassersteinSignedCurvature(t=4, n_pairs=6, seed=0,
                                     compute_midpoint=False)
    est.fit(X=X, idx=[0])
    assert np.nanmean(est.orc_) < 0


def test_wrapper_classes_return_arrays(small_sphere):
    orc = DiffusionORC(t=4, n_pairs=4, seed=0, compute_midpoint=False)
    k1 = orc.fit_transform(X=small_sphere, idx=[0])
    assert k1.shape == (1,)
    mid = MidpointEntropyCurvature(t=4, n_pairs=4, seed=0)
    k2 = mid.fit_transform(X=small_sphere, idx=[0])
    assert k2.shape == (1,)
    assert np.isfinite(k2).all()


def test_precomputed_measure_passthrough(small_sphere):
    """Supplying M (successor-measure stand-in) skips diffusion powers."""
    n = small_sphere.shape[0]
    rng = np.random.default_rng(0)
    M = rng.random((n, n)) ** 4  # sparse-ish synthetic measures
    est = WassersteinSignedCurvature(t=4, n_pairs=4, seed=0,
                                     compute_midpoint=False)
    est.fit(X=small_sphere, M=M, idx=[0])
    assert est.orc_.shape == (1,)
    assert np.isfinite(est.orc_).all()
