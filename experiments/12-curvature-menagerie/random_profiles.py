"""Random warped-product profile factory — honest generalization training set.

Samples f-profiles beyond the fixed battery (dumbbell beta=0.8, necklace b=0.7)
so an integrator trained on them is tested on genuinely held-out families. Four
families: dumbbell (variable neck depth), necklace (variable pearls k and depth),
bumpy_cap (sphere base times random interior gaussian bumps -> S^d topology),
bumpy_periodic (positive cosine harmonics -> boundary-free tube). Every profile
is a smooth positive f with the cap/periodicity WarpedProduct requires, so the
certified per-point scalar-curvature field is well defined.

Exclusions (so the E1 held-out families never leak into training): dumbbell
beta near {0.5, 0.8} and necklace (k=2, b near {0.55, 0.7}).
"""

from __future__ import annotations

import numpy as np


def _dumbbell(rng):
    beta = rng.uniform(0.4, 0.85)
    while abs(beta - 0.8) < 0.03 or abs(beta - 0.5) < 0.03:
        beta = rng.uniform(0.4, 0.85)
    L = np.pi
    w = L / 10.0 * rng.uniform(0.7, 1.4)

    def f(r):
        r = np.clip(r, 0, L)
        return (L / np.pi) * np.sin(np.pi * r / L) * (
            1 - beta * np.exp(-((r - L / 2) ** 2) / (2 * w ** 2)))
    return f, L, False, dict(family="dumbbell", beta=round(float(beta), 3),
                             w=round(float(w), 3))


def _necklace(rng):
    k = int(rng.choice([1, 2, 3]))
    b = rng.uniform(0.4, 0.8)
    while k == 2 and (abs(b - 0.7) < 0.03 or abs(b - 0.55) < 0.03):
        b = rng.uniform(0.4, 0.8)
    L, a = 2 * np.pi, 1.0
    phase = rng.uniform(0, 2 * np.pi)

    def f(r):
        r = np.asarray(r, dtype=float)
        return a + b * np.cos(2 * np.pi * k * r / L + phase)
    return f, L, True, dict(family="necklace", b=round(float(b), 3), k=k)


def _bumpy_cap(rng):
    """Sphere base (L/pi) sin(pi r/L) modulated by interior gaussian bumps."""
    L = np.pi
    nb = int(rng.integers(2, 4))
    centers = rng.uniform(0.25 * L, 0.75 * L, nb)
    amps = rng.uniform(-0.5, 0.5, nb)
    widths = rng.uniform(0.06 * L, 0.16 * L, nb)

    def f(r):
        r = np.clip(r, 0, L)
        mod = 1.0 + sum(a * np.exp(-((r - c) ** 2) / (2 * wj ** 2))
                        for a, c, wj in zip(amps, centers, widths))
        mod = np.maximum(mod, 0.15)
        return (L / np.pi) * np.sin(np.pi * r / L) * mod
    return f, L, False, dict(family="bumpy_cap", nb=nb,
                             amp=round(float(np.abs(amps).max()), 3))


def _bumpy_periodic(rng):
    """Positive cosine tube: a + sum c_k cos(2pi k r/L + phase), a>sum|c_k|."""
    L = 2 * np.pi
    ks = sorted(int(x) for x in rng.choice([1, 2, 3, 4], size=int(rng.integers(2, 4)),
                                           replace=False))
    cs = rng.uniform(0.15, 0.45, len(ks))
    a = float(np.sum(cs)) + rng.uniform(0.15, 0.4)
    phases = rng.uniform(0, 2 * np.pi, len(ks))

    def f(r):
        r = np.asarray(r, dtype=float)
        return a + sum(c * np.cos(2 * np.pi * k * r / L + p)
                       for c, k, p in zip(cs, ks, phases))
    return f, L, True, dict(family="bumpy_periodic", ks=str(ks),
                            a=round(a, 3))


_FAMILIES = [_dumbbell, _necklace, _bumpy_cap, _bumpy_periodic]


def random_profile(rng) -> tuple:
    """Return (f, L, periodic, params) for a randomly chosen held-out family."""
    return _FAMILIES[int(rng.integers(len(_FAMILIES)))](rng)
