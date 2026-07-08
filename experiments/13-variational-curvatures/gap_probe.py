"""V3 probe (exact level): does the sandwich gap H(S+|s) - I1(s) localize
the slip-noise field eta(x)? (It should: the gap is the conditional entropy
H(S+|a,s), whose spatially varying part is the noise.)"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent / "../11-successor-curvatures"))
from benchmark_kmetric_colosseum import affinity_from_D
from diffusion_curvature.menagerie import WarpedProduct, necklace_profile
from vmi_noise import (KNN, KSLIP, N, entropy_rows, exact_I1, resolvent,
                       sample_with_u)

warnings.filterwarnings("ignore")

f, L = necklace_profile(b=0.7)
wp = WarpedProduct(f, L, d=3, periodic=True)
for wseed in (0, 1):
    D, ks, u1 = sample_with_u(wp, N, wseed)
    s = np.median(D[np.triu_indices_from(D, 1)])
    D, ks = D / s, ks * s**2
    W = affinity_from_D(D, k=KNN)
    P = W / np.maximum(W.sum(1, keepdims=True), 1e-30)
    nbr25 = np.argsort(D, axis=1)[:, 1:KSLIP + 1]
    anchors = np.arange(0, N, 7)          # broad anchor set
    for eta_max in (0.4, 0.8):
        eta = eta_max * (1 + u1) / 2
        U25 = np.zeros((N, N))
        np.put_along_axis(U25, nbr25, 1.0 / KSLIP, axis=1)
        P_noisy = (1 - eta)[:, None] * P + eta[:, None] * U25
        Mres = resolvent(P_noisy)
        H = entropy_rows(Mres[anchors])
        I1 = exact_I1(P_noisy, Mres, eta, nbr25, anchors)
        gap = H - I1
        print(f"wseed={wseed} eta_max={eta_max}: "
              f"r(gap, eta)={pearsonr(gap, eta[anchors])[0]:+.3f}  "
              f"r(H, eta)={pearsonr(H, eta[anchors])[0]:+.3f}  "
              f"r(gap, ks)={pearsonr(gap, ks[anchors])[0]:+.3f}",
              flush=True)
