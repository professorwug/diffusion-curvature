"""Unified-design probe: TD-InfoNCE + input-noise augmentation (td_aug).

Pick the augmentation scale on three diagnostic cells:
  nk2/d3/w0/clean  — the pathological clean cell (td stuck at ~0.21-0.25)
  nk2/d3/w0/iso15  — td's home turf (must not regress)
  nk2/d3/w0/hd64   — the embedding regime (must not regress)
alpha in {0, 0.5, 1, 2} x median step length; 2 net seeds.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent / "../11-successor-curvatures"))
from diffusion_curvature.variational import TDInfoNCE
from noise_benchmark import pooled_dv, prepare_unit

warnings.filterwarnings("ignore")

DEV = "cuda:1" if torch.cuda.is_available() else "cpu"
CELLS = [dict(profile="nk2", d=3, wseed=0, noise=n)
         for n in ("clean", "iso15", "hd64")]
ALPHAS = (0.0, 0.5, 1.0, 2.0)

caches = {}
for unit in CELLS:
    prep = prepare_unit(unit, caches)
    X, kt_w = prep["X"], prep["kt_w"]
    groups, gamma_c = prep["groups"], prep["gamma_c"]
    for alpha in ALPHAS:
        vals = []
        for ns in (0, 1):
            est = TDInfoNCE(gamma=gamma_c, z_dim=128, features="coords",
                            hidden=256, n_epochs=300, batch_size=4096,
                            n_candidates=511, lr=3e-4, holdout_frac=0.5,
                            lags_per_step=16, aug_scale=alpha,
                            device=DEV, seed=ns).fit(
                prep["traj_idx"], prep["n_pts"], X=X)
            vals.append(pooled_dv(est, groups, rng=ns))
        v = np.nanmean(vals, axis=0)
        mm = np.isfinite(v) & np.isfinite(kt_w)
        r = pearsonr(v[mm], kt_w[mm])[0]
        print(f"{unit['noise']:<6} alpha={alpha}: r={r:+.3f} "
              f"(jit={est.jitter_:.4f})", flush=True)
