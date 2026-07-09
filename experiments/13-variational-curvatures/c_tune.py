"""C-series critic tuning pass — fair-play iteration for vmi_coord and
td_infonce (the graph/count channels got their scale swept; the critics get
epochs/z/lr swept). Same walks as c_series (deterministic regeneration).

Output: processed_data/c_tune.csv
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent / "../11-successor-curvatures"))
from c_series import (DT_NORM, N_ORACLE, NT, POOL_STATES, SPREAD_T, T,
                      pooled_dv)
from diffusion_curvature.continuous_walks import brownian_walks, chi_embed
from diffusion_curvature.menagerie import WarpedProduct, necklace_profile
from diffusion_curvature.variational import VMI, TDInfoNCE

warnings.filterwarnings("ignore")

DEV = "cuda:1" if torch.cuda.is_available() else "cpu"
NET_SEEDS = (0, 1, 2, 3)

CONFIGS = [
    ("vmi_e150_z64", VMI, dict(z_dim=64, n_epochs=150, lr=1e-3,
                               lags_per_step=16)),
    ("vmi_e150_z128", VMI, dict(z_dim=128, n_epochs=150, lr=3e-4,
                                lags_per_step=16)),
    ("td_e300_z64", TDInfoNCE, dict(z_dim=64, n_epochs=300, lr=1e-3)),
    ("td_e300_z128", TDInfoNCE, dict(z_dim=128, n_epochs=300, lr=3e-4)),
]


def main():
    rows = []
    for d in (3, 4):
        f, L = necklace_profile(b=0.7)
        wp = WarpedProduct(f, L, d=d, periodic=True)
        m = wp.sample(N_ORACLE, rng=100 + d)
        s = np.median(m["D"][np.triu_indices(N_ORACLE, 1)])
        gamma_c = 1 - d * DT_NORM / SPREAD_T
        for wseed in (0, 1):
            walks = brownian_walks(wp, NT, T, dt=DT_NORM * s**2, rng=wseed)
            chi = chi_embed(wp, walks["r"], walks["u"]) / s
            n_pts = NT * (T + 1)
            X = chi.reshape(n_pts, -1)
            traj_idx = np.arange(n_pts).reshape(NT, T + 1)
            tree = cKDTree(X)
            tw = brownian_walks(wp, 8, 1200, dt=DT_NORM * s**2,
                                rng=1000 + wseed)
            tchi = (chi_embed(wp, tw["r"], tw["u"]) / s).reshape(-1,
                                                                 X.shape[1])
            tks = (tw["ks"] * s**2).ravel()
            torder = np.argsort(tks)
            targets = torder[(np.linspace(0.02, 0.98, 48)
                              * (len(tks) - 1)).astype(int)]
            Xt_ev, kt_w = tchi[targets], tks[targets]
            _, grp = tree.query(Xt_ev, k=POOL_STATES)
            groups = list(grp)

            def rw(v):
                mm = np.isfinite(v) & np.isfinite(kt_w)
                return (pearsonr(v[mm], kt_w[mm])[0]
                        if mm.sum() > 8 else np.nan)

            for name, cls, kw in CONFIGS:
                t0 = time.time()
                vals = []
                for ns in NET_SEEDS:
                    est = cls(gamma=gamma_c, features="coords", hidden=256,
                              batch_size=4096, holdout_frac=0.5,
                              device=DEV, seed=ns, **kw).fit(
                        traj_idx, n_pts, X=X)
                    vals.append(pooled_dv(est, groups, rng=ns))
                per_seed = [rw(v) for v in vals]
                row = dict(d=d, wseed=wseed, config=name,
                           r_ens=rw(np.nanmean(vals, axis=0)),
                           r_seed_mean=float(np.nanmean(per_seed)),
                           r_seed_min=float(np.nanmin(per_seed)),
                           r_seed_max=float(np.nanmax(per_seed)),
                           t=round(time.time() - t0, 1))
                rows.append(row)
                print(" ".join(f"{k}={v:.3f}" if isinstance(v, float)
                               else f"{k}={v}" for k, v in row.items()),
                      flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("processed_data/c_tune.csv", index=False)
    print("\n=== tuning summary (ensemble r, mean over cells) ===")
    print(df.groupby("config")[["r_ens", "r_seed_mean"]].mean().round(3))


if __name__ == "__main__":
    main()
