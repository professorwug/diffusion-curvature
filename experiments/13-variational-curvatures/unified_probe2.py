"""Verify the lag-displacement jitter rule (aug_scale=0.5) on the three
diagnostic cells before the full army."""
import sys, warnings
from pathlib import Path
import numpy as np, torch
from scipy.stats import pearsonr
sys.path.insert(0, str(Path(__file__).parent / "../11-successor-curvatures"))
from diffusion_curvature.variational import TDInfoNCE
from noise_benchmark import pooled_dv, prepare_unit
warnings.filterwarnings("ignore")
DEV = "cuda:1" if torch.cuda.is_available() else "cpu"
caches = {}
for noise in ("clean", "iso15", "hd64"):
    unit = dict(profile="nk2", d=3, wseed=0, noise=noise)
    prep = prepare_unit(unit, caches)
    vals = []
    for ns in (0, 1):
        est = TDInfoNCE(gamma=prep["gamma_c"], z_dim=128, features="coords",
                        hidden=256, n_epochs=300, batch_size=4096,
                        n_candidates=511, lr=3e-4, holdout_frac=0.5,
                        lags_per_step=16, aug_scale=0.5,
                        device=DEV, seed=ns).fit(
            prep["traj_idx"], prep["n_pts"], X=prep["X"])
        vals.append(pooled_dv(est, prep["groups"], rng=ns))
    v = np.nanmean(vals, axis=0)
    kt = prep["kt_w"]
    mm = np.isfinite(v) & np.isfinite(kt)
    print(f"{noise:<6} aug=0.5xlag: r={pearsonr(v[mm], kt[mm])[0]:+.3f} "
          f"(jit={est.jitter_:.3f})", flush=True)
