"""Calibrate the noise-structure CV detector across the six settings."""
import sys, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent / "../11-successor-curvatures"))
from diffusion_curvature.variational import noise_structure_cv
from noise_benchmark import prepare_unit
warnings.filterwarnings("ignore")
caches = {}
for noise in ("clean", "iso05", "iso15", "hd64", "hetero", "ar1"):
    for wseed in (0, 1):
        unit = dict(profile="nk2", d=4, wseed=wseed, noise=noise)
        prep = prepare_unit(unit, caches)
        cv = noise_structure_cv(prep["X"], prep["traj_idx"], rng=7)
        print(f"{noise:<7} w{wseed}: CV={cv:.3f}", flush=True)
