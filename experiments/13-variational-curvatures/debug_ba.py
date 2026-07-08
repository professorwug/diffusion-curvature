"""Debug the BA decoder on the small exact chain: separate target mismatch
(mixture-of-entropies vs entropy-of-mixture) from estimator error
(decoder misfit / readout variance)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2].parent))
sys.path.insert(0, str(Path(__file__).parent / "tests"))
from test_13_variational_curvatures import (_exact_mi, _two_scale_chain,
                                            _walks)

from diffusion_curvature.variational import BADecoder

GAMMA = 0.9
P, rng = _two_scale_chain()
n = P.shape[0]
traj = _walks(P, nt=20, T=800, rng=rng)

# exact targets
H_M = np.log(n) - _exact_mi(P, GAMMA)          # entropy of resolvent row
Pk = np.eye(n)
H_mix = np.zeros(n)                             # E_n[H(P^n(s,.))], n~Geom
w_total = 0.0
for k in range(1, 120):
    Pk = Pk @ P
    w = (1 - GAMMA) * GAMMA ** (k - 1)
    Pc = np.clip(Pk, 1e-30, 1)
    H_mix += w * (-(Pc * np.log(Pc)).sum(1))
    w_total += w
H_mix /= w_total

print(f"r(H_mix, H_M) = {np.corrcoef(H_mix, H_M)[0,1]:+.3f}")
print(f"means: H_M {H_M.mean():.3f}  H_mix {H_mix.mean():.3f}")

for epochs, z in [(20, 16), (60, 32)]:
    est = BADecoder(gamma=GAMMA, z_dim=z, n_epochs=epochs, batch_size=4096,
                    lr=1e-2, device="cpu", seed=0).fit(traj, n)
    out = est.ba_field(np.arange(n))
    m = np.isfinite(out.ba)
    print(f"\nepochs={epochs} z={z}: mean BA={np.nanmean(out.ba):.3f} "
          f"(target H_mix mean {H_mix.mean():.3f})")
    print(f"  r(BA, H_mix)={np.corrcoef(out.ba[m], H_mix[m])[0,1]:+.3f}  "
          f"r(BA, H_M)={np.corrcoef(out.ba[m], H_M[m])[0,1]:+.3f}  "
          f"n_paths min/med={out.n_paths.min():.0f}/{np.median(out.n_paths):.0f}")
    # two net seeds to gauge seed noise
    out2 = BADecoder(gamma=GAMMA, z_dim=z, n_epochs=epochs, batch_size=4096,
                     lr=1e-2, device="cpu", seed=1).fit(traj, n).ba_field(
        np.arange(n))
    print(f"  r(BA_s0, BA_s1)={np.corrcoef(out.ba[m], out2.ba[m])[0,1]:+.3f}")
