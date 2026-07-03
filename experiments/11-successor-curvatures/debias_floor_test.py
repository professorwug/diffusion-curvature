"""Split-sample noise-floor debiasing of the W1-contraction ORC.

Hypothesis: the sparse-regime positive zero-drift (+0.23..+0.35 at d>=4) is
shared-mass noise transporting for free; the transport cost between two
INDEPENDENT estimates of the same measure (walks split in half) measures that
noise floor delta_i directly, and subtracting it should re-zero the plane and
unlock negative-curvature detection.

Per (dataset in {plane, sphere, saddle} at d=4, n_traj in {50, 200}):
  - gamma=0.98 ensemble on all walks -> kernel distances (D_graph)
  - gamma=0.8 ensembles on all walks, half A, half B -> measures M, MA, MB
  - per seed: kappa raw + floor delta_i -> corrections
      lin       : 1 - (W1 - delta_i) / d
      lin_sqrt2 : 1 - (W1 - delta_i/sqrt(2)) / d   (half-sample noise ~ sqrt2 x full)
      quad      : 1 - sqrt(max(W1^2 - delta_i^2/2, 0)) / d
    and a W2-transport arm (raw + lin_sqrt2).
  - unlearned control: traj_dorc raw on the visited cloud.

Usage: pixi run python debias_floor_test.py [--device cuda:0] [--nts 50 200]
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd
import pygsp

from diffusion_curvature.successor.ensemble import EnsembleFBTrainer
from diffusion_curvature.trajectory_utils import subsample_trajectories
from diffusion_curvature.wasserstein_signed import WassersteinSignedCurvature

from benchmark_kmetric_colosseum import knn_distance_graph, potential_distances
from fb_kernel_ablation_d4 import build_Xd, relu_rownorm

warnings.filterwarnings("ignore")

N_POINTS = 2000
D_INTRINSIC = 4
KNN = 10
TRAJ_LEN = 50
Z_DIM = 64
N_EPOCHS = 300
SEEDS = (7, 8, 9, 10)
N_ANCHORS = 30
GAMMA_DIST, GAMMA_MEAS = 0.98, 0.8


def corrected_kappas(est) -> dict[str, float]:
    """Raw kappa, far-pair self-calibrated kappa, and diagnostics.

    (The split-half floor correction was tested first and moved the plane the
    WRONG way — it measures independent-noise transport, which inflates W1,
    not the shared-mass contamination, which deflates it. Far-pair
    calibration targets the shared-mass channel: kappa on far pairs ~ the
    contamination fraction rho, and kappa* = 1 - (1-kappa)/(1-rho).)
    """
    return {
        "raw": float(np.nanmean(est.orc_)),
        "cal": float(np.nanmean(est.orc_calibrated_)),
        "far": float(np.nanmean(est.far_)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--nts", type=int, nargs="+", default=[50, 200])
    ap.add_argument("--out", default="processed_data/debias_floor_test.csv")
    args = ap.parse_args()

    rows = []
    for ds in ("plane", "sphere", "saddle"):
        X, ks = build_Xd(ds, D_INTRINSIC, seed=7)
        X32 = X.astype(np.float32)
        anchors = np.random.default_rng(7).choice(
            N_POINTS, N_ANCHORS, replace=False).tolist()
        G = pygsp.graphs.NNGraph(np.asarray(X, dtype=np.float64), k=KNN)

        for nt in args.nts:
            t0 = time.time()
            traj_idx = subsample_trajectories(
                G, n_trajectories=nt, length=TRAJ_LEN, rng=7)
            coverage = len(np.unique(traj_idx)) / N_POINTS
            traj = X32[traj_idx]
            half = nt // 2
            tA, tB = traj[:half], traj[half:]

            def train(gamma, data):
                e = EnsembleFBTrainer(obs_dim=X.shape[1], seeds=SEEDS,
                                      z_dim=Z_DIM, gamma=gamma,
                                      n_epochs=N_EPOCHS, device=args.device)
                e.fit(data)
                return np.maximum(e.raw_kernels(X32), 0.0)

            K_dist = train(GAMMA_DIST, traj)
            K_meas = train(GAMMA_MEAS, traj)
            K_A = train(GAMMA_MEAS, tA)
            K_B = train(GAMMA_MEAS, tB)

            per_variant: dict[str, list[float]] = {}
            for s in range(len(SEEDS)):
                Gk = knn_distance_graph(potential_distances(K_dist[s]))
                M = relu_rownorm(K_meas[s])
                MA, MB = relu_rownorm(K_A[s]), relu_rownorm(K_B[s])
                for transport in ("w1", "w2"):
                    est = WassersteinSignedCurvature(
                        n_pairs=8, seed=0, compute_midpoint=False,
                        transport=transport, far_quantiles=(0.5, 0.7))
                    est.fit(M=M, M2=(MA, MB), D_graph=Gk, idx=anchors)
                    for cname, v in corrected_kappas(est).items():
                        per_variant.setdefault(f"{transport}_{cname}", []).append(v)
                    per_variant.setdefault(f"{transport}_floor", []).append(
                        float(np.nanmean(est.floor_)))

            # unlearned control on the visited cloud (raw + far-calibrated)
            V = np.unique(traj_idx)
            est = WassersteinSignedCurvature(t="auto", knn=KNN, n_pairs=8,
                                             seed=0, compute_midpoint=False,
                                             far_quantiles=(0.5, 0.7))
            anchor_pos = [int(np.argmin(np.linalg.norm(
                X[V] - X[a], axis=1))) for a in anchors[:10]]
            est.fit(X=X[V], idx=anchor_pos)
            per_variant["control_raw"] = [float(np.nanmean(est.orc_))]
            per_variant["control_cal"] = [float(np.nanmean(est.orc_calibrated_))]

            for name, vals in per_variant.items():
                v = np.asarray(vals, dtype=float)
                v = v[np.isfinite(v)]
                rows.append(dict(
                    dataset=ds, n_traj=nt, ks_true=ks, variant=name,
                    coverage=round(coverage, 3),
                    orc=float(v.mean()) if v.size else np.nan,
                    orc_std=float(v.std()) if v.size else np.nan,
                    n=len(v)))
            done = {r["variant"]: r["orc"] for r in rows
                    if r["dataset"] == ds and r["n_traj"] == nt}
            print(f"{ds} nt={nt} cov={coverage:.2f} ({time.time()-t0:.0f}s): "
                  + " ".join(f"{k}={v:+.3f}" for k, v in done.items()
                             if not k.endswith("floor")), flush=True)

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    df = pd.DataFrame(rows)
    for nt in args.nts:
        print(f"\n=== n_traj={nt}: plane / sphere / saddle by variant ===")
        piv = df[df.n_traj == nt].pivot_table(
            index="variant", columns="dataset", values="orc")
        piv["sphere-plane"] = piv.sphere - piv.plane
        piv["plane-saddle"] = piv.plane - piv.saddle
        print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
