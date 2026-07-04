"""RL-warp v3 — the FAITHFUL option-(b): geometry inferred from the optimal
policy's traces (exact successor limit), then measured with the full
standard channel stack.

Pipeline per instance:
  1. P_pi = Doob-tilted kNN chain toward positive-proxy goals (as v2)
  2. M = (1-gamma) (I - gamma P_pi)^{-1}   -- exact successor kernel
     (infinite-trace limit of Q-learning traces + successor machinery)
  3. D_warp = Dijkstra re-metrized -log potential distances of M_sym
     (the FB-validated recipe, applied to the policy's successor kernel)
  4. ALL channels (kappa, frac, ruler) computed from D_warp alone
Readout: field Pearson of channels-on-D_warp vs ks, compared to
channels-on-D (unwarped); amplification = does the warped METRIC separate
sign more strongly? Random-goals ablation as before.

Usage: pixi run python rl_warp_v3_successor.py [--device cuda:0]
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse.csgraph import dijkstra
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent / "../11-successor-curvatures"))

from benchmark_kmetric_colosseum import (_SimpleG, affinity_from_D,
                                         knn_distance_graph,
                                         potential_distances)
from diffusing_fraction_colosseum import edge_fractions
from diffusion_curvature.wasserstein_signed import WassersteinSignedCurvature
from graph_ablation_ruler import cak_W
from build_battery import materialize

warnings.filterwarnings("ignore")

KNN = 10
BETA = 6.0
GOAL_Q = 0.04
GAMMAS = (0.95, 0.99)
T_ENT = 4
T_FRAC = 16
K_EDGE = 16
N_EVAL = 48


def channels_on_D(D: np.ndarray, anchors, device: str) -> dict:
    n = D.shape[0]
    W = affinity_from_D(D, k=KNN)
    est = WassersteinSignedCurvature(t=8, n_pairs=8, seed=0,
                                     compute_midpoint=False)
    est.fit(G=_SimpleG(W), D_graph=knn_distance_graph(D, k=KNN),
            idx=list(anchors))
    kappa = np.asarray(est.orc_)
    P10 = torch.as_tensor(W / np.maximum(W.sum(1, keepdims=True), 1e-30),
                          dtype=torch.float32, device=device)
    order = np.argsort(D[anchors], axis=1)
    edges = [(int(a), int(j)) for q, a in enumerate(anchors)
             for j in order[q, 1:K_EDGE + 1]]
    fr = edge_fractions(P10, edges, (T_FRAC,))
    frac = fr[T_FRAC].reshape(len(anchors), K_EDGE).mean(axis=1)
    W_cak, _ = cak_W(D, max(n // 4, 20))
    Pc = torch.as_tensor(W_cak / np.maximum(W_cak.sum(1, keepdims=True),
                                            1e-30),
                         dtype=torch.float32, device=device)
    with torch.no_grad():
        rows = torch.zeros((len(anchors), n), device=device)
        for q, a in enumerate(anchors):
            rows[q, a] = 1.0
        for _ in range(T_ENT):
            rows = rows @ Pc
        pr = rows.clamp_min(1e-12)
        ent = (-(pr * pr.log()).sum(1)).cpu().numpy()
    return dict(kappa=kappa, frac=frac, ent=ent)


def warp_metric(P_np: np.ndarray, V: np.ndarray, gamma: float,
                device: str) -> np.ndarray:
    scale = max(np.median(V[V > 0]), 1e-9)
    tilt = np.exp(-BETA * (V[None, :] - V[:, None]) / scale)
    Pw = P_np * tilt
    Pw = Pw / np.maximum(Pw.sum(1, keepdims=True), 1e-30)
    n = Pw.shape[0]
    Pw_t = torch.as_tensor(Pw, dtype=torch.float64, device=device)
    M = (1 - gamma) * torch.linalg.inv(
        torch.eye(n, dtype=torch.float64, device=device) - gamma * Pw_t)
    M = np.maximum(M.cpu().numpy(), 0.0)
    M_sym = 0.5 * (M + M.T)
    return potential_distances(M_sym)


def run_one(recipe, device, goal_mode="proxy"):
    inst = materialize(recipe)
    D, ks = inst["D"], inst["ks_field"]
    n = len(ks)
    rng = np.random.default_rng(recipe["seed"] + 99)
    W = affinity_from_D(D, k=KNN)
    P_np = W / np.maximum(W.sum(1, keepdims=True), 1e-30)

    W_cak, _ = cak_W(D, max(n // 4, 20))
    Pc = torch.as_tensor(W_cak / np.maximum(W_cak.sum(1, keepdims=True),
                                            1e-30),
                         dtype=torch.float32, device=device)
    with torch.no_grad():
        Pt = torch.matrix_power(Pc, T_ENT)
        Hn = (-(Pt.clamp_min(1e-12) * Pt.clamp_min(1e-12).log())
              .sum(1)).cpu().numpy()
    n_goals = max(int(GOAL_Q * n), 8)
    goals = (np.argsort(Hn)[:n_goals] if goal_mode == "proxy"
             else rng.choice(n, n_goals, replace=False))
    goal_acc = float((ks[goals] > 0).mean())
    V = dijkstra(knn_distance_graph(D, k=KNN), directed=False,
                 indices=goals).min(axis=0)
    V = np.where(np.isfinite(V), V, np.nanmax(V[np.isfinite(V)]))

    order = np.argsort(ks)
    anchors = order[(np.linspace(0.02, 0.98, N_EVAL) * (n - 1)).astype(int)]
    base = channels_on_D(D, anchors, device)
    out = []
    for gamma in GAMMAS:
        Dw = warp_metric(P_np, V, gamma, device)
        warp = channels_on_D(Dw, anchors, device)
        for q, a in enumerate(anchors):
            out.append(dict(anchor=int(a), ks=float(ks[a]), gamma=gamma,
                            V=float(V[a]), goal_acc=goal_acc,
                            **{f"{c}0": float(base[c][q]) for c in base},
                            **{f"{c}w": float(warp[c][q]) for c in warp}))
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out",
                    default="processed_data/rl_warp_v3.csv")
    args = ap.parse_args()
    cells = ([dict(kind="dumbbell", dim=d, beta=0.8) for d in (3, 4)]
             + [dict(kind="necklace", dim=d, b=0.7) for d in (3, 4)])
    frames = []
    for base in cells:
        for seed in (0, 1, 2):
            for mode in ("proxy", "random"):
                rec = dict(**base, seed=seed, noise=0.0, dataset="x")
                t0 = time.time()
                try:
                    df = run_one(rec, args.device, goal_mode=mode)
                except Exception as e:
                    print(f"  [err] {base['kind']} d={base['dim']} "
                          f"s={seed} {mode}: {e}", flush=True)
                    continue
                df["kind"], df["dim"], df["seed"], df["mode"] = \
                    base["kind"], base["dim"], seed, mode
                frames.append(df)
                g99 = df[df.gamma == 0.99]
                msg = " ".join(
                    f"{c}: {pearsonr(o * g99[c + '0'], g99.ks)[0]:+.2f}->"
                    f"{pearsonr(o * g99[c + 'w'], g99.ks)[0]:+.2f}"
                    for c, o in [("kappa", 1), ("frac", -1), ("ent", -1)])
                print(f"{base['kind']:<9} d={base['dim']} s={seed} "
                      f"{mode:<6} (g=.99): {msg} ({time.time()-t0:.0f}s)",
                      flush=True)
    allf = pd.concat(frames, ignore_index=True)
    allf.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print("\n=== field Pearson: unwarped -> warped-metric (mean over seeds) ===")
    res = []
    for (kind, d, mode, gamma), g in allf.groupby(
            ["kind", "dim", "mode", "gamma"]):
        row = dict(kind=kind, dim=d, mode=mode, gamma=gamma,
                   goal_acc=round(g.goal_acc.mean(), 2))
        for c, o in [("kappa", 1), ("frac", -1), ("ent", -1)]:
            r0 = np.mean([pearsonr(o * gi[c + "0"], gi.ks)[0]
                          for _, gi in g.groupby("seed")])
            rw = np.mean([pearsonr(o * gi[c + "w"], gi.ks)[0]
                          for _, gi in g.groupby("seed")])
            row[c] = f"{r0:+.2f}->{rw:+.2f}"
        res.append(row)
    print(pd.DataFrame(res).to_string(index=False))


if __name__ == "__main__":
    main()
