"""RL-warp v2 on menagerie manifolds: does goal-directed warping amplify
curvature as read by our channels?

Warp: goals = top-q positive-proxy points (lowest cak entropy; placement
accuracy CHECKED against ks_field). V = multi-source geodesic time.
P_pi(i,j) prop. P(i,j) * exp(-beta (V(j)-V(i))/scale)  — Doob-style tilt,
no absorption, channels stay valid.

Readouts per eval point (stratified by ks):
  d_ent  = ent(P_pi) - ent(P)     (t=4 rows at the point)
  d_frac = frac(P_pi) - frac(P)   (16 edges, t=16)
Pre-registered (user's hypothesis): amplification = corr(-d_ent, ks) > 0
and corr(-d_frac, ks) > 0 (positive regions concentrate, negative diffuse),
AND true-goal warping must beat the random-goals ablation.

Usage: pixi run python rl_warp_menagerie.py [--device cuda:0]
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

from benchmark_kmetric_colosseum import affinity_from_D, knn_distance_graph
from diffusing_fraction_colosseum import edge_fractions
from graph_ablation_ruler import cak_W
from build_battery import materialize

warnings.filterwarnings("ignore")

KNN = 10
BETA = 6.0
GOAL_Q = 0.04
T_ENT = 4
T_FRAC = 16
K_EDGE = 16
N_EVAL = 48


def ent_and_frac(P: torch.Tensor, anchors, D, device):
    n = P.shape[0]
    with torch.no_grad():
        rows = torch.zeros((len(anchors), n), device=device)
        for q, a in enumerate(anchors):
            rows[q, a] = 1.0
        for _ in range(T_ENT):
            rows = rows @ P
        pr = rows.clamp_min(1e-12)
        ent = (-(pr * pr.log()).sum(1)).cpu().numpy()
    order = np.argsort(D[anchors], axis=1)
    edges = [(int(a), int(j)) for q, a in enumerate(anchors)
             for j in order[q, 1:K_EDGE + 1]]
    fr = edge_fractions(P, edges, (T_FRAC,))
    frac = fr[T_FRAC].reshape(len(anchors), K_EDGE).mean(axis=1)
    return ent, frac


def run_one(recipe, device, goal_mode="proxy"):
    inst = materialize(recipe)
    D, ks = inst["D"], inst["ks_field"]
    n = len(ks)
    rng = np.random.default_rng(recipe["seed"] + 99)

    W = affinity_from_D(D, k=KNN)
    P_np = W / np.maximum(W.sum(1, keepdims=True), 1e-30)
    P = torch.as_tensor(P_np, dtype=torch.float32, device=device)

    # positive-proxy goals (or random ablation)
    W_cak, _ = cak_W(D, max(n // 4, 20))
    Pc = torch.as_tensor(W_cak / np.maximum(W_cak.sum(1, keepdims=True),
                                            1e-30),
                         dtype=torch.float32, device=device)
    with torch.no_grad():
        Pt = torch.matrix_power(Pc, T_ENT)
        Hn = (-(Pt.clamp_min(1e-12) * Pt.clamp_min(1e-12).log())
              .sum(1)).cpu().numpy()
    n_goals = max(int(GOAL_Q * n), 8)
    if goal_mode == "proxy":
        goals = np.argsort(Hn)[:n_goals]
    else:
        goals = rng.choice(n, n_goals, replace=False)
    goal_acc = float((ks[goals] > 0).mean())

    Gk = knn_distance_graph(D, k=KNN)
    V = dijkstra(Gk, directed=False, indices=goals).min(axis=0)
    V = np.where(np.isfinite(V), V, np.nanmax(V[np.isfinite(V)]))
    scale = max(np.median(V[V > 0]), 1e-9)

    # Doob-style tilt on the kNN chain (no absorption)
    tilt = np.exp(-BETA * (V[None, :] - V[:, None]) / scale)
    Pw_np = P_np * tilt
    Pw_np = Pw_np / np.maximum(Pw_np.sum(1, keepdims=True), 1e-30)
    Pw = torch.as_tensor(Pw_np, dtype=torch.float32, device=device)

    # eval points stratified by ks
    order = np.argsort(ks)
    anchors = order[(np.linspace(0.02, 0.98, N_EVAL)
                     * (n - 1)).astype(int)]
    ent0, frac0 = ent_and_frac(P, anchors, D, device)
    entw, fracw = ent_and_frac(Pw, anchors, D, device)
    return pd.DataFrame(dict(
        anchor=anchors, ks=ks[anchors], V=V[anchors],
        ent0=ent0, entw=entw, frac0=frac0, fracw=fracw,
        d_ent=entw - ent0, d_frac=fracw - frac0)), goal_acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="processed_data/rl_warp_menagerie.csv")
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
                    df, acc = run_one(rec, args.device, goal_mode=mode)
                except Exception as e:
                    print(f"  [err] {base['kind']} d={base['dim']} s={seed} "
                          f"{mode}: {e}", flush=True)
                    continue
                df["kind"], df["dim"], df["seed"], df["mode"] = \
                    base["kind"], base["dim"], seed, mode
                df["goal_acc"] = acc
                frames.append(df)
                amp_e = pearsonr(-df.d_ent, df.ks)[0]
                amp_f = pearsonr(-df.d_frac, df.ks)[0]
                base_e = pearsonr(-df.ent0, df.ks)[0]
                warp_e = pearsonr(-df.entw, df.ks)[0]
                print(f"{base['kind']:<9} d={base['dim']} s={seed} "
                      f"{mode:<6}: goal_acc={acc:.2f} "
                      f"amp(-dEnt)={amp_e:+.2f} amp(-dFrac)={amp_f:+.2f} "
                      f"| field r ent {base_e:+.2f} -> {warp_e:+.2f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    allf = pd.concat(frames, ignore_index=True)
    allf.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print("\n=== amplification corr(-dCh, ks) mean over seeds ===")
    res = []
    for (kind, d, mode), g in allf.groupby(["kind", "dim", "mode"]):
        by_seed_e = [pearsonr(-gi.d_ent, gi.ks)[0]
                     for _, gi in g.groupby("seed")]
        by_seed_f = [pearsonr(-gi.d_frac, gi.ks)[0]
                     for _, gi in g.groupby("seed")]
        fe0 = [pearsonr(-gi.ent0, gi.ks)[0] for _, gi in g.groupby("seed")]
        few = [pearsonr(-gi.entw, gi.ks)[0] for _, gi in g.groupby("seed")]
        res.append(dict(kind=kind, dim=d, mode=mode,
                        amp_ent=np.mean(by_seed_e),
                        amp_frac=np.mean(by_seed_f),
                        field_ent_before=np.mean(fe0),
                        field_ent_after=np.mean(few),
                        goal_acc=g.goal_acc.mean()))
    print(pd.DataFrame(res).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
