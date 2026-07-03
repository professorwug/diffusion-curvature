"""Probe: per-edge sign distribution of the probability-space entropy
derivative ("diffusing edges"), sparse vs dense, with interior-anchor
boundary control. See zettel results 2026-07-03; produced the finding that
the SIGN of dH/deps is carried by the upper tail of the per-edge
distribution, not the mean.

Usage: pixi run python diffusing_edge_probe.py [--interior] [--device cuda:0]
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
from sklearn.metrics import pairwise_distances

from diffusion_curvature.datasets import plane, rejection_sample_from_saddle, sphere
from benchmark_kmetric_colosseum import affinity_from_D
from successor_ricci_ladder import anchor_edges, prob_edge_response

warnings.filterwarnings("ignore")


def build(ds, d, n, seed):
    np.random.seed(seed)
    if ds == "plane":
        return np.hstack([plane(n, dim=d), np.zeros((n, 1))])
    if ds == "sphere":
        X, _ = sphere(n, d=d, seed=seed)
        return np.asarray(X)
    X, _ = rejection_sample_from_saddle(n, d)
    return np.asarray(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interior", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--ns", type=int, nargs="+", default=[200, 300, 500, 1000])
    ap.add_argument("--ts", type=int, nargs="+", default=[4, 8, 16])
    args = ap.parse_args()

    print("fraction of POSITIVE-derivative (diffusing) kNN edges"
          + (" [interior anchors]" if args.interior else ""))
    for d in args.dims:
        for n in args.ns:
            for t in args.ts:
                vals = {}
                for ds in ("plane", "sphere", "saddle"):
                    fr = []
                    for seed in (1, 2, 3):
                        X = build(ds, d, n, seed)
                        Xc = np.asarray(X, dtype=np.float64)
                        pool = np.arange(n)
                        if args.interior:
                            r_c = np.linalg.norm(Xc - Xc.mean(0), axis=1)
                            pool = np.where(r_c <= np.quantile(r_c, 0.4))[0]
                        rng = np.random.default_rng(seed)
                        anchors = rng.choice(
                            pool, min(40, len(pool)), replace=False).tolist()
                        D = pairwise_distances(Xc)
                        W = affinity_from_D(D, k=10)
                        edges = [(a, j) for a, js in
                                 anchor_edges(D, anchors, 4).items() for j in js]
                        r = prob_edge_response(W, t, edges, args.device)
                        fr.append(float((r > 0).mean()))
                    vals[ds] = (np.mean(fr), np.std(fr))
                p, s, sa = (vals[k][0] for k in ("plane", "sphere", "saddle"))
                mono = "  <<< MONO" if (sa > p > s) else ""
                print(f"d={d} n={n:4d} t={t:2d}: "
                      f"plane={p:.2f}±{vals['plane'][1]:.2f} "
                      f"sphere={s:.2f}±{vals['sphere'][1]:.2f} "
                      f"saddle={sa:.2f}±{vals['saddle'][1]:.2f}{mono}")


if __name__ == "__main__":
    main()
