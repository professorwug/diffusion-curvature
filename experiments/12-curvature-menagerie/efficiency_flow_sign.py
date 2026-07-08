"""Sign-recovery analysis of the efficiency flow (user's target).

Per-point dH saved; two flow variants (asymmetric clip as before, symmetric
clip); sign accuracy at three zeros: dH=0 (native), dH=flat-null drift
(one-number universal calibration), dH=per-manifold median (relative).
Balanced sign accuracy per manifold; flat-null drift reported.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "../11-successor-curvatures"))
from efficiency_flow import (build_manifold, skeleton, edge_traffic,
                             geodesics, sent_H, ETA, K_STEPS)

warnings.filterwarnings("ignore")

def run_flow(D, ks, seed, clip_lo, clip_hi):
    n = D.shape[0]
    rng = np.random.default_rng(seed)
    edges, lengths = skeleton(D)
    total0 = lengths.sum()
    order = np.argsort(ks)
    anchors = order[(np.linspace(0.02, 0.98, 40) * (n - 1)).astype(int)]
    H0 = sent_H(geodesics(edges, lengths, n), anchors)
    for k in range(K_STEPS):
        tr = edge_traffic(edges, lengths, n, rng)
        t_hat = np.clip(tr / max(tr.mean(), 1e-12) - 1.0, clip_lo, clip_hi)
        lengths = lengths * (1.0 - ETA * t_hat)
        lengths = np.maximum(lengths, 1e-6)
        lengths *= total0 / lengths.sum()
    Hk = sent_H(geodesics(edges, lengths, n), anchors)
    return anchors, Hk - H0

frames = []
for variant, (lo, hi) in [("asym", (-0.8, 3.0)), ("sym", (-1.0, 1.0))]:
    for mani in ("dumbbell3", "dumbbell4", "necklace3", "torus_rev",
                 "flat_null"):
        for seed in (0, 1):
            D, ks = build_manifold(mani, seed)
            anchors, dH = run_flow(D, ks, seed, lo, hi)
            frames.append(pd.DataFrame(dict(
                variant=variant, mani=mani, seed=seed,
                ks=ks[anchors], dH=dH)))
            print(f"{variant} {mani:<10} s={seed}: dH range "
                  f"[{dH.min():+.3f}, {dH.max():+.3f}] med={np.median(dH):+.3f}",
                  flush=True)
df = pd.concat(frames, ignore_index=True)
df.to_csv("processed_data/efficiency_flow_sign.csv", index=False)

print("\n=== balanced sign accuracy per zero ===")
for variant, gv in df.groupby("variant"):
    flat_drift = gv[gv.mani == "flat_null"].dH.median()
    print(f"\n--- variant={variant} (flat-null drift = {flat_drift:+.3f}) ---")
    for mani, g in gv.groupby("mani"):
        if mani == "flat_null" or g.ks.std() == 0:
            continue
        for zname, z in [("native0", 0.0), ("flatnull", flat_drift),
                         ("median", g.dH.median())]:
            pos, neg = g[g.ks > 0], g[g.ks < 0]
            if len(pos) and len(neg):
                bal = 0.5 * ((pos.dH > z).mean() + (neg.dH < z).mean())
                print(f"  {mani:<10} zero={zname:<8} balsign={bal:.2f}")
