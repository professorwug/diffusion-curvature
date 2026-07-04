"""Full menagerie battery run: v6 channel set, field-level scoring.

Units = recipe indices, sharded CONTIGUOUSLY (recipes are profile-grouped,
so each worker builds few warped-product tables). Per eval point: kappa
(Diffusion ORC t=8, D-native), frac (t=16), ent_cak (t in {2,4}).

Summarize: (a) tier2 within-manifold field Pearson per (kind, dim, noise);
(b) tier1 cross-manifold Pearson + balanced sign at the torus-referenced
zero; per-channel and simple torus-z composite.

Usage:
  pixi run python menagerie_channels.py run --worker-id K --num-workers W --device cuda:X
  pixi run python menagerie_channels.py summarize
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent / "../11-successor-curvatures"))

from diffusion_curvature.wasserstein_signed import WassersteinSignedCurvature
from benchmark_kmetric_colosseum import (_SimpleG, affinity_from_D,
                                         knn_distance_graph)
from diffusing_fraction_colosseum import edge_fractions
from graph_ablation_ruler import cak_W
from build_battery import materialize

warnings.filterwarnings("ignore")

RECIPES = Path("processed_data/menagerie_recipes.joblib")
OUT_TPL = "processed_data/menagerie_ch_w{wid}.csv"
KNN = 10
T_FRAC = 16
K_EDGE = 16


def channels_at(D: np.ndarray, anchors: np.ndarray, device: str) -> dict:
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
    P = torch.as_tensor(W_cak / np.maximum(W_cak.sum(1, keepdims=True),
                                           1e-30),
                        dtype=torch.float32, device=device)
    ent = {}
    with torch.no_grad():
        rows = torch.zeros((len(anchors), n), device=device)
        for q, a in enumerate(anchors):
            rows[q, a] = 1.0
        for step in range(1, 5):
            rows = rows @ P
            if step in (2, 4):
                pr = rows.clamp_min(1e-12)
                ent[step] = (-(pr * pr.log()).sum(1)).cpu().numpy()
    return dict(kappa=kappa, frac=frac, ent2=ent[2], ent4=ent[4])


def run_worker(args) -> None:
    recipes = joblib.load(RECIPES)
    W = args.num_workers
    lo = (len(recipes) * args.worker_id) // W
    hi = (len(recipes) * (args.worker_id + 1)) // W
    out = Path(OUT_TPL.format(wid=args.worker_id))
    done = set()
    if out.exists() and out.stat().st_size > 0:
        done = set(pd.read_csv(out, usecols=["instance"]).instance)
    header = out.exists() and out.stat().st_size > 0
    print(f"[w{args.worker_id}] recipes {lo}..{hi}", flush=True)
    t0 = time.time()
    for i in range(lo, hi):
        if i in done:
            continue
        rec = recipes[i]
        err = ""
        rows = []
        try:
            inst = materialize(rec)
            ch = channels_at(inst["D"], inst["eval_idx"], args.device)
            for q, a in enumerate(inst["eval_idx"]):
                rows.append(dict(
                    instance=i, point=int(a), kind=rec["kind"],
                    dataset=rec["dataset"], dim=rec["dim"],
                    noise=rec["noise"], seed=rec["seed"], name=inst["name"],
                    ks_true=float(inst["ks_field"][a]),
                    kappa=float(ch["kappa"][q]), frac=float(ch["frac"][q]),
                    ent2=float(ch["ent2"][q]), ent4=float(ch["ent4"][q])))
        except Exception as e:
            err = str(e)[:200]
            rows = [dict(instance=i, point=-1, kind=rec["kind"],
                         dataset=rec["dataset"], dim=rec["dim"],
                         noise=rec["noise"], seed=rec["seed"], name="",
                         ks_true=np.nan, kappa=np.nan, frac=np.nan,
                         ent2=np.nan, ent4=np.nan)]
            print(f"  [err] {i}: {err}", flush=True)
        pd.DataFrame(rows).to_csv(out, mode="a", index=False,
                                  header=not header)
        header = True
    print(f"[w{args.worker_id}] done in {(time.time()-t0)/60:.1f} min",
          flush=True)


def run_summarize() -> None:
    from scipy.stats import pearsonr
    df = pd.concat([pd.read_csv(p) for p in
                    sorted(Path("processed_data").glob("menagerie_ch_w*.csv"))],
                   ignore_index=True).drop_duplicates(
        ["instance", "point"], keep="last")
    df.to_csv("processed_data/menagerie_channels.csv", index=False)
    print(f"rows: {len(df)}; instances: {df.instance.nunique()}")
    chans = [("kappa", +1), ("frac", -1), ("ent2", -1), ("ent4", -1)]

    print("\n=== TIER 2 within-manifold field Pearson (mean over instances) ===")
    t2 = df[df.kind.isin(["dumbbell", "necklace"])]
    for kind in ("dumbbell", "necklace"):
        g0 = t2[t2.kind == kind]
        for nz, gn in g0.groupby("noise"):
            row = []
            for d in (3, 4, 5, 6):
                gd = gn[gn.dim == d]
                vals = {}
                for ch, orient in chans:
                    rs = []
                    for _, gi in gd.groupby("instance"):
                        v = orient * pd.to_numeric(gi[ch], errors="coerce")
                        m = np.isfinite(v) & np.isfinite(gi.ks_true)
                        if m.sum() > 8 and v[m].std() > 0:
                            rs.append(pearsonr(v[m], gi.ks_true[m])[0])
                    vals[ch] = np.mean(rs) if rs else np.nan
                row.append(f"d{d}: " + "/".join(
                    f"{vals[c]:+.2f}" for c, _ in chans))
            print(f"{kind:<9} nz={nz:<5} " + "  ".join(row))

    print("\n=== TIER 1 cross-manifold (per-instance means; torus-zero balanced sign) ===")
    t1 = df[~df.kind.isin(["dumbbell", "necklace"])]
    inst = t1.groupby("instance").agg(
        dim=("dim", "first"), noise=("noise", "first"),
        kind=("kind", "first"), ks=("ks_true", "mean"),
        **{c: (c, "mean") for c, _ in chans}).reset_index()
    for nz, g in inst.groupby("noise"):
        row = []
        for d in (2, 3, 4, 5, 6):
            gd = g[g.dim == d]
            tor = gd[gd.kind == "torus"]
            parts = []
            for ch, orient in chans:
                v = orient * pd.to_numeric(gd[ch], errors="coerce")
                m = np.isfinite(v)
                r = (pearsonr(v[m], gd.ks[m])[0]
                     if m.sum() > 4 and v[m].std() > 0 else np.nan)
                zero = orient * tor[ch].mean() if len(tor) else np.nan
                pos = m & (gd.ks > 1e-9)
                neg = m & (gd.ks < -1e-9)
                bal = (0.5 * ((v[pos] > zero).mean() + (v[neg] < zero).mean())
                       if pos.any() and neg.any() and np.isfinite(zero)
                       else np.nan)
                parts.append(f"{r:+.2f}|{bal:.2f}")
            row.append(f"d{d}: " + " ".join(parts))
        print(f"nz={nz}: " + "   ".join(row))
    print("(channel order: kappa | frac | ent2 | ent4)")

    v2 = df[df.dataset == "tier1v2"]
    if len(v2):
        print("\n=== TIER 1 v2 (EQUAL DENSITY) cross-manifold: Pearson | torus-zero balanced sign ===")
        iv2 = v2.groupby("instance").agg(
            dim=("dim", "first"), noise=("noise", "first"),
            kind=("kind", "first"), ks=("ks_true", "mean"),
            **{c: (c, "mean") for c, _ in chans}).reset_index()
        for nz, g in iv2.groupby("noise"):
            row = []
            for d in (2, 3, 4, 5, 6):
                gd = g[g.dim == d]
                tor = gd[gd.kind == "torus"]
                parts = []
                for ch, orient in chans:
                    v = orient * pd.to_numeric(gd[ch], errors="coerce")
                    m = np.isfinite(v)
                    r = (pearsonr(v[m], gd.ks[m])[0]
                         if m.sum() > 4 and v[m].std() > 0 else np.nan)
                    zero = orient * tor[ch].mean() if len(tor) else np.nan
                    pos = m & (gd.ks > 1e-9)
                    neg = m & (gd.ks < -1e-9)
                    bal = (0.5 * ((v[pos] > zero).mean()
                                  + (v[neg] < zero).mean())
                           if pos.any() and neg.any() and np.isfinite(zero)
                           else np.nan)
                    parts.append(f"{r:+.2f}|{bal:.2f}")
                row.append(f"d{d}: " + " ".join(parts))
            print(f"nz={nz}: " + "   ".join(row))
        print("\n  v2 per-kind means (nz=0):")
        print(iv2[iv2.noise == 0].groupby(["kind", "dim"])
              [["ks", "kappa", "frac", "ent4"]].mean().round(2).to_string())

    print("\n=== TIER 1 per-kind channel means (nz=0, diagnosing the inversion) ===")
    g0 = inst[inst.noise == 0.0]
    print(g0.groupby(["kind", "dim"])[["ks", "kappa", "frac", "ent4"]]
          .mean().round(2).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("run")
    w.add_argument("--worker-id", type=int, default=0)
    w.add_argument("--num-workers", type=int, default=2)
    w.add_argument("--device", default="cuda:0")
    sub.add_parser("summarize")
    args = ap.parse_args()
    if args.cmd == "run":
        run_worker(args)
    else:
        run_summarize()


if __name__ == "__main__":
    main()
