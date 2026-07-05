"""Resolvent successor entropy (H0, gamma=0.97) as a battery channel —
the sleeper from succ_ent_warp. Separate shards (sent_w*.csv), merged with
menagerie_channels by (instance, point) in its summarize.

Usage: run --worker-id K --num-workers W --device cuda:X | summarize
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent / "../11-successor-curvatures"))
from benchmark_kmetric_colosseum import affinity_from_D
from build_battery import materialize

warnings.filterwarnings("ignore")

RECIPES = Path("processed_data/menagerie_recipes.joblib")
OUT_TPL = "processed_data/sent_w{wid}.csv"
GAMMA = 0.97
KNN = 10


def sent_at(D, anchors, device):
    n = D.shape[0]
    W = affinity_from_D(D, k=KNN)
    P = W / np.maximum(W.sum(1, keepdims=True), 1e-30)
    Pt = torch.as_tensor(P, dtype=torch.float64, device=device)
    M = (1 - GAMMA) * torch.linalg.inv(
        torch.eye(n, dtype=torch.float64, device=device) - GAMMA * Pt)
    M = np.maximum(M.cpu().numpy(), 0.0)
    R = M[anchors]
    R = R / np.maximum(R.sum(1, keepdims=True), 1e-30)
    R = np.clip(R, 1e-15, 1)
    return -(R * np.log(R)).sum(1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("run")
    w.add_argument("--worker-id", type=int, default=0)
    w.add_argument("--num-workers", type=int, default=2)
    w.add_argument("--device", default="cuda:0")
    sub.add_parser("summarize")
    args = ap.parse_args()
    if args.cmd == "summarize":
        from scipy.stats import pearsonr
        df = pd.concat([pd.read_csv(p) for p in
                        sorted(Path("processed_data").glob("sent_w*.csv"))],
                       ignore_index=True).drop_duplicates(
            ["instance", "point"], keep="last")
        df.to_csv("processed_data/sent_channel.csv", index=False)
        t2 = df[df.kind.isin(["dumbbell", "necklace"])]
        print("=== sent (resolvent entropy) field Pearson, tier2, per (kind,dim,noise) ===")
        for (kind, nz), g in t2.groupby(["kind", "noise"]):
            row = []
            for d in (3, 4, 5, 6):
                rs = []
                for _, gi in g[g.dim == d].groupby("instance"):
                    v = -pd.to_numeric(gi.sent, errors="coerce")
                    m = np.isfinite(v) & np.isfinite(gi.ks_true)
                    if m.sum() > 8 and v[m].std() > 0:
                        rs.append(pearsonr(v[m], gi.ks_true[m])[0])
                row.append(f"d{d}: {np.mean(rs):+.2f}" if rs else f"d{d}: -")
            print(f"{kind:<9} nz={nz:<5} " + "  ".join(row))
        return
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
    for i in range(lo, hi):
        if i in done:
            continue
        rec = recipes[i]
        rows = []
        try:
            inst = materialize(rec)
            se = sent_at(inst["D"], inst["eval_idx"], args.device)
            for q, a in enumerate(inst["eval_idx"]):
                rows.append(dict(instance=i, point=int(a), kind=rec["kind"],
                                 dim=rec["dim"], noise=rec["noise"],
                                 seed=rec["seed"],
                                 ks_true=float(inst["ks_field"][a]),
                                 sent=float(se[q])))
        except Exception as e:
            print(f"  [err] {i}: {str(e)[:120]}", flush=True)
            continue
        pd.DataFrame(rows).to_csv(out, mode="a", index=False,
                                  header=not header)
        header = True
    print(f"[w{args.worker_id}] done", flush=True)


if __name__ == "__main__":
    main()
