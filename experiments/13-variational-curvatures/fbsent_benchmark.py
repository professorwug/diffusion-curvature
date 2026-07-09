"""FBSENT on the N-suite: the TD-trained FB successor entropy as a third
function-approximation contender.

Same 72 units as noise_benchmark (identical walks/noise/targets via
prepare_unit). Channel: EnsembleFBTrainer (4 vmapped seeds, Bellman-gap TD
loss, gamma = gamma_c) -> successor rows min(F1 B^T, F2 B^T) over an 8192-
state corpus -> softmax with spread-rule auto-tau (exp-11 convention) ->
row entropy, pooled over the same 32-state groups, ensembled at the
ESTIMATE level. Orientation: -H.

Question answered vs noise_benchmark's r_td: which function approximation is
better under noise — FB (Bellman-gap loss + post-hoc softmax readout) or
TD-InfoNCE (loss matched to the density-ratio readout)?

Army pattern: run --worker-id K --num-workers W --device cuda:X | summarize
(merges with noise_bench.csv).
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from torch.func import functional_call, vmap

from diffusion_curvature.successor.ensemble import EnsembleFBTrainer
from noise_benchmark import prepare_unit, units

warnings.filterwarnings("ignore")

FB_SEEDS = (0, 1, 2, 3)
Z_DIM = 64
N_EPOCHS = 300
CORPUS = 8192
SPREAD_FRACTION = 0.3
OUT_TPL = "processed_data/fbsent_w{wid}.csv"


def fb_rows(est, X_anchor, X_corpus):
    """(S, A, C) symmetric-free raw scores min(F1 B^T, F2 B^T) per seed."""
    o = torch.as_tensor(X_anchor, dtype=torch.float32, device=est.device)
    c = torch.as_tensor(X_corpus, dtype=torch.float32, device=est.device)

    def _one(pF, pB):
        F1, F2 = functional_call(est._baseF, (pF, est.bF), (o,))
        Bc = functional_call(est._baseB, (pB, est.bB), (c,))
        return torch.minimum(F1 @ Bc.T, F2 @ Bc.T)

    with torch.no_grad():
        return vmap(_one)(est.pF, est.pB).cpu().numpy()


def pick_tau_spread(K, D, rng):
    """Bisect tau so median measure spread ~ SPREAD_FRACTION * median dist
    (exp-11 se_tau_auto rule); K, D: (probes, C)."""
    target = SPREAD_FRACTION * float(np.median(D))
    scale = max(float(np.std(K)), 1e-12)
    lo, hi = 1e-4 * scale, 1e4 * scale

    def spread(tau):
        x = K / tau
        x = x - x.max(1, keepdims=True)
        P = np.exp(x)
        P /= P.sum(1, keepdims=True)
        return float(np.median((P * D).sum(1)))

    for _ in range(30):
        mid = np.sqrt(lo * hi)
        if spread(mid) > target:
            hi = mid
        else:
            lo = mid
    return float(np.sqrt(lo * hi))


def run_unit(unit, caches, device):
    prep = prepare_unit(unit, caches)
    X, kt_w = prep["X"], prep["kt_w"]
    groups, gamma_c = prep["groups"], prep["gamma_c"]
    n_pts = prep["n_pts"]
    traj_obs = X[prep["traj_idx"]]                  # (NT, T+1, D)
    t0 = time.time()
    est = EnsembleFBTrainer(obs_dim=X.shape[1], seeds=FB_SEEDS, z_dim=Z_DIM,
                            hidden_dim=256, gamma=gamma_c,
                            n_epochs=N_EPOCHS, batch_size=1024,
                            device=device).fit(traj_obs)
    rng = np.random.default_rng(99 + unit["wseed"])
    corpus_idx = rng.choice(n_pts, CORPUS, replace=False)
    anchors = np.unique(np.concatenate(groups))
    loc = {a: i for i, a in enumerate(anchors)}
    K_all = fb_rows(est, X[anchors], X[corpus_idx])     # (S, A, C)
    D_ac = np.linalg.norm(X[anchors][:, None] - X[corpus_idx][None],
                          axis=-1)
    probe = rng.choice(len(anchors), min(20, len(anchors)), replace=False)
    fields = []
    taus = []
    for s_i in range(len(FB_SEEDS)):
        K = K_all[s_i]
        tau = pick_tau_spread(K[probe], D_ac[probe], rng)
        taus.append(tau)
        x = K / tau
        x = x - x.max(1, keepdims=True)
        P = np.exp(x)
        P /= P.sum(1, keepdims=True)
        H = -(np.clip(P, 1e-30, 1) * np.log(np.clip(P, 1e-30, 1))).sum(1)
        fields.append(-H)
    field = np.mean(fields, axis=0)                    # estimate-level mean
    grp_field = np.array([field[[loc[m] for m in g]].mean()
                          for g in groups])
    mm = np.isfinite(grp_field) & np.isfinite(kt_w)
    row = dict(unit)
    row["gamma_c"] = round(gamma_c, 4)
    row["r_fbsent"] = (pearsonr(grp_field[mm], kt_w[mm])[0]
                       if mm.sum() > 8 else np.nan)
    row["tau_med"] = float(np.median(taus))
    row["t_fb"] = round(time.time() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("run")
    w.add_argument("--worker-id", type=int, default=0)
    w.add_argument("--num-workers", type=int, default=1)
    w.add_argument("--device", default="cuda:0")
    w.add_argument("--test-run", action="store_true")
    sub.add_parser("summarize")
    args = ap.parse_args()
    if args.cmd == "summarize":
        fb = pd.concat([pd.read_csv(p) for p in
                        sorted(Path("processed_data").glob(
                            "fbsent_w*.csv"))], ignore_index=True)
        fb = fb.drop_duplicates(["profile", "d", "wseed", "noise"],
                                keep="last")
        fb.to_csv("processed_data/fbsent.csv", index=False)
        nb = pd.read_csv("processed_data/noise_bench.csv")
        df = nb.merge(fb[["profile", "d", "wseed", "noise", "r_fbsent"]],
                      on=["profile", "d", "wseed", "noise"], how="left")
        df.to_csv("processed_data/noise_bench_full.csv", index=False)
        print(f"{fb.shape[0]}/72 fbsent units")
        pd.set_option("display.width", 200)
        print(df.groupby(["noise", "d"])[
            ["r_sent_bin", "r_sent_knn", "r_fbsent", "r_td"]]
            .mean().round(3))
        return
    us = units()
    if args.test_run:
        us = [u for u in us if u["profile"] == "nk2" and u["d"] == 3
              and u["wseed"] == 0 and u["noise"] in ("clean", "iso15")]
    W = args.num_workers
    mine = [(i, u) for i, u in enumerate(us) if i % W == args.worker_id]
    out = Path(OUT_TPL.format(wid=args.worker_id))
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists() and out.stat().st_size > 0:
        dfd = pd.read_csv(out)
        done = set(zip(dfd.profile, dfd.d, dfd.wseed, dfd.noise))
    header = out.exists() and out.stat().st_size > 0
    caches = {}
    print(f"[w{args.worker_id}] {len(mine)} units", flush=True)
    for i, u in mine:
        if (u["profile"], u["d"], u["wseed"], u["noise"]) in done:
            continue
        try:
            row = run_unit(u, caches, args.device)
        except Exception as e:
            print(f"  [err] unit {i}: {str(e)[:150]}", flush=True)
            continue
        pd.DataFrame([row]).to_csv(out, mode="a", index=False,
                                   header=not header)
        header = True
        print("  " + " ".join(f"{k}={v:.3f}" if isinstance(v, float)
                              else f"{k}={v}" for k, v in row.items()),
              flush=True)
    print(f"[w{args.worker_id}] done", flush=True)


if __name__ == "__main__":
    main()
