"""Entropy-response Ricci flow: exaggerate curvature, read sign from drift.

Flow design (edges = union-kNN skeleton, k=10; what flows = edge LENGTHS):
  1. skeleton lengths d_e from euclidean distances; adaptive bandwidths
     sigma_i FROZEN from the initial graph (so length changes move the kernel)
  2. affinity w_e = exp(-d_e^2 / (sigma_a sigma_b)); P = rownorm
  3. per-edge signed driver: batched probability-space entropy response
     r_e = d[H_a + H_b]/d eps at t=8 (concentrating < 0, diffusing > 0)
  4. flow: d_e <- d_e * (1 + eta * clip(r_e / median|r|, -2, 2));
     renormalize total length (zero-sum: flat cannot drift globally)
  5. repeat K steps, recomputing responses on the flowed operator

Readouts per anchor (30 random), per step:
  len_drift : mean log(d_e(k)/d_e(0)) over the anchor's incident edges
              (direct Ricci readout: contraction = positive curvature)
  ent_drift : entropy of P^4 row at the anchor, tracked over steps

Sign predictions: sphere len_drift < 0 (edges contract), saddle > 0,
plane ~ 0; separations should GROW with flow steps if the flow exaggerates.

Usage: pixi run python ricci_flow_ladder.py [--device cuda:0]
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd
import torch

from fb_kernel_ablation_d4 import build_Xd

warnings.filterwarnings("ignore")

N_POINTS = 1500
KNN = 10
T_RESP = 8
T_ENT = 4
EPS_RESP = 0.05
ETA = 0.1
K_STEPS = 8
N_ANCHORS = 30
DIMS = (2, 3, 4, 6)
DATASETS = ("plane", "sphere", "saddle")
SEEDS = (1, 2, 3)


def build_skeleton(X: np.ndarray):
    """Union-kNN edge list with euclidean lengths + frozen bandwidths."""
    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(X)
    n = D.shape[0]
    idx = np.argsort(D, axis=1)[:, 1:KNN + 1]
    pairs = set()
    for i in range(n):
        for j in idx[i]:
            pairs.add((min(i, int(j)), max(i, int(j))))
    edges = np.array(sorted(pairs))
    lengths = D[edges[:, 0], edges[:, 1]].astype(np.float64)
    sigma = np.partition(D, KNN, axis=1)[:, KNN].astype(np.float64)
    return edges, lengths, sigma


def operator_from(edges, lengths, sigma, n, device):
    w = np.exp(-(lengths**2) / (sigma[edges[:, 0]] * sigma[edges[:, 1]]))
    W = np.zeros((n, n))
    W[edges[:, 0], edges[:, 1]] = w
    W[edges[:, 1], edges[:, 0]] = w
    P = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-30)
    return torch.as_tensor(P, dtype=torch.float32, device=device)


def batched_edge_response(P: torch.Tensor, edges: np.ndarray,
                          t: int = T_RESP, eps: float = EPS_RESP,
                          block: int = 4096) -> np.ndarray:
    """dH/deps for every edge (probability-space bump at both endpoints),
    vectorized: each edge contributes 2 propagated rows with its own rank-2
    correction, gathered/scattered per step."""
    n = P.shape[0]
    device = P.device
    out = np.empty(len(edges))
    with torch.no_grad():
        for s in range(0, len(edges), block):
            eb = edges[s:s + block]
            B = len(eb)
            a = torch.as_tensor(eb[:, 0], device=device)
            j = torch.as_tensor(eb[:, 1], device=device)
            # perturbed rows: for edge e, rows a_e and j_e
            src = torch.cat([a, j])                       # (2B,)
            ea = torch.cat([a, a])                        # per-row edge endpoint a
            ej = torch.cat([j, j])                        # per-row edge endpoint j
            rows0 = torch.zeros((2 * B, n), device=device)
            rows0[torch.arange(2 * B, device=device), src] = 1.0
            rows_p = rows0.clone()
            rows_u = rows0.clone()
            Pa = P[a]                                     # (B, n)
            Pj = P[j]
            Pa2 = torch.cat([Pa, Pa])                     # (2B, n)
            Pj2 = torch.cat([Pj, Pj])
            for _ in range(t):
                rows_u = rows_u @ P
                base = rows_p @ P
                ca = rows_p.gather(1, ea.unsqueeze(1)).squeeze(1)  # mass at a
                cj = rows_p.gather(1, ej.unsqueeze(1)).squeeze(1)
                corr = -eps * (ca.unsqueeze(1) * Pa2 + cj.unsqueeze(1) * Pj2)
                corr.scatter_add_(1, ej.unsqueeze(1).expand(-1, 1),
                                  (eps * ca).unsqueeze(1))
                corr.scatter_add_(1, ea.unsqueeze(1).expand(-1, 1),
                                  (eps * cj).unsqueeze(1))
                rows_p = base + corr
            pu = rows_u.clamp_min(1e-12)
            pp = rows_p.clamp_min(1e-12)
            Hu = -(pu * pu.log()).sum(dim=1)
            Hp = -(pp * pp.log()).sum(dim=1)
            dH = (Hp - Hu) / eps                          # (2B,)
            out[s:s + block] = (dH[:B] + dH[B:]).cpu().numpy()
    return out


def run_ladder(device: str, out_csv: str) -> None:
    rows = []
    for d in DIMS:
        for ds in DATASETS:
            for seed in SEEDS:
                np.random.seed(seed)
                X, ks = build_Xd(ds, d, seed=seed)
                X = np.asarray(X[:N_POINTS], dtype=np.float64)
                rng = np.random.default_rng(seed)
                anchors = rng.choice(len(X), N_ANCHORS, replace=False)
                t0 = time.time()
                edges, lengths, sigma = build_skeleton(X)
                len0 = lengths.copy()
                total0 = lengths.sum()
                inc = {a: np.where((edges[:, 0] == a)
                                   | (edges[:, 1] == a))[0] for a in anchors}
                node_inc = [[] for _ in range(len(X))]
                for ei, (u, v) in enumerate(edges):
                    node_inc[u].append(ei)
                    node_inc[v].append(ei)
                node_inc = [np.asarray(e) for e in node_inc]
                traj_len, traj_ent = [], []
                for k in range(K_STEPS + 1):
                    P = operator_from(edges, lengths, sigma, len(X), device)
                    with torch.no_grad():
                        rr = torch.zeros((len(anchors), len(X)), device=device)
                        for q, a in enumerate(anchors):
                            rr[q, a] = 1.0
                        for _ in range(T_ENT):
                            rr = rr @ P
                        pr = rr.clamp_min(1e-12)
                        H = (-(pr * pr.log()).sum(dim=1)).cpu().numpy()
                    ldrift = np.array([np.log(lengths[inc[a]]
                                              / len0[inc[a]]).mean()
                                       for a in anchors])
                    traj_ent.append(H)
                    traj_len.append(ldrift)
                    if k == K_STEPS:
                        break
                    r = batched_edge_response(P, edges)
                    # tail-targeted flow: per node, stretch its most-diffusing
                    # incident edge, shrink its most-concentrating; others
                    # untouched (per-node action, condensation-resistant)
                    factor = np.ones(len(edges))
                    n_nodes = len(X)
                    for node in range(n_nodes):
                        einc = node_inc[node]
                        if len(einc) < 2:
                            continue
                        rv = r[einc]
                        factor[einc[np.argmax(rv)]] *= (1.0 + ETA)
                        factor[einc[np.argmin(rv)]] *= (1.0 - ETA)
                    lengths = lengths * factor
                    lengths *= total0 / lengths.sum()
                ent_slope = np.polyfit(np.arange(K_STEPS + 1),
                                       np.stack(traj_ent), 1)[0]
                len_slope = np.polyfit(np.arange(K_STEPS + 1),
                                       np.stack(traj_len), 1)[0]
                rows.append(dict(
                    dim=d, dataset=ds, seed=seed, ks_true=ks,
                    ent_slope=float(np.mean(ent_slope)),
                    len_slope=float(np.mean(len_slope)),
                    ent_final_minus_init=float(np.mean(traj_ent[-1]
                                                       - traj_ent[0])),
                    secs=round(time.time() - t0, 1)))
                print(f"d={d} {ds:<7} s={seed}: len_slope={rows[-1]['len_slope']:+.5f} "
                      f"ent_slope={rows[-1]['ent_slope']:+.4f} "
                      f"({rows[-1]['secs']}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    for metric in ("len_slope", "ent_slope"):
        piv = df.pivot_table(index="dim", columns="dataset", values=metric)
        piv["sph-pl"] = piv.sphere - piv.plane
        piv["pl-sad"] = piv.plane - piv.saddle
        print(f"\n=== {metric} (mean over seeds) ===")
        print(piv.round(5).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="processed_data/ricci_flow_ladder.csv")
    args = ap.parse_args()
    run_ladder(args.device, args.out)


if __name__ == "__main__":
    main()
