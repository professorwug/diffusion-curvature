"""Local flattening flow — a gradient-flow curvature estimator.

Concept (user's design). The V3 second-moment *defect*

    delta(i,j) = mu_i^T D2 mu_j - 1/2 mu_i^T D2 mu_i
               - 1/2 mu_j^T D2 mu_j - dg(i,j)^2

is exactly 0 in flat space, so its zero-set IS flatness. Instead of reading
delta statically, we build, around each evaluation point p, a LOCAL PATCH
(geodesic ball) and run a few steps of gradient flow on the patch's edge
lengths minimizing

    F = sum_{j in band} delta(p, j)^2            (center-anchored)
        or  sum_{(i,j) in band} delta(i,j)^2      (all-pairs)

Open patches CAN flatten (no Gauss-Bonnet obstruction). We read EARLY RATES:

  (a) deformation rate: mean d log(L)/d step over edges incident to p.
      Pre-registered sign: stretching (+) = +K, shrinking (-) = -K
      (flattening a spherical cap requires stretching; a hyperbolic
      crumple shrinking).
  (b) channel rate: d(sent)/d step at p, sent = resolvent successor entropy
      (gamma=0.97) on the flowed patch geometry. Pre-registered: +K stretches
      the patch -> entropy at p RISES.

Differentiable geometry: the shortest-path STRUCTURE (predecessor trees) is
frozen at step 0; geodesic distances = sums of current edge lengths along the
frozen paths (linear map B @ L, autograd-friendly). Gauge: mean patch edge
length renormalized each step, so the readout measures p's edges *relative* to
the patch (kills the trivial global-scale mode).

Subcommands:
  python local_flatten_flow.py small          # hyperparameter grid search
  python local_flatten_flow.py rederivation   # mu-fixed vs recompute mini-test
  python local_flatten_flow.py full           # headline run on best config
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import dijkstra

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
EXP11 = REPO / "experiments" / "11-successor-curvatures"
sys.path.insert(0, str(EXP11))

from graph_ablation_ruler import cak_W  # noqa: E402
from benchmark_kmetric_colosseum import affinity_from_D  # noqa: E402

# reuse the validated static machinery
from v3_defect import (  # noqa: E402
    build_manifold, measure_spread, pearson, balanced_sign, pick_eval_points,
)

PROC = HERE / "processed_data"
PROC.mkdir(exist_ok=True)

GAMMA_SENT = 0.97
KNN_SENT = 10


# ---------------------------------------------------------------------------
# Patch construction
# ---------------------------------------------------------------------------

def build_patch(D: np.ndarray, p: int, radius: float) -> np.ndarray:
    """Local indices of points within geodesic `radius` of p (p included)."""
    idx = np.where(D[p] <= radius)[0]
    if p not in idx:
        idx = np.concatenate([idx, [p]])
    return np.sort(idx)


def patch_graph(Dp: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Symmetric kNN edge list on a patch distance submatrix.

    Returns (edges (E,2) local indices u<v, L0 (E,), edge_id (m,m) int with -1
    for non-edges). Guaranteed connected by union with an MST fallback."""
    m = Dp.shape[0]
    kk = min(k, m - 1)
    nn = np.argsort(Dp, axis=1)[:, 1:kk + 1]
    pairs = set()
    for i in range(m):
        for j in nn[i]:
            pairs.add((min(i, int(j)), max(i, int(j))))
    # ensure connectivity via an MST over the full patch metric
    mst = sp.csgraph.minimum_spanning_tree(sp.csr_matrix(Dp)).tocoo()
    for u, v in zip(mst.row, mst.col):
        pairs.add((min(int(u), int(v)), max(int(u), int(v))))
    edges = np.array(sorted(pairs), dtype=int)
    L0 = Dp[edges[:, 0], edges[:, 1]].astype(float)
    edge_id = -np.ones((m, m), dtype=np.int64)
    edge_id[edges[:, 0], edges[:, 1]] = np.arange(len(edges))
    edge_id[edges[:, 1], edges[:, 0]] = np.arange(len(edges))
    return edges, L0, edge_id


def path_incidence(edges: np.ndarray, L0: np.ndarray, edge_id: np.ndarray,
                   m: int) -> sp.csr_matrix:
    """Frozen shortest-path incidence B (m*m, E): row s*m+t lists the edges on
    the initial shortest path s->t. D_geo_flat = B @ L is then linear in L."""
    E = len(edges)
    G = sp.csr_matrix((np.concatenate([L0, L0]),
                       (np.concatenate([edges[:, 0], edges[:, 1]]),
                        np.concatenate([edges[:, 1], edges[:, 0]]))),
                      shape=(m, m))
    _, pred = dijkstra(G, directed=False, return_predecessors=True)
    rows, cols = [], []
    targets = np.arange(m)
    for s in range(m):
        node = targets.copy()
        active = node != s
        # guard against any disconnected node (shouldn't happen after MST)
        active &= pred[s] != -9999
        steps = 0
        while active.any() and steps < m:
            p_ = pred[s, node]
            good = active & (p_ >= 0)
            e = edge_id[p_[good], node[good]]
            rows.append(s * m + targets[good])
            cols.append(e)
            node = np.where(good, p_, node)
            active = active & good & (node != s)
            steps += 1
    if rows:
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
    else:
        rows, cols = np.array([], int), np.array([], int)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)),
                         shape=(m * m, E))


# ---------------------------------------------------------------------------
# Diffusion measures + channel
# ---------------------------------------------------------------------------

def patch_P(Dp: np.ndarray, kernel: str) -> np.ndarray:
    m = Dp.shape[0]
    if kernel == "cak":
        W, _ = cak_W(Dp, target=max(6, m // 4))
    else:
        W = affinity_from_D(Dp, k=min(10, m - 1))
    return W / np.maximum(W.sum(1, keepdims=True), 1e-30)


def diffusion_power(P: np.ndarray, t: int) -> np.ndarray:
    M = P.copy()
    for _ in range(int(np.log2(t))):
        M = M @ M
    return M


def sent_point(Dp: np.ndarray, p_local: int) -> float:
    """Resolvent successor entropy (gamma=0.97) at one patch point."""
    m = Dp.shape[0]
    W = affinity_from_D(Dp, k=min(KNN_SENT, m - 1))
    P = W / np.maximum(W.sum(1, keepdims=True), 1e-30)
    Mres = (1 - GAMMA_SENT) * np.linalg.inv(
        np.eye(m) - GAMMA_SENT * P)
    r = np.maximum(Mres[p_local], 0.0)
    r = r / max(r.sum(), 1e-30)
    r = np.clip(r, 1e-15, 1)
    return float(-(r * np.log(r)).sum())


# ---------------------------------------------------------------------------
# The flow (per eval point)
# ---------------------------------------------------------------------------

def flow_point(D: np.ndarray, p: int, ks_p: float, cfg: dict,
               device: str = "cpu") -> dict | None:
    """Run the local flattening flow around p and return readouts.

    cfg keys: kernel, t, band, radius_q (quantile of D[p]), n_pairs, boundary
    ('free'|'clamped'), objective ('center'|'allpairs'), recompute_mu (bool),
    n_steps, target_rel."""
    n = D.shape[0]
    radius = float(np.quantile(D[p], cfg["radius_q"]))
    patch = build_patch(D, p, radius)
    m = len(patch)
    if m < 12:
        return None
    lp = int(np.where(patch == p)[0][0])
    Dp = D[np.ix_(patch, patch)].astype(float)

    edges, L0, edge_id = patch_graph(Dp, k=cfg.get("k", 8))
    B = path_incidence(edges, L0, edge_id, m)
    Bt = torch.sparse_coo_tensor(
        np.vstack(B.nonzero()), torch.ones(B.nnz, dtype=torch.float64),
        size=B.shape, device=device).coalesce()

    # step-0 measures and band
    P0 = patch_P(Dp, cfg["kernel"])
    M0 = diffusion_power(P0, cfg["t"])
    spread = measure_spread(M0, Dp)
    target = cfg["band"] * spread
    diff = np.abs(Dp[lp] - target)
    diff[lp] = np.inf
    npair = min(cfg["n_pairs"], m - 1)
    band_j = np.argpartition(diff, npair)[:npair]
    # patch adequacy: is the band inside the ball?
    band_ok = float(np.mean(Dp[lp, band_j] <= radius * 1.001))

    mu = torch.as_tensor(M0, dtype=torch.float64, device=device)  # (m,m)

    # boundary clamp mask over edges
    if cfg["boundary"] == "clamped":
        shell = Dp[lp] > 0.8 * radius
        frozen = shell[edges[:, 0]] | shell[edges[:, 1]]
    else:
        frozen = np.zeros(len(edges), dtype=bool)
    free_mask = torch.as_tensor(~frozen, dtype=torch.float64, device=device)

    theta0 = torch.log(torch.as_tensor(L0, dtype=torch.float64, device=device))
    theta = theta0.clone().requires_grad_(True)

    def geodesic2(theta_):
        L = torch.exp(theta_)
        dflat = torch.sparse.mm(Bt, L[:, None]).squeeze(1)
        Dg = dflat.reshape(m, m)
        Dg = 0.5 * (Dg + Dg.t())
        return Dg * Dg  # D2

    def objective(D2, mu_cur):
        if cfg["objective"] == "center":
            sel = np.concatenate([[lp], band_j])
            Ms = mu_cur[sel]                      # (1+np, m)
            AA = Ms @ D2 @ Ms.t()
            d2pj = D2[lp, band_j]
            dd = AA[0, 1:] - 0.5 * AA[0, 0] - 0.5 * torch.diagonal(AA)[1:] - d2pj
            return (dd * dd).sum()
        else:  # allpairs: every patch point i, its band neighbours j
            A = mu_cur @ D2 @ mu_cur.t()
            a = torch.diagonal(A)
            delta = A - 0.5 * a[:, None] - 0.5 * a[None, :] - D2
            # mask to band annulus (relative to each i)
            band_lo, band_hi = 0.6 * target, 1.6 * target
            mask = torch.as_tensor(
                ((Dp >= band_lo) & (Dp <= band_hi)).astype(float),
                dtype=torch.float64, device=device)
            return (delta * delta * mask).sum() / max(mask.sum().item(), 1.0)

    # calibrate eta so the max first-step |d theta| ~ target_rel
    D2 = geodesic2(theta)
    if cfg["recompute_mu"]:
        mu_cur = _recompute_mu(D2, cfg, device)
    else:
        mu_cur = mu
    F0 = objective(D2, mu_cur)
    (g0,) = torch.autograd.grad(F0, theta)
    g0 = g0 * free_mask
    gmax = float(g0.abs().max())
    if gmax < 1e-30:
        return dict(kind_ok=band_ok, m=m, rate_deform=0.0, rate_channel=0.0,
                    static_score=0.0, ks=ks_p, delocal_r=float("nan"))
    eta = cfg["target_rel"] / gmax

    # record step-0 channel
    sent0 = sent_point(_D2_to_D(D2), lp)
    static_score = _static_score(D2, mu, lp, band_j)  # mean(-delta) raw

    theta_hist = [theta.detach().clone()]
    for _ in range(cfg["n_steps"]):
        D2 = geodesic2(theta)
        mu_cur = _recompute_mu(D2, cfg, device) if cfg["recompute_mu"] else mu
        F = objective(D2, mu_cur)
        (g,) = torch.autograd.grad(F, theta)
        with torch.no_grad():
            theta = theta - eta * g * free_mask
            # gauge: renormalize mean edge length to L0 mean
            logscale = torch.log(torch.exp(theta).mean()
                                 / torch.exp(theta0).mean())
            theta = theta - logscale
        theta = theta.requires_grad_(True)
        theta_hist.append(theta.detach().clone())

    # readouts
    inc = (edges[:, 0] == lp) | (edges[:, 1] == lp)
    inc_t = torch.as_tensor(inc, dtype=torch.bool, device=device)
    dtheta = (theta_hist[-1] - theta_hist[0]) / cfg["n_steps"]
    rate_deform = float(dtheta[inc_t].mean()) if inc.any() else 0.0

    D2_final = geodesic2(theta).detach()
    sentF = sent_point(_D2_to_D(D2_final), lp)
    rate_channel = (sentF - sent0) / cfg["n_steps"]

    # delocalization diagnostic: per-edge deformation vs analytic ks proxy.
    # (we lack per-edge ks; use correlation of |dtheta| with edge radius from p
    #  as a coarse localization proxy — reported, not scored)
    edge_rad = 0.5 * (Dp[lp, edges[:, 0]] + Dp[lp, edges[:, 1]])
    delocal_r = pearson(np.abs(dtheta.cpu().numpy()), edge_rad)

    return dict(kind_ok=band_ok, m=m, rate_deform=rate_deform,
                rate_channel=rate_channel, static_score=static_score,
                ks=ks_p, delocal_r=delocal_r)


def _recompute_mu(D2, cfg, device):
    """Differentiable measure recompute from flowed D2 (adaptive gaussian,
    sigma treated as constant from initial neighbours)."""
    m = D2.shape[0]
    Dg = torch.sqrt(torch.clamp(D2, min=0.0))
    k = min(10, m - 1)
    sig = torch.kthvalue(Dg, k + 1, dim=1).values.detach().clamp_min(1e-9)
    W = torch.exp(-D2 / (2 * sig[None, :] ** 2)) / sig[None, :]
    W = 0.5 * (W + W.t())
    P = W / W.sum(1, keepdim=True).clamp_min(1e-30)
    M = P
    for _ in range(int(np.log2(cfg["t"]))):
        M = M @ M
    return M


def _D2_to_D(D2):
    return torch.sqrt(torch.clamp(D2, min=0.0)).detach().cpu().numpy()


def _static_score(D2, mu, lp, band_j):
    """V3 raw score at p: mean over band of -delta(p,j), measures fixed."""
    sel = np.concatenate([[lp], band_j])
    Ms = mu[sel]
    AA = (Ms @ D2.detach() @ Ms.t())
    d2pj = D2.detach()[lp, band_j]
    dd = AA[0, 1:] - 0.5 * AA[0, 0] - 0.5 * torch.diagonal(AA)[1:] - d2pj
    return float((-dd).mean())


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def _run_manifold(kind, d, n, seed, cfg, n_eval, device):
    man = build_manifold(kind, d, n, seed)
    D, ks = man["D"], man["ks"]
    rng = np.random.default_rng(7000 + seed)
    eval_idx = pick_eval_points(ks, n_eval, rng)
    rows = []
    for i in eval_idx:
        out = flow_point(D, int(i), float(ks[i]), cfg, device)
        if out is None:
            continue
        out.update(kind=kind, dim=d, seed=seed, point=int(i))
        rows.append(out)
    return rows


def _summ(df: pd.DataFrame, group_extra: list[str]) -> pd.DataFrame:
    out = []
    for keys, g in df.groupby(group_extra + ["kind", "dim"]):
        rec = dict(zip(group_extra + ["kind", "dim"], keys if isinstance(keys, tuple) else (keys,)))
        for readout in ("rate_deform", "rate_channel"):
            if g["kind"].iloc[0] == "torus":
                rec[f"{readout}_flatmed"] = float(g[readout].abs().median())
            else:
                rec[f"{readout}_r"] = pearson(g[readout], g.ks)
                rec[f"{readout}_bal"] = balanced_sign(g[readout].values, g.ks.values)
        rec["static_r"] = pearson(g.static_score, g.ks)
        # re-derivation: deformation rate vs static defect score
        rec["deform_vs_static"] = pearson(g.rate_deform, g.static_score)
        rec["band_ok"] = float(g.kind_ok.mean())
        rec["m_med"] = float(g.m.median())
        rec["delocal_r"] = float(g.delocal_r.mean())
        out.append(rec)
    return pd.DataFrame(out)


def run_small(device: str) -> None:
    n, n_eval, seed = 600, 20, 0
    manifolds = [("dumbbell", 3), ("necklace", 3), ("torus", 3)]
    base = dict(kernel="cak", t=2, band=2.0, n_pairs=16, k=8,
                recompute_mu=False, n_steps=3, target_rel=0.02)
    t0 = time.time()
    rows = []
    for rq in (0.05, 0.10, 0.20):
        for boundary in ("free", "clamped"):
            for objective in ("center", "allpairs"):
                cfg = dict(base, radius_q=rq, boundary=boundary,
                           objective=objective)
                cname = f"rq{rq}|{boundary}|{objective}"
                for kind, d in manifolds:
                    for r in _run_manifold(kind, d, n, seed, cfg, n_eval, device):
                        r["config"] = cname
                        rows.append(r)
                print(f"[small] {cname} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(PROC / "local_flatten_small_points.csv", index=False)
    summ = _summ(df, ["config"])
    summ.to_csv(PROC / "local_flatten_small_summary.csv", index=False)
    with pd.option_context("display.width", 240, "display.max_columns", None):
        print(summ.sort_values(["config", "kind"]).to_string(index=False))


def run_rederivation(device: str) -> None:
    """mu-fixed vs mu-recompute on the strongest small-scale cell."""
    n, n_eval, seed = 600, 20, 0
    manifolds = [("dumbbell", 3), ("necklace", 3), ("torus", 3)]
    base = dict(kernel="cak", t=2, band=2.0, n_pairs=16, k=8, radius_q=0.10,
                boundary="free", objective="allpairs", n_steps=3, target_rel=0.02)
    rows = []
    for recompute in (False, True):
        cfg = dict(base, recompute_mu=recompute)
        cname = "recompute" if recompute else "fixed"
        for kind, d in manifolds:
            for r in _run_manifold(kind, d, n, seed, cfg, n_eval, device):
                r["config"] = cname
                rows.append(r)
        print(f"[rederiv] {cname} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(PROC / "local_flatten_rederiv_points.csv", index=False)
    summ = _summ(df, ["config"])
    summ.to_csv(PROC / "local_flatten_rederiv_summary.csv", index=False)
    with pd.option_context("display.width", 240, "display.max_columns", None):
        print(summ.sort_values(["config", "kind"]).to_string(index=False))


def run_full(device: str) -> None:
    n, n_eval = 1200, 40
    seeds = [0, 1]
    dims = [3, 4, 5]
    manifolds = [("dumbbell", d) for d in dims] + \
                [("necklace", d) for d in dims + [6]] + \
                [("torus", d) for d in dims]
    base = dict(kernel="cak", t=2, band=2.0, n_pairs=16, k=8, radius_q=0.10,
                boundary="free", recompute_mu=False, n_steps=3, target_rel=0.02)
    configs = {
        "allpairs": dict(base, objective="allpairs"),  # necklace-strong winner
        "center": dict(base, objective="center"),      # pre-registered comparator
    }
    t0 = time.time()
    rows = []
    for kind, d in manifolds:
        for seed in seeds:
            for cname, cfg in configs.items():
                for r in _run_manifold(kind, d, n, seed, cfg, n_eval, device):
                    r["config"] = cname
                    rows.append(r)
        print(f"[full] {kind}{d} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(PROC / "local_flatten_full_points.csv", index=False)
    summ = _summ(df, ["config"])
    summ.to_csv(PROC / "local_flatten_full_summary.csv", index=False)
    with pd.option_context("display.width", 260, "display.max_columns", None):
        print(summ.sort_values(["config", "kind", "dim"]).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("small", "rederivation", "full"):
        s = sub.add_parser(name)
        s.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.cmd == "small":
        run_small(args.device)
    elif args.cmd == "rederivation":
        run_rederivation(args.device)
    elif args.cmd == "full":
        run_full(args.device)


if __name__ == "__main__":
    main()
