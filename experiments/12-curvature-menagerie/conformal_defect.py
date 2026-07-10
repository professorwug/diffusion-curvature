"""V3-conformal — fit a global flattening factor, read its Laplacian.

Global static repair of the V3 second-moment defect. Parametrize a per-node
log conformal factor u in R^n (mean-zero gauge). Squared distances scale as

    D'2[a,b] = exp(u_a + u_b) * D2[a,b]        (frozen-structure, 1st order)

i.e. E(u) = diag(e^u) D2 diag(e^u). Measures mu (cak n/4, t=2 — the validated
V3 config) are FROZEN at u=0 (pure metric response). The u-dependent defect is
the V3 defect with D2 -> E(u):

    delta_u(i,j) = mu_i^T E(u) mu_j - 1/2 mu_i^T E mu_i
                 - 1/2 mu_j^T E mu_j - E[i,j] .

Fit u by minimizing F(u) = sum_{pairs} delta_u(i,j)^2 + lambda * u^T L u over the
V3 pair set (n_pairs per eval point at band 2 x spread), torch autograd
(Adam + LBFGS). Mean-zero enforced structurally: u = v - mean(v).

READOUT: K_hat_i = -(L u)_i, L = graph Laplacian of the kNN affinity graph
(unnormalized or random-walk). Pre-registered sign: flattening a +K bulb needs
local stretch -> u peaks -> Delta u < 0 -> K_hat > 0. Secondary readouts: raw
-u, and the fitted residual |delta_u|.

Subcommands:
  python conformal_defect.py select   # hyperparameter selection (small ladder)
  python conformal_defect.py full      # headline ladder on best config
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
EXP11 = REPO / "experiments" / "11-successor-curvatures"
sys.path.insert(0, str(EXP11))

from graph_ablation_ruler import cak_W  # noqa: E402
from benchmark_kmetric_colosseum import affinity_from_D  # noqa: E402
from v3_defect import (  # noqa: E402
    build_manifold, measure_spread, pearson, balanced_sign, pick_eval_points,
    _apply_noise,
)

PROC = HERE / "processed_data"
PROC.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Frozen V3 ingredients
# ---------------------------------------------------------------------------

def build_measures(D: np.ndarray, t: int = 2) -> np.ndarray:
    """Frozen diffusion measures: P = cak n/4 kernel, M = P^t (repeated sq)."""
    n = D.shape[0]
    W, _ = cak_W(D, target=max(4, n // 4))
    P = W / np.maximum(W.sum(1, keepdims=True), 1e-30)
    M = P
    for _ in range(int(np.log2(t))):
        M = M @ M
    return M


def build_pairs(D: np.ndarray, eval_idx: np.ndarray, target: float,
                n_pairs: int) -> tuple[np.ndarray, np.ndarray]:
    """Directed pair index arrays: for each eval point i, the n_pairs points
    nearest the target radius. Returns (rows, cols) flat index arrays."""
    n = D.shape[0]
    rows, cols = [], []
    for i in eval_idx:
        diff = np.abs(D[i] - target)
        diff[i] = np.inf
        j = np.argpartition(diff, n_pairs)[:n_pairs]
        rows.append(np.full(n_pairs, i))
        cols.append(j)
    return np.concatenate(rows), np.concatenate(cols)


def laplacians(D: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Unnormalized (Deg - W) and random-walk (I - Dinv W) Laplacians of the
    adaptive kNN affinity graph."""
    W = affinity_from_D(D, k=k)
    np.fill_diagonal(W, 0.0)
    deg = W.sum(1)
    L_un = np.diag(deg) - W
    L_rw = np.eye(D.shape[0]) - W / np.maximum(deg[:, None], 1e-30)
    return L_un, L_rw


# ---------------------------------------------------------------------------
# The conformal fit
# ---------------------------------------------------------------------------

def fit_u(M: np.ndarray, D2: np.ndarray, rows: np.ndarray, cols: np.ndarray,
          lam: float, Lreg: np.ndarray | None, device: str,
          adam_steps: int = 500, lbfgs_steps: int = 40,
          lr: float = 0.05) -> tuple[np.ndarray, dict]:
    """Fit the mean-zero log conformal factor. Returns (u, info)."""
    n = M.shape[0]
    Mt = torch.as_tensor(M, dtype=torch.float64, device=device)
    D2t = torch.as_tensor(D2, dtype=torch.float64, device=device)
    r = torch.as_tensor(rows, dtype=torch.long, device=device)
    c = torch.as_tensor(cols, dtype=torch.long, device=device)
    Lt = None if (Lreg is None or lam == 0) else \
        torch.as_tensor(Lreg, dtype=torch.float64, device=device)
    v = torch.zeros(n, dtype=torch.float64, device=device, requires_grad=True)

    def data_loss(u):
        eu = torch.exp(u)
        E = eu[:, None] * eu[None, :] * D2t
        A = Mt @ E @ Mt.t()
        a = torch.diagonal(A)
        delta = A - 0.5 * a[:, None] - 0.5 * a[None, :] - E
        dsel = delta[r, c]
        return (dsel * dsel).sum(), dsel

    def total_loss():
        u = v - v.mean()
        fd, _ = data_loss(u)
        if Lt is not None:
            fd = fd + lam * (u @ (Lt @ u))
        return fd

    with torch.no_grad():
        f0, d0 = data_loss(v - v.mean())
        f0 = float(f0)
        resid0 = float(d0.abs().mean())

    opt = torch.optim.Adam([v], lr=lr)
    for _ in range(adam_steps):
        opt.zero_grad()
        loss = total_loss()
        loss.backward()
        opt.step()

    opt2 = torch.optim.LBFGS([v], max_iter=lbfgs_steps, line_search_fn="strong_wolfe")

    def closure():
        opt2.zero_grad()
        loss = total_loss()
        loss.backward()
        return loss
    if lbfgs_steps > 0:
        opt2.step(closure)

    with torch.no_grad():
        u = (v - v.mean())
        fF, dF = data_loss(u)
        info = dict(f0=f0, fF=float(fF), resid0=resid0,
                    residF=float(dF.abs().mean()),
                    rel_resid=float((fF / max(f0, 1e-30)) ** 0.5))
        return u.cpu().numpy(), info


# ---------------------------------------------------------------------------
# Readouts + evaluation
# ---------------------------------------------------------------------------

def readouts(u: np.ndarray, L_un: np.ndarray, L_rw: np.ndarray) -> dict:
    return dict(K_unnorm=-(L_un @ u), K_rw=-(L_rw @ u), neg_u=-u, u=u)


def eval_cell(kind: str, d: int, n: int, seed: int, noise: float, cfg: dict,
              device: str) -> pd.DataFrame:
    man = build_manifold(kind, d, n, seed)
    D0, ks = man["D"], man["ks"]
    rng = np.random.default_rng(3000 + seed)
    D = _apply_noise(D0, noise, rng) if noise > 0 else D0
    D2 = D ** 2
    M = build_measures(D, t=cfg["t"])
    spread = measure_spread(M, D)
    target = cfg["band"] * spread
    # eval coverage
    erng = np.random.default_rng(2000 + seed)
    if cfg["coverage"] == "all":
        eval_idx = np.arange(n)
    else:
        eval_idx = pick_eval_points(ks, int(cfg["coverage"]), erng)
    rows, cols = build_pairs(D, eval_idx, target, cfg["n_pairs"])
    L_un, L_rw = laplacians(D, k=cfg["lap_k"])
    Lreg = L_un if cfg["lam"] > 0 else None
    u, info = fit_u(M, D2, rows, cols, cfg["lam"], Lreg, device,
                    adam_steps=cfg["adam_steps"], lbfgs_steps=cfg["lbfgs_steps"])
    R = readouts(u, L_un, L_rw)
    recs = []
    for q in eval_idx:
        recs.append(dict(kind=kind, dim=d, noise=noise, seed=seed, point=int(q),
                         ks=float(ks[q]),
                         K_unnorm=float(R["K_unnorm"][q]),
                         K_rw=float(R["K_rw"][q]),
                         neg_u=float(R["neg_u"][q]),
                         resid=float(info["residF"]),
                         rel_resid=float(info["rel_resid"])))
    return pd.DataFrame(recs)


# reference: v3 static score for the re-derivation check (measures fixed, u=0)
def v3_static_score(kind, d, n, seed, noise, cfg, device):
    man = build_manifold(kind, d, n, seed)
    D0, ks = man["D"], man["ks"]
    rng = np.random.default_rng(3000 + seed)
    D = _apply_noise(D0, noise, rng) if noise > 0 else D0
    D2 = D ** 2
    M = build_measures(D, t=cfg["t"])
    spread = measure_spread(M, D)
    target = cfg["band"] * spread
    erng = np.random.default_rng(2000 + seed)
    eval_idx = (np.arange(n) if cfg["coverage"] == "all"
                else pick_eval_points(ks, int(cfg["coverage"]), erng))
    A = M @ D2 @ M.T
    a = np.diag(A)
    delta = A - 0.5 * a[:, None] - 0.5 * a[None, :] - D2
    scores = {}
    for i in eval_idx:
        diff = np.abs(D[i] - target); diff[i] = np.inf
        j = np.argpartition(diff, cfg["n_pairs"])[:cfg["n_pairs"]]
        scores[int(i)] = float(np.mean(-delta[i, j]))
    return scores


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def _summarize(df: pd.DataFrame, static: dict, key_cols: list[str]) -> pd.DataFrame:
    out = []
    for keys, g in df.groupby(key_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rec = dict(zip(key_cols, keys))
        kind = g.kind.iloc[0]
        for col in ("K_unnorm", "K_rw", "neg_u"):
            if kind == "torus":
                rec[f"{col}_flatmed"] = float(g[col].abs().median())
                rec[f"{col}_flatstd"] = float(g[col].std())
            else:
                rec[f"{col}_r"] = pearson(g[col], g.ks)
                rec[f"{col}_bal"] = balanced_sign(g[col].values, g.ks.values)
        # re-derivation vs v3 static (primary readout K_unnorm)
        skey = (kind, g.dim.iloc[0], g.noise.iloc[0], g.seed.iloc[0])
        sc = static.get(skey, {})
        sv = np.array([sc.get(p, np.nan) for p in g.point])
        rec["static_r"] = pearson(sv, g.ks.values)
        rec["Kun_vs_static"] = pearson(g.K_unnorm.values, sv)
        rec["rel_resid"] = float(g.rel_resid.mean())
        out.append(rec)
    return pd.DataFrame(out)


SELECT_CFG = dict(t=2, band=2.0, adam_steps=600, lbfgs_steps=50)


def run_select(device: str) -> None:
    n = 1400
    cells = [("dumbbell", 3), ("dumbbell", 4), ("necklace", 3),
             ("necklace", 4), ("necklace", 5), ("torus", 3)]
    seed, noise = 0, 0.0
    # lambda is inert (probed separately: no effect on residual or readout at d5),
    # so the live knobs are pair-set size, eval coverage, and a lambda spot-check.
    grid = []
    for lam in (0.0, 1e-1):
        for n_pairs in (8, 16):
            for coverage in ("200", "all"):
                grid.append(dict(lam=lam, n_pairs=n_pairs, coverage=coverage,
                                 lap_k=10))
    t0 = time.time()
    all_rows, static = [], {}
    for gi, gc in enumerate(grid):
        cfg = dict(SELECT_CFG, **gc)
        cname = f"lam{gc['lam']}|np{gc['n_pairs']}|cov{gc['coverage']}"
        for kind, d in cells:
            skey = (kind, d, noise, seed)
            if skey not in static:
                static[skey] = v3_static_score(kind, d, n, seed, noise, cfg, device)
            df = eval_cell(kind, d, n, seed, noise, cfg, device)
            df["config"] = cname
            all_rows.append(df)
        print(f"[select] {cname} ({time.time()-t0:.0f}s)", flush=True)
    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(PROC / "conformal_defect_select_points.csv", index=False)
    summ = _summarize(full, static, ["config", "kind", "dim"])
    summ.to_csv(PROC / "conformal_defect_select_summary.csv", index=False)
    with pd.option_context("display.width", 280, "display.max_columns", None):
        print(summ.sort_values(["config", "kind", "dim"]).to_string(index=False))


def run_full(device: str, cfg_over: dict | None = None) -> None:
    n = 1400
    seeds = [0, 1]
    noises = [0.0, 0.15]
    cells = [("torus", 3)] + \
            [("dumbbell", d) for d in (3, 4, 5, 6)] + \
            [("necklace", d) for d in (3, 4, 5, 6)]
    # selected config: 200 stratified eval points (fits + reads far better than
    # 'all'; fewer constraints let u actually flatten), 16 pairs, lambda inert
    # (kept 0), K_unnorm the primary readout (quietest null).
    cfg = dict(SELECT_CFG, lam=0.0, n_pairs=16, coverage="200", lap_k=10)
    if cfg_over:
        cfg.update(cfg_over)
    t0 = time.time()
    all_rows, static = [], {}
    for kind, d in cells:
        for seed in seeds:
            for noise in noises:
                skey = (kind, d, noise, seed)
                static[skey] = v3_static_score(kind, d, n, seed, noise, cfg, device)
                df = eval_cell(kind, d, n, seed, noise, cfg, device)
                all_rows.append(df)
        print(f"[full] {kind}{d} ({time.time()-t0:.0f}s)", flush=True)
    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(PROC / "conformal_defect_full_points.csv", index=False)
    summ = _summarize(full, static, ["kind", "dim", "noise"])
    summ.to_csv(PROC / "conformal_defect_full_summary.csv", index=False)
    with pd.option_context("display.width", 280, "display.max_columns", None):
        print(summ.sort_values(["kind", "dim", "noise"]).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("select", "full"):
        s = sub.add_parser(name)
        s.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.cmd == "select":
        run_select(args.device)
    elif args.cmd == "full":
        run_full(args.device)


if __name__ == "__main__":
    main()
