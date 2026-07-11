"""Deployment-gauntlet ablation #2: DISTANCE (oracle geodesics vs estimation chain).

Replaces the oracle distance matrix with the full deployment chain (embed ->
cdist -> kNN(10) -> Dijkstra geodesics -> suite) and measures the cost against
certified ks fields. TRUE d everywhere (dimension was abl-1).

Branches per instance:
  A oracle           : exact WarpedProduct geodesics.
  B embedded-clean   : rotational embedding in R^{d+1} (|f'|<=1) -> Dijkstra.
  C embedded+ambient : B + Gaussian ambient noise at {0.5, 1.7} x median-kNN.
  D embedded+hd64    : B + smooth random lift to R^64 + 0.1 noise (codim killer).
  E nugget-corrected : variogram-nugget floor subtracted from D^2 before graph.

Embeddability: a warped product g=dr^2+f^2 gΩ embeds as a surface of revolution
X=(f(r)u, z(r)), z(r)=∫sqrt(1-f'^2), ONLY when |f'|<=1 (else 1-f'^2<0 — no
isometric embedding at any codim). The strong signed battery profiles
(dumbbell beta 0.65/0.8, necklace b 0.55/0.7) FAIL this; only dumbbell beta<=0.5
qualifies, so the embedded branches run on dumbbell beta=0.45 (signed) + flat
nulls. This is documented as a finding.

References are regenerated per branch through the same corruption pipeline.

Usage: python dist_ablation.py refs | run | summarize
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
EXP11 = REPO / "experiments" / "11-successor-curvatures"
sys.path.insert(0, str(EXP11))
sys.path.insert(0, str(HERE))

import v3_defect as v3  # noqa: E402
import composite_suite as cs  # noqa: E402
from diffusion_curvature.menagerie import dumbbell_profile, WarpedProduct  # noqa: E402

PROC = HERE / "processed_data"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DUMB_BETA = 0.45
DIMS = [3, 4, 5, 6]
BRANCHES = ["A", "B", "C05", "C17", "D", "E_C17", "E_D"]


# ---------------------------------------------------------------------------
# Rotational embedding + corruptions + graph geodesics
# ---------------------------------------------------------------------------

def _z_of_r(wp: WarpedProduct) -> np.ndarray:
    """Height profile z(r)=∫_0^r sqrt(1-f'^2) on the wp grid."""
    fp = np.gradient(wp.fg, wp.rg)
    integrand = np.sqrt(np.clip(1 - fp**2, 0, None))
    return np.concatenate([[0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1])
                                          * np.diff(wp.rg))])


def sample_embedded(wp: WarpedProduct, n: int, d: int, seed: int):
    """Sample (r,u) ~ manifold, return X in R^{d+1} and ks per point."""
    rng = np.random.default_rng(seed)
    pdf = np.maximum(wp.fg, 0) ** (d - 1)
    cdf = np.cumsum(pdf); cdf /= cdf[-1]
    r = np.interp(rng.random(n), cdf, wp.rg)
    u = rng.normal(size=(n, d)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    fr = np.interp(r, wp.rg, wp.fg)
    zr = np.interp(r, wp.rg, _z_of_r(wp))
    X = np.hstack([fr[:, None] * u, zr[:, None]])          # R^{d+1}
    return X, wp.scalar(r)


def flat_plane(n: int, d: int, seed: int):
    rng = np.random.default_rng(seed)
    X = np.hstack([rng.random((n, d)), np.zeros((n, 1))])
    return X, np.zeros(n)


def corrupt(X: np.ndarray, branch: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(1000 + seed)
    if branch in ("B", "E_C17", "E_D"):
        base = X
    else:
        base = X
    if branch.startswith("C") or branch == "E_C17":
        # sigma relative to median kNN(10) distance in the clean embedding
        from scipy.spatial.distance import cdist
        Dm = cdist(X, X); nn = np.sort(Dm, axis=1)[:, 10]
        med = float(np.median(nn))
        sig = (0.5 if branch == "C05" else 1.7) * med
        return X + sig * rng.normal(size=X.shape)
    if branch == "D" or branch == "E_D":
        D0 = X.shape[1]
        rl = np.random.default_rng(777 + D0)
        W1 = rl.normal(size=(D0, 32)) / np.sqrt(D0); b1 = rl.normal(size=32) * 0.3
        W2 = rl.normal(size=(32, 64)) / np.sqrt(32)
        Xl = np.tanh(X @ W1 * 3.0 + b1) @ W2
        sub = rl.choice(len(Xl), min(2000, len(Xl)), replace=False)
        from scipy.spatial.distance import cdist
        sc = float(np.median(cdist(Xl[sub], Xl[sub])[np.triu_indices(len(sub), 1)]))
        Xl = Xl / sc
        return Xl + 0.1 * rng.normal(size=Xl.shape)
    return base


def graph_D(X: np.ndarray, nugget: bool = False) -> np.ndarray:
    """cdist -> (optional nugget subtraction) -> kNN(10) Dijkstra geodesics."""
    Xt = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
    D = torch.cdist(Xt, Xt).cpu().numpy().astype(np.float64)
    if nugget:
        c0 = _estimate_nugget(D)
        D2 = np.maximum(D**2 - c0, 0.0)
        D = np.sqrt(D2)
        np.fill_diagonal(D, 0.0)
    return v3._graph_geodesic(D, k=10)


def _estimate_nugget(D: np.ndarray) -> float:
    """Adapted variogram nugget: additive ambient noise inflates every off-diag
    squared distance by a constant floor c0 = 2*D_amb*sigma^2. Estimate it as
    the intercept of a robust line fit of squared kNN distance vs neighbor rank,
    extrapolated to rank 0 (the noise floor as true separation -> 0)."""
    Ds = np.sort(D, axis=1)[:, 1:9] ** 2        # squared 1st..8th NN distances
    ranks = np.arange(1, 9, dtype=float)
    # per-point linear fit sq-dist ~ a + b*rank; nugget = median intercept a
    x = ranks - ranks.mean()
    b = (Ds * x).sum(1) / (x @ x)
    a = Ds.mean(1) - b * ranks.mean()
    return float(max(np.median(a), 0.0))


# ---------------------------------------------------------------------------
# Per-branch references (embedded flat plane through the same corruption)
# ---------------------------------------------------------------------------

def run_refs(reps: int = 8, n: int = 1400) -> None:
    t0 = time.time()
    store = {}
    for branch in BRANCHES:
        if branch == "A":
            continue
        rows = []
        for d in DIMS:
            for rep in range(reps):
                Xc, _ = flat_plane(n, d, seed=500 + 13 * d + rep)
                Xc = corrupt(Xc, branch, seed=500 + 13 * d + rep)
                D = graph_D(Xc, nugget=branch.startswith("E"))
                a = int(np.argmin(np.linalg.norm(Xc - Xc.mean(0), axis=1)))
                fr = cs.extract_raw(D, np.array([a]), d)
                fr["n_bucket"] = n
                rows.append(fr)
            print(f"[refs] {branch} d={d} ({time.time()-t0:.0f}s)", flush=True)
        df = pd.concat(rows, ignore_index=True)
        store[branch] = df.groupby(["n_bucket", "true_dim"])[list(cs.CHANNELS)].agg(
            ["mean", "std"])
    joblib.dump(store, PROC / "dist_refs.joblib")
    print(f"[refs] done in {time.time()-t0:.0f}s")


def _flatz(fr: pd.DataFrame, ref) -> pd.DataFrame:
    df = fr.copy()
    for i, row in df.iterrows():
        key = (cs._bucket(int(row["n_points"])), int(row["true_dim"]))
        for c in cs.CHANNELS:
            try:
                mu = ref.loc[key, (c, "mean")]; sd = ref.loc[key, (c, "std")]
            except KeyError:
                mu, sd = 0.0, 1.0
            df.at[i, c] = (row[c] - mu) / (abs(sd) + 1e-9)
    return df


# ---------------------------------------------------------------------------
# Main run: all branches x manifolds
# ---------------------------------------------------------------------------

def run() -> None:
    refs = joblib.load(PROC / "dist_refs.joblib")
    ref_tor = cs.load_flatref(embedded=False)
    model = joblib.load(PROC / "suite_model.joblib")
    t0 = time.time()
    rows = []
    manifolds = [("dumbbell", d, s) for d in DIMS for s in (0, 1)] + \
                [("flat", d, 0) for d in DIMS]
    for kind, d, seed in manifolds:
        if kind == "dumbbell":
            f, L = dumbbell_profile(beta=DUMB_BETA)
            wp = WarpedProduct(f, L, d=d)
            Xclean, ks = sample_embedded(wp, 1400, d, seed=seed)
            D_oracle = wp.sample(1400, rng=seed)  # oracle uses its own sample
        else:
            Xclean, ks = flat_plane(1400, d, seed)
            D_oracle = None
        for branch in BRANCHES:
            if branch == "A":
                if kind == "flat":
                    continue
                D = D_oracle["D"]; ks_b = D_oracle["ks_field"]
                Dn = D / max(np.median(D[~np.eye(D.shape[0], dtype=bool)]), 1e-12)
                fr = cs.extract_raw(Dn, v3.pick_eval_points(ks_b, 24, np.random.default_rng(seed)), d)
                kv = ks_b[v3.pick_eval_points(ks_b, 24, np.random.default_rng(seed))]
                ref = ref_tor
            else:
                Xc = corrupt(Xclean, branch, seed=seed)
                D = graph_D(Xc, nugget=branch.startswith("E"))
                Dn = D / max(np.median(D[~np.eye(D.shape[0], dtype=bool)]), 1e-12)
                eidx = (v3.pick_eval_points(ks, 24, np.random.default_rng(seed))
                        if kind == "dumbbell" else
                        np.random.default_rng(seed).choice(1400, 24, replace=False))
                fr = cs.extract_raw(Dn, eidx, d)
                kv = ks[eidx]
                ref = refs[branch]
            frz = _flatz(fr, ref)
            X = frz[list(cs.CHANNELS) + cs.CONTEXT].values
            isign = model["clf"].predict_proba(X)[:, 1] - 0.5
            imag = model["reg"].predict(X)
            for q in range(len(kv)):
                rows.append(dict(kind=kind, dim=d, seed=seed, branch=branch,
                                 ks=float(kv[q]), integ_sign=isign[q], integ_mag=imag[q],
                                 **{c: frz[c].values[q] for c in cs.CHANNELS}))
        print(f"[run] {kind} d{d} s{seed} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(PROC / "dist_ablation_points.csv", index=False)
    _summarize(df)
    print(f"[run] {len(df)} points in {time.time()-t0:.0f}s")


def _summarize(df: pd.DataFrame) -> None:
    chans = list(cs.CHANNELS)
    dumb = df[df.kind == "dumbbell"]
    rows = []
    for (d, branch), g in dumb.groupby(["dim", "branch"]):
        rec = dict(dim=int(d), branch=branch,
                   r_integ=v3.pearson(g.integ_mag, g.ks),
                   sign_integ=v3.balanced_sign(g.integ_sign.values, g.ks.values))
        for c in chans:
            rec[f"r_{c}"] = v3.pearson(g[c], g.ks)
        rows.append(rec)
    sdf = pd.DataFrame(rows)
    sdf.to_csv(PROC / "dist_ablation_summary.csv", index=False)
    # flat null per branch (does the chain make flat read curved?)
    flat = df[df.kind == "flat"]
    fn = flat.groupby(["dim", "branch"])[["v4_m60", "v4_m120", "v3_defect"]].agg(
        lambda s: float(np.median(np.abs(s))))
    fn.to_csv(PROC / "dist_ablation_flatnull.csv")
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", None)
    print("\n=== DISTANCE ablation: dumbbell beta=0.45, integrator + V4 + diffusion field r ===")
    piv = sdf.pivot_table(index="dim", columns="branch",
                          values=["r_integ", "r_v4_m60", "r_ent_cak"])
    print(piv.round(2).to_string())
    print("\n=== flat-null |median z| per branch (hallucination: chain reads flat as curved?) ===")
    print(fn.round(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("refs"); sub.add_parser("run")
    sub.add_parser("summarize")
    args = ap.parse_args()
    if args.cmd == "refs":
        run_refs()
    elif args.cmd == "run":
        run()
    elif args.cmd == "summarize":
        _summarize(pd.read_csv(PROC / "dist_ablation_points.csv"))


if __name__ == "__main__":
    main()
