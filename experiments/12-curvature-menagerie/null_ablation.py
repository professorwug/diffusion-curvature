"""Deployment-gauntlet ablation #3: THE DATA-DRIVEN NULL.

Sign classification needs the flat reference that sets the channel zero. Every
reference so far used ORACLE parameters (true d, matched n, known noise). Here
we SYNTHESIZE the reference from the data alone and measure whether balanced sign
survives. Dimension was isolated in abl-1, so V4 scores use TRUE d throughout;
the ablation is the REFERENCE ONLY.

Reference branches:
  R-oracle        : embedded flat planes at true d, matched n (current practice).
  R-estimated     : per instance, estimate d-hat (Levina-Bickel, CEIL-clamped up
                    per the abl-1 safety law), sigma-hat (local-PCA residual std
                    in the top-d-hat complement, m=30), density (median kNN
                    radius); synthesize planes of dim d-hat in R^{D_ambient} with
                    matched density + sigma-hat noise, through the instance's
                    SAME chain (plain), z-reference.
  R-estimated-PCA : same, but the abl-2 PCA-denoise chain end to end (data+ref).

Reports balanced sign per channel + integrator under each reference, the flat-null
false-sign rate (hallucination under estimated refs), and the d-hat / sigma-hat
error distributions on cells where sign degrades.

Usage: python null_ablation.py colosseum | menagerie | summarize
"""
from __future__ import annotations

import sys, time
from pathlib import Path
import joblib, numpy as np, pandas as pd, torch
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
EXP11 = HERE.parent / "11-successor-curvatures"
sys.path.insert(0, str(EXP11)); sys.path.insert(0, str(HERE))
import v3_defect as v3, composite_suite as cs, dim_ablation as dim_ab, dist_estimators as de

PROC = HERE / "processed_data"
DEVICE = "cpu"
CH = list(cs.CHANNELS)


# ---------------------------------------------------------------------------
# Parameter estimation from data alone
# ---------------------------------------------------------------------------

def estimate_params(X: np.ndarray, m: int = 30) -> dict:
    n, Damb = X.shape
    D = cdist(X, X)
    order = np.argsort(D, axis=1)
    dhat = int(np.ceil(dim_ab.levina_bickel(D, 10)))       # CEIL: never floor
    dhat = int(np.clip(dhat, 1, Damb - 1))
    nn1 = np.median(np.sort(D, axis=1)[:, 1])              # median 1-NN distance
    # sigma-hat: residual std in the top-dhat complement of local PCA (m NN)
    res = []
    step = max(1, n // 250)
    for i in range(0, n, step):
        nb = X[order[i, :m]]; c = nb.mean(0)
        _, S, Vt = np.linalg.svd(nb - c, full_matrices=False)
        comp = (nb - c) @ Vt[dhat:].T                      # complement coords
        if comp.size:
            res.append(float(np.mean(np.var(comp, axis=0))))
    sigma = float(np.sqrt(np.mean(res))) if res else 0.0
    sigma_rel = sigma / max(nn1, 1e-9)                     # noise-to-spacing ratio
    knn_r = float(np.median(np.sort(D, axis=1)[:, 10]))
    dens = knn_r / max(np.median(D[~np.eye(n, dtype=bool)]), 1e-9)
    return dict(dhat=dhat, sigma_rel=round(sigma_rel, 3), density=round(dens, 4),
                Damb=Damb, n=n)


# ---------------------------------------------------------------------------
# Reference synthesis (bucketed cache)
# ---------------------------------------------------------------------------

_REF_CACHE: dict = {}


def synth_ref(true_d: int, dhat: int, sigma_rel: float, Damb: int, n: int,
              chain: str, reps: int = 6) -> pd.DataFrame:
    key = (true_d, dhat, round(sigma_rel, 1), Damb, chain)
    if key in _REF_CACHE:
        return _REF_CACHE[key]
    rows = []
    for rep in range(reps):
        rng = np.random.default_rng(600 + 17 * dhat + rep)
        Y = rng.random((n, dhat))
        X = np.hstack([Y, np.zeros((n, Damb - dhat))]) if Damb > dhat else Y[:, :Damb]
        nn1 = np.median(np.sort(cdist(X, X), axis=1)[:, 1])
        X = X + (sigma_rel * nn1) * rng.normal(size=X.shape)
        if chain == "pca":
            D = de.est_pca(X, dhat)
        else:
            D = de._knn_dijkstra(cdist(X, X))
        D = D / max(np.median(D[~np.eye(n, dtype=bool)]), 1e-9)
        a = rng.choice(n, 3, replace=False)
        fr = cs.extract_raw(D, a, true_d); fr["n_bucket"] = n
        rows.append(fr)
    df = pd.concat(rows, ignore_index=True)
    ref = df.groupby(["n_bucket", "true_dim"])[CH].agg(["mean", "std"])
    _REF_CACHE[key] = ref
    return ref


def _flatz_predict(fr: pd.DataFrame, ref, model) -> tuple:
    dfz = fr.copy()
    for i, row in dfz.iterrows():
        for c in CH:
            try:
                mu = ref.loc[(cs._bucket(int(row.n_points)), int(row.true_dim)), (c, "mean")]
                sd = ref.loc[(cs._bucket(int(row.n_points)), int(row.true_dim)), (c, "std")]
            except KeyError:
                mu, sd = 0.0, 1.0
            dfz.at[i, c] = (row[c] - mu) / (abs(sd) + 1e-9)
    Xf = dfz[CH + cs.CONTEXT].values
    isign = model["clf"].predict_proba(Xf)[:, 1] - 0.5
    return dfz, isign


# ---------------------------------------------------------------------------
# Colosseum: the deployment archetype (headline number)
# ---------------------------------------------------------------------------

def run_colosseum(pca: bool = True) -> None:
    model = joblib.load(PROC / "suite_model.joblib")
    ref_oracle = cs.load_flatref(embedded=True)
    raw = pd.read_csv(PROC / "suite_e3_raw.csv")   # plain-chain channels at true d
    inst = joblib.load(EXP11 / "processed_data" / "signed_battery.joblib")
    t0 = time.time(); rows = []
    todo = [(i, z) for i, z in enumerate(inst)
            if z["dataset"] == "colosseum" and z["m"] < 15]
    for k, (i, z) in enumerate(todo):
        d = int(z["dim"]); X = np.asarray(z["X"], float)
        src = raw[raw.inst_id == i]
        if not len(src):
            continue
        p = estimate_params(X)
        rec = dict(inst_id=i, dim=d, noise=float(z["noise"]), ks=float(z["ks_true"]),
                   dhat=p["dhat"], dhat_err=p["dhat"] - d, sigma_rel=p["sigma_rel"])
        # R-oracle (reuse cached plain channels)
        fr = src[CH + cs.CONTEXT + ["n_points", "true_dim"]].copy()
        dfz, isg = _flatz_predict(fr, ref_oracle, model)
        rec["integ_oracle"] = float(isg[0])
        for c in ("v4_m60", "ent_cak", "sent"):
            rec[f"{c}_oracle"] = float(dfz[c].values[0])
        # R-estimated (synthesized plain reference)
        ref_e = synth_ref(d, p["dhat"], p["sigma_rel"], p["Damb"], p["n"], "plain")
        dfz, isg = _flatz_predict(fr, ref_e, model)
        rec["integ_est"] = float(isg[0])
        for c in ("v4_m60", "ent_cak", "sent"):
            rec[f"{c}_est"] = float(dfz[c].values[0])
        # R-estimated-PCA (re-extract data through PCA chain + PCA reference)
        if pca:
            Dp = de.est_pca(X, p["dhat"]); Dp = Dp / max(np.median(Dp[~np.eye(len(X), dtype=bool)]), 1e-9)
            frp = cs.extract_raw(Dp, np.array([0]), d)
            ref_ep = synth_ref(d, p["dhat"], p["sigma_rel"], p["Damb"], p["n"], "pca")
            dfz, isg = _flatz_predict(frp, ref_ep, model)
            rec["integ_estpca"] = float(isg[0])
            for c in ("v4_m60", "ent_cak", "sent"):
                rec[f"{c}_estpca"] = float(dfz[c].values[0])
        rows.append(rec)
        if (k + 1) % 25 == 0:
            print(f"[colosseum] {k+1}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(PROC / "null_colosseum_points.csv", index=False)
    _summ_colosseum(df)
    print(f"[colosseum] {len(df)} instances in {time.time()-t0:.0f}s")


def _bsign(score, ks):
    return v3.balanced_sign(np.asarray(score, float), np.asarray(ks, float))


def _summ_colosseum(df: pd.DataFrame) -> None:
    variants = [("oracle", "R-oracle"), ("est", "R-estimated"), ("estpca", "R-estimated-PCA")]
    rows = []
    for d, g in df.groupby("dim"):
        rec = dict(dim=int(d), n=len(g), dhat_err_absmean=float(g.dhat_err.abs().mean()),
                   sigma_rel_mean=float(g.sigma_rel.mean()))
        for suf, _ in variants:
            if f"integ_{suf}" not in g:
                continue
            rec[f"sign_integ_{suf}"] = _bsign(g[f"integ_{suf}"].values, g.ks.values)
            for c in ("v4_m60", "ent_cak", "sent"):
                rec[f"sign_{c}_{suf}"] = _bsign(g[f"{c}_{suf}"].values, g.ks.values)
        rows.append(rec)
    sdf = pd.DataFrame(rows); sdf.to_csv(PROC / "null_colosseum_summary.csv", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", None)
    print("\n=== COLOSSEUM balanced sign: R-oracle vs R-estimated vs R-estimated-PCA ===")
    cols = ["dim", "n", "dhat_err_absmean", "sigma_rel_mean",
            "sign_integ_oracle", "sign_integ_est", "sign_integ_estpca",
            "sign_v4_m60_oracle", "sign_v4_m60_est", "sign_ent_cak_oracle", "sign_ent_cak_est"]
    cols = [c for c in cols if c in sdf.columns]
    print(sdf[cols].round(3).to_string(index=False))
    # decisive number: mean over dims, integrator
    for suf in ("oracle", "est", "estpca"):
        col = f"sign_integ_{suf}"
        if col in sdf:
            print(f"  mean integrator sign [{suf}] = {sdf[col].mean():.3f}")


# families for the false-positive analysis
RIGIDITY = ["v4_m60", "v4_m120", "v3_defect"]     # embedding-rigidity channels
DIFFUSION = ["ent_cak", "sent", "frac", "kappa"]  # diffusion channels


def run_datashaped(n_sample: int = 40) -> None:
    """Data-shaped null: coordinate-shuffle real colosseum clouds (destroys the
    joint manifold, preserves per-coordinate marginals). A well-behaved pipeline
    reads manifold-less data as ~flat; strong curvature = false positive. Scored
    against the instance's own R-estimated reference."""
    model = joblib.load(PROC / "suite_model.joblib")
    inst = joblib.load(EXP11 / "processed_data" / "signed_battery.joblib")
    todo = [(i, z) for i, z in enumerate(inst)
            if z["dataset"] == "colosseum" and z["m"] < 15]
    rng = np.random.default_rng(0)
    pick = [todo[k] for k in rng.choice(len(todo), min(n_sample, len(todo)), replace=False)]
    t0 = time.time(); rows = []
    for j, (i, z) in enumerate(pick):
        d = int(z["dim"]); X = np.asarray(z["X"], float).copy()
        for col in range(X.shape[1]):                       # per-coordinate shuffle
            X[:, col] = X[rng.permutation(len(X)), col]
        p = estimate_params(X)
        D = de._knn_dijkstra(cdist(X, X)); D = D / max(np.median(D[~np.eye(len(X), dtype=bool)]), 1e-9)
        fr = cs.extract_raw(D, np.array([0]), d)
        ref = synth_ref(d, p["dhat"], p["sigma_rel"], p["Damb"], p["n"], "plain")
        dfz, isg = _flatz_predict(fr, ref, model)
        rows.append(dict(inst_id=i, dim=d, dhat=p["dhat"],
                         v4_m60=float(dfz.v4_m60.values[0]), v3_defect=float(dfz.v3_defect.values[0]),
                         ent_cak=float(dfz.ent_cak.values[0]), sent=float(dfz.sent.values[0]),
                         integ_sign=float(isg[0])))
        if (j + 1) % 10 == 0:
            print(f"[datashaped] {j+1}/{len(pick)} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows); df.to_csv(PROC / "null_datashaped_points.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== DATA-SHAPED NULL (coordinate-shuffled clouds; |z| should be small) ===")
    print("family      median|z|   frac|z|>1.5 (false-curvature rate)")
    for fam, cols in (("rigidity(V4/V3)", ["v4_m60", "v3_defect"]),
                      ("diffusion", ["ent_cak", "sent"])):
        vals = np.abs(df[cols].values.ravel())
        print(f"  {fam:16s} {np.median(vals):.3f}      {np.mean(vals > 1.5):.3f}")
    print(f"  integrator |sign| median={df.integ_sign.abs().median():.3f}  "
          f"frac|sign|>0.25={np.mean(df.integ_sign.abs() > 0.25):.3f}")
    print(f"[datashaped] {len(df)} shuffled clouds in {time.time()-t0:.0f}s")


def main():
    import argparse
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("colosseum"); c.add_argument("--no-pca", action="store_true")
    sub.add_parser("datashaped")
    sub.add_parser("summarize")
    a = ap.parse_args()
    if a.cmd == "colosseum":
        run_colosseum(pca=not a.no_pca)
    elif a.cmd == "datashaped":
        run_datashaped()
    elif a.cmd == "summarize":
        _summ_colosseum(pd.read_csv(PROC / "null_colosseum_points.csv"))


if __name__ == "__main__":
    main()
