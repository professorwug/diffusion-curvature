"""D-abl-2 part 2: integrator + balanced-sign downstream of the estimator bakeoff
with branch/estimator-MATCHED references (flat planes through the same estimator
chain). Focused grid: dumbbell beta=0.45, d in {3,4,5}, 2 seeds, branches
B/C05/D, estimators plain/pca/diffusion_t4/heatgeo_t2 (fb field-r already logged
separately -- its per-instance training is too heavy for the reference planes).
"""
from __future__ import annotations

import sys, time
from pathlib import Path
import joblib, numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "11-successor-curvatures"))
import v3_defect as v3, composite_suite as cs, dist_ablation as da, dist_estimators as de
from diffusion_curvature.menagerie import dumbbell_profile, WarpedProduct

PROC = HERE / "processed_data"
ESTS = ["plain", "pca", "diffusion_t4", "heatgeo_t2"]
BRANCHES = ["B", "C05", "D"]
DIMS = [3, 4, 5]


def _norm(D):
    return D / max(np.median(D[~np.eye(D.shape[0], dtype=bool)]), 1e-12)


def build_refs(reps=5):
    t0 = time.time(); store = {}
    for est in ESTS:
        for br in BRANCHES:
            rows = []
            for d in DIMS:
                for rep in range(reps):
                    Xf, _ = da.flat_plane(1400, d, seed=900 + 13 * d + rep)
                    Xc = da.corrupt(Xf, br, seed=900 + 13 * d + rep)
                    D = _norm(de.ESTIMATORS[est](Xc, d))
                    a = int(np.argmin(np.linalg.norm(Xc - Xc.mean(0), axis=1)))
                    fr = cs.extract_raw(D, np.array([a]), d); fr["n_bucket"] = 1400
                    rows.append(fr)
                print(f"[refs] {est}/{br}/d{d} ({time.time()-t0:.0f}s)", flush=True)
            df = pd.concat(rows, ignore_index=True)
            store[(est, br)] = df.groupby(["n_bucket", "true_dim"])[list(cs.CHANNELS)].agg(["mean", "std"])
    joblib.dump(store, PROC / "dist_est_down_refs.joblib")
    print(f"[refs] done {time.time()-t0:.0f}s")


def _flatz(fr, ref):
    df = fr.copy()
    for i, row in df.iterrows():
        for c in cs.CHANNELS:
            try:
                mu = ref.loc[(1400, int(row.true_dim)), (c, "mean")]; sd = ref.loc[(1400, int(row.true_dim)), (c, "std")]
            except KeyError:
                mu, sd = 0.0, 1.0
            df.at[i, c] = (row[c] - mu) / (abs(sd) + 1e-9)
    return df


def run():
    refs = joblib.load(PROC / "dist_est_down_refs.joblib")
    model = joblib.load(PROC / "suite_model.joblib")
    t0 = time.time(); rows = []
    for d in DIMS:
        f, L = dumbbell_profile(beta=da.DUMB_BETA); wp = WarpedProduct(f, L, d=d)
        for seed in (0, 1):
            X, ks = da.sample_embedded(wp, 1400, d, seed=seed)
            e = v3.pick_eval_points(ks, 24, np.random.default_rng(seed))
            for br in BRANCHES:
                Xc = da.corrupt(X, br, seed)
                for est in ESTS:
                    try:
                        D = _norm(de.ESTIMATORS[est](Xc, d)); fr = cs.extract_raw(D, e, d)
                        frz = _flatz(fr, refs[(est, br)])
                        Xf = frz[list(cs.CHANNELS) + cs.CONTEXT].values
                        isign = model["clf"].predict_proba(Xf)[:, 1] - 0.5
                        imag = model["reg"].predict(Xf)
                        for q in range(len(e)):
                            rows.append(dict(d=d, seed=seed, branch=br, est=est, ks=float(ks[e][q]),
                                             integ_sign=isign[q], integ_mag=imag[q],
                                             v4_m60=frz.v4_m60.values[q], ent_cak=frz.ent_cak.values[q],
                                             sent=frz.sent.values[q]))
                    except Exception as ex:
                        print(f"  [err] d{d} {br} {est}: {str(ex)[:60]}", flush=True)
            print(f"[run] d{d} s{seed} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows); df.to_csv(PROC / "dist_est_down_points.csv", index=False)
    summarize(df); print(f"[run] done {time.time()-t0:.0f}s")


def summarize(df):
    rows = []
    for (d, br, est), g in df.groupby(["d", "branch", "est"]):
        rec = dict(d=d, branch=br, est=est)
        for ch in ("v4_m60", "ent_cak", "sent"):
            rec[f"r_{ch}"] = v3.pearson(g[ch], g.ks)
            rec[f"s_{ch}"] = v3.balanced_sign(g[ch].values, g.ks.values)
        rec["r_integ"] = v3.pearson(g.integ_mag, g.ks)
        rec["s_integ"] = v3.balanced_sign(g.integ_sign.values, g.ks.values)
        rows.append(rec)
    sdf = pd.DataFrame(rows); sdf.to_csv(PROC / "dist_est_down_summary.csv", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", None)
    for br in BRANCHES:
        print(f"\n=== branch {br}: field r / balanced sign (mean over d3-5) per estimator ===")
        m = sdf[sdf.branch == br].groupby("est")[["r_v4_m60", "s_v4_m60", "r_ent_cak", "s_ent_cak",
                                                  "r_sent", "s_sent", "r_integ", "s_integ"]].mean()
        print(m.round(2).to_string())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["refs", "run"])
    a = ap.parse_args()
    build_refs() if a.cmd == "refs" else run()
