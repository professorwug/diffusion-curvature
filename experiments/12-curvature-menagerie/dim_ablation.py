"""Deployment-gauntlet ablation #1: DIMENSION (known d vs inferred d-hat).

d enters the suite in exactly two places (audited): V4 m60/m120's rank split
`readouts(eigs, d)`, and the flat-reference lookup keyed on (n, d). All other
channels are d-free (their raw values are identical known-vs-inferred; only
their z-score shifts if the reference row changes). This module measures the
cost of replacing oracle d with an estimated d-hat.

Stage A: bakeoff of three D-native dimension estimators (TwoNN, Levina-Bickel
MLE at k in {10,20}, local-PCA/Gram eigengap via V4 patches) across all regimes.
Stage B (dim_ablation_stageB.py): re-score V4 + integrator + channels under
true d vs round(d-hat), with the 4-way corruption/reference decomposition.

Stage A saves, per eval point, the top-K Gram eigenvalues + eigensum for m=60
and m=120 patches so Stage B can recompute V4(d) for any d without re-extraction.

Usage: python dim_ablation.py stageA
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
import v4_mds_strain as v4  # noqa: E402
import build_battery as bb  # noqa: E402

PROC = HERE / "processed_data"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
KEIG = 10  # top eigenvalues stored per patch


# ---------------------------------------------------------------------------
# D-native dimension estimators
# ---------------------------------------------------------------------------

def twonn(D: np.ndarray, discard: float = 0.1) -> float:
    """Facco TwoNN: mu = r2/r1 per point; MLE slope of log(1-F) vs log(mu),
    discarding the top `discard` tail. d_hat = N / sum(log mu) over kept."""
    Ds = np.sort(D, axis=1)
    r1 = Ds[:, 1]
    r2 = Ds[:, 2]
    mu = r2 / np.maximum(r1, 1e-12)
    mu = mu[mu > 1.0 + 1e-9]
    mu = np.sort(mu)
    keep = mu[:max(2, int((1 - discard) * len(mu)))]
    return float(len(keep) / np.sum(np.log(keep)))


def levina_bickel(D: np.ndarray, k: int = 10) -> float:
    """Levina-Bickel local MLE with MacKay-Ghahramani correction; averaged over
    points. m_k(x) = [1/(k-2) sum_{j=1}^{k-1} log(T_k/T_j)]^{-1}."""
    Ds = np.sort(D, axis=1)[:, 1:k + 1]          # k nearest, exclude self
    Tk = Ds[:, -1][:, None]
    Tj = np.maximum(Ds[:, :-1], 1e-12)
    logs = np.log(Tk / Tj).sum(axis=1)           # sum over j=1..k-1
    m = (k - 2) / np.maximum(logs, 1e-12)        # per-point estimate
    return float(np.mean(m))


def eigengap_dhat(D2: np.ndarray, order: np.ndarray, eval_idx: np.ndarray,
                  m: int = 60) -> tuple[float, list]:
    """Median over eval points of the V4 top-eigengap rank (dmax=8). Returns
    (d_hat, list of per-point (topK eigs, eigsum) for m and 2m patches)."""
    n = D2.shape[0]
    dhats = []
    eigrows = []
    for a in eval_idx:
        rec = {}
        for mm, tag in ((m, "60"), (min(2 * m, n - 1), "120")):
            patch = order[a, :mm]
            eigs = v4.gram_eigs(D2[np.ix_(patch, patch)])
            rec[tag] = (eigs[:KEIG], float(eigs.sum()))
            if tag == "60":
                dhats.append(v4.estimate_rank(eigs, 8))
        eigrows.append(rec)
    return float(np.median(dhats)), eigrows


# ---------------------------------------------------------------------------
# Instance iteration across regimes
# ---------------------------------------------------------------------------

def _menagerie_instances(n: int = 1400):
    """tier2 (dumbbell/necklace d3-6, noise {0,.05,.15}, 2 seeds) + tier1v2."""
    out = []
    for kind in ("dumbbell", "necklace"):
        for d in (3, 4, 5, 6):
            for seed in (0, 1):
                man = v4.build_manifold(kind, d, n, seed)
                for noise in (0.0, 0.05, 0.15):
                    out.append(("tier2", kind, d, noise, seed, man, None))
    for j, r in enumerate(bb.tier1_v2_recipes()):
        rec = dict(r); rec.setdefault("seed", 0); rec.setdefault("noise", 0.0)
        try:
            inst = bb.materialize(rec)
        except Exception:
            continue
        man = dict(D=inst["D"], ks=inst["ks_field"], eval_idx=inst["eval_idx"])
        out.append(("tier1v2", rec["kind"], rec["dim"], 0.0, 0, man,
                    ("materialized", j, rec)))
    return out


def run_stageA() -> None:
    t0 = time.time()
    rows = []
    eig_store = {}   # (regime, kind, d, noise, seed, point) -> eig record

    # ---- menagerie ----
    for gid, (regime, kind, d, noise, seed, man, mat) in enumerate(
            _menagerie_instances()):
        recipe = None
        if mat is None:                      # tier2: apply noise to cached D
            D0, ks = man["D"], man["ks"]
            D = v3._apply_noise(D0, noise, np.random.default_rng(3000 + seed))
            eidx = v3.pick_eval_points(ks, 24, np.random.default_rng(2000 + seed))
        else:                                # tier1v2: already materialized
            D, ks, eidx = man["D"], man["ks"], man["eval_idx"]
            recipe = mat[2]
        D = D / max(np.median(D[~np.eye(D.shape[0], dtype=bool)]), 1e-12)
        D2 = D ** 2
        order = np.argsort(D, axis=1)
        dh_eg, eigrows = eigengap_dhat(D2, order, eidx)
        iid = f"{regime}-{kind}-d{d}-nz{noise}-s{seed}-g{gid}"
        rows.append(dict(regime=regime, kind=kind, true_d=d, noise=noise,
                         seed=seed, n=D.shape[0],
                         dhat_twonn=twonn(D), dhat_lb10=levina_bickel(D, 10),
                         dhat_lb20=levina_bickel(D, 20), dhat_eigengap=dh_eg,
                         inst_id=iid))
        eig_store[iid] = dict(eval_idx=eidx.tolist(), ks=ks[eidx].tolist(),
                              eigs=eigrows, recipe=recipe)
        if len(rows) % 20 == 0:
            print(f"[stageA] menagerie {len(rows)} ({time.time()-t0:.0f}s)", flush=True)

    # ---- battery (colosseum m<15 + sadspheres) ----
    inst_b = joblib.load(EXP11 / "processed_data" / "signed_battery.joblib")
    for i, z in enumerate(inst_b):
        if not (z["dataset"] == "sadspheres"
                or (z["dataset"] == "colosseum" and z["m"] < 15)):
            continue
        X = np.asarray(z["X"], dtype=np.float64)
        Xt = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
        D = torch.cdist(Xt, Xt).cpu().numpy().astype(np.float64)
        D = D / max(np.median(D[~np.eye(D.shape[0], dtype=bool)]), 1e-12)
        D2 = D ** 2
        order = np.argsort(D, axis=1)
        eidx = np.array([0])
        dh_eg, eigrows = eigengap_dhat(D2, order, eidx)
        iid = f"{z['dataset']}-{i}"
        rows.append(dict(regime=z["dataset"], kind=z.get("shape", "") or "colosseum",
                         true_d=int(z["dim"]), noise=float(z["noise"]), seed=0,
                         n=D.shape[0], dhat_twonn=twonn(D),
                         dhat_lb10=levina_bickel(D, 10), dhat_lb20=levina_bickel(D, 20),
                         dhat_eigengap=dh_eg, inst_id=iid, bat_idx=i))
        eig_store[iid] = dict(eval_idx=[0], ks=[float(z["ks_true"])], eigs=eigrows)
        if len(rows) % 100 == 0:
            print(f"[stageA] +battery {len(rows)} ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(PROC / "dim_stageA_dhat.csv", index=False)
    joblib.dump(eig_store, PROC / "dim_eig_store.joblib")
    _summarize_stageA(df)
    print(f"[stageA] {len(df)} instances in {time.time()-t0:.0f}s")


def _summarize_stageA(df: pd.DataFrame) -> None:
    ests = ["dhat_twonn", "dhat_lb10", "dhat_lb20", "dhat_eigengap"]
    rows = []
    for (regime, d), g in df.groupby(["regime", "true_d"]):
        rec = dict(regime=regime, true_d=int(d), n=len(g))
        for e in ests:
            rec[f"{e}_mean"] = float(g[e].mean())
            rec[f"{e}_acc"] = float((np.round(g[e]) == d).mean())
            rec[f"{e}_bias"] = float((g[e] - d).mean())
        rows.append(rec)
    sdf = pd.DataFrame(rows)
    sdf.to_csv(PROC / "dim_stageA_summary.csv", index=False)
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", None)
    print("\n=== Stage A: round(d-hat) accuracy per (regime, true_d) ===")
    acc = sdf[["regime", "true_d", "n"] + [f"{e}_acc" for e in ests]]
    print(acc.round(3).to_string(index=False))
    print("\n=== global round-accuracy and mean signed bias per estimator ===")
    for e in ests:
        acc_all = float((np.round(df[e]) == df.true_d).mean())
        bias_all = float((df[e] - df.true_d).mean())
        print(f"  {e:16s} acc={acc_all:.3f} bias={bias_all:+.3f}")
    print("\n=== per-regime winner (by round-accuracy) ===")
    for regime, g in df.groupby("regime"):
        accs = {e: float((np.round(g[e]) == g.true_d).mean()) for e in ests}
        best = max(accs, key=accs.get)
        print(f"  {regime:12s} winner={best} ({accs[best]:.3f}) | "
              + " ".join(f"{e.split('_')[1]}:{accs[e]:.2f}" for e in ests))


# ---------------------------------------------------------------------------
# Stage B: known-d vs inferred-d channel ablation
# ---------------------------------------------------------------------------

# per-regime dimension estimator winner (Stage A verdict)
WINNER = {"tier2": "dhat_eigengap", "tier1v2": "dhat_lb10",
          "colosseum": "dhat_lb10", "sadspheres": "dhat_lb10"}
CHANNELS = ["v4_m60", "v4_m120", "v3_defect", "sent", "ent_cak", "frac", "kappa"]
DFREE = ["v3_defect", "sent", "ent_cak", "frac", "kappa"]
CONTEXT = ["est_dim", "spread", "knn_radius", "patch_frac"]
FEATURES = CHANNELS + CONTEXT


def _v4_at_d(eig_topk, eigsum, d):
    d = int(max(1, min(d, len(eig_topk) - 1)))
    top = float(np.sum(eig_topk[:d]))
    resid = eigsum - top
    return -(resid / max(top, 1e-12))          # oriented: higher = more +K


def _zscore_row(raw: dict, nb: int, ref_d: int, flatref) -> dict:
    out = dict(raw)
    for c in CHANNELS:
        try:
            mu = flatref.loc[(nb, ref_d), (c, "mean")]
            sd = flatref.loc[(nb, ref_d), (c, "std")]
        except KeyError:
            mu, sd = 0.0, 1.0
        out[c] = (raw[c] - mu) / (abs(sd) + 1e-9)
    return out


def run_stageB() -> None:
    import composite_suite as cs
    ref_tor = cs.load_flatref(embedded=False)
    ref_emb = cs.load_flatref(embedded=True)
    models = {"warped": joblib.load(PROC / "suite_model.joblib"),
              "aug": joblib.load(PROC / "suite_model_aug.joblib")}
    dhat = pd.read_csv(PROC / "dim_stageA_dhat.csv").set_index("inst_id")
    store = joblib.load(PROC / "dim_eig_store.joblib")
    e3 = pd.read_csv(PROC / "suite_e3_raw.csv").set_index("inst_id")
    e4 = pd.read_csv(PROC / "suite_e4_raw.csv").set_index("inst_id")
    t0 = time.time()

    rows = []          # per eval-point, with A/B/C/D integrator + channel scores
    for iid, meta in store.items():
        drow = dhat.loc[iid]
        regime = drow["regime"]
        true_d = int(drow["true_d"])
        rdh = int(round(float(drow[WINNER[regime]])))
        nb = cs._bucket(int(drow["n"]))
        flatref = ref_emb if regime in ("colosseum", "sadspheres") else ref_tor
        ks_list = meta["ks"]

        # d-free channels + context per eval point
        if regime in ("colosseum", "sadspheres"):
            src = (e3 if regime == "colosseum" else e4).loc[drow["bat_idx"]]
            dfree_ctx = [{**{c: float(src[c]) for c in DFREE},
                          **{c: float(src[c]) for c in CONTEXT}}]
        else:
            dfree_ctx = _menagerie_dfree(regime, drow, meta.get("recipe"), cs)

        for q, eigrec in enumerate(meta["eigs"]):
            e60, s60 = np.asarray(eigrec["60"][0]), eigrec["60"][1]
            e120, s120 = np.asarray(eigrec["120"][0]), eigrec["120"][1]
            base = dict(dfree_ctx[min(q, len(dfree_ctx) - 1)])
            ks = ks_list[q]

            def feat(v4_d):
                r = dict(base)
                r["v4_m60"] = _v4_at_d(e60, s60, v4_d)
                r["v4_m120"] = _v4_at_d(e120, s120, v4_d)
                return r

            variants = {"A": (true_d, true_d), "B": (rdh, rdh),
                        "C": (true_d, rdh), "D": (rdh, true_d)}
            rec = dict(regime=regime, kind=drow["kind"], true_d=true_d,
                       noise=float(drow["noise"]), rdhat=rdh,
                       dhat_err=rdh - true_d, ks=ks, iid=iid)
            for name, (v4_d, ref_d) in variants.items():
                z = _zscore_row(feat(v4_d), nb, ref_d, flatref)
                X = np.array([[z[f] for f in FEATURES]])
                for mname, mdl in models.items():
                    rec[f"integ_sign_{name}_{mname}"] = float(
                        mdl["clf"].predict_proba(X)[0, 1] - 0.5)
                    rec[f"integ_mag_{name}_{mname}"] = float(mdl["reg"].predict(X)[0])
                for c in ("v4_m60", "v4_m120", *DFREE):
                    rec[f"{c}_{name}"] = z[c]
            rows.append(rec)
        if len(rows) % 500 == 0:
            print(f"[stageB] {len(rows)} points ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(PROC / "dim_stageB_points.csv", index=False)
    _summarize_stageB(df)
    print(f"[stageB] {len(df)} eval points in {time.time()-t0:.0f}s")


def _menagerie_dfree(regime, drow, recipe, cs):
    """Recompute d-free channels + context for a menagerie instance (cheap,
    n=1400); returns list of per-eval-point dicts aligned with stored eigs."""
    true_d = int(drow["true_d"])
    if regime == "tier2":
        man = v4.build_manifold(drow["kind"], true_d, 1400, int(drow["seed"]))
        D0, ks = man["D"], man["ks"]
        D = v3._apply_noise(D0, float(drow["noise"]),
                            np.random.default_rng(3000 + int(drow["seed"])))
        eidx = v3.pick_eval_points(ks, 24, np.random.default_rng(2000 + int(drow["seed"])))
    else:  # tier1v2 — exact recipe stored in Stage A
        rec = dict(recipe); rec.setdefault("seed", 0); rec.setdefault("noise", 0.0)
        inst = bb.materialize(rec); D = inst["D"]; eidx = inst["eval_idx"]
    fr = cs.extract_raw(D, np.asarray(eidx), true_d)
    return [{**{c: float(fr[c].values[q]) for c in DFREE},
             **{c: float(fr[c].values[q]) for c in CONTEXT}}
            for q in range(len(eidx))]


def _fr_sign(g, mag_col, sign_col):
    return (v3.pearson(g[mag_col], g.ks), v3.balanced_sign(g[sign_col].values, g.ks.values))


def _summarize_stageB(df: pd.DataFrame) -> None:
    # tier2 (warped) has within-manifold ks variation -> group by kind+noise;
    # homogeneous/embedded regimes have constant ks per instance -> pool
    # cross-manifold by dim so ks varies.
    rows = []
    for regime, dr in df.groupby("regime"):
        gcols = (["kind", "true_d", "noise"] if regime == "tier2" else ["true_d"])
        for keys, g in dr.groupby(gcols):
            keys = keys if isinstance(keys, tuple) else (keys,)
            rec = dict(regime=regime, true_d=int(g.true_d.iloc[0]), n=len(g))
            if regime == "tier2":
                rec.update(kind=keys[0], noise=keys[2])
            for name in ("A", "B", "C", "D"):
                r, s = _fr_sign(g, f"integ_mag_{name}_warped", f"integ_sign_{name}_warped")
                rec[f"r_integ_{name}"] = r
                rec[f"sign_integ_{name}"] = s
            _, sB_aug = _fr_sign(g, "integ_mag_B_aug", "integ_sign_B_aug")
            rec["sign_integ_B_aug"] = sB_aug
            for c in ("v4_m60", "v4_m120"):
                rec[f"r_{c}_A"] = v3.pearson(g[f"{c}_A"], g.ks)
                rec[f"r_{c}_B"] = v3.pearson(g[f"{c}_B"], g.ks)
            rec["dhat_err_absmean"] = float(g.dhat_err.abs().mean())
            rows.append(rec)
    sdf = pd.DataFrame(rows).sort_values(["regime", "true_d"])
    sdf.to_csv(PROC / "dim_stageB_summary.csv", index=False)
    pd.set_option("display.width", 320); pd.set_option("display.max_columns", None)
    print("\n=== Stage B: integrator (warped) KNOWN(A) vs INFERRED(B) + decomposition ===")
    print("   C = ref-miscalibration (true-d V4, d-hat ref); D = V4-corruption (d-hat V4, true-d ref)")
    show = ["regime", "true_d", "n", "dhat_err_absmean",
            "r_integ_A", "r_integ_B", "r_integ_C", "r_integ_D",
            "sign_integ_A", "sign_integ_B", "sign_integ_B_aug"]
    print(sdf[show].round(3).to_string(index=False))
    print("\n=== deployment verdict per regime (mean |Δ| known→inferred; signed drop) ===")
    for regime, g in sdf.groupby("regime"):
        dr = float((g.r_integ_A - g.r_integ_B).abs().mean())
        drop = float((g.r_integ_A - g.r_integ_B).clip(lower=0).mean())  # degradation only
        ds = float((g.sign_integ_A - g.sign_integ_B).abs().mean())
        tag = ("READY" if dr <= 0.05 and ds <= 0.05
               else "DEGRADED" if dr <= 0.15 and ds <= 0.15 else "BLOCKED")
        print(f"  {regime:11s} |Δr|={dr:.3f} (degrade-only {drop:.3f}) |Δsign|={ds:.3f} -> {tag}")


def run_halluc() -> None:
    """Flat-null hallucination check: score a flat torus at wrong assumed d.
    Under-estimation blows up V4 (flat reads curved); over-estimation is mild."""
    import composite_suite as cs
    from diffusion_curvature.menagerie import torus_flat
    ref = cs.load_flatref(embedded=False)
    rows = []
    for true_d in (3, 4, 5):
        m = torus_flat(1400, true_d, rng=99)
        D = m["D"]; D = D / np.median(D[~np.eye(1400, dtype=bool)]); D2 = D ** 2
        order = np.argsort(D, axis=1)
        idx = np.random.default_rng(0).choice(1400, 20, replace=False)
        eigs = [v4.gram_eigs(D2[np.ix_(order[a, :60], order[a, :60])]) for a in idx]
        for assumed in (true_d - 1, true_d, true_d + 1):
            vals = []
            for e in eigs:
                raw = -(e[assumed:].sum() / max(e[:assumed].sum(), 1e-12))
                mu = ref.loc[(1400, assumed), ("v4_m60", "mean")]
                sd = ref.loc[(1400, assumed), ("v4_m60", "std")]
                vals.append((raw - mu) / (abs(sd) + 1e-9))
            rows.append(dict(true_d=true_d, assumed_d=assumed,
                             delta=assumed - true_d,
                             abs_z_median=float(np.median(np.abs(vals)))))
    hdf = pd.DataFrame(rows)
    hdf.to_csv(PROC / "dim_halluc_check.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== FLAT-NULL HALLUCINATION: |z(V4_m60)| of a flat torus at assumed d ===")
    print(hdf.to_string(index=False))
    print("under-estimation (delta=-1) blows up; correct ~0; over-estimation mild")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stageA")
    sub.add_parser("stageB")
    sub.add_parser("summarize")
    sub.add_parser("halluc")
    args = ap.parse_args()
    if args.cmd == "stageA":
        run_stageA()
    elif args.cmd == "stageB":
        run_stageB()
    elif args.cmd == "summarize":
        _summarize_stageB(pd.read_csv(PROC / "dim_stageB_points.csv"))
    elif args.cmd == "halluc":
        run_halluc()


if __name__ == "__main__":
    main()
