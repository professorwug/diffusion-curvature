"""Channel-suite composite with learned integration — the assembly milestone.

Seven signed-curvature channels + four context features per evaluation point,
all computed from a distance matrix alone (menagerie-native), oriented so
higher = more positive curvature, then z-scored against a flat-torus reference
at matching (n, dim). A small gradient-boosted integrator fuses them into a
sign classifier and a magnitude regressor; isotonic-on-best-channel and every
single channel are reported as baselines.

Channels:
  v4_m60, v4_m120 : MDS-strain resid_n (sign=-1), patch m=60 / m=120
  v3_defect       : second-moment defect (cak n/4, t=2, band 2x spread, raw)
  sent            : resolvent successor entropy (gamma=0.97)
  ent_cak         : cak-kernel t=4 diffusion entropy
  frac            : diffusing-edge fraction t=16
  kappa           : Diffusion ORC t=8
Context: est_dim (eigengap), spread (measure radius), knn_radius (density
proxy), patch_frac (m=120 ball radius / median — wrap diagnostic).

Subcommands: flatref | train | e1 | e2 | e3 | e4 | all
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
EXP11 = REPO / "experiments" / "11-successor-curvatures"
sys.path.insert(0, str(EXP11))
sys.path.insert(0, str(HERE))

import v3_defect as v3  # noqa: E402
import v4_mds_strain as v4  # noqa: E402
from sent_channel import sent_at  # noqa: E402
from menagerie_channels import channels_at  # noqa: E402
from diffusion_curvature.menagerie import torus_flat, WarpedProduct  # noqa: E402
import build_battery as bb  # noqa: E402
import random_profiles as rp  # noqa: E402

warnings.filterwarnings("ignore")

PROC = HERE / "processed_data"
PROC.mkdir(exist_ok=True)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# set from CLI --aug; selects the homogeneous-augmented model + suffixed outputs
_AUG = False


def _model_path() -> Path:
    return PROC / ("suite_model_aug.joblib" if _AUG else "suite_model.joblib")


def _out(name: str) -> Path:
    return PROC / (f"suite_{name}_aug.csv" if _AUG else f"suite_{name}.csv")

# channel name -> orientation so that higher = more positive curvature
CHANNELS = {
    "v4_m60": +1, "v4_m120": +1, "v3_defect": +1,
    "sent": -1, "ent_cak": -1, "frac": -1, "kappa": +1,
}
CONTEXT = ["est_dim", "spread", "knn_radius", "patch_frac"]
FEATURES = list(CHANNELS) + CONTEXT
N_BUCKETS = [1400, 2000, 3000]


# ---------------------------------------------------------------------------
# Feature extraction (raw, oriented; z-scoring applied later with flat ref)
# ---------------------------------------------------------------------------

def _norm_D(D: np.ndarray) -> np.ndarray:
    off = D[~np.eye(D.shape[0], dtype=bool)]
    s = float(np.median(off))
    return D / max(s, 1e-12)


def extract_raw(D: np.ndarray, anchors: np.ndarray, true_dim: int,
                device: str = DEVICE) -> pd.DataFrame:
    """Per-anchor oriented raw channel + context values from a distance matrix."""
    D = _norm_D(np.asarray(D, dtype=float))
    n = D.shape[0]
    anchors = np.asarray(anchors)
    D2 = D ** 2
    order = np.argsort(D, axis=1)

    # --- V4 MDS-strain (m=60, m=120) + est_dim + patch_frac ---
    v4_m60 = np.full(len(anchors), np.nan)
    v4_m120 = np.full(len(anchors), np.nan)
    est_dim = np.full(len(anchors), np.nan)
    patch_frac = np.full(len(anchors), np.nan)
    knn_radius = np.full(len(anchors), np.nan)
    for q, a in enumerate(anchors):
        for m, store in ((60, "60"), (min(120, n - 1), "120")):
            patch = order[a, :m]
            eigs = v4.gram_eigs(D2[np.ix_(patch, patch)])
            rn = v4.readouts(eigs, true_dim)["resid_n"]
            if store == "60":
                v4_m60[q] = -rn
                est_dim[q] = v4.estimate_rank(eigs, 8)
            else:
                v4_m120[q] = -rn
                patch_frac[q] = float(D[a, patch[-1]])
        knn_radius[q] = float(D[a, order[a, 10]])

    # --- V3 second-moment defect (cak n/4, t=2, band 2x spread, raw) ---
    P = v3.build_P(D, "cak_n4")
    M = v3.diffusion_powers(P, [2])[2]
    spread = v3.measure_spread(M, D)
    delta = v3.defect_matrix(M, D2)
    v3sc = v3.point_scores(delta, D, anchors, 2.0 * spread, "raw", spread, 16)
    spread_i = (M[anchors] * D[anchors]).sum(axis=1)

    # --- entropy / fraction / ORC (device) ---
    try:
        ch = channels_at(D, anchors, device)
        kappa, frac, ent_cak = ch["kappa"], ch["frac"], ch["ent4"]
    except Exception:
        kappa = frac = ent_cak = np.full(len(anchors), np.nan)
    try:
        sent = sent_at(D, anchors, device)
    except Exception:
        sent = np.full(len(anchors), np.nan)

    raw = dict(v4_m60=v4_m60, v4_m120=v4_m120, v3_defect=np.asarray(v3sc),
               sent=np.asarray(sent, float), ent_cak=np.asarray(ent_cak, float),
               frac=np.asarray(frac, float), kappa=np.asarray(kappa, float),
               est_dim=est_dim, spread=spread_i, knn_radius=knn_radius,
               patch_frac=patch_frac)
    df = pd.DataFrame(raw)
    for c, orient in CHANNELS.items():
        df[c] = orient * df[c]
    df["anchor"] = anchors
    df["n_points"] = n
    df["true_dim"] = true_dim
    return df


def _bucket(n: int) -> int:
    return int(min(N_BUCKETS, key=lambda b: abs(b - n)))


# ---------------------------------------------------------------------------
# Flat-torus reference table
# ---------------------------------------------------------------------------

def run_flatref(reps: int = 8) -> None:
    buckets = {1400: [3, 4, 5, 6], 2000: [2, 3, 4, 5, 6], 3000: [2, 3, 4, 5, 6]}
    t0 = time.time()
    rows = []
    for n, dims in buckets.items():
        for d in dims:
            for rep in range(reps):
                rng = np.random.default_rng(7000 + 13 * d + rep)
                m = torus_flat(n, d, rng=rng)
                D = m["D"]
                anchors = rng.choice(n, 3, replace=False)
                fr = extract_raw(D, anchors, d)
                fr["n_bucket"] = n
                rows.append(fr)
            print(f"[flatref] n={n} d={d} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.concat(rows, ignore_index=True)
    stats = df.groupby(["n_bucket", "true_dim"])[list(CHANNELS)].agg(
        ["mean", "std"])
    stats.to_pickle(PROC / "suite_flatref.pkl")
    print(f"[flatref] {len(df)} rows, {len(stats)} cells in {time.time()-t0:.0f}s")


def load_flatref() -> pd.DataFrame:
    return pd.read_pickle(PROC / "suite_flatref.pkl")


def apply_flatz(df: pd.DataFrame, flatref: pd.DataFrame) -> pd.DataFrame:
    """z-score each channel against the flat-torus reference at matching (n,d)."""
    df = df.copy()
    for i, row in df.iterrows():
        key = (_bucket(int(row["n_points"])), int(row["true_dim"]))
        for c in CHANNELS:
            try:
                mu = flatref.loc[key, (c, "mean")]
                sd = flatref.loc[key, (c, "std")]
            except KeyError:
                mu, sd = 0.0, 1.0
            df.at[i, c] = (row[c] - mu) / (abs(sd) + 1e-9)
    return df


# ---------------------------------------------------------------------------
# Training on random-profile manifolds
# ---------------------------------------------------------------------------

def _build_random_manifold(idx: int, d: int, n: int, seed: int,
                           n_grid_r: int = 360, n_grid_phi: int = 220) -> dict:
    rng = np.random.default_rng(seed)
    f, L, periodic, params = rp.random_profile(rng)
    wp = WarpedProduct(f, L, d, n_grid_r=n_grid_r, n_grid_phi=n_grid_phi,
                       periodic=periodic)
    m = wp.sample(n, rng=rng)
    D, ks = v3._normalize(np.asarray(m["D"], float),
                          np.asarray(m["ks_field"], float))
    return dict(D=D, ks=ks, params=params, dim=d)


def _homogeneous_training(n: int, n_eval: int) -> list:
    """Constant-curvature manifolds (sphere +K, hyperbolic -K, torus 0) at
    varied magnitude — teaches the integrator the homogeneous / cross-manifold
    regime that warped signed fields never present."""
    from diffusion_curvature.menagerie import sphere_sf, hyperbolic
    specs = []
    for d in (2, 3, 4, 5, 6):
        for r in (0.6, 1.0, 1.6):
            specs.append(("sphere", d, dict(r=r)))
        for kappa in (0.5, 1.0, 2.0):
            specs.append(("hyperbolic", d, dict(kappa=kappa)))
        specs.append(("torus", d, {}))
    frames = []
    for j, (kind, d, kw) in enumerate(specs):
        rng = np.random.default_rng(50000 + j)
        if kind == "sphere":
            m = sphere_sf(n, d, rng=rng, **kw)
        elif kind == "hyperbolic":
            m = hyperbolic(n, d, rng=rng, **kw)
        else:
            m = torus_flat(n, d, rng=rng)
        D, ks = v3._normalize(np.asarray(m["D"], float),
                              np.asarray(m["ks_field"], float))
        eidx = rng.choice(n, min(n_eval, n), replace=False)
        fr = extract_raw(D, eidx, d)
        fr["ks"] = ks[eidx]
        fr["manifold"] = 10000 + j
        fr["family"] = f"homog_{kind}"
        frames.append(fr)
    return frames


def run_train(n_manifolds: int = 72, n: int = 1400, n_eval: int = 40,
              aug: bool = False, reuse: bool = False) -> None:
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                   HistGradientBoostingRegressor)
    from sklearn.isotonic import IsotonicRegression
    flatref = load_flatref()
    dims = [3, 4, 5, 6]
    t0 = time.time()

    reuse_csv = PROC / "suite_train_points.csv"
    if reuse and reuse_csv.exists():
        # warped features already computed & flat-z-scored; only add homogeneous
        dfz_warp = pd.read_csv(reuse_csv)
        print(f"[train] reusing {len(dfz_warp)} warped feature rows", flush=True)
        homog = _homogeneous_training(n, n_eval)
        df_homog = apply_flatz(pd.concat(homog, ignore_index=True), flatref)
        dfz = pd.concat([dfz_warp, df_homog], ignore_index=True)
        df = dfz  # for the manifold/family bookkeeping below
        _finish_train(dfz, df, aug, t0, HistGradientBoostingClassifier,
                      HistGradientBoostingRegressor, IsotonicRegression)
        return

    frames = []
    for i in range(n_manifolds):
        d = dims[i % len(dims)]
        seed = 40000 + i
        try:
            man = _build_random_manifold(i, d, n, seed)
        except Exception as e:
            print(f"  [train] manifold {i} d{d} failed: {str(e)[:80]}", flush=True)
            continue
        ks = man["ks"]
        eidx = v3.pick_eval_points(ks, n_eval, np.random.default_rng(seed))
        fr = extract_raw(man["D"], eidx, d)
        fr["ks"] = ks[eidx]
        fr["manifold"] = i
        fr["family"] = man["params"]["family"]
        frames.append(fr)
        if (i + 1) % 8 == 0:
            print(f"[train] {i+1}/{n_manifolds} ({time.time()-t0:.0f}s)", flush=True)
    if aug:
        print("[train] adding homogeneous constant-K manifolds", flush=True)
        frames += _homogeneous_training(n, n_eval)
    df = pd.concat(frames, ignore_index=True)
    dfz = apply_flatz(df, flatref)
    _finish_train(dfz, df, aug, t0, HistGradientBoostingClassifier,
                  HistGradientBoostingRegressor, IsotonicRegression)


def _finish_train(dfz, df, aug, t0, ClfCls, RegCls, IsoCls) -> None:
    # labels: drop near-zero |ks| for the sign head; keep all for magnitude
    ks = dfz["ks"].values
    eps = 0.05 * np.median(np.abs(ks))
    sign_mask = np.abs(ks) > eps
    X = dfz[FEATURES].values
    y_sign = (ks > 0).astype(int)
    y_mag = ks / (np.std(ks) + 1e-9)

    clf = ClfCls(max_depth=4, max_iter=300, learning_rate=0.06,
                 l2_regularization=1.0, random_state=0)
    clf.fit(X[sign_mask], y_sign[sign_mask])
    reg = RegCls(max_depth=4, max_iter=300, learning_rate=0.06,
                 l2_regularization=1.0, random_state=0)
    reg.fit(X, y_mag)

    # isotonic on best single channel (by train sign accuracy)
    best_ch, best_acc = None, -1
    for c in CHANNELS:
        v = dfz[c].values
        m = np.isfinite(v) & sign_mask
        acc = max(((v[m] > 0) == y_sign[m]).mean(),
                  ((v[m] < 0) == y_sign[m]).mean())
        if acc > best_acc:
            best_acc, best_ch = acc, c
    iso = IsoCls(out_of_bounds="clip")
    vv = dfz[best_ch].values
    mm = np.isfinite(vv)
    iso.fit(vv[mm], y_mag[mm])

    perm = _perm_importance(clf, X[sign_mask], y_sign[sign_mask])
    mpath = PROC / ("suite_model_aug.joblib" if aug else "suite_model.joblib")
    joblib.dump(dict(clf=clf, reg=reg, iso=iso, best_ch=best_ch,
                     features=FEATURES, perm=perm), mpath)
    if not aug:
        dfz.to_csv(PROC / "suite_train_points.csv", index=False)
    else:
        dfz.to_csv(PROC / "suite_train_points_aug.csv", index=False)
    fam = sorted(df.family.unique()) if "family" in df else "n/a"
    print(f"[train] {len(dfz)} points, families={fam}")
    print(f"[train] best single channel = {best_ch} (train sign acc {best_acc:.3f})")
    print("[train] permutation importance (sign head):")
    for c, imp in sorted(perm.items(), key=lambda kv: -kv[1]):
        print(f"    {c:<12} {imp:+.4f}")
    print(f"[train] done in {time.time()-t0:.0f}s")


def _perm_importance(clf, X, y, n_rep: int = 5) -> dict:
    from sklearn.metrics import accuracy_score
    rng = np.random.default_rng(0)
    base = accuracy_score(y, clf.predict(X))
    out = {}
    for j, c in enumerate(FEATURES):
        drops = []
        for _ in range(n_rep):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - accuracy_score(y, clf.predict(Xp)))
        out[c] = float(np.mean(drops))
    return out


# ---------------------------------------------------------------------------
# Prediction + scoring helpers
# ---------------------------------------------------------------------------

def predict(dfz: pd.DataFrame, model: dict) -> dict:
    X = dfz[model["features"]].values
    p_sign = model["clf"].predict_proba(X)[:, 1] - 0.5  # >0 => +K
    mag = model["reg"].predict(X)
    iso = model["iso"].predict(dfz[model["best_ch"]].values) - 0.5
    return dict(integ_sign=p_sign, integ_mag=mag, iso=iso)


def _pearson(a, b):
    return v3.pearson(np.asarray(a, float), np.asarray(b, float))


def _sign_acc(score, ks):
    score, ks = np.asarray(score, float), np.asarray(ks, float)
    m = np.isfinite(score) & np.isfinite(ks) & (np.abs(ks) > 1e-9)
    if m.sum() == 0:
        return np.nan
    return v3.balanced_sign(score[m], ks[m])


# ---------------------------------------------------------------------------
# E1 — held-out battery families (menagerie dumbbell beta=0.8 / necklace b=0.7)
# ---------------------------------------------------------------------------

def run_e1() -> None:
    model = joblib.load(_model_path())
    flatref = load_flatref()
    n, n_eval, seeds = 1400, 40, [0, 1]
    t0 = time.time()
    rows = []
    for kind in ("dumbbell", "necklace"):
        for d in (3, 4, 5, 6):
            for seed in seeds:
                man = v4.build_manifold(kind, d, n, seed)
                D0, ks = man["D"], man["ks"]
                eidx = v3.pick_eval_points(ks, n_eval, np.random.default_rng(2000 + seed))
                for noise in (0.0, 0.15):
                    D = v3._apply_noise(D0, noise, np.random.default_rng(3000 + seed))
                    fr = extract_raw(D, eidx, d)
                    fr["ks"] = ks[eidx]
                    fr = apply_flatz(fr, flatref)
                    pr = predict(fr, model)
                    for q in range(len(eidx)):
                        rows.append(dict(kind=kind, dim=d, noise=noise, seed=seed,
                                         ks=float(fr.ks.values[q]),
                                         integ_sign=pr["integ_sign"][q],
                                         integ_mag=pr["integ_mag"][q],
                                         iso=pr["iso"][q],
                                         **{c: fr[c].values[q] for c in CHANNELS}))
            print(f"[e1] {kind}{d} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(_out("e1_points"), index=False)
    _report_field(df, "E1 held-out families", _out("e1_summary"))


def _report_field(df: pd.DataFrame, title: str, out: Path) -> None:
    """Field Pearson + balanced sign per (kind,dim,noise): integrator, iso,
    single channels."""
    rows = []
    cols = ["integ_mag", "iso"] + list(CHANNELS)
    for (kind, d, noise), g in df.groupby(["kind", "dim", "noise"]):
        rec = dict(kind=kind, dim=d, noise=noise)
        rec["r_integ"] = _pearson(g.integ_mag, g.ks)
        rec["sign_integ"] = _sign_acc(g.integ_sign, g.ks)
        rec["r_iso"] = _pearson(g.iso, g.ks)
        for c in CHANNELS:
            rec[f"r_{c}"] = _pearson(g[c], g.ks)
            rec[f"sign_{c}"] = _sign_acc(g[c], g.ks)
        # best single channel per cell (by |r|)
        rs = {c: rec[f"r_{c}"] for c in CHANNELS}
        rec["best_ch"] = max(rs, key=lambda c: abs(rs[c]) if np.isfinite(rs[c]) else -1)
        rec["r_best_single"] = rs[rec["best_ch"]]
        rows.append(rec)
    sdf = pd.DataFrame(rows).sort_values(["kind", "dim", "noise"])
    sdf.to_csv(out, index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", None)
    show = ["kind", "dim", "noise", "r_integ", "sign_integ", "r_iso",
            "r_v4_m60", "sign_v4_m60", "r_v4_m120", "sign_v4_m120",
            "r_v3_defect", "r_sent", "best_ch", "r_best_single"]
    print(f"\n=== {title} ===")
    print(sdf[show].round(3).to_string(index=False))


# ---------------------------------------------------------------------------
# E2 — tier1v2 homogeneous (equal density sphere / hyperbolic / torus)
# ---------------------------------------------------------------------------

def run_e2() -> None:
    model = joblib.load(_model_path())
    flatref = load_flatref()
    recs = bb.tier1_v2_recipes()
    t0 = time.time()
    rows = []
    for r in recs:
        rec = dict(r); rec.setdefault("seed", 0); rec.setdefault("noise", 0.0)
        try:
            inst = bb.materialize(rec)
        except Exception as e:
            print(f"  [e2] {rec['kind']}d{rec['dim']} failed: {str(e)[:80]}")
            continue
        eidx = inst["eval_idx"]
        fr = extract_raw(inst["D"], eidx, rec["dim"])
        fr["ks"] = inst["ks_field"][eidx]
        fr = apply_flatz(fr, flatref)
        pr = predict(fr, model)
        for q in range(len(eidx)):
            rows.append(dict(kind=rec["kind"], dim=rec["dim"],
                             ks=float(fr.ks.values[q]),
                             integ_sign=pr["integ_sign"][q],
                             integ_mag=pr["integ_mag"][q], iso=pr["iso"][q],
                             **{c: fr[c].values[q] for c in CHANNELS}))
        print(f"[e2] {rec['kind']}d{rec['dim']} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(_out("e2_points"), index=False)
    # cross-manifold per dim: balanced sign of integrator vs sign(ks)
    rows2 = []
    for d, g in df.groupby("dim"):
        rec = dict(dim=d, n_pts=len(g))
        rec["sign_integ"] = _sign_acc(g.integ_sign, g.ks)
        rec["r_integ"] = _pearson(g.integ_mag, g.ks)
        for c in CHANNELS:
            rec[f"sign_{c}"] = _sign_acc(g[c], g.ks)
        rows2.append(rec)
    sdf = pd.DataFrame(rows2)
    sdf.to_csv(_out("e2_summary"), index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", None)
    print("\n=== E2 tier1v2 homogeneous (cross-manifold balanced sign per dim) ===")
    print(sdf.round(3).to_string(index=False))


# ---------------------------------------------------------------------------
# E3 / E4 — battery transfer (colosseum, sadspheres)
# ---------------------------------------------------------------------------

def _battery_features(instances, keep_fn, device: str,
                      ckpt: Path) -> pd.DataFrame:
    """Origin-anchored raw features per battery instance, checkpointed to `ckpt`
    (append per instance, resumable) so a kill loses at most one instance."""
    t0 = time.time()
    done = set()
    if ckpt.exists() and ckpt.stat().st_size > 0:
        done = set(pd.read_csv(ckpt, usecols=["name"]).name.astype(str))
    header = not (ckpt.exists() and ckpt.stat().st_size > 0)
    n_new = 0
    for i, inst in enumerate(instances):
        if not keep_fn(inst) or str(inst.get("name", "")) in done:
            continue
        X = np.asarray(inst["X"], dtype=np.float64)
        Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
        D = torch.cdist(Xt, Xt).cpu().numpy().astype(np.float64)
        try:
            fr = extract_raw(D, np.array([0]), int(inst["dim"]), device)
        except Exception as e:
            print(f"  [batt] {i} failed: {str(e)[:80]}", flush=True)
            continue
        fr["ks"] = float(inst["ks_true"])
        fr["dim"] = int(inst["dim"])
        fr["shape"] = inst.get("shape", "")
        fr["name"] = str(inst.get("name", ""))
        fr.to_csv(ckpt, mode="a", index=False, header=header)
        header = False
        n_new += 1
        if n_new % 25 == 0:
            print(f"  [batt] {n_new} new ({time.time()-t0:.0f}s)", flush=True)
    return pd.read_csv(ckpt)


def run_e3(device: str = DEVICE) -> None:
    model = joblib.load(_model_path())
    flatref = load_flatref()
    instances = joblib.load(EXP11 / "processed_data" / "signed_battery.joblib")
    df = _battery_features(
        instances, lambda z: z["dataset"] == "colosseum" and z["m"] < 15, device,
        PROC / "suite_e3_raw.csv")
    dfz = apply_flatz(df, flatref)
    pr = predict(dfz, model)
    for k, v in pr.items():
        dfz[k] = v
    dfz.to_csv(_out("e3_points"), index=False)
    rows = []
    for d, g in dfz.groupby("dim"):
        rec = dict(dim=int(d), n=len(g))
        rec["r_integ"] = _pearson(g.integ_mag, g.ks)
        rec["sign_integ"] = _sign_acc(g.integ_sign, g.ks)
        rec["r_iso"] = _pearson(g.iso, g.ks)
        for c in CHANNELS:
            rec[f"r_{c}"] = _pearson(g[c], g.ks)
            rec[f"sign_{c}"] = _sign_acc(g[c], g.ks)
        rows.append(rec)
    sdf = pd.DataFrame(rows)
    sdf.to_csv(_out("e3_summary"), index=False)
    pd.set_option("display.width", 280); pd.set_option("display.max_columns", None)
    print("\n=== E3 COLOSSEUM TRANSFER (per-dim Pearson | balanced sign) ===")
    print("v6 baseline Pearson .60/.53/.80/.51/.64 sign .70/.51/.52/.76/.93 (d2-6)")
    show = ["dim", "n", "r_integ", "sign_integ", "r_iso",
            "r_v4_m60", "sign_v4_m60", "r_v4_m120", "sign_v4_m120",
            "r_v3_defect", "r_kappa", "sign_kappa", "r_sent", "r_ent_cak"]
    print(sdf[show].round(3).to_string(index=False))


def run_e4(device: str = DEVICE) -> None:
    from sklearn.metrics import roc_auc_score
    model = joblib.load(_model_path())
    flatref = load_flatref()
    instances = joblib.load(EXP11 / "processed_data" / "signed_battery.joblib")
    df = _battery_features(instances, lambda z: z["dataset"] == "sadspheres", device,
                          PROC / "suite_e4_raw.csv")
    dfz = apply_flatz(df, flatref)
    pr = predict(dfz, model)
    for k, v in pr.items():
        dfz[k] = v
    dfz.to_csv(_out("e4_points"), index=False)

    def auc(score, ks):
        score, ks = np.asarray(score, float), np.asarray(ks, float)
        m = np.isfinite(score) & (np.abs(ks) > 1e-9)
        y = (ks[m] > 0).astype(int)
        if len(np.unique(y)) < 2:
            return np.nan
        return roc_auc_score(y, score[m])

    rows = []
    for d, g in dfz.groupby("dim"):
        rec = dict(dim=int(d), n=len(g))
        rec["auc_integ"] = auc(g.integ_sign, g.ks)
        rec["sign_integ"] = _sign_acc(g.integ_sign, g.ks)
        for c in CHANNELS:
            rec[f"auc_{c}"] = auc(g[c], g.ks)
        rows.append(rec)
    sdf = pd.DataFrame(rows)
    sdf.to_csv(_out("e4_summary"), index=False)
    pd.set_option("display.width", 280); pd.set_option("display.max_columns", None)
    print("\n=== E4 SADSPHERES (AUC + balanced sign-at-zero) ===")
    show = ["dim", "n", "auc_integ", "sign_integ", "auc_v4_m60", "auc_v4_m120",
            "auc_v3_defect", "auc_kappa", "auc_sent", "auc_ent_cak", "auc_frac"]
    print(sdf[show].round(3).to_string(index=False))


def main() -> None:
    global _AUG
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", action="store_true",
                    help="use the homogeneous-augmented model + suffixed outputs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("flatref")
    tr = sub.add_parser("train"); tr.add_argument("--n-manifolds", type=int, default=72)
    tr.add_argument("--reuse", action="store_true",
                    help="reuse cached warped features from suite_train_points.csv")
    sub.add_parser("e1"); sub.add_parser("e2")
    e3 = sub.add_parser("e3"); e3.add_argument("--device", default=DEVICE)
    e4 = sub.add_parser("e4"); e4.add_argument("--device", default=DEVICE)
    args = ap.parse_args()
    _AUG = args.aug
    if args.cmd == "flatref":
        run_flatref()
    elif args.cmd == "train":
        run_train(n_manifolds=args.n_manifolds, aug=args.aug, reuse=args.reuse)
    elif args.cmd == "e1":
        run_e1()
    elif args.cmd == "e2":
        run_e2()
    elif args.cmd == "e3":
        run_e3(device=args.device)
    elif args.cmd == "e4":
        run_e4(device=args.device)


if __name__ == "__main__":
    main()
