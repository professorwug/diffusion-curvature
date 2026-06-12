"""Benchmark the natively-signed Wasserstein curvatures on Colosseum + SadSpheres.

Unlike the earlier iid benchmarks, the battery is generated ONCE with a fixed
seed and persisted, so every method (and every worker) sees identical
manifolds — enabling paired comparisons.

Methods:
  - signed_orc_t8 / signed_orc_phys_t8     (new; one fit, two readouts)
  - signed_orc_t16 / signed_orc_phys_t16   (new)
  - dc_wasserstein_ollivier, orc, laziness_knn_t5, hickock  (baselines, re-run
    on the same manifolds via benchmark_all_on_iid implementations)

Usage:
  pixi run python benchmark_signed_on_iid.py prepare
  pixi run python benchmark_signed_on_iid.py run --worker-id K --num-workers 8
  pixi run python benchmark_signed_on_iid.py merge
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.10")

import joblib
import numpy as np
import pandas as pd

BATTERY_PATH = Path("processed_data/signed_battery.joblib")
OUT_REF = Path("processed_data/signed_metrics.csv")
SHARD_TPL = "processed_data/signed_metrics_w{wid}.csv"
SEED = 1234

COLOSSEUM_KW = dict(
    intrinsic_dims=[2, 3, 4, 5, 6],
    codimensions=[1],
    noise_levels=[0.01, 0.05, 0.1, 0.2],
    num_manifolds_per_dim=30,
    n_samples_rule=lambda d: 3000,
)
SADSPHERES_KW = dict(
    dimension=[2, 3, 4, 5, 6],
    num_pointclouds=20,
    num_points=2000,
    noise_level=0,
    include_planes=True,
)

_ROW_SCHEMA = [
    "dataset", "instance", "method", "ks_hat", "elapsed_s", "err",
    "name", "dim", "codim", "noise", "m", "shape", "ks_true",
]


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------


def prepare(args) -> None:
    from diffusion_curvature.colosseum import CurvatureColosseum
    from diffusion_curvature.sadspheres import SadSpheres

    np.random.seed(SEED)
    instances = []

    print("Building SadSpheres…", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ss = SadSpheres(**SADSPHERES_KW)
    for i in range(len(ss)):
        obj = ss.DS[i].obj
        ks = obj["ks"]
        ks = float(np.mean(np.asarray(ks))) if not np.isscalar(ks) else float(ks)
        name = ss.names[i]
        instances.append(dict(
            dataset="sadspheres", X=np.asarray(obj["X"], dtype=np.float32),
            ks_true=ks, dim=int(obj["d"]), codim=np.nan, noise=0.0, m=np.nan,
            shape=name.split("-", 1)[1] if "-" in name else name, name=name,
        ))
    print(f"  {len(instances)} sadsphere instances", flush=True)

    print("Building Colosseum (slow: sympy + rejection sampling)…", flush=True)
    t0 = time.time()
    cc = CurvatureColosseum(**COLOSSEUM_KW, save_directory=".signed-colosseum")
    for i in range(len(cc)):
        obj = cc.DS[i].obj
        instances.append(dict(
            dataset="colosseum", X=np.asarray(obj["X"], dtype=np.float32),
            ks_true=float(obj["ks"]), dim=int(obj["d"]), codim=int(obj["c"]),
            noise=float(obj["noise"]), m=int(obj["m"]), shape="",
            name=cc.names[i],
        ))
    print(f"  colosseum built in {time.time()-t0:.0f}s; total {len(instances)}",
          flush=True)

    BATTERY_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(instances, BATTERY_PATH, compress=3)
    print(f"wrote {BATTERY_PATH}", flush=True)


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------


def _signed_fits(X: np.ndarray, ts=(8, 16)) -> dict[str, float]:
    from diffusion_curvature.wasserstein_signed import WassersteinSignedCurvature
    out = {}
    for t in ts:
        suffix = "auto" if t == "auto" else f"t{t}"
        est = WassersteinSignedCurvature(
            t=t, knn=10, n_pairs=8, seed=0, compute_midpoint=False,
        )
        est.fit(X=X.astype(np.float64), idx=[0])
        out[f"signed_orc_{suffix}"] = float(est.orc_[0])
        out[f"signed_orc_phys_{suffix}"] = float(est.orc_phys_[0])
    return out


def _baseline_fns():
    """(name, fn) pairs from the v1 benchmark module, run on identical data."""
    import benchmark_all_on_iid as bv1
    return [
        ("dc_wasserstein_ollivier", bv1.run_dc_wasserstein_ollivier),
        ("orc", bv1.run_orc),
        ("laziness_knn_t5", bv1.run_laziness_knn_t5),
        ("hickock", bv1.run_hickock),
    ]


# ---------------------------------------------------------------------------
# Worker / merge (same shard pattern as benchmark_all_on_iid_v2)
# ---------------------------------------------------------------------------


def _existing_done(csv_path: Path) -> set[tuple[str, int, str]]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    df = pd.read_csv(csv_path, usecols=["dataset", "instance", "method"])
    return set(zip(df["dataset"], df["instance"].astype(int), df["method"]))


def _write_row(csv_path: Path, row: dict, header_written: bool) -> bool:
    complete = {k: row.get(k, "") for k in _ROW_SCHEMA}
    pd.DataFrame([complete]).to_csv(
        csv_path, mode="a", index=False, header=not header_written,
    )
    return True


def run_worker(args) -> None:
    instances = joblib.load(BATTERY_PATH)
    done = _existing_done(OUT_REF) | _existing_done(Path(args.out))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header_written = out.exists() and out.stat().st_size > 0
    baselines = _baseline_fns() if not args.skip_baselines else []

    mine = [i for i in range(len(instances)) if i % args.num_workers == args.worker_id]
    print(f"[w{args.worker_id}] {len(mine)} instances "
          f"({len(done)} rows already done)", flush=True)

    t0 = time.time()
    for prog, i in enumerate(mine, 1):
        inst = instances[i]
        X = inst["X"]
        meta = {k: inst[k] for k in
                ("name", "dim", "codim", "noise", "m", "shape", "ks_true")}
        ds = inst["dataset"]

        # --- new signed methods (one fit per t, two readouts) ---
        ts = [t if t == "auto" else int(t) for t in args.ts.split(",")]
        names = [f"signed_orc_{'auto' if t == 'auto' else f't{t}'}" for t in ts]
        new_needed = [
            m for nm in names for m in (nm, nm.replace("orc_", "orc_phys_"))
            if (ds, i, m) not in done
        ]
        if new_needed:
            t_start = time.time()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    vals = _signed_fits(X, ts=ts)
                err = ""
            except Exception as e:
                vals = {m: float("nan") for m in new_needed}
                err = str(e)[:200]
                print(f"  [err] {ds}[{i}] signed: {err}", flush=True)
            el = round(time.time() - t_start, 2)
            for m in new_needed:
                header_written = _write_row(out, {
                    "dataset": ds, "instance": i, "method": m,
                    "ks_hat": vals.get(m, float("nan")),
                    "elapsed_s": el, "err": err, **meta,
                }, header_written)

        # --- baselines ---
        for name, fn in baselines:
            if (ds, i, name) in done:
                continue
            t_start = time.time()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    val = fn(X, int(inst["dim"]))
                err = ""
            except Exception as e:
                val = float("nan")
                err = str(e)[:200]
                print(f"  [err] {ds}[{i}] {name}: {err}", flush=True)
            header_written = _write_row(out, {
                "dataset": ds, "instance": i, "method": name, "ks_hat": val,
                "elapsed_s": round(time.time() - t_start, 2), "err": err, **meta,
            }, header_written)

        rate = prog / max((time.time() - t0) / 60, 1e-9)
        print(f"  [w{args.worker_id}] {prog}/{len(mine)} inst={i} "
              f"({ds} d={inst['dim']}) rate={rate:.2f}/min "
              f"eta={(len(mine)-prog)/max(rate,1e-9):.0f}min", flush=True)

    print(f"[w{args.worker_id}] done in {(time.time()-t0)/60:.1f} min", flush=True)


def run_merge(args) -> None:
    frames = []
    if OUT_REF.exists():
        frames.append(pd.read_csv(OUT_REF))
    for p in sorted(Path("processed_data").glob("signed_metrics_w*.csv")):
        frames.append(pd.read_csv(p))
    if not frames:
        print("nothing to merge")
        return
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(["dataset", "instance", "method"], keep="last")
    df.to_csv(OUT_REF, index=False)
    print(f"wrote {OUT_REF} ({len(df)} rows)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    w = sub.add_parser("run")
    w.add_argument("--worker-id", type=int, required=True)
    w.add_argument("--num-workers", type=int, default=8)
    w.add_argument("--out", default=None)
    w.add_argument("--skip-baselines", action="store_true")
    w.add_argument("--ts", default="8,16",
                   help="comma list of diffusion times; 'auto' allowed")
    sub.add_parser("merge")
    args = p.parse_args()

    if args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "run":
        args.out = args.out or SHARD_TPL.format(wid=args.worker_id)
        run_worker(args)
    else:
        run_merge(args)


if __name__ == "__main__":
    main()
