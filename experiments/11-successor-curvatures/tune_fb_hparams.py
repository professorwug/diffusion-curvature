"""Optuna hyperparameter search for FBTrainer (via SuccessorEntropyCurvature).

Two studies, one per dataset:
- sadspheres : ROC-AUC of per-instance ⟨ks_hat⟩ predicting sphere vs saddle
- colosseum  : Spearman of per-instance ⟨ks_hat⟩ vs true scalar curvature

Each trial trains FB with the suggested params on every instance of a fixed
evaluation config and aggregates the config-level score. Studies are persisted
to SQLite (resumable via `load_if_exists=True`).

Usage:
    # each runs independently — point to a shared or separate SQLite file
    pixi run python tune_fb_hparams.py --dataset sadspheres --n-trials 40
    pixi run python tune_fb_hparams.py --dataset colosseum  --n-trials 40

Best params are also written to `processed_data/optuna/<study>_best.json`.

Defaults favour a small, fast evaluation config (n_samples=500, 10 instances)
suitable for many trials. Scale up with `--n-samples / --num-instances` for
a final, higher-fidelity tuning pass.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import scipy.stats
from sklearn.metrics import roc_auc_score

from diffusion_curvature.successor import SuccessorEntropyCurvature
from diffusion_curvature.tau_datasets import TauColosseum, TauSadSpheres


# ---------------------------------------------------------------------------
# Trial evaluation
# ---------------------------------------------------------------------------


def _score_instance(inst: dict[str, Any], params: dict[str, Any], seed: int) -> float:
    """Train FB on one instance; return per-instance ⟨ks_hat⟩ over visited nodes."""
    sec = SuccessorEntropyCurvature(**params, seed=seed)
    k_hat = sec.fit_transform(X=inst["X"], trajectories=inst["trajectories"])
    visited = np.unique(np.asarray(inst["trajectories_idx"]).ravel())
    vals = np.asarray(k_hat, dtype=float)[visited]
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else float("nan")


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    return dict(
        z_dim=trial.suggest_int("z_dim", 2, 8),
        hidden_dim=trial.suggest_categorical("hidden_dim", [64, 128, 256, 512]),
        gamma=trial.suggest_float("gamma", 0.2, 0.95),
        lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        n_epochs=trial.suggest_int("n_epochs", 50, 300, step=50),
        ortho_coef=trial.suggest_float("ortho_coef", 0.1, 10.0, log=True),
        tau_polyak=trial.suggest_float("tau_polyak", 1e-3, 5e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
    )


def objective_sadspheres(trial: optuna.Trial, ds: TauSadSpheres) -> float:
    params = suggest_params(trial)
    scores: list[float] = []
    labels: list[int] = []
    for i in range(len(ds)):
        inst = ds.get_item(i)
        try:
            s = _score_instance(inst, params, seed=42 + i)
        except Exception as e:
            trial.set_user_attr("error", f"inst {i}: {e}")
            return 0.5
        if not np.isfinite(s):
            return 0.5
        scores.append(s)
        labels.append(1 if float(np.mean(inst["ks"])) > 0 else 0)
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return 0.5


def objective_colosseum(trial: optuna.Trial, ds: TauColosseum) -> float:
    params = suggest_params(trial)
    scores: list[float] = []
    truths: list[float] = []
    for i in range(len(ds)):
        inst = ds.get_item(i)
        try:
            s = _score_instance(inst, params, seed=42 + i)
        except Exception as e:
            trial.set_user_attr("error", f"inst {i}: {e}")
            return 0.0
        if not np.isfinite(s):
            return 0.0
        scores.append(s)
        truths.append(float(np.mean(inst["ks"])))
    try:
        r, _ = scipy.stats.spearmanr(scores, truths)
        return float(r) if np.isfinite(r) else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Dataset construction (cached per-config via Optuna's set_user_attr is overkill;
# we just rebuild once per CLI invocation)
# ---------------------------------------------------------------------------


def build_dataset(
    dataset: str,
    n_samples: int,
    n_trajectories: int,
    traj_length: int,
    num_instances: int,
    seed: int,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if dataset == "sadspheres":
            return TauSadSpheres(
                n_samples=n_samples,
                n_trajectories=n_trajectories,
                traj_length=traj_length,
                dimension=2,
                num_pointclouds=num_instances,
                knn=min(10, n_samples - 2),
                seed=seed,
                save_directory="/tmp/.tau-ss-tune",
            )
        if dataset == "colosseum":
            return TauColosseum(
                n_samples=n_samples,
                n_trajectories=n_trajectories,
                traj_length=traj_length,
                intrinsic_dims=[2],
                codimensions=[1],
                noise_levels=[0.0],
                num_manifolds_per_dim=num_instances,
                knn=min(10, n_samples - 2),
                seed=seed,
                save_directory="/tmp/.tau-cc-tune",
            )
    raise ValueError(dataset)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["sadspheres", "colosseum"], required=True)
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--n-trajectories", type=int, default=200)
    p.add_argument("--traj-length", type=int, default=50)
    p.add_argument("--num-instances", type=int, default=10,
                   help="SadSpheres num_pointclouds or Colosseum num_manifolds_per_dim")
    p.add_argument("--storage", default="processed_data/optuna/studies.db")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    Path(args.storage).parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{args.storage}"
    study_name = f"fb_{args.dataset}"

    print(f"Building {args.dataset} eval set "
          f"(n_samples={args.n_samples}, n_trajectories={args.n_trajectories}, "
          f"traj_length={args.traj_length}, instances={args.num_instances})")
    ds = build_dataset(
        args.dataset, args.n_samples, args.n_trajectories,
        args.traj_length, args.num_instances, args.seed,
    )

    if args.dataset == "sadspheres":
        objective = lambda t: objective_sadspheres(t, ds)
    else:
        objective = lambda t: objective_colosseum(t, ds)

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    n_completed_before = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    print(f"Study '{study_name}' has {n_completed_before} completed trials. "
          f"Running {args.n_trials} more.")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    print(f"\nBest value: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    out = Path(f"processed_data/optuna/{study_name}_best.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {
            "study_name": study_name,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "eval_config": dict(
                n_samples=args.n_samples,
                n_trajectories=args.n_trajectories,
                traj_length=args.traj_length,
                num_instances=args.num_instances,
            ),
        },
        indent=2,
    ))
    print(f"Wrote best config → {out}")


if __name__ == "__main__":
    main()
