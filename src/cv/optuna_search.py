"""
Optuna hyperparameter search with walk-forward CV objective.

The objective function maximizes the **mean Out-Of-Fold (OOF) Sharpe Ratio**
across 5 expanding-window folds. This ensures the selected hyperparameters
generalize across market regimes, not just a single train/val split.

Uses Optuna's `MedianPruner` to aggressively prune underperforming trials
after each fold — if the cumulative OOF Sharpe is below the median of
completed trials at the same fold step, the trial is killed early.

Public API
----------
- `sample_hyperparameters`: sample one trial from the search space.
- `optuna_objective`: objective function compatible with `optuna.study.optimize`.
- `run_optuna_search`: end-to-end helper that creates a study, runs the
  search, and returns the best hyperparameters.

Search Space
------------
    lr:         float in [1e-4, 1e-2] (log-uniform)
    units:      int in {32, 64, 128}
    dropout:    float in {0.0, 0.2, 0.4}
    num_layers: int in {1, 2}
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import optuna

from src.cv.walk_forward import (
    walk_forward_splits,
    evaluate_oof_metrics,
    WalkForwardResult,
)
from src.models.lstm import make_model_factory

logger = logging.getLogger(__name__)

# Suppress TF logs during search
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


# ---------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------
def sample_hyperparameters(trial) -> dict:
    """Sample one set of hyperparameters from the Optuna search space."""
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "units": trial.suggest_categorical("units", [32, 64, 128]),
        "dropout": trial.suggest_categorical("dropout", [0.0, 0.2, 0.4]),
        "num_layers": trial.suggest_categorical("num_layers", [1, 2]),
    }


# ---------------------------------------------------------------------
# Objective — walk-forward CV with per-fold pruning
# ---------------------------------------------------------------------
def optuna_objective(
    trial,
    X: np.ndarray,
    y: np.ndarray,
    close: np.ndarray,
    dates: np.ndarray,
    n_features: int,
    n_folds: int = 5,
    min_train: int = 400,
    val_size: int = 60,
    epochs: int = 15,
    batch_size: int = 32,
    class_weight: Optional[dict] = None,
    fee: float = 0.001,
    threshold: float = 0.5,
) -> float:
    """Optuna objective: maximize mean OOF Sharpe across walk-forward folds.

    Uses `walk_forward_splits()` from Step 1. After each fold, reports the
    cumulative mean OOF Sharpe to Optuna for pruning. If the trial is pruned,
    raises `optuna.TrialPruned` (caught by Optuna internally).

    Parameters
    ----------
    trial : optuna.Trial
    X, y, close, dates : np.ndarray
        Full dataset (train+val+test concatenated). The walk-forward splits
        operate on indices into this array.
    n_features : int
        Number of input features (X.shape[-1]).
    n_folds, min_train, val_size : int
        Walk-forward CV parameters.
    epochs : int
        Max epochs per fold (EarlyStopping may reduce this).
    batch_size : int
    class_weight : dict, optional
    fee, threshold : float
        Backtest parameters for OOF Sharpe computation.

    Returns
    -------
    float
        Mean OOF Sharpe ratio across all folds.
    """
    import tensorflow as tf
    from tensorflow.keras import callbacks

    tf.get_logger().setLevel("ERROR")

    hp = sample_hyperparameters(trial)
    model_factory = make_model_factory(hp, n_features)

    fold_sharpes: list[float] = []
    n = len(X)

    for fold in walk_forward_splits(n=n, n_folds=n_folds, min_train=min_train, val_size=val_size):
        X_tr, y_tr = X[fold.train_idx], y[fold.train_idx]
        X_va, y_va = X[fold.val_idx], y[fold.val_idx]
        close_va = close[fold.val_idx]

        # Build a fresh model for each fold
        tf.keras.backend.clear_session()
        model = model_factory()

        cb = [
            callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=5, restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
            ),
        ]

        model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            class_weight=class_weight,
            callbacks=cb,
        )

        y_prob_va = model.predict(X_va, verbose=0).ravel()
        metrics = evaluate_oof_metrics(
            y_true=y_va, y_prob=y_prob_va,
            close=close_va, threshold=threshold, fee=fee,
        )
        fold_sharpes.append(metrics["sharpe"])

        # Report cumulative mean OOF Sharpe to Optuna for pruning
        cumulative_mean = float(np.mean(fold_sharpes))
        trial.report(cumulative_mean, step=fold.fold_num)

        # If Optuna decides to prune, raise immediately
        if trial.should_prune():
            logger.info(
                f"Trial {trial.number} pruned after fold {fold.fold_num} "
                f"(cumulative OOF Sharpe = {cumulative_mean:+.3f})"
            )
            raise optuna.TrialPruned()

    mean_sharpe = float(np.mean(fold_sharpes))
    logger.info(
        f"Trial {trial.number} completed: hp={hp}  "
        f"OOF Sharpe = {mean_sharpe:+.3f}  "
        f"per-fold = {[f'{s:+.2f}' for s in fold_sharpes]}"
    )
    return mean_sharpe


# ---------------------------------------------------------------------
# End-to-end search runner
# ---------------------------------------------------------------------
def run_optuna_search(
    X: np.ndarray,
    y: np.ndarray,
    close: np.ndarray,
    dates: np.ndarray,
    n_features: int,
    n_trials: int = 15,
    n_folds: int = 5,
    min_train: int = 400,
    val_size: int = 60,
    epochs: int = 15,
    batch_size: int = 32,
    class_weight: Optional[dict] = None,
    fee: float = 0.001,
    threshold: float = 0.5,
    pruner_n_startup: int = 3,
    pruner_n_warmup: int = 2,
    output_path: Optional[Path] = None,
    seed: int = 42,
) -> dict:
    """Run a full Optuna hyperparameter search.

    Parameters
    ----------
    X, y, close, dates : np.ndarray
        Full dataset.
    n_features : int
    n_trials : int
        Number of Optuna trials. Default 15.
    n_folds, min_train, val_size : int
        Walk-forward CV parameters.
    epochs, batch_size : int
        Per-fold training parameters.
    class_weight : dict, optional
    fee, threshold : float
        Backtest parameters for OOF Sharpe.
    pruner_n_startup : int
        Number of trials to run without pruning (warmup). Default 3.
    pruner_n_warmup : int
        Number of folds to complete before pruning kicks in. Default 2.
    output_path : Path, optional
        If provided, save best params as JSON to this path.
    seed : int

    Returns
    -------
    dict
        Best hyperparameters.
    """
    import optuna
    import tensorflow as tf

    optuna.logging.set_verbosity(optuna.logging.INFO)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # MedianPruner: aggressively prune trials whose intermediate values
    # are below the median of completed trials at the same step.
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=pruner_n_startup,
        n_warmup_steps=pruner_n_warmup,
        interval_steps=1,
    )

    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        study_name="lstm_walk_forward_sharpe",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    # Suppress TF per-trial logging
    tf.get_logger().setLevel("ERROR")

    study.optimize(
        lambda trial: optuna_objective(
            trial=trial,
            X=X, y=y, close=close, dates=dates,
            n_features=n_features,
            n_folds=n_folds,
            min_train=min_train,
            val_size=val_size,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            fee=fee,
            threshold=threshold,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    best_params = best.params
    best_value = best.value

    print(f"\n{'='*60}")
    print(f"Optuna search complete — {len(study.trials)} trials")
    print(f"Best OOF Sharpe: {best_value:+.4f}")
    print(f"Best params: {best_params}")
    print(f"{'='*60}")

    # Save to JSON
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "best_params": best_params,
            "best_oof_sharpe": float(best_value),
            "n_trials": len(study.trials),
            "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "n_complete": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "search_space": {
                "lr": "log-uniform [1e-4, 1e-2]",
                "units": [32, 64, 128],
                "dropout": [0.0, 0.2, 0.4],
                "num_layers": [1, 2],
            },
            "walk_forward_config": {
                "n_folds": n_folds,
                "min_train": min_train,
                "val_size": val_size,
            },
        }
        with output_path.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved best params → {output_path}")

    return best_params
