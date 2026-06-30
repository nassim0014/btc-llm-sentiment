"""Run Optuna hyperparameter search end-to-end.

This script:
  1. Loads the feature bundle from Notebook 03
  2. Reconstructs the full close-price array
  3. Runs the Optuna search with walk-forward CV objective
  4. Saves best params to outputs/best_optuna_params.json

Usage:
    python3 scripts/run_optuna_search.py [--n-trials 15] [--epochs 15]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

INTERIM = ROOT / "notebooks" / "interim"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)


def main(n_trials: int = 15, epochs: int = 15) -> None:
    print("=" * 60)
    print("Optuna Walk-Forward Hyperparameter Search")
    print("=" * 60)

    # Load features — use safe loader with SHA256 integrity check
    from src.utils.safe_pickle import safe_load_bundle
    bundle = safe_load_bundle()

    X = np.concatenate([bundle["train_x"], bundle["val_x"], bundle["test_x"]], axis=0)
    y = np.concatenate([bundle["train_y"], bundle["val_y"], bundle["test_y"]], axis=0)
    dates = np.concatenate([bundle["train_dates"], bundle["val_dates"], bundle["test_dates"]], axis=0)

    merged = pd.read_parquet(INTERIM / "merged_with_llm_sentiment.parquet").sort_values("date").reset_index(drop=True)
    close = merged["close"].values

    n_features = X.shape[-1]
    print(f"X: {X.shape}  y: {y.shape}  close: {close.shape}  n_features: {n_features}")

    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights, strict=False)}
    print(f"Class weights: {class_weight}")

    from src.cv.optuna_search import run_optuna_search

    best_params = run_optuna_search(
        X=X, y=y, close=close, dates=dates,
        n_features=n_features,
        n_trials=n_trials,
        n_folds=5,
        min_train=400,
        val_size=60,
        epochs=epochs,
        batch_size=32,
        class_weight=class_weight,
        fee=0.001,
        threshold=0.5,
        pruner_n_startup=3,
        pruner_n_warmup=2,
        output_path=OUTPUTS / "best_optuna_params.json",
        seed=42,
    )

    print(f"\nBest params: {best_params}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()
    main(n_trials=args.n_trials, epochs=args.epochs)
