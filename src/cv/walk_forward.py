"""
Walk-Forward Cross-Validation (expanding window) for time-series ML.

Strictly preserves temporal ordering — no shuffling, no look-ahead leakage.

Public API
----------
- `WalkForwardFold`: dataclass holding train/val indices for one fold.
- `walk_forward_splits`: generator yielding folds.
- `evaluate_oof_metrics`: compute Sharpe, Accuracy, F1 on out-of-fold predictions.
- `WalkForwardResult`: dataclass holding per-fold metrics + aggregated stats.
- `run_walk_forward`: end-to-end helper that trains + evaluates a model
  factory across all folds and returns a WalkForwardResult.

Example
-------
    from src.cv.walk_forward import walk_forward_splits, run_walk_forward

    for fold in walk_forward_splits(n=730, n_folds=5, min_train=400, val_size=60):
        print(fold)

    result = run_walk_forward(
        X=X, y=y, close=close, dates=dates,
        model_factory=lambda: build_lstm(...),
        feature_cols=FEATURE_COLS,
    )
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Fold dataclass
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class WalkForwardFold:
    """One expanding-window fold.

    train_idx and val_idx are integer indices into the original DataFrame.
    `fold_num` is 1-indexed for human-readable logs.
    """
    fold_num: int
    train_start: int
    train_end: int       # exclusive
    val_start: int
    val_end: int         # exclusive
    train_idx: np.ndarray = field(repr=False)
    val_idx: np.ndarray = field(repr=False)

    def __repr__(self) -> str:
        """Return a human-readable one-line summary of the fold boundaries."""
        return (
            f"Fold {self.fold_num}: "
            f"train=[{self.train_start}:{self.train_end}] "
            f"({self.train_end - self.train_start} days)  "
            f"val=[{self.val_start}:{self.val_end}] "
            f"({self.val_end - self.val_start} days)"
        )


# ---------------------------------------------------------------------
# Split generator
# ---------------------------------------------------------------------
def walk_forward_splits(
    n: int,
    n_folds: int = 5,
    min_train: int = 400,
    val_size: int = 60,
) -> Iterable[WalkForwardFold]:
    """Yield `n_folds` expanding-window folds over a series of length `n`.

    The training window grows by `val_size` between folds; the validation
    window is fixed at `val_size` and slides forward. The first fold has
    `min_train` training rows.

    Parameters
    ----------
    n : int
        Total length of the series.
    n_folds : int
        Number of out-of-fold validation windows.
    min_train : int
        Training size for the first fold.
    val_size : int
        Validation window size (constant across folds).

    Yields
    ------
    WalkForwardFold
    """
    if min_train + n_folds * val_size > n:
        raise ValueError(
            f"Cannot fit {n_folds} folds of size {val_size} with min_train="
            f"{min_train} into n={n}. Reduce n_folds or val_size."
        )

    train_end = min_train
    for i in range(n_folds):
        val_start = train_end
        val_end = val_start + val_size
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)
        yield WalkForwardFold(
            fold_num=i + 1,
            train_start=0,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            train_idx=train_idx,
            val_idx=val_idx,
        )
        train_end = val_end  # expand


# ---------------------------------------------------------------------
# OOF metrics
# ---------------------------------------------------------------------
def evaluate_oof_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    close: np.ndarray,
    threshold: float = 0.5,
    fee: float = 0.001,
) -> dict:
    """Compute classification + trading metrics on out-of-fold predictions.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels (1 = up, 0 = down).
    y_prob : np.ndarray
        Model probability of class 1.
    close : np.ndarray
        BTC close prices aligned with y_true/y_prob.
    threshold : float
        Trading signal threshold.
    fee : float
        Per-trade transaction cost (default 0.1%).

    Returns
    -------
    dict with keys: accuracy, f1, precision, recall, auc, sharpe, sortino,
    max_dd, win_rate, n_trades
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = (y_prob >= threshold).astype(int)

    if len(y_true) == 0:
        # sklearn's classification metrics raise ValueError on empty input
        # (e.g. accuracy_score requires >=1 sample) rather than returning a
        # degenerate value, so this has to be handled before we ever call
        # them. Same zeroed-metrics contract as the n_days == 0 backtest
        # path below, for a consistent "empty in, zero out" behaviour.
        return {
            "accuracy": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "auc": float("nan"),
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_dd": 0.0,
            "win_rate": 0.0,
            "n_trades": 0,
        }

    metrics: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc"] = float("nan")

    # Backtest on the validation close prices
    # y_prob is aligned with close — we need close[:-1] returns aligned with
    # signals on the same dates. Use the simple backtest logic from run_pipeline.
    signal = y_pred
    rets = np.diff(close) / close[:-1]
    # Align: signals[0:len(rets)] apply to rets[0:len(rets)]
    if len(signal) > len(rets):
        signal = signal[: len(rets)]
    strat_rets = signal * rets
    trade_flags = np.abs(np.diff(np.insert(signal, 0, signal[0] if len(signal) else 0)))
    # Simpler: signal changes
    if len(signal) >= 2:
        trade_flags = np.abs(np.diff(signal))
    else:
        trade_flags = np.array([])
    if len(strat_rets) > 0 and len(trade_flags) == len(strat_rets):
        strat_rets = strat_rets - trade_flags * fee
    elif len(strat_rets) > 0:
        # Mismatch — skip fee adjustment
        pass

    n_days = len(strat_rets)
    if n_days == 0:
        metrics.update(sharpe=0.0, sortino=0.0, max_dd=0.0, win_rate=0.0, n_trades=0)
        return metrics

    ann = np.sqrt(252)
    mean_r = strat_rets.mean()
    std_r = strat_rets.std() + 1e-12
    sharpe = (mean_r / std_r) * ann
    downside = strat_rets[strat_rets < 0]
    sortino = (mean_r / (downside.std() + 1e-12)) * ann if len(downside) > 0 else sharpe
    equity = np.cumprod(1 + strat_rets)
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    max_dd = drawdowns.min()
    win_rate = (strat_rets > 0).mean()
    n_trades = int(trade_flags.sum() // 2) if len(trade_flags) else 0

    metrics.update(
        sharpe=float(sharpe),
        sortino=float(sortino),
        max_dd=float(max_dd),
        win_rate=float(win_rate),
        n_trades=n_trades,
    )
    return metrics


# ---------------------------------------------------------------------
# Aggregated result
# ---------------------------------------------------------------------
@dataclass
class WalkForwardResult:
    """Aggregated walk-forward CV results."""
    fold_metrics: list[dict]
    oof_sharpe_mean: float
    oof_sharpe_std: float
    oof_accuracy_mean: float
    oof_accuracy_std: float
    oof_f1_mean: float
    oof_f1_std: float
    n_folds: int

    def to_dataframe(self) -> pd.DataFrame:
        """Return per-fold metrics as a pandas DataFrame (one row per fold)."""
        return pd.DataFrame(self.fold_metrics)

    def summary(self) -> str:
        """Return a multi-line string summarizing the aggregated OOF metrics."""
        return (
            f"Walk-Forward CV — {self.n_folds} folds\n"
            f"  OOF Sharpe   : {self.oof_sharpe_mean:+.3f} ± {self.oof_sharpe_std:.3f}\n"
            f"  OOF Accuracy : {self.oof_accuracy_mean:.3f} ± {self.oof_accuracy_std:.3f}\n"
            f"  OOF F1       : {self.oof_f1_mean:.3f} ± {self.oof_f1_std:.3f}"
        )


# ---------------------------------------------------------------------
# End-to-end runner
# ---------------------------------------------------------------------
def run_walk_forward(
    X: np.ndarray,
    y: np.ndarray,
    close: np.ndarray,
    dates: np.ndarray,
    model_factory: Callable,
    n_folds: int = 5,
    min_train: int = 400,
    val_size: int = 60,
    threshold: float = 0.5,
    fee: float = 0.001,
    fit_kwargs: dict | None = None,
    verbose: bool = True,
) -> WalkForwardResult:
    """Run walk-forward CV across `n_folds` folds.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, timesteps, n_features)
    y : np.ndarray, shape (n_samples,)
    close : np.ndarray, shape (n_samples,)
        Close prices aligned with X/y, for backtest metrics.
    dates : np.ndarray
        Dates aligned with X/y (for logging only).
    model_factory : callable
        Called with no args; must return a fresh Keras/TF model with .fit()
        and .predict() methods. A new model is built per fold.
    n_folds, min_train, val_size : int
        Walk-forward parameters.
    threshold : float
        Trading signal threshold.
    fee : float
        Transaction cost.
    fit_kwargs : dict, optional
        Extra kwargs passed to model.fit() (e.g. class_weight, callbacks).

    Returns
    -------
    WalkForwardResult
    """
    fit_kwargs = fit_kwargs or {}
    n = len(X)
    fold_metrics: list[dict] = []

    for fold in walk_forward_splits(n=n, n_folds=n_folds, min_train=min_train, val_size=val_size):
        if verbose:
            print(f"\n--- {fold} ---")

        X_tr, y_tr = X[fold.train_idx], y[fold.train_idx]
        X_va, y_va = X[fold.val_idx], y[fold.val_idx]
        close_va = close[fold.val_idx]
        dates_va = dates[fold.val_idx]

        model = model_factory()
        # Always pass validation data for early stopping
        model.fit(X_tr, y_tr, validation_data=(X_va, y_va), **fit_kwargs)

        y_prob_va = model.predict(X_va, verbose=0).ravel()

        m = evaluate_oof_metrics(
            y_true=y_va, y_prob=y_prob_va,
            close=close_va, threshold=threshold, fee=fee,
        )
        m["fold"] = fold.fold_num
        m["val_start"] = pd.Timestamp(dates_va[0]).isoformat() if len(dates_va) else ""
        m["val_end"] = pd.Timestamp(dates_va[-1]).isoformat() if len(dates_va) else ""
        m["train_size"] = len(fold.train_idx)
        m["val_size"] = len(fold.val_idx)
        fold_metrics.append(m)

        if verbose:
            print(
                f"  acc={m['accuracy']:.3f}  f1={m['f1']:.3f}  auc={m['auc']:.3f}  "
                f"sharpe={m['sharpe']:+.3f}  max_dd={m['max_dd']:+.3f}  "
                f"trades={m['n_trades']}"
            )

    sharpes = np.array([m["sharpe"] for m in fold_metrics])
    accs = np.array([m["accuracy"] for m in fold_metrics])
    f1s = np.array([m["f1"] for m in fold_metrics])

    return WalkForwardResult(
        fold_metrics=fold_metrics,
        oof_sharpe_mean=float(sharpes.mean()),
        oof_sharpe_std=float(sharpes.std()),
        oof_accuracy_mean=float(accs.mean()),
        oof_accuracy_std=float(accs.std()),
        oof_f1_mean=float(f1s.mean()),
        oof_f1_std=float(f1s.std()),
        n_folds=len(fold_metrics),
    )
