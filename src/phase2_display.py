"""Display helpers extracted from pages/2_🚀_Phase_2_Deep_Dive.py (item 5).

Pure-logic helpers for the Phase 2 Deep-Dive page: Optuna trial
statistics, SHAP feature sorting, and data availability checks.
No Streamlit dependencies so they can be unit-tested.
"""
from __future__ import annotations

import pandas as pd


def has_data(df: pd.DataFrame | None) -> bool:
    """True if the DataFrame is not None and has at least one row."""
    return df is not None and len(df) > 0


def compute_optuna_stats(best_params: dict | None) -> dict:
    """Compute summary statistics from the Optuna best_params JSON.

    Args:
        best_params: The parsed JSON from best_optuna_params.json, or None.

    Returns:
        Dict with keys: best_oof_sharpe (float), total_trials (int),
        pruned_trials (int), pruned_pct (float). Returns zeros if
        best_params is None.
    """
    if best_params is None:
        return {
            "best_oof_sharpe": 0.0,
            "total_trials": 0,
            "pruned_trials": 0,
            "pruned_pct": 0.0,
        }

    best_oof_sharpe = best_params.get("best_oof_sharpe", 0)
    n_complete = best_params.get("n_complete", 0)
    n_pruned = best_params.get("n_pruned", 0)
    total = n_complete + n_pruned
    pruned_pct = (n_pruned / total * 100) if total > 0 else 0.0

    return {
        "best_oof_sharpe": best_oof_sharpe,
        "total_trials": total,
        "pruned_trials": n_pruned,
        "pruned_pct": pruned_pct,
    }


def sort_shap_by_importance(
    shap_df: pd.DataFrame,
    ascending: bool = True,
) -> pd.DataFrame:
    """Sort the SHAP feature-importance DataFrame by mean_abs_shap.

    Args:
        shap_df: DataFrame with 'feature' and 'mean_abs_shap' columns.
        ascending: True for ascending (smallest first), False for descending.

    Returns:
        Sorted copy of the DataFrame.
    """
    if "mean_abs_shap" not in shap_df.columns:
        return shap_df.copy()
    return shap_df.sort_values("mean_abs_shap", ascending=ascending).copy()


def top_n_features(
    shap_df: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """Return the top N features by mean_abs_shap (descending).

    Args:
        shap_df: DataFrame with 'feature' and 'mean_abs_shap' columns.
        n: Number of top features to return.

    Returns:
        DataFrame with the top N rows, sorted descending.
    """
    return sort_shap_by_importance(shap_df, ascending=False).head(n)
