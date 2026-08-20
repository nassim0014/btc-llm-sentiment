"""Tests for src/phase2_display.py — extracted from pages/2 (item 5)."""
from __future__ import annotations

import pandas as pd

from src.phase2_display import (
    compute_optuna_stats,
    has_data,
    sort_shap_by_importance,
    top_n_features,
)


class TestHasData:
    def test_none_returns_false(self):
        assert has_data(None) is False

    def test_empty_dataframe_returns_false(self):
        assert has_data(pd.DataFrame()) is False

    def test_non_empty_returns_true(self):
        assert has_data(pd.DataFrame({"a": [1]})) is True


class TestComputeOptunaStats:
    def test_none_returns_zeros(self):
        result = compute_optuna_stats(None)
        assert result == {
            "best_oof_sharpe": 0.0,
            "total_trials": 0,
            "pruned_trials": 0,
            "pruned_pct": 0.0,
        }

    def test_computes_stats_from_params(self):
        params = {
            "best_oof_sharpe": 1.5,
            "n_complete": 80,
            "n_pruned": 20,
        }
        result = compute_optuna_stats(params)
        assert result["best_oof_sharpe"] == 1.5
        assert result["total_trials"] == 100
        assert result["pruned_trials"] == 20
        assert result["pruned_pct"] == 20.0

    def test_handles_zero_total(self):
        """When total trials = 0, pruned_pct is 0 (no division by zero)."""
        result = compute_optuna_stats({"n_complete": 0, "n_pruned": 0})
        assert result["pruned_pct"] == 0.0

    def test_handles_missing_keys(self):
        """Missing keys default to 0."""
        result = compute_optuna_stats({})
        assert result["best_oof_sharpe"] == 0
        assert result["total_trials"] == 0
        assert result["pruned_pct"] == 0.0

    def test_pruned_pct_calculation(self):
        """50 pruned out of 200 total = 25%."""
        params = {"n_complete": 150, "n_pruned": 50}
        result = compute_optuna_stats(params)
        assert result["pruned_pct"] == 25.0


class TestSortShapByImportance:
    def test_sorts_ascending(self):
        df = pd.DataFrame({
            "feature": ["a", "b", "c"],
            "mean_abs_shap": [0.3, 0.1, 0.2],
        })
        result = sort_shap_by_importance(df, ascending=True)
        assert list(result["feature"]) == ["b", "c", "a"]

    def test_sorts_descending(self):
        df = pd.DataFrame({
            "feature": ["a", "b", "c"],
            "mean_abs_shap": [0.3, 0.1, 0.2],
        })
        result = sort_shap_by_importance(df, ascending=False)
        assert list(result["feature"]) == ["a", "c", "b"]

    def test_does_not_mutate_original(self):
        df = pd.DataFrame({"feature": ["a", "b"], "mean_abs_shap": [0.3, 0.1]})
        sort_shap_by_importance(df)
        assert list(df["feature"]) == ["a", "b"]  # original unchanged

    def test_missing_column_returns_copy(self):
        """If mean_abs_shap column is missing, returns a copy unchanged."""
        df = pd.DataFrame({"feature": ["a", "b"]})
        result = sort_shap_by_importance(df)
        assert list(result["feature"]) == ["a", "b"]


class TestTopNFeatures:
    def test_returns_top_n_descending(self):
        df = pd.DataFrame({
            "feature": ["a", "b", "c", "d"],
            "mean_abs_shap": [0.1, 0.4, 0.3, 0.2],
        })
        result = top_n_features(df, n=2)
        assert len(result) == 2
        assert list(result["feature"]) == ["b", "c"]  # highest first

    def test_n_larger_than_df(self):
        """n > len(df) returns all rows."""
        df = pd.DataFrame({
            "feature": ["a", "b"],
            "mean_abs_shap": [0.1, 0.2],
        })
        result = top_n_features(df, n=10)
        assert len(result) == 2

    def test_default_n_is_10(self):
        """Default n is 10."""
        df = pd.DataFrame({
            "feature": [f"f{i}" for i in range(15)],
            "mean_abs_shap": [float(i) for i in range(15)],
        })
        result = top_n_features(df)
        assert len(result) == 10
