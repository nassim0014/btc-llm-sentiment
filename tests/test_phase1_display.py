"""Tests for src/phase1_display.py — extracted from pages/1 (item 5)."""
from __future__ import annotations

import pandas as pd

from src.phase1_display import (
    METRIC_LABELS,
    METRIC_OPTIONS,
    format_display_columns,
    format_metric_label,
    has_data,
)


class TestMetricConfig:
    def test_metric_options_has_six_entries(self):
        assert len(METRIC_OPTIONS) == 6

    def test_metric_labels_covers_all_options(self):
        """Every option has a label."""
        for opt in METRIC_OPTIONS:
            assert opt in METRIC_LABELS

    def test_metric_labels_are_human_readable(self):
        """Labels don't contain underscores."""
        for label in METRIC_LABELS.values():
            assert "_" not in label


class TestFormatMetricLabel:
    def test_known_key_returns_label(self):
        assert format_metric_label("total_return_pct") == "Total Return (%)"
        assert format_metric_label("n_trades") == "Number of Trades"

    def test_unknown_key_falls_back_to_title_case(self):
        assert format_metric_label("custom_metric") == "Custom Metric"

    def test_unknown_key_with_no_underscores(self):
        assert format_metric_label("custom") == "Custom"


class TestFormatDisplayColumns:
    def test_converts_snake_to_title(self):
        df = pd.DataFrame({"total_return_pct": [1.0], "n_trades": [5]})
        result = format_display_columns(df)
        assert list(result.columns) == ["Total Return Pct", "N Trades"]

    def test_does_not_mutate_original(self):
        df = pd.DataFrame({"my_col": [1]})
        format_display_columns(df)
        assert list(df.columns) == ["my_col"]  # original unchanged

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = format_display_columns(df)
        assert list(result.columns) == []


class TestHasData:
    def test_none_returns_false(self):
        assert has_data(None) is False

    def test_empty_dataframe_returns_false(self):
        assert has_data(pd.DataFrame()) is False

    def test_non_empty_returns_true(self):
        assert has_data(pd.DataFrame({"a": [1]})) is True
