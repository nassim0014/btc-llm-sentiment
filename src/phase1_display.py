"""Display helpers extracted from pages/1_🔬_Phase_1_Deep_Dive.py (item 5).

Pure-logic helpers for the Phase 1 Deep-Dive page: metric labels,
column-name formatting, and data availability checks. No Streamlit
dependencies so they can be unit-tested.
"""
from __future__ import annotations

import pandas as pd

# The metrics shown in the trading-metrics comparison dropdown.
METRIC_OPTIONS = [
    "total_return_pct",
    "annualized_sharpe",
    "annualized_sortino",
    "max_drawdown_pct",
    "win_rate_pct",
    "n_trades",
]

# Human-readable labels for each metric key.
METRIC_LABELS = {
    "total_return_pct": "Total Return (%)",
    "annualized_sharpe": "Annualized Sharpe",
    "annualized_sortino": "Annualized Sortino",
    "max_drawdown_pct": "Max Drawdown (%)",
    "win_rate_pct": "Win Rate (%)",
    "n_trades": "Number of Trades",
}


def format_metric_label(key: str) -> str:
    """Return the human-readable label for a metric key.

    Falls back to the key itself (with underscores → spaces, title-cased)
    if the key isn't in METRIC_LABELS.
    """
    return METRIC_LABELS.get(key, key.replace("_", " ").title())


def format_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the DataFrame with human-readable column names.

    Converts snake_case column names to Title Case (e.g.
    'total_return_pct' → 'Total Return Pct').
    """
    display = df.copy()
    display.columns = [c.replace("_", " ").title() for c in display.columns]
    return display


def has_data(df: pd.DataFrame | None) -> bool:
    """True if the DataFrame is not None and has at least one row."""
    return df is not None and len(df) > 0
