"""
Phase 1 Deep-Dive — Data, Sentiment, LSTM, Backtest

Interactive visualizations for the Phase 1 pipeline:
  - Portfolio equity curves (Plotly line chart)
  - Trading metrics comparison (Plotly bar charts)
  - Per-strategy breakdown
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.phase1_display import (
    METRIC_OPTIONS,
    format_display_columns,
    format_metric_label,
    has_data,
)

st.set_page_config(page_title="Phase 1 Deep-Dive", page_icon="🔬", layout="wide")

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"


@st.cache_data(ttl=300)
def load_csv(filename: str) -> pd.DataFrame | None:
    path = OUTPUTS / filename
    if not path.exists():
        st.warning(f"Data file not found: `outputs/{filename}`. Run the Phase 1 pipeline first.")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to load `outputs/{filename}`: {e}")
        return None


st.title("🔬 Phase 1 — Data → Sentiment → LSTM → Backtest")
st.markdown("Deep-dive into the Phase 1 pipeline: 4 LSTM configurations vs Buy & Hold with threshold-optimized trading signals.")

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
portfolio = load_csv("portfolio_values_over_time.csv")
fmc = load_csv("final_model_comparison.csv")
trading_metrics = load_csv("trading_metrics.csv")

# ---------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("📈 Portfolio Equity Curves (Test Window)")

if portfolio is not None and len(portfolio) > 0:
    portfolio["date"] = pd.to_datetime(portfolio["date"])

    fig = go.Figure()
    strategy_cols = [c for c in portfolio.columns if c != "date"]
    colors = px.colors.qualitative.Set2[: len(strategy_cols)]

    for i, col in enumerate(strategy_cols):
        fig.add_trace(
            go.Scatter(
                x=portfolio["date"],
                y=portfolio[col],
                mode="lines",
                name=col.replace("_", " ").title(),
                line=dict(color=colors[i], width=2),
                hovertemplate=f"<b>{col}</b><br>Date: %{{x|%Y-%m-%d}}<br>Equity: %{{y:.4f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Portfolio Value Over Time (Start = 1.0)",
        xaxis_title="Date",
        yaxis_title="Equity",
        hovermode="x unified",
        template="plotly_dark",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Portfolio equity curve data not available.")

# ---------------------------------------------------------------------
# Trading metrics bar charts
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Trading Metrics Comparison")

if has_data(trading_metrics):
    selected_metric = st.selectbox(
        "Select metric to visualize:",
        options=METRIC_OPTIONS,
        format_func=format_metric_label,
    )

    fig = px.bar(
        trading_metrics,
        x="strategy",
        y=selected_metric,
        color="strategy",
        title=f"{format_metric_label(selected_metric)} by Strategy",
        template="plotly_dark",
        text_auto=".2f",
    )
    fig.update_layout(
        xaxis_title="Strategy",
        yaxis_title=format_metric_label(selected_metric),
        showlegend=False,
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Also show the full table
    st.markdown("#### Full Trading Metrics Table")
    st.dataframe(format_display_columns(trading_metrics), use_container_width=True, hide_index=True)
else:
    st.warning("Trading metrics data not available.")

# ---------------------------------------------------------------------
# Final model comparison
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("🏆 Final Model Comparison")

if fmc is not None and len(fmc) > 0:
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            fmc,
            x="strategy",
            y="final_portfolio_value",
            color="strategy",
            title="Final Portfolio Value",
            template="plotly_dark",
            text_auto=".4f",
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            fmc,
            x="strategy",
            y="sharpe_ratio",
            color="strategy",
            title="Sharpe Ratio",
            template="plotly_dark",
            text_auto=".2f",
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Model comparison data not available.")

st.markdown("---")
st.caption("Phase 1 — Data loading, LLM sentiment scoring, feature engineering, LSTM fine-tuning, and backtesting.")
