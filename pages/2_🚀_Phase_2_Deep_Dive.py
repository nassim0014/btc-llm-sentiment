"""
Phase 2 Deep-Dive — Walk-Forward CV, Optuna, Risk Management, SHAP

Interactive visualizations for the Phase 2 pipeline:
  - Risk-managed equity curve with position sizing (Plotly)
  - Walk-forward OOF metrics (Plotly bar chart)
  - Optuna hyperparameter search results
  - SHAP feature importance (Plotly bar chart + regime comparison)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Phase 2 Deep-Dive", page_icon="🚀", layout="wide")

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"


@st.cache_data(ttl=300)
def load_csv(filename: str) -> pd.DataFrame | None:
    path = OUTPUTS / filename
    if not path.exists():
        st.warning(f"Data file not found: `outputs/{filename}`. Run the Phase 2 pipeline first.")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to load `outputs/{filename}`: {e}")
        return None


@st.cache_data(ttl=300)
def load_json(filename: str) -> dict | None:
    path = OUTPUTS / filename
    if not path.exists():
        st.warning(f"Data file not found: `outputs/{filename}`.")
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load `outputs/{filename}`: {e}")
        return None


st.title("🚀 Phase 2 — Advanced Upgrades")
st.markdown("Walk-Forward CV → Optuna HPO → Risk-Managed Backtest → SHAP Interpretability")

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
equity_curve = load_csv("risk_managed_equity_curve.csv")
strategy_cmp = load_csv("strategy_comparison.csv")
rm_results = load_csv("risk_managed_backtest_results.csv")
shap_importance = load_csv("shap_feature_importance.csv")
best_params = load_json("best_optuna_params.json")

# ---------------------------------------------------------------------
# Risk-managed equity curve
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("🛡️ Risk-Managed Backtest — Equity Curve & Position Sizing")

if equity_curve is not None and len(equity_curve) > 0:
    equity_curve["date"] = pd.to_datetime(equity_curve["date"])

    fig = go.Figure()

    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=equity_curve["date"],
            y=equity_curve["equity"],
            mode="lines",
            name="Equity",
            line=dict(color="#00d4ff", width=2.5),
            yaxis="y",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Equity: %{y:.4f}<extra></extra>",
        )
    )

    # Position size on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=equity_curve["date"],
            y=equity_curve["position"],
            mode="lines",
            name="Position Size",
            line=dict(color="#f59e0b", width=1.5, dash="dot"),
            yaxis="y2",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Position: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Equity Curve (Blue) + Position Size (Orange, Kelly × Vol-Target)",
        xaxis_title="Date",
        yaxis=dict(title="Equity", side="left"),
        yaxis2=dict(title="Position Size", side="right", overlaying="y", range=[0, 1]),
        hovermode="x unified",
        template="plotly_dark",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Kelly fraction and vol-target factor
    col1, col2 = st.columns(2)

    with col1:
        fig_kelly = px.line(
            equity_curve,
            x="date",
            y="kelly_fraction",
            title="Kelly Fraction Sizing Over Time",
            template="plotly_dark",
        )
        fig_kelly.update_layout(height=350, xaxis_title="Date", yaxis_title="Kelly Fraction")
        st.plotly_chart(fig_kelly, use_container_width=True)

    with col2:
        fig_vol = px.line(
            equity_curve,
            x="date",
            y="vol_target_factor",
            title="Volatility Target Factor Over Time",
            template="plotly_dark",
        )
        fig_vol.update_layout(height=350, xaxis_title="Date", yaxis_title="Vol-Target Factor")
        st.plotly_chart(fig_vol, use_container_width=True)
else:
    st.warning("Risk-managed equity curve data not available.")

# ---------------------------------------------------------------------
# Strategy comparison
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Strategy Comparison — Simple vs Risk-Managed vs Buy & Hold")

if strategy_cmp is not None and len(strategy_cmp) > 0:
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            strategy_cmp,
            x="strategy",
            y="final_value",
            color="strategy",
            title="Final Portfolio Value",
            template="plotly_dark",
            text_auto=".4f",
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            strategy_cmp,
            x="strategy",
            y="sharpe",
            color="strategy",
            title="Sharpe Ratio",
            template="plotly_dark",
            text_auto=".2f",
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Full Comparison Table")
    st.dataframe(
        strategy_cmp.style.format({
            "final_value": "{:.4f}",
            "sharpe": "{:.4f}",
            "max_dd": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.warning("Strategy comparison data not available.")

# ---------------------------------------------------------------------
# Optuna results
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("⚙️ Optuna Hyperparameter Search Results")

if best_params is not None:
    from src.phase2_display import compute_optuna_stats

    stats = compute_optuna_stats(best_params)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Best OOF Sharpe", f"{stats['best_oof_sharpe']:.4f}")

    with col2:
        st.metric("Total Trials", stats["total_trials"])

    with col3:
        st.metric("Pruned Trials", f"{stats['pruned_trials']} ({stats['pruned_pct']:.0f}%)")

    st.markdown("#### Best Hyperparameters")
    bp = best_params.get("best_params", {})
    bp_display = pd.DataFrame([
        {"Hyperparameter": "Learning Rate", "Value": f"{bp.get('lr', 0):.6f}"},
        {"Hyperparameter": "LSTM Units", "Value": bp.get("units", "N/A")},
        {"Hyperparameter": "Dropout", "Value": bp.get("dropout", "N/A")},
        {"Hyperparameter": "Number of Layers", "Value": bp.get("num_layers", "N/A")},
    ])
    st.dataframe(bp_display, use_container_width=True, hide_index=True)

    st.markdown("#### Search Space & Walk-Forward Config")
    col1, col2 = st.columns(2)
    with col1:
        st.json(best_params.get("search_space", {}))
    with col2:
        st.json(best_params.get("walk_forward_config", {}))
else:
    st.warning("Optuna parameters not available.")

# ---------------------------------------------------------------------
# SHAP feature importance
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("🔍 SHAP Feature Importance")

if shap_importance is not None and len(shap_importance) > 0:
    from src.phase2_display import sort_shap_by_importance, top_n_features

    # Bar chart
    fig = px.bar(
        sort_shap_by_importance(shap_importance, ascending=True),
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        title="Global Feature Importance (Mean |SHAP| Value)",
        template="plotly_dark",
        color="mean_abs_shap",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        height=600,
        xaxis_title="Mean |SHAP|",
        yaxis_title="Feature",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Top 10 Features by Mean |SHAP|")
    st.dataframe(
        top_n_features(shap_importance, n=10).style.format({"mean_abs_shap": "{:.6f}"}),
        use_container_width=True,
        hide_index=True,
    )

    # SHAP summary image
    shap_summary_path = OUTPUTS / "shap_summary.png"
    if shap_summary_path.exists():
        st.markdown("#### SHAP Beeswarm Summary Plot")
        st.image(str(shap_summary_path), use_container_width=True)

    # Regime comparison image
    shap_regime_path = OUTPUTS / "shap_regime_comparison.png"
    if shap_regime_path.exists():
        st.markdown("#### SHAP Regime Comparison (High-Vol vs Low-Vol)")
        st.image(str(shap_regime_path), use_container_width=True)
else:
    st.warning("SHAP feature importance data not available.")

st.markdown("---")
st.caption("Phase 2 — Walk-forward CV, Optuna HPO, Kelly + vol-targeting + DD circuit breaker, SHAP interpretability.")
