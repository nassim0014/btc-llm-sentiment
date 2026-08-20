"""
Backtest Simulator — Interactive parameter tuning.

Lets users adjust Kelly fraction, volatility targeting, and drawdown circuit
breaker parameters interactively and see how the equity curve changes in
real-time. Uses the same risk_managed_backtest() function as the pipeline
but with user-controlled parameters.
"""
from __future__ import annotations

from pathlib import Path

from src.backtest.simulator import run_backtest

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Backtest Simulator", page_icon="🎛️", layout="wide")

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"


# ---------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_equity_curve() -> pd.DataFrame | None:
    """Load the risk-managed equity curve with raw probabilities."""
    path = OUTPUTS / "risk_managed_equity_curve.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_strategy_comparison() -> pd.DataFrame | None:
    """Load the original strategy comparison for reference."""
    path = OUTPUTS / "strategy_comparison.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# ---------------------------------------------------------------------
# Interactive backtest function (parameterized)
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------
st.title("🎛️ Backtest Simulator")
st.markdown("Adjust risk management parameters interactively and see how the equity curve responds. Uses the same LSTM model probabilities as the pipeline backtest.")

st.markdown("---")

# Load data
equity_data = load_equity_curve()

if equity_data is None or len(equity_data) == 0:
    st.error("⚠️ Risk-managed equity curve data not found. Run the Phase 2 pipeline first to generate `outputs/risk_managed_equity_curve.csv`.")
    st.stop()

st.success(f"✅ Loaded {len(equity_data)} days of backtest data with model probabilities.")

# Extract probabilities + close prices
equity_data["date"] = pd.to_datetime(equity_data["date"])
prob = equity_data["raw_prob"].values
# Reconstruct close from equity (we need the actual BTC closes for the backtest)
# The equity column is the strategy equity, not BTC price. We'll use the raw_prob
# and reconstruct a synthetic close from the daily returns.
# Actually, we can compute close from the position + equity + daily returns:
# daily_return[t] = position[t-1] * (close[t]/close[t-1] - 1) - cost
# This is complex. Instead, let's fetch the actual BTC close for the test window.
# For now, use a simpler approach: we have the equity curve, let's just vary
# the position sizing on the same probability stream.

# Simpler approach: reconstruct BTC returns from the equity curve
# We know equity[t] = equity[t-1] * (1 + position[t-1] * btc_ret[t-1] - cost)
# So btc_ret[t-1] = (equity[t]/equity[t-1] - 1 + cost) / position[t-1]  (when position > 0)
# This is approximate but works for the simulator.

# Actually, the cleanest approach: fetch the test-window BTC closes
@st.cache_data(ttl=300)
def fetch_test_btc() -> np.ndarray | None:
    """Fetch BTC closes for the test window (Sep-Dec 2024)."""
    try:
        import yfinance as yf
        btc = yf.download("BTC-USD", start="2024-09-01", end="2024-12-31",
                          auto_adjust=False, progress=False)
        if isinstance(btc.columns, pd.MultiIndex):
            btc.columns = [" ".join(c).strip() for c in btc.columns]
        close_col = [c for c in btc.columns if "Close" in c][0]
        # Align with equity_data length
        closes = btc[close_col].values
        # Trim to match the equity curve length
        n = len(equity_data)
        if len(closes) > n:
            closes = closes[-n:]
        elif len(closes) < n:
            # Pad from the front
            closes = np.pad(closes, (n - len(closes), 0), mode="edge")
        return closes
    except Exception:
        return None


with st.spinner("Fetching BTC prices for the test window..."):
    close = fetch_test_btc()

if close is None:
    st.warning("⚠️ Could not fetch BTC prices. Using approximate reconstruction from equity curve.")
    # Fallback: reconstruct from the original equity
    close = np.cumprod(1 + np.diff(equity_data["equity"].values, prepend=1.0)) * 100
    st.info(f"Using {len(close)} synthetic close prices for simulation.")

# ---------------------------------------------------------------------
# Interactive controls
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("⚙️ Risk Management Parameters")

col1, col2, col3, col4 = st.columns(4)

with col1:
    threshold = st.slider(
        "Signal Threshold",
        min_value=0.30,
        max_value=0.70,
        value=0.50,
        step=0.05,
        help="Probability above which to go long. Lower = more aggressive.",
    )

with col2:
    target_vol = st.slider(
        "Target Annual Volatility",
        min_value=0.10,
        max_value=0.60,
        value=0.20,
        step=0.05,
        help="Target annualized volatility for position sizing.",
    )

with col3:
    max_dd = st.slider(
        "Max Drawdown Breaker",
        min_value=0.05,
        max_value=0.30,
        value=0.15,
        step=0.05,
        help="Drawdown level that triggers the circuit breaker (flattens position).",
    )

with col4:
    kelly_cap = st.slider(
        "Kelly Cap",
        min_value=0.25,
        max_value=1.00,
        value=1.00,
        step=0.25,
        help="Maximum position size from Kelly fraction. 1.0 = full Kelly.",
    )

col5, col6, col7 = st.columns(3)

with col5:
    use_vol_target = st.checkbox("Enable Vol-Targeting", value=True)

with col6:
    use_breaker = st.checkbox("Enable Circuit Breaker", value=True)

with col7:
    fee = st.slider("Transaction Fee (%)", min_value=0.0, max_value=0.5, value=0.1, step=0.05) / 100

# ---------------------------------------------------------------------
# Run the backtest with user parameters
# ---------------------------------------------------------------------
st.markdown("---")

with st.spinner("Running backtest with your parameters..."):
    result = run_backtest(
        prob=prob,
        close=close,
        threshold=threshold,
        fee=fee,
        target_annual_vol=target_vol,
        vol_lookback=20,
        max_drawdown_pct=max_dd,
        kelly_cap=kelly_cap,
        use_vol_target=use_vol_target,
        use_circuit_breaker=use_breaker,
    )

# ---------------------------------------------------------------------
# Display metrics
# ---------------------------------------------------------------------
st.subheader("📊 Backtest Results")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Final Value", f"{result['final_value']:.4f}")
col2.metric("Sharpe Ratio", f"{result['sharpe']:.2f}")
col3.metric("Sortino Ratio", f"{result['sortino']:.2f}")
col4.metric("Max Drawdown", f"{result['max_dd']*100:.2f}%")
col5.metric("Win Rate", f"{result['win_rate']*100:.1f}%")

st.metric("Number of Trades", result["n_trades"])
if result["breaker_triggered"]:
    st.error(f"🚨 Circuit breaker triggered on day {result['breaker_trigger_day']}")
else:
    st.success("✅ Circuit breaker was NOT triggered")

# ---------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("📈 Equity Curve (Your Parameters vs Original)")

fig = go.Figure()

# User's equity curve
fig.add_trace(
    go.Scatter(
        x=equity_data["date"],
        y=result["equity"],
        mode="lines",
        name="Your Backtest",
        line=dict(color="#00d4ff", width=2.5),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Equity: %{y:.4f}<extra></extra>",
    )
)

# Original equity curve (from the pipeline)
fig.add_trace(
    go.Scatter(
        x=equity_data["date"],
        y=equity_data["equity"],
        mode="lines",
        name="Original (Pipeline)",
        line=dict(color="#f59e0b", width=2, dash="dash"),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Equity: %{y:.4f}<extra></extra>",
    )
)

fig.update_layout(
    title="Equity Curve Comparison",
    xaxis_title="Date",
    yaxis_title="Equity",
    template="plotly_dark",
    height=500,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("🎚️ Position Sizing Over Time")

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=equity_data["date"],
        y=result["positions"],
        mode="lines",
        name="Position Size",
        line=dict(color="#3b82f6", width=2),
        fill="tozeroy",
        fillcolor="rgba(59, 130, 246, 0.2)",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Position: %{y:.4f}<extra></extra>",
    )
)
fig.update_layout(
    title="Position Size (Kelly × Vol-Target)",
    xaxis_title="Date",
    yaxis_title="Position Size",
    template="plotly_dark",
    height=350,
    yaxis=dict(range=[0, 1]),
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# Drawdown chart
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("📉 Drawdown Chart")

equity = result["equity"]
running_max = np.maximum.accumulate(equity)
drawdown = (equity - running_max) / running_max * 100

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=equity_data["date"],
        y=drawdown,
        mode="lines",
        name="Drawdown",
        line=dict(color="#dc2626", width=2),
        fill="tozeroy",
        fillcolor="rgba(220, 38, 38, 0.2)",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<extra></extra>",
    )
)
if use_breaker:
    fig.add_hline(
        y=-max_dd * 100,
        line_dash="dash",
        line_color="white",
        annotation_text=f"Circuit Breaker (-{max_dd*100:.0f}%)",
    )
fig.update_layout(
    title="Portfolio Drawdown",
    xaxis_title="Date",
    yaxis_title="Drawdown (%)",
    template="plotly_dark",
    height=350,
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# Parameter sensitivity (cached in session_state)
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("🔬 Parameter Sensitivity Analysis")

sensitivity_param = st.selectbox(
    "Select parameter to analyze:",
    ["threshold", "target_vol", "max_dd", "kelly_cap"],
    format_func=lambda x: {
        "threshold": "Signal Threshold",
        "target_vol": "Target Volatility",
        "max_dd": "Max Drawdown Breaker",
        "kelly_cap": "Kelly Cap",
    }[x],
)

# Build a cache key from the current backtest config + the selected param.
# If any of these change, we recompute; otherwise we reuse session_state.
cache_key = (
    sensitivity_param,
    round(threshold, 4),
    round(target_vol, 4),
    round(max_dd, 4),
    round(kelly_cap, 4),
    round(fee, 6),
    use_vol_target,
    use_breaker,
)

# Check if we have cached results for this exact configuration
if "sens_cache" not in st.session_state:
    st.session_state["sens_cache"] = {}

if cache_key in st.session_state["sens_cache"]:
    sens_df = st.session_state["sens_cache"][cache_key]
    # Display a small note that results are cached
    st.caption("📦 Results loaded from session cache (adjust a slider to recompute)")
else:
    # Run sensitivity scan (only if not cached)
    with st.spinner(f"Scanning {sensitivity_param}..."):
        if sensitivity_param == "threshold":
            param_values = np.arange(0.30, 0.71, 0.05)
            results = []
            for v in param_values:
                r = run_backtest(prob, close, threshold=v, fee=fee,
                                 target_annual_vol=target_vol, max_drawdown_pct=max_dd,
                                 kelly_cap=kelly_cap, use_vol_target=use_vol_target,
                                 use_circuit_breaker=use_breaker)
                results.append({"value": v, "sharpe": r["sharpe"], "final_value": r["final_value"],
                                "max_dd": r["max_dd"], "n_trades": r["n_trades"]})
        elif sensitivity_param == "target_vol":
            param_values = np.arange(0.10, 0.61, 0.05)
            results = []
            for v in param_values:
                r = run_backtest(prob, close, threshold=threshold, fee=fee,
                                 target_annual_vol=v, max_drawdown_pct=max_dd,
                                 kelly_cap=kelly_cap, use_vol_target=use_vol_target,
                                 use_circuit_breaker=use_breaker)
                results.append({"value": v, "sharpe": r["sharpe"], "final_value": r["final_value"],
                                "max_dd": r["max_dd"], "n_trades": r["n_trades"]})
        elif sensitivity_param == "max_dd":
            param_values = np.arange(0.05, 0.31, 0.025)
            results = []
            for v in param_values:
                r = run_backtest(prob, close, threshold=threshold, fee=fee,
                                 target_annual_vol=target_vol, max_drawdown_pct=v,
                                 kelly_cap=kelly_cap, use_vol_target=use_vol_target,
                                 use_circuit_breaker=use_breaker)
                results.append({"value": v, "sharpe": r["sharpe"], "final_value": r["final_value"],
                                "max_dd": r["max_dd"], "n_trades": r["n_trades"]})
        else:  # kelly_cap
            param_values = np.arange(0.25, 1.01, 0.05)
            results = []
            for v in param_values:
                r = run_backtest(prob, close, threshold=threshold, fee=fee,
                                 target_annual_vol=target_vol, max_drawdown_pct=max_dd,
                                 kelly_cap=v, use_vol_target=use_vol_target,
                                 use_circuit_breaker=use_breaker)
                results.append({"value": v, "sharpe": r["sharpe"], "final_value": r["final_value"],
                                "max_dd": r["max_dd"], "n_trades": r["n_trades"]})

        sens_df = pd.DataFrame(results)
        # Store in session_state for reuse
        st.session_state["sens_cache"][cache_key] = sens_df

# Plot sensitivity
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=sens_df["value"],
        y=sens_df["sharpe"],
        mode="lines+markers",
        name="Sharpe Ratio",
        line=dict(color="#00d4ff", width=2),
        yaxis="y",
    )
)
fig.add_trace(
    go.Scatter(
        x=sens_df["value"],
        y=sens_df["final_value"],
        mode="lines+markers",
        name="Final Value",
        line=dict(color="#f59e0b", width=2),
        yaxis="y2",
    )
)
fig.update_layout(
    title=f"Sensitivity to {sensitivity_param.replace('_', ' ').title()}",
    xaxis_title=sensitivity_param.replace("_", " ").title(),
    yaxis=dict(title="Sharpe Ratio", side="left"),
    yaxis2=dict(title="Final Value", side="right", overlaying="y"),
    template="plotly_dark",
    height=400,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

st.dataframe(sens_df.round(4), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("🎛️ Backtest Simulator — Interactive parameter tuning using the same LSTM probabilities as the pipeline. The 'Original (Pipeline)' line shows the default parameters for comparison.")
