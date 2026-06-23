"""
BTC Sentiment-Driven LSTM Trading Pipeline — Streamlit Dashboard

Main entry point. Renders the Overview/Dashboard page with high-level metrics
and provides navigation to the Phase 1 and Phase 2 deep-dive pages via the
Streamlit `pages/` directory.

Usage:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="BTC Sentiment LSTM Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


# ---------------------------------------------------------------------
# Data loaders (with error handling)
# ---------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_csv(filename: str) -> Optional[pd.DataFrame]:
    """Load a CSV from outputs/. Return None and show a warning if missing."""
    path = OUTPUTS / filename
    if not path.exists():
        st.warning(f"Data file not found: `outputs/{filename}`. Run the pipeline first.")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to load `outputs/{filename}`: {e}")
        return None


@st.cache_data(ttl=300)
def load_json(filename: str) -> Optional[dict]:
    """Load a JSON file from outputs/."""
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


# ---------------------------------------------------------------------
# Live BTC price ticker (yfinance)
# ---------------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_live_btc() -> Optional[dict]:
    """Fetch live BTC-USD price + 7-day history via yfinance.
    
    Returns a dict with: price, change_24h_pct, history (DataFrame).
    Returns None on failure (caller shows a warning).
    """
    try:
        import yfinance as yf
        btc = yf.download("BTC-USD", period="7d", interval="1h", progress=False, auto_adjust=False)
        if btc is None or len(btc) == 0:
            return None
        # Flatten MultiIndex if present (newer yfinance versions)
        if isinstance(btc.columns, pd.MultiIndex):
            btc.columns = [" ".join(c).strip() for c in btc.columns]
        # Get the latest close price
        close_col = [c for c in btc.columns if "Close" in c][0]
        current_price = float(btc[close_col].iloc[-1])
        # 24h change: compare last value to ~24h ago (24 hourly candles)
        if len(btc) >= 24:
            prev_price = float(btc[close_col].iloc[-24])
            change_pct = ((current_price - prev_price) / prev_price) * 100
        else:
            change_pct = 0.0
        # Prepare history for sparkline
        history = btc[[close_col]].copy()
        history.columns = ["price"]
        history.index.name = "datetime"
        return {
            "price": current_price,
            "change_24h_pct": change_pct,
            "history": history.reset_index(),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------
# Live crypto news sentiment ticker
# ---------------------------------------------------------------------
@st.cache_data(ttl=600)
def fetch_live_sentiment() -> Optional[dict]:
    """Fetch the latest crypto news headlines and compute a sentiment score.

    Uses the pre-computed TextBlob sentiment embedded in the cryptonews.csv
    (same data source as the pipeline). Returns the latest headline, its
    sentiment score, and a 7-day daily sentiment trend.

    Returns None on failure.
    """
    try:
        import requests
        from io import StringIO
        import ast

        NEWS_URL = "https://raw.githubusercontent.com/nassim0014/btc-llm-sentiment/main/Data/cryptonews.csv"
        r = requests.get(NEWS_URL, timeout=30)
        r.raise_for_status()
        news = pd.read_csv(StringIO(r.text))

        # Parse the embedded sentiment dict
        def parse_sentiment(s):
            try:
                d = ast.literal_eval(s) if isinstance(s, str) else {}
                return float(d.get("polarity", 0.0))
            except Exception:
                return 0.0

        news["sentiment"] = news["sentiment"].apply(parse_sentiment)
        news["date"] = pd.to_datetime(news["date"], format="mixed", utc=True, errors="coerce")
        news = news.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        # Latest headline
        latest = news.iloc[-1]
        latest_headline = str(latest.get("title", "N/A"))[:120]
        latest_sentiment = float(latest["sentiment"])

        # Daily sentiment trend (last 7 days with data)
        news["date_day"] = news["date"].dt.tz_convert(None).dt.floor("D")
        daily = news.groupby("date_day")["sentiment"].mean().tail(7).reset_index()
        daily.columns = ["date", "sentiment"]

        # Sentiment label
        if latest_sentiment > 0.2:
            label = "🟢 Bullish"
        elif latest_sentiment < -0.2:
            label = "🔴 Bearish"
        else:
            label = "🟡 Neutral"

        return {
            "latest_headline": latest_headline,
            "latest_sentiment": latest_sentiment,
            "label": label,
            "daily_trend": daily,
            "source_date": str(latest["date"]),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("📈 BTC Sentiment-Driven LSTM Trading Dashboard")
st.markdown(
    "An end-to-end ML pipeline combining **FinBERT news sentiment** with "
    "**LSTM price forecasting**, **Optuna hyperparameter optimization**, "
    "**Kelly-based risk management**, and **SHAP interpretability**."
)

# ---------------------------------------------------------------------
# Live BTC price ticker
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("🔴 Live BTC-USD Price")

live_btc = fetch_live_btc()

if live_btc is not None:
    col_price, col_change, col_chart = st.columns([1, 1, 4])

    with col_price:
        st.metric(
            label="BTC-USD",
            value=f"${live_btc['price']:,.2f}",
        )

    with col_change:
        change = live_btc["change_24h_pct"]
        st.metric(
            label="24h Change",
            value=f"{change:+.2f}%",
            delta=f"{change:+.2f}%",
            delta_color="normal" if change >= 0 else "inverse",
        )

    with col_chart:
        import plotly.graph_objects as go
        hist = live_btc["history"]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=hist["datetime"],
                y=hist["price"],
                mode="lines",
                name="BTC-USD",
                line=dict(color="#00d4ff" if change >= 0 else "#ff4b4b", width=2),
                fill="tozeroy",
                fillcolor="rgba(0, 212, 255, 0.1)" if change >= 0 else "rgba(255, 75, 75, 0.1)",
                hovertemplate="Time: %{x|%Y-%m-%d %H:%M}<br>Price: $%{y:,.2f}<extra></extra>",
            )
        )
        fig.update_layout(
            template="plotly_dark",
            height=200,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, visible=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption(f"📅 Data via yfinance • Cached for 5 minutes • Last updated: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}")
else:
    st.warning("⚠️ Could not fetch live BTC price. The yfinance API may be rate-limited or unavailable. Pipeline results below are still available.")

# ---------------------------------------------------------------------
# Live crypto news sentiment ticker
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("📰 Live Crypto News Sentiment")

live_sentiment = fetch_live_sentiment()

if live_sentiment is not None:
    col_sent, col_label, col_trend = st.columns([3, 1, 3])

    with col_sent:
        st.markdown("**Latest Headline:**")
        st.markdown(f"*\"{live_sentiment['latest_headline']}...\"*")
        st.caption(f"📅 Source date: {live_sentiment['source_date'][:19]}")

    with col_label:
        sent_val = live_sentiment["latest_sentiment"]
        st.metric(
            label="Sentiment Score",
            value=f"{sent_val:+.3f}",
        )
        st.markdown(f"**{live_sentiment['label']}**")

    with col_trend:
        import plotly.graph_objects as go
        trend = live_sentiment["daily_trend"]
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=trend["date"],
                y=trend["sentiment"],
                marker_color=["#16a34a" if v >= 0 else "#dc2626" for v in trend["sentiment"]],
                name="Daily Sentiment",
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Sentiment: %{y:.3f}<extra></extra>",
            )
        )
        fig.update_layout(
            title="7-Day Sentiment Trend",
            template="plotly_dark",
            height=200,
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis=dict(title="", range=[-1, 1], zeroline=True, zerolinecolor="rgba(255,255,255,0.3)"),
            xaxis=dict(showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption("📰 Data from cryptonews.csv (GitHub raw) • TextBlob sentiment • Cached for 10 minutes")
else:
    st.warning("⚠️ Could not fetch live news sentiment. The cryptonews.csv may be unavailable.")

# ---------------------------------------------------------------------
# Load key data
# ---------------------------------------------------------------------
fmc = load_csv("final_model_comparison.csv")
sc = load_csv("strategy_comparison.csv")
rm = load_csv("risk_managed_backtest_results.csv")
best_params = load_json("best_optuna_params.json")
trading_metrics = load_csv("trading_metrics.csv")

# ---------------------------------------------------------------------
# Top-level KPI metrics
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("🎯 Key Performance Metrics")

if fmc is not None and len(fmc) > 0:
    # Find best LSTM strategy and Buy & Hold
    lstm_strategies = fmc[fmc["strategy"] != "buy_hold"].copy()
    best_lstm = lstm_strategies.loc[lstm_strategies["sharpe_ratio"].idxmax()] if len(lstm_strategies) > 0 else None
    buy_hold = fmc[fmc["strategy"] == "buy_hold"].iloc[0] if (fmc["strategy"] == "buy_hold").any() else None

    col1, col2, col3, col4, col5 = st.columns(5)

    if best_lstm is not None:
        col1.metric(
            label=f"Best LSTM Sharpe ({best_lstm['strategy']})",
            value=f"{best_lstm['sharpe_ratio']:.2f}",
            delta=f"{best_lstm['final_portfolio_value']:.4f}",
        )
    if buy_hold is not None:
        col2.metric(
            label="Buy & Hold Sharpe",
            value=f"{buy_hold['sharpe_ratio']:.2f}",
            delta=f"{buy_hold['final_portfolio_value']:.4f}",
        )
    if best_lstm is not None:
        col3.metric(
            label="Best LSTM Max Drawdown",
            value=f"{best_lstm['max_drawdown']*100:.2f}%",
            delta=f"{(best_lstm['max_drawdown'] - buy_hold['max_drawdown'])*100:.2f}pp vs B&H" if buy_hold is not None else None,
            delta_color="inverse",
        )
    if best_params is not None:
        col4.metric(
            label="Optuna Best OOF Sharpe",
            value=f"{best_params.get('best_oof_sharpe', 0):.2f}",
            delta=f"{best_params.get('n_complete', 0)} trials",
        )
    if rm is not None and len(rm) > 0:
        col5.metric(
            label="Risk-Managed Max DD",
            value=f"{rm.iloc[0]['max_drawdown_pct']:.2f}%",
            delta=f"{rm.iloc[0]['n_trades']} trades",
        )
else:
    st.info("Run the pipeline to populate the dashboard metrics.")

# ---------------------------------------------------------------------
# Strategy comparison table
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Strategy Comparison — Phase 1 (LSTM Configs vs Buy & Hold)")

if fmc is not None and len(fmc) > 0:
    display_df = fmc.copy()
    # Format percentages
    display_df["max_drawdown"] = (display_df["max_drawdown"] * 100).round(2).astype(str) + "%"
    display_df["win_rate"] = (display_df["win_rate"] * 100).round(2).astype(str) + "%"
    display_df["final_portfolio_value"] = display_df["final_portfolio_value"].round(4)
    display_df["sharpe_ratio"] = display_df["sharpe_ratio"].round(4)
    display_df["sortino_ratio"] = display_df["sortino_ratio"].round(4)
    display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]
    st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.warning("Strategy comparison data not available.")

# ---------------------------------------------------------------------
# Risk-managed vs Simple vs Buy & Hold
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("🛡️ Phase 2: Risk-Managed Backtest Comparison")

if sc is not None and len(sc) > 0:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.dataframe(
            sc.style.format({
                "final_value": "{:.4f}",
                "sharpe": "{:.4f}",
                "max_dd": "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    with col_right:
        if rm is not None and len(rm) > 0:
            r = rm.iloc[0]
            st.metric("Risk-Managed Final Value", f"{r['final_portfolio_value']:.4f}")
            st.metric("Risk-Managed Sharpe", f"{r['annualized_sharpe']:.4f}")
            st.metric("Circuit Breaker Triggered", "Yes" if r["circuit_breaker_triggered"] else "No")
else:
    st.warning("Strategy comparison data not available.")

# ---------------------------------------------------------------------
# Best Optuna hyperparameters
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("⚙️ Optuna Best Hyperparameters")

if best_params is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        bp = best_params.get("best_params", {})
        st.json({
            "Learning Rate": f"{bp.get('lr', 0):.6f}",
            "LSTM Units": bp.get("units", "N/A"),
            "Dropout": bp.get("dropout", "N/A"),
            "Number of Layers": bp.get("num_layers", "N/A"),
        })

    with col2:
        st.json({
            "Best OOF Sharpe": f"{best_params.get('best_oof_sharpe', 0):.4f}",
            "Total Trials": best_params.get("n_complete", 0) + best_params.get("n_pruned", 0),
            "Completed Trials": best_params.get("n_complete", 0),
            "Pruned Trials": best_params.get("n_pruned", 0),
            "Walk-Forward Folds": best_params.get("walk_forward_config", {}).get("n_folds", "N/A"),
        })
else:
    st.warning("Optuna parameters not available.")

# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "📋 **Navigation:** Use the sidebar to explore **Phase 1** (data → sentiment → LSTM → backtest) "
    "and **Phase 2** (walk-forward CV → Optuna → risk-managed backtest → SHAP) deep-dives."
)
st.caption("BTC Sentiment-Driven LSTM Trading Pipeline — Built with Streamlit + Plotly")
