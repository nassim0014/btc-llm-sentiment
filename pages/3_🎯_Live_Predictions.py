"""
Live Predictions — Fetch today's BTC price + news, run the trained LSTM model,
and display a live up/down prediction.

This page loads the saved Optuna-tuned LSTM model, fetches the latest BTC
OHLCV data and crypto news, engineers the same 23 features used in training,
and runs the model to produce a directional prediction for the next 5 days.

If the model is not available (e.g., on Streamlit Cloud without the .keras
file committed), the page still shows live BTC price, sentiment data, and
feature engineering — but displays "Model Required" instead of the prediction.
"""
from __future__ import annotations

import ast
from io import StringIO
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from src.live_features import engineer_features

st.set_page_config(page_title="Live Predictions", page_icon="🎯", layout="wide")

# Use centralized config + safe pickle loader
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import NEWS_URL, find_model
from src.utils.safe_pickle import SafePickleError, safe_load_bundle

OUTPUTS = ROOT / "outputs"
INTERIM = ROOT / "notebooks" / "interim"
DATA = ROOT / "Data"

# Backwards-compat aliases used by the rest of this file
MODEL_PATHS = [INTERIM / "best_optuna_model.keras", ROOT / "models" / "best_optuna_model.keras", ROOT / "best_optuna_model.keras"]
BUNDLE_PATHS = [INTERIM / "features_for_lstm.pkl", ROOT / "models" / "features_for_lstm.pkl"]


# ---------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------
# find_model and find_bundle are now imported from src.config


@st.cache_data(ttl=300)
def load_feature_bundle() -> dict | None:
    """Load the feature bundle (scaler + feature columns + test_close).

    Uses the safe pickle loader with SHA256 integrity verification and
    a restricted unpickler to defend against tampered .pkl files.
    """
    try:
        return safe_load_bundle()
    except SafePickleError as e:
        st.error(f"Refused to load feature bundle (integrity check failed): {e}")
        return None
    except Exception:
        return None


@st.cache_resource
def load_lstm_model():
    """Load the saved Optuna-tuned LSTM model."""
    path = find_model()
    if path is None:
        return None
    try:
        import os
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        return tf.keras.models.load_model(str(path))
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_latest_btc(days: int = 60) -> pd.DataFrame | None:
    """Fetch the latest BTC-USD daily candles via yfinance."""
    try:
        import yfinance as yf
        btc = yf.download("BTC-USD", period=f"{days}d", auto_adjust=False, progress=False)
        if btc is None or len(btc) == 0:
            return None
        if isinstance(btc.columns, pd.MultiIndex):
            btc.columns = [" ".join(c).strip() for c in btc.columns]
        btc = btc.reset_index().rename(columns={"Date": "date"})
        btc.columns = [c.replace(" BTC-USD", "").lower() for c in btc.columns]
        btc["date"] = pd.to_datetime(btc["date"]).dt.floor("D")
        return btc
    except Exception:
        return None


@st.cache_data(ttl=600)
def fetch_latest_news_sentiment() -> pd.DataFrame | None:
    """Fetch the latest crypto news + parse sentiment from cryptonews.csv."""
    try:
        r = requests.get(
            NEWS_URL,
            timeout=60,
            headers={"User-Agent": "btc-llm-sentiment/1.0 (+https://github.com/nassim0014/btc-llm-sentiment)"},
        )
        r.raise_for_status()
        news = pd.read_csv(StringIO(r.text))
        news["date"] = pd.to_datetime(news["date"], format="mixed", utc=True, errors="coerce")
        news = news.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        def parse_sentiment(s):
            try:
                d = ast.literal_eval(s) if isinstance(s, str) else {}
                return pd.Series({
                    "sentiment_class": d.get("class", "neutral"),
                    "sentiment_polarity": float(d.get("polarity", 0.0)),
                    "sentiment_subjectivity": float(d.get("subjectivity", 0.0)),
                })
            except Exception:
                return pd.Series({"sentiment_class": "neutral", "sentiment_polarity": 0.0, "sentiment_subjectivity": 0.0})

        sent = news["sentiment"].apply(parse_sentiment)
        news = pd.concat([news.drop(columns=["sentiment"]), sent], axis=1)
        news["llm_sentiment"] = news["sentiment_polarity"].astype(float)
        return news
    except Exception:
        return None


# ---------------------------------------------------------------------
# Feature engineering (same as the pipeline)
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------
# Title + refresh button in the same row
col_title, col_refresh = st.columns([8, 1])
with col_title:
    st.title("🎯 Live Predictions")
with col_refresh:
    st.write("")  # vertical spacer to align with title
    if st.button("🔄 Refresh", help="Clear cache and fetch fresh BTC price + news data"):
        # Clear the cached data for this page's fetch functions
        fetch_latest_btc.clear()
        fetch_latest_news_sentiment.clear()
        st.rerun()

st.markdown("Fetches the latest BTC price + crypto news, engineers features, and runs the trained LSTM model to predict the next 5-day directional move.")

st.markdown("---")

# ---------------------------------------------------------------------
# Check model availability (non-fatal — continue even if missing)
# ---------------------------------------------------------------------
model = load_lstm_model()
bundle = load_feature_bundle()
model_available = model is not None and bundle is not None

if not model_available:
    st.warning("⚠️ Trained LSTM model not found in this deployment.")
    st.info("""
    **To enable live predictions:**
    - Run the Phase 2 Optuna pipeline locally to generate the model
    - The model file (`best_optuna_model.keras`) is 122KB and should be committed to the repo
    - Place it at: `notebooks/interim/best_optuna_model.keras`

    **Below:** Live BTC price, news sentiment, and feature engineering are still available —
    only the final LSTM prediction requires the model.
    """)
    st.markdown("---")

if model_available:
    st.success(f"✅ Model loaded: {model.name} | Features: {len(bundle['feature_cols'])}")

# ---------------------------------------------------------------------
# Fetch live data (always shown, even without the model)
# ---------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    with st.spinner("Fetching latest BTC-USD prices..."):
        btc = fetch_latest_btc(days=60)
    if btc is not None:
        st.success(f"✅ Fetched {len(btc)} days of BTC-USD data (latest: {btc['date'].iloc[-1].date()})")
    else:
        st.error("⚠️ Could not fetch BTC price data.")

with col2:
    with st.spinner("Fetching latest crypto news..."):
        news = fetch_latest_news_sentiment()
    if news is not None:
        st.success(f"✅ Fetched {len(news):,} news headlines (latest: {news['date'].iloc[-1].strftime('%Y-%m-%d')})")
    else:
        st.error("⚠️ Could not fetch crypto news.")

if btc is None or news is None:
    st.stop()

# ---------------------------------------------------------------------
# Feature engineering (always shown)
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("🔧 Feature Engineering")

with st.spinner("Engineering features (technical + sentiment)..."):
    df = engineer_features(btc, news)

if df is None or len(df) < 30:
    st.error("⚠️ Could not engineer features. Need at least 30 days of data.")
    st.stop()

st.success(f"✅ Engineered features for {df['date'].iloc[-1].date()}")

# Show latest feature values (if we have the feature column list)
if model_available:
    FEATURE_COLS = bundle["feature_cols"]
    df_features = df[FEATURE_COLS].fillna(0)
    latest_features = df_features.iloc[-1:]

    with st.expander("📋 View latest feature values"):
        feature_display = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Value": latest_features.values[0],
        })
        st.dataframe(feature_display, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# LSTM Prediction (only if model is available)
# ---------------------------------------------------------------------
if model_available:
    st.markdown("---")
    st.subheader("🤖 LSTM Prediction")

    scaler = bundle["scaler"]
    latest_scaled = scaler.transform(latest_features)
    latest_3d = latest_scaled.reshape(1, 1, len(FEATURE_COLS))

    with st.spinner("Running the LSTM model..."):
        try:
            prob_up = float(model.predict(latest_3d, verbose=0).ravel()[0])
        except Exception as e:
            st.error(f"⚠️ Model prediction failed: {e}")
            prob_up = None

    if prob_up is not None:
        # Display prediction
        col_pred, col_conf, col_signal = st.columns(3)

        with col_pred:
            if prob_up >= 0.5:
                st.metric("Prediction", "📈 UP", delta=f"{prob_up*100:.1f}% confidence")
            else:
                st.metric("Prediction", "📉 DOWN", delta=f"{(1-prob_up)*100:.1f}% confidence", delta_color="inverse")

        with col_conf:
            st.metric("P(Up)", f"{prob_up*100:.1f}%")
            st.metric("P(Down)", f"{(1-prob_up)*100:.1f}%")

        with col_signal:
            threshold = 0.5
            if prob_up >= threshold:
                signal = "🟢 LONG"
            else:
                signal = "🔴 CASH"
            st.markdown(f"### {signal}")
            st.caption(f"Threshold: {threshold:.2f}")

        # Prediction gauge
        st.markdown("---")
        st.subheader("📊 Prediction Confidence Gauge")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_up * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Probability of BTC going UP in 5 days"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#00d4ff"},
                "steps": [
                    {"range": [0, 30], "color": "#dc2626"},
                    {"range": [30, 70], "color": "#f59e0b"},
                    {"range": [70, 100], "color": "#16a34a"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ))
        fig.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)

else:
    # Model not available — show placeholder
    st.markdown("---")
    st.subheader("🤖 LSTM Prediction")
    st.info("📋 **Model Required** — The trained LSTM model is not available in this deployment. See the warning above for instructions on how to enable live predictions.")
    st.markdown("### ⏳ Prediction: *Model Required*")

# ---------------------------------------------------------------------
# Recent price chart (always shown)
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("📈 Recent BTC-USD Price (30 days)")

fig = go.Figure()
last_30 = btc.tail(30)
fig.add_trace(
    go.Candlestick(
        x=last_30["date"],
        open=last_30["open"],
        high=last_30["high"],
        low=last_30["low"],
        close=last_30["close"],
        name="BTC-USD",
        increasing_line_color="#16a34a",
        decreasing_line_color="#dc2626",
    )
)
fig.update_layout(
    template="plotly_dark",
    height=400,
    xaxis_rangeslider_visible=False,
    xaxis_title="Date",
    yaxis_title="Price (USD)",
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# Recent sentiment (always shown)
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("📰 Recent News Sentiment (7 days)")

news["date_day"] = news["date"].dt.tz_convert(None).dt.floor("D")
daily_sent = news.groupby("date_day")["llm_sentiment"].agg(["mean", "size"]).tail(7).reset_index()

if len(daily_sent) > 0:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=daily_sent["date_day"],
            y=daily_sent["mean"],
            marker_color=["#16a34a" if v >= 0 else "#dc2626" for v in daily_sent["mean"]],
            name="Daily Sentiment",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Sentiment: %{y:.3f}<br>Headlines: %{customdata}<extra></extra>",
            customdata=daily_sent["size"],
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=300,
        yaxis=dict(title="Sentiment Score", range=[-1, 1], zeroline=True),
        xaxis_title="Date",
    )
    st.plotly_chart(fig, use_container_width=True)

# Latest headlines (always shown)
st.markdown("#### Latest 5 Headlines")
latest_news = news.tail(5)[["date", "title", "llm_sentiment"]].copy()
latest_news["date"] = latest_news["date"].dt.strftime("%Y-%m-%d %H:%M")
latest_news.columns = ["Date", "Headline", "Sentiment"]
st.dataframe(latest_news, use_container_width=True, hide_index=True)

st.markdown("---")
status = "with LSTM prediction" if model_available else "without LSTM prediction (model not found)"
st.caption(f"🎯 Live Predictions — Uses the Optuna-tuned LSTM model with the same 23 features (technical + sentiment) as training. Running {status}.")
