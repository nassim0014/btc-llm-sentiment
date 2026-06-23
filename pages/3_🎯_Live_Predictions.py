"""
Live Predictions — Fetch today's BTC price + news, run the trained LSTM model,
and display a live up/down prediction.

This page loads the saved Optuna-tuned LSTM model, fetches the latest BTC
OHLCV data and crypto news, engineers the same 23 features used in training,
and runs the model to produce a directional prediction for the next 5 days.
"""
from __future__ import annotations

import ast
import json
import pickle
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Live Predictions", page_icon="🎯", layout="wide")

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
INTERIM = ROOT / "notebooks" / "interim"
DATA = ROOT / "Data"


# ---------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_feature_bundle() -> Optional[dict]:
    """Load the feature bundle (scaler + feature columns + test_close)."""
    path = INTERIM / "features_for_lstm.pkl"
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        return None


@st.cache_resource
def load_lstm_model():
    """Load the saved Optuna-tuned LSTM model."""
    model_path = INTERIM / "best_optuna_model.keras"
    if not model_path.exists():
        return None
    try:
        import os
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        return tf.keras.models.load_model(model_path)
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_latest_btc(days: int = 60) -> Optional[pd.DataFrame]:
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
def fetch_latest_news_sentiment() -> Optional[pd.DataFrame]:
    """Fetch the latest crypto news + parse sentiment from cryptonews.csv."""
    try:
        url = "https://raw.githubusercontent.com/nassim0014/btc-llm-sentiment/main/Data/cryptonews.csv"
        r = requests.get(url, timeout=60)
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
def engineer_features(btc: pd.DataFrame, news: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Engineer the 23 features used by the LSTM model."""
    try:
        # Aggregate news to daily
        news["date_day"] = news["date"].dt.tz_convert(None).dt.floor("D")
        daily_news = (
            news.groupby("date_day")
            .agg(
                news_count=("title", "size"),
                mean_polarity=("sentiment_polarity", "mean"),
                mean_subjectivity=("sentiment_subjectivity", "mean"),
                neg_share=("sentiment_class", lambda s: (s == "negative").mean()),
                pos_share=("sentiment_class", lambda s: (s == "positive").mean()),
                llm_sentiment_mean=("llm_sentiment", "mean"),
                llm_sentiment_std=("llm_sentiment", "std"),
                llm_headline_count=("llm_sentiment", "size"),
                llm_pos_share=("llm_sentiment", lambda s: (s > 0.3).mean()),
                llm_neg_share=("llm_sentiment", lambda s: (s < -0.3).mean()),
            )
            .reset_index()
            .rename(columns={"date_day": "date"})
            .fillna({"llm_sentiment_std": 0})
        )

        df = pd.merge(btc, daily_news, on="date", how="left")
        fill_cols = [
            "news_count", "mean_polarity", "neg_share", "pos_share",
            "llm_sentiment_mean", "llm_sentiment_std", "llm_headline_count",
            "llm_pos_share", "llm_neg_share",
        ]
        df[fill_cols] = df[fill_cols].fillna(0)
        df = df.sort_values("date").reset_index(drop=True)

        # Technical indicators
        def rsi(close, period=14):
            delta = close.diff()
            gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
            return 100 - (100 / (1 + gain / (loss + 1e-12)))

        df["ret_1d"] = np.log(df["close"] / df["close"].shift(1))
        df["ret_3d"] = np.log(df["close"] / df["close"].shift(3))
        df["ret_7d"] = np.log(df["close"] / df["close"].shift(7))
        df["vol_7d"] = df["ret_1d"].rolling(7).std()
        df["vol_21d"] = df["ret_1d"].rolling(21).std()
        df["rsi_14"] = rsi(df["close"], 14)

        ema_fast = df["close"].ewm(span=12, adjust=False).mean()
        ema_slow = df["close"].ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        df["macd_line"] = macd_line
        df["macd_hist"] = macd_line - macd_line.ewm(span=9, adjust=False).mean()

        bb_mid = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_pct_b"] = (df["close"] - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-12)
        df["bb_width"] = (4 * bb_std) / (bb_mid + 1e-12)

        for lag in [1, 2, 3, 5]:
            df[f"llm_sent_lag{lag}"] = df["llm_sentiment_mean"].shift(lag)
            df[f"llm_pos_share_lag{lag}"] = df["llm_pos_share"].shift(lag)
        df["llm_sent_3d_ma"] = df["llm_sentiment_mean"].rolling(3).mean().shift(1)
        df["llm_sent_5d_ma"] = df["llm_sentiment_mean"].rolling(5).mean().shift(1)

        return df
    except Exception:
        return None


# ---------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------
st.title("🎯 Live Predictions")
st.markdown("Fetches the latest BTC price + crypto news, engineers features, and runs the trained LSTM model to predict the next 5-day directional move.")

st.markdown("---")

# Load model + feature bundle
with st.spinner("Loading the trained LSTM model..."):
    model = load_lstm_model()
    bundle = load_feature_bundle()

if model is None:
    st.error("⚠️ Trained LSTM model not found. Run the Phase 2 pipeline first to generate `notebooks/interim/best_optuna_model.keras`.")
    st.stop()

if bundle is None:
    st.error("⚠️ Feature bundle not found. Run `scripts/generate_interim_features.py` first.")
    st.stop()

st.success(f"✅ Model loaded: {model.name} | Features: {len(bundle['feature_cols'])}")

# Fetch live data
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

# Engineer features
st.markdown("---")
st.subheader("🔧 Feature Engineering")

with st.spinner("Engineering 23 features (technical + sentiment)..."):
    df = engineer_features(btc, news)

if df is None or len(df) < 30:
    st.error("⚠️ Could not engineer features. Need at least 30 days of data.")
    st.stop()

# Get the latest row with all features filled
FEATURE_COLS = bundle["feature_cols"]
scaler = bundle["scaler"]

df_features = df[FEATURE_COLS].fillna(0)
latest_features = df_features.iloc[-1:]

st.success(f"✅ Engineered {len(FEATURE_COLS)} features for {df['date'].iloc[-1].date()}")

# Show latest feature values
with st.expander("📋 View latest feature values"):
    feature_display = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Value": latest_features.values[0],
    })
    st.dataframe(feature_display, use_container_width=True, hide_index=True)

# Scale features + reshape for LSTM
latest_scaled = scaler.transform(latest_features)
latest_3d = latest_scaled.reshape(1, 1, len(FEATURE_COLS))

# Run prediction
st.markdown("---")
st.subheader("🤖 LSTM Prediction")

with st.spinner("Running the LSTM model..."):
    try:
        prob_up = float(model.predict(latest_3d, verbose=0).ravel()[0])
    except Exception as e:
        st.error(f"⚠️ Model prediction failed: {e}")
        st.stop()

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
    # Trading signal based on threshold
    threshold = 0.5
    if prob_up >= threshold:
        signal = "🟢 LONG"
        signal_color = "green"
    else:
        signal = "🔴 CASH"
        signal_color = "red"
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

# Recent price chart
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

# Recent sentiment
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

# Latest headlines
st.markdown("#### Latest 5 Headlines")
latest_news = news.tail(5)[["date", "title", "llm_sentiment"]].copy()
latest_news["date"] = latest_news["date"].dt.strftime("%Y-%m-%d %H:%M")
latest_news.columns = ["Date", "Headline", "Sentiment"]
st.dataframe(latest_news, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("🎯 Live Predictions — Uses the Optuna-tuned LSTM model with the same 23 features (technical + sentiment) as training. Prediction is for the 5-day forward directional move.")
