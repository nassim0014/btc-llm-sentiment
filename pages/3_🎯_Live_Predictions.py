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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Live Predictions", page_icon="🎯", layout="wide")

# Use centralized config + safe pickle loader
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import NEWS_URL, find_bundle, find_model
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
    """Load the saved Optuna-tuned LSTM model.

    Returns (model, error_message). On success, error_message is None.
    On failure, model is None and error_message explains what went wrong
    so the UI can show it instead of a generic 'not found' message.
    """
    path = find_model()
    if path is None:
        return None, "Model file not found in any expected location. Searched: " + ", ".join(str(p) for p in MODEL_PATHS)
    try:
        import os
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        model = tf.keras.models.load_model(str(path))
        return model, None
    except ImportError as e:
        return None, f"TensorFlow could not be imported: {e}. Check that `tensorflow-cpu` is in requirements.txt and installed on Streamlit Cloud."
    except Exception as e:
        return None, f"TensorFlow failed to load the model: {type(e).__name__}: {e}"


@st.cache_data(ttl=300)
def fetch_latest_btc(days: int = 60) -> tuple[pd.DataFrame | None, str | None]:
    """Fetch the latest BTC-USD daily candles via yfinance.

    Returns (df, error_message). On success, error_message is None.
    On failure, df is None and error_message explains what went wrong.
    """
    try:
        import yfinance as yf
        btc = yf.download("BTC-USD", period=f"{days}d", auto_adjust=False, progress=False)
        if btc is None or len(btc) == 0:
            return None, "yfinance returned empty data — Yahoo Finance may be rate-limiting or down."
        if isinstance(btc.columns, pd.MultiIndex):
            btc.columns = [" ".join(c).strip() for c in btc.columns]
        btc = btc.reset_index().rename(columns={"Date": "date"})
        btc.columns = [c.replace(" BTC-USD", "").lower() for c in btc.columns]
        btc["date"] = pd.to_datetime(btc["date"]).dt.floor("D")
        return btc, None
    except ImportError:
        return None, "yfinance is not installed. Add `yfinance` to requirements.txt."
    except Exception as e:
        return None, f"yfinance error: {type(e).__name__}: {e}"


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
def engineer_features(btc: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame | None:
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
model, model_error = load_lstm_model()
bundle = load_feature_bundle()
model_available = model is not None and bundle is not None

if not model_available:
    # Show the ACTUAL error instead of a generic "not found" message.
    # The most common causes are:
    #   1. TensorFlow not installed on Streamlit Cloud (requirements.txt issue)
    #   2. TensorFlow version mismatch (model saved with 2.17, runtime has older)
    #   3. Model file genuinely missing (shouldn't happen — it's committed)
    st.warning("⚠️ Trained LSTM model could not be loaded in this deployment.")
    if model is None and model_error:
        st.error(f"**Model load error:** `{model_error}`")
    if bundle is None:
        st.error("**Feature bundle error:** Could not load `features_for_lstm.pkl`. See logs above for SHA256 or unpickler errors.")
    st.info("""
    **To enable live predictions:**
    - If the error above says TensorFlow could not be imported, check that
      `tensorflow-cpu` is listed in `requirements.txt` (it is, but Streamlit
      Cloud may have failed to install it — check the app logs).
    - If the error says the model file is not found, run the Phase 2 Optuna
      pipeline locally and commit the file to `notebooks/interim/`.
    - If the error is a TensorFlow version mismatch, retrain the model with
      the same TF version that Streamlit Cloud uses.

    **Below:** Live BTC price, news sentiment, and feature engineering are
    still available — only the final LSTM prediction requires the model.
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
        btc, btc_error = fetch_latest_btc(days=60)
    if btc is not None:
        st.success(f"✅ Fetched {len(btc)} days of BTC-USD data (latest: {btc['date'].iloc[-1].date()})")
    else:
        st.error("⚠️ Could not fetch BTC price data.")
        if btc_error:
            st.caption(f"**Error:** `{btc_error}`")

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
status = "with LSTM prediction" if model_available else "without LSTM prediction (model not loaded)"
st.caption(f"🎯 Live Predictions — Uses the Optuna-tuned LSTM model with the same 23 features (technical + sentiment) as training. Running {status}.")

# ---------------------------------------------------------------------
# Diagnostics — show installed package versions + file paths for debugging
# ---------------------------------------------------------------------
with st.expander("🔧 Diagnostics (click to expand)", expanded=False):
    st.markdown("**Installed package versions:**")
    import importlib

    pkgs = ["tensorflow", "yfinance", "numpy", "pandas", "scikit_learn", "streamlit", "plotly"]
    version_rows = []
    for pkg in pkgs:
        try:
            mod = importlib.import_module(pkg.replace("-", "_"))
            ver = getattr(mod, "__version__", "unknown")
            version_rows.append({"Package": pkg, "Version": ver, "Status": "✅ installed"})
        except ImportError:
            version_rows.append({"Package": pkg, "Version": "—", "Status": "❌ NOT installed"})
    st.dataframe(version_rows, use_container_width=True, hide_index=True)

    st.markdown("**Model artifact paths:**")
    model_path = find_model()
    bundle_path = find_bundle()
    path_rows = [
        {"Artifact": "LSTM model (.keras)", "Expected path": str(MODEL_PATHS[0]), "Found": "✅" if model_path else "❌", "Resolved": str(model_path) if model_path else "—"},
        {"Artifact": "Feature bundle (.pkl)", "Expected path": str(BUNDLE_PATHS[0]), "Found": "✅" if bundle_path else "❌", "Resolved": str(bundle_path) if bundle_path else "—"},
    ]
    st.dataframe(path_rows, use_container_width=True, hide_index=True)

    if model_path:
        st.caption(f"Model file size: {model_path.stat().st_size / 1024:.1f} KB")
    if bundle_path:
        st.caption(f"Bundle file size: {bundle_path.stat().st_size / 1024:.1f} KB")

    st.markdown("**Errors captured this session:**")
    if model_error:
        st.code(f"Model: {model_error}", language="text")
    else:
        st.caption("Model: no error (loaded successfully or not attempted)")
    if btc_error:
        st.code(f"BTC fetch: {btc_error}", language="text")
    else:
        st.caption("BTC fetch: no error")
