"""Feature engineering extracted from pages/3_🎯_Live_Predictions.py (item 5).

The `engineer_features` function builds the 23 features used by the LSTM
model: news aggregation + technical indicators (RSI, MACD, Bollinger
Bands, returns, volatility) + LLM sentiment lags.

Pure-logic (takes DataFrames, returns a DataFrame) — no Streamlit
dependencies, so it can be unit-tested.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-12)))


def engineer_features(btc: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame | None:
    """Engineer the 23 features used by the LSTM model.

    Args:
        btc: DataFrame with at least 'date' and 'close' columns.
        news: DataFrame with 'date', 'sentiment_polarity',
            'sentiment_subjectivity', 'sentiment_class',
            'llm_sentiment', and 'title' columns.

    Returns:
        DataFrame with all 23 features, or None on failure.
    """
    try:
        # Aggregate news to daily
        news = news.copy()
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
        df["ret_1d"] = np.log(df["close"] / df["close"].shift(1))
        df["ret_3d"] = np.log(df["close"] / df["close"].shift(3))
        df["ret_7d"] = np.log(df["close"] / df["close"].shift(7))
        df["vol_7d"] = df["ret_1d"].rolling(7).std()
        df["vol_21d"] = df["ret_1d"].rolling(21).std()
        df["rsi_14"] = _rsi(df["close"], 14)

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
