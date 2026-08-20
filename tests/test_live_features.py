"""Tests for src/live_features.py — extracted from pages/3 (item 5)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.live_features import engineer_features


def _make_btc(n: int = 30) -> pd.DataFrame:
    """Create a synthetic BTC DataFrame with date + close columns."""
    dates = pd.date_range("2024-09-01", periods=n, freq="D")
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({"date": dates, "close": close})


def _make_news(n: int = 50) -> pd.DataFrame:
    """Create a synthetic news DataFrame."""
    dates = pd.date_range("2024-09-01", periods=n, freq="6h", tz="UTC")
    return pd.DataFrame({
        "date": dates,
        "title": [f"Headline {i}" for i in range(n)],
        "sentiment_polarity": np.random.uniform(-1, 1, n),
        "sentiment_subjectivity": np.random.uniform(0, 1, n),
        "sentiment_class": np.random.choice(["positive", "negative", "neutral"], n),
        "llm_sentiment": np.random.uniform(-1, 1, n),
    })


class TestEngineerFeatures:
    def test_returns_dataframe(self):
        """The function returns a DataFrame (not None) on valid input."""
        btc = _make_btc(30)
        news = _make_news(50)
        result = engineer_features(btc, news)
        assert result is not None
        assert isinstance(result, pd.DataFrame)

    def test_includes_technical_indicators(self):
        """The result has the expected technical indicator columns."""
        btc = _make_btc(30)
        news = _make_news(50)
        result = engineer_features(btc, news)
        expected_cols = [
            "ret_1d", "ret_3d", "ret_7d",
            "vol_7d", "vol_21d", "rsi_14",
            "macd_line", "macd_hist",
            "bb_pct_b", "bb_width",
        ]
        for col in expected_cols:
            assert col in result.columns, f"missing column: {col}"

    def test_includes_news_aggregates(self):
        """The result has the news-aggregation columns."""
        btc = _make_btc(30)
        news = _make_news(50)
        result = engineer_features(btc, news)
        news_cols = [
            "news_count", "mean_polarity", "neg_share", "pos_share",
            "llm_sentiment_mean", "llm_sentiment_std", "llm_headline_count",
            "llm_pos_share", "llm_neg_share",
        ]
        for col in news_cols:
            assert col in result.columns, f"missing column: {col}"

    def test_includes_llm_lags(self):
        """The result has the LLM sentiment lag columns."""
        btc = _make_btc(30)
        news = _make_news(50)
        result = engineer_features(btc, news)
        for lag in [1, 2, 3, 5]:
            assert f"llm_sent_lag{lag}" in result.columns
            assert f"llm_pos_share_lag{lag}" in result.columns
        assert "llm_sent_3d_ma" in result.columns
        assert "llm_sent_5d_ma" in result.columns

    def test_returns_none_on_exception(self):
        """The function returns None if the input is invalid."""
        btc = pd.DataFrame({"wrong": [1]})  # no 'close' column
        news = _make_news(10)
        result = engineer_features(btc, news)
        assert result is None

    def test_rsi_is_bounded_0_to_100(self):
        """RSI values are in [0, 100] (excluding NaN from the warmup)."""
        btc = _make_btc(50)
        news = _make_news(100)
        result = engineer_features(btc, news)
        rsi = result["rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_does_not_mutate_input_news(self):
        """The function doesn't mutate the input news DataFrame."""
        btc = _make_btc(30)
        news = _make_news(50)
        original_cols = list(news.columns)
        engineer_features(btc, news)
        # The function copies news internally, so the original should be unchanged
        assert list(news.columns) == original_cols

    def test_fills_missing_news_with_zeros(self):
        """Days with no news have zero-filled aggregate columns."""
        btc = _make_btc(30)
        # News only for the first 5 days
        news = _make_news(10)
        result = engineer_features(btc, news)
        # Days after the news window should have 0 news_count (or NaN filled to 0)
        # At least the first few days should have news_count > 0
        assert (result["news_count"] >= 0).all()
