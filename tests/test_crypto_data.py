"""Tests for the crypto data fetcher with yfinance + CoinGecko + Binance fallback.

These tests verify the fallback logic without requiring network access
(the actual API calls are tested manually / in CI with network).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.crypto_data import (
    fetch_btc_close,
    fetch_btc_current,
    fetch_btc_range,
)

# ────────────────────────────────────────────────────────────
# 1. Fallback logic — yfinance success → no fallback called
# ────────────────────────────────────────────────────────────


def test_fetch_btc_close_uses_yfinance_when_available():
    """When yfinance succeeds, CoinGecko should not be called."""
    fake_df = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=3), "close": [100, 101, 102]})
    with patch("src.utils.crypto_data._fetch_yfinance_daily", return_value=fake_df) as mock_yf, \
         patch("src.utils.crypto_data._fetch_coingecko_daily") as mock_cg:
        df, source = fetch_btc_close(days=3)
        assert df is not None
        assert source == "yfinance"
        assert len(df) == 3
        mock_yf.assert_called_once_with(3)
        mock_cg.assert_not_called()


def test_fetch_btc_current_uses_yfinance_when_available():
    """When yfinance succeeds, CoinGecko should not be called."""
    fake_data = {"price": 50000.0, "change_24h_pct": 2.5, "history": None, "source": "yfinance"}
    with patch("src.utils.crypto_data._fetch_yfinance_current", return_value=fake_data) as mock_yf, \
         patch("src.utils.crypto_data._fetch_coingecko_current") as mock_cg:
        data, source = fetch_btc_current()
        assert data is not None
        assert source == "yfinance"
        assert data["price"] == 50000.0
        mock_yf.assert_called_once()
        mock_cg.assert_not_called()


# ────────────────────────────────────────────────────────────
# 2. Fallback logic — yfinance fails → CoinGecko called
# ────────────────────────────────────────────────────────────


def test_fetch_btc_close_falls_back_to_coingecko():
    """When yfinance returns None, CoinGecko should be tried."""
    fake_df = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=3), "close": [100, 101, 102]})
    with patch("src.utils.crypto_data._fetch_yfinance_daily", return_value=None) as mock_yf, \
         patch("src.utils.crypto_data._fetch_coingecko_daily", return_value=fake_df) as mock_cg:
        df, source = fetch_btc_close(days=3)
        assert df is not None
        assert source == "coingecko"
        mock_yf.assert_called_once()
        mock_cg.assert_called_once()


def test_fetch_btc_current_falls_back_to_coingecko():
    """When yfinance returns None, CoinGecko should be tried."""
    fake_data = {"price": 50000.0, "change_24h_pct": 2.5, "history": None, "source": "coingecko"}
    with patch("src.utils.crypto_data._fetch_yfinance_current", return_value=None), \
         patch("src.utils.crypto_data._fetch_coingecko_current", return_value=fake_data) as mock_cg:
        data, source = fetch_btc_current()
        assert data is not None
        assert source == "coingecko"
        mock_cg.assert_called_once()


# ────────────────────────────────────────────────────────────
# 3. Fallback logic — all sources fail → returns error
# ────────────────────────────────────────────────────────────


def test_fetch_btc_close_returns_error_when_all_fail():
    """When both yfinance and CoinGecko return None, should return (None, error)."""
    with patch("src.utils.crypto_data._fetch_yfinance_daily", return_value=None), \
         patch("src.utils.crypto_data._fetch_coingecko_daily", return_value=None):
        df, source = fetch_btc_close(days=3)
        assert df is None
        assert "failed" in source.lower() or "both" in source.lower()


def test_fetch_btc_current_returns_error_when_all_fail():
    with patch("src.utils.crypto_data._fetch_yfinance_current", return_value=None), \
         patch("src.utils.crypto_data._fetch_coingecko_current", return_value=None):
        data, source = fetch_btc_current()
        assert data is None
        assert "failed" in source.lower() or "both" in source.lower()


# ────────────────────────────────────────────────────────────
# 4. Range fallback — yfinance → CoinGecko → Binance
# ────────────────────────────────────────────────────────────


def test_fetch_btc_range_uses_yfinance_first():
    fake_df = pd.DataFrame({"date": pd.date_range("2024-09-01", periods=10), "close": range(10)})
    with patch("src.utils.crypto_data._fetch_yfinance_range", return_value=fake_df) as mock_yf, \
         patch("src.utils.crypto_data._fetch_coingecko_range") as mock_cg, \
         patch("src.utils.crypto_data._fetch_binance_range") as mock_bn:
        df, source = fetch_btc_range("2024-09-01", "2024-09-10")
        assert df is not None
        assert source == "yfinance"
        mock_yf.assert_called_once()
        mock_cg.assert_not_called()
        mock_bn.assert_not_called()


def test_fetch_btc_range_falls_back_to_coingecko_then_binance():
    """When yfinance and CoinGecko both fail, Binance should be tried."""
    fake_df = pd.DataFrame({"date": pd.date_range("2024-09-01", periods=10), "close": range(10)})
    with patch("src.utils.crypto_data._fetch_yfinance_range", return_value=None), \
         patch("src.utils.crypto_data._fetch_coingecko_range", return_value=None), \
         patch("src.utils.crypto_data._fetch_binance_range", return_value=fake_df) as mock_bn:
        df, source = fetch_btc_range("2024-09-01", "2024-09-10")
        assert df is not None
        assert source == "binance"
        mock_bn.assert_called_once()


def test_fetch_btc_range_all_fail_returns_error():
    with patch("src.utils.crypto_data._fetch_yfinance_range", return_value=None), \
         patch("src.utils.crypto_data._fetch_coingecko_range", return_value=None), \
         patch("src.utils.crypto_data._fetch_binance_range", return_value=None):
        df, source = fetch_btc_range("2024-09-01", "2024-09-10")
        assert df is None
        assert "failed" in source.lower() or "all" in source.lower()


# ────────────────────────────────────────────────────────────
# 5. DataFrame structure validation
# ────────────────────────────────────────────────────────────


def test_fetched_dataframe_has_required_columns():
    """All fetchers must return a DataFrame with 'date' and 'close' columns."""
    fake_df = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=5), "close": [100, 101, 102, 103, 104]})
    with patch("src.utils.crypto_data._fetch_yfinance_daily", return_value=fake_df):
        df, _ = fetch_btc_close(days=5)
        assert "date" in df.columns
        assert "close" in df.columns
        assert len(df) == 5
