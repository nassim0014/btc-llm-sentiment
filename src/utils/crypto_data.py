"""BTC price data fetcher with yfinance + CoinGecko fallback.

## Why this module exists

Yahoo Finance (yfinance) frequently rate-limits cloud provider IP ranges
(AWS, GCP, Azure — including Streamlit Cloud). When that happens, every
page that fetches BTC prices breaks simultaneously. This module provides
a single `fetch_btc_close()` function that tries yfinance first, then
falls back to the free CoinGecko API (no API key required, reliable,
doesn't rate-limit cloud IPs).

## Usage

    from src.utils.crypto_data import fetch_btc_close, fetch_btc_current

    # Historical daily candles (DataFrame with 'date' and 'close' columns)
    df = fetch_btc_close(days=60)

    # Current price + 24h change (dict)
    info = fetch_btc_current()

Both functions return None on total failure (both sources down). The
CoinGecko fallback handles the common Streamlit Cloud case where yfinance
is rate-limited but the rest of the internet is fine.
"""
from __future__ import annotations

import logging
from datetime import UTC

import pandas as pd

from src.config import REQUEST_TIMEOUT_SECONDS, USER_AGENT

logger = logging.getLogger(__name__)

# CoinGecko free API — no key required, 10-30 calls/min rate limit.
# Docs: https://www.coingecko.com/api/documentation
# NOTE: free tier limited to past 365 days of historical data.
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# Binance public API — no key required, full historical data, reliable.
# Docs: https://binance-docs.github.io/apidocs/spot/en/
# Used as a third fallback for date ranges older than 365 days
# (CoinGecko free tier doesn't support those).
BINANCE_BASE = "https://api.binance.com"


# ────────────────────────────────────────────────────────────
# yfinance primary
# ────────────────────────────────────────────────────────────


def _fetch_yfinance_daily(days: int) -> pd.DataFrame | None:
    """Fetch daily BTC-USD candles via yfinance. Returns None on any failure."""
    try:
        import yfinance as yf

        btc = yf.download("BTC-USD", period=f"{days}d", auto_adjust=False, progress=False)
        if btc is None or len(btc) == 0:
            return None
        if isinstance(btc.columns, pd.MultiIndex):
            btc.columns = [" ".join(c).strip() for c in btc.columns]
        close_col = [c for c in btc.columns if "Close" in c][0]
        btc = btc.reset_index().rename(columns={"Date": "date"})
        btc = btc[["date", close_col]].rename(columns={close_col: "close"})
        btc["date"] = pd.to_datetime(btc["date"]).dt.floor("D")
        btc["close"] = pd.to_numeric(btc["close"], errors="coerce")
        btc = btc.dropna(subset=["close"])
        return btc if len(btc) > 0 else None
    except Exception as e:
        logger.info(f"yfinance daily failed: {type(e).__name__}: {e}")
        return None


def _fetch_yfinance_range(start: str, end: str) -> pd.DataFrame | None:
    """Fetch BTC-USD daily candles for a date range via yfinance."""
    try:
        import yfinance as yf

        btc = yf.download("BTC-USD", start=start, end=end, auto_adjust=False, progress=False)
        if btc is None or len(btc) == 0:
            return None
        if isinstance(btc.columns, pd.MultiIndex):
            btc.columns = [" ".join(c).strip() for c in btc.columns]
        close_col = [c for c in btc.columns if "Close" in c][0]
        btc = btc.reset_index().rename(columns={"Date": "date"})
        btc = btc[["date", close_col]].rename(columns={close_col: "close"})
        btc["date"] = pd.to_datetime(btc["date"]).dt.floor("D")
        btc["close"] = pd.to_numeric(btc["close"], errors="coerce")
        btc = btc.dropna(subset=["close"])
        return btc if len(btc) > 0 else None
    except Exception as e:
        logger.info(f"yfinance range failed: {type(e).__name__}: {e}")
        return None


def _fetch_yfinance_current() -> dict | None:
    """Fetch current BTC price + 24h change via yfinance (hourly candles)."""
    try:
        import yfinance as yf

        btc = yf.download("BTC-USD", period="7d", interval="1h", progress=False, auto_adjust=False)
        if btc is None or len(btc) == 0:
            return None
        if isinstance(btc.columns, pd.MultiIndex):
            btc.columns = [" ".join(c).strip() for c in btc.columns]
        close_col = [c for c in btc.columns if "Close" in c][0]
        current_price = float(btc[close_col].iloc[-1])
        if len(btc) >= 24:
            prev_price = float(btc[close_col].iloc[-24])
            change_pct = ((current_price - prev_price) / prev_price) * 100
        else:
            change_pct = 0.0
        history = btc[[close_col]].copy()
        history.columns = ["price"]
        history.index.name = "datetime"
        return {
            "price": current_price,
            "change_24h_pct": change_pct,
            "history": history.reset_index(),
            "source": "yfinance",
        }
    except Exception as e:
        logger.info(f"yfinance current failed: {type(e).__name__}: {e}")
        return None


# ────────────────────────────────────────────────────────────
# CoinGecko fallback
# ────────────────────────────────────────────────────────────


def _fetch_coingecko_daily(days: int) -> pd.DataFrame | None:
    """Fetch daily BTC-USD candles via CoinGecko. Returns None on failure."""
    try:
        import requests

        # CoinGecko /coins/{id}/market_chart — returns hourly prices for up to 90 days.
        # We request daily granularity by using /range with daily interval.
        # For simplicity, use /market_chart with days=days and downsample.
        url = f"{COINGECKO_BASE}/coins/bitcoin/market_chart"
        params = {"vs_currency": "usd", "days": str(days), "interval": "daily"}
        r = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers={"User-Agent": USER_AGENT, "accept": "application/json"},
        )
        r.raise_for_status()
        data = r.json()
        prices = data.get("prices", [])
        if not prices:
            return None
        df = pd.DataFrame(prices, columns=["timestamp_ms", "close"])
        df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.floor("D")
        df = df[["date", "close"]].sort_values("date").reset_index(drop=True)
        # Deduplicate by date (CoinGecko may return multiple points per day)
        df = df.groupby("date", as_index=False).last()
        return df if len(df) > 0 else None
    except Exception as e:
        logger.info(f"CoinGecko daily failed: {type(e).__name__}: {e}")
        return None


def _fetch_coingecko_range(start: str, end: str) -> pd.DataFrame | None:
    """Fetch BTC-USD daily candles for a date range via CoinGecko.

    NOTE: CoinGecko free API only supports the past 365 days. For older
    ranges, this returns None and the caller should try Binance instead.
    """
    try:
        from datetime import datetime

        import requests

        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
        end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC)
        start_ts = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())

        url = f"{COINGECKO_BASE}/coins/bitcoin/market_chart/range"
        params = {"vs_currency": "usd", "from": start_ts, "to": end_ts}
        r = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers={"User-Agent": USER_AGENT, "accept": "application/json"},
        )
        r.raise_for_status()
        data = r.json()
        prices = data.get("prices", [])
        if not prices:
            return None
        df = pd.DataFrame(prices, columns=["timestamp_ms", "close"])
        df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.floor("D")
        df = df[["date", "close"]].sort_values("date").reset_index(drop=True)
        df = df.groupby("date", as_index=False).last()
        return df if len(df) > 0 else None
    except Exception as e:
        logger.info(f"CoinGecko range failed: {type(e).__name__}: {e}")
        return None


def _fetch_binance_range(start: str, end: str) -> pd.DataFrame | None:
    """Fetch BTC-USDT daily candles for a date range via Binance.

    Binance's klines endpoint returns full historical data with no API
    key required. Used as a fallback when CoinGecko's 365-day limit is
    exceeded.

    Returns a DataFrame with 'date' and 'close' columns.
    """
    try:
        from datetime import datetime

        import requests

        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
        end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC)
        start_ts = int(start_dt.timestamp() * 1000)  # Binance uses ms
        end_ts = int(end_dt.timestamp() * 1000)

        url = f"{BINANCE_BASE}/api/v3/klines"
        params = {
            "symbol": "BTCUSDT",
            "interval": "1d",
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": 1000,
        }
        r = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers={"User-Agent": USER_AGENT},
        )
        r.raise_for_status()
        # Binance returns an array of arrays:
        # [open_time, open, high, low, close, volume, close_time, ...]
        rows = r.json()
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ])
        df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.floor("D")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df[["date", "close"]].sort_values("date").reset_index(drop=True)
        df = df.dropna(subset=["close"])
        return df if len(df) > 0 else None
    except Exception as e:
        logger.info(f"Binance range failed: {type(e).__name__}: {e}")
        return None


def _fetch_coingecko_current() -> dict | None:
    """Fetch current BTC price + 24h change via CoinGecko."""
    try:
        import requests

        url = f"{COINGECKO_BASE}/simple/price"
        params = {
            "ids": "bitcoin",
            "vs_currencies": "usd",
            "include_24hr_change": "true",
            "include_last_updated_at": "true",
        }
        r = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers={"User-Agent": USER_AGENT, "accept": "application/json"},
        )
        r.raise_for_status()
        data = r.json()
        btc = data.get("bitcoin", {})
        price = btc.get("usd")
        change_24h = btc.get("usd_24h_change", 0.0)
        if price is None:
            return None
        # Also fetch 7-day history for the sparkline
        hist_df = _fetch_coingecko_daily(7)
        history = None
        if hist_df is not None and len(hist_df) > 0:
            history = pd.DataFrame({
                "datetime": hist_df["date"],
                "price": hist_df["close"],
            })
        return {
            "price": float(price),
            "change_24h_pct": float(change_24h),
            "history": history,
            "source": "coingecko",
        }
    except Exception as e:
        logger.info(f"CoinGecko current failed: {type(e).__name__}: {e}")
        return None


# ────────────────────────────────────────────────────────────
# Public API — try yfinance, fall back to CoinGecko
# ────────────────────────────────────────────────────────────


def fetch_btc_close(days: int = 60) -> tuple[pd.DataFrame | None, str | None]:
    """Fetch daily BTC-USD close prices for the last `days` days.

    Tries yfinance first; on failure falls back to CoinGecko.

    Returns (df, source) where source is 'yfinance' or 'coingecko'.
    Returns (None, error) if both sources fail.
    """
    df = _fetch_yfinance_daily(days)
    if df is not None:
        return df, "yfinance"

    df = _fetch_coingecko_daily(days)
    if df is not None:
        return df, "coingecko"

    return None, "Both yfinance and CoinGecko failed — check network connectivity."


def fetch_btc_range(start: str, end: str) -> tuple[pd.DataFrame | None, str | None]:
    """Fetch daily BTC-USD close prices for a date range.

    Tries yfinance → CoinGecko (if within 365 days) → Binance.

    Args:
        start: YYYY-MM-DD
        end: YYYY-MM-DD

    Returns (df, source) or (None, error). Source is one of:
    'yfinance', 'coingecko', 'binance'.
    """
    df = _fetch_yfinance_range(start, end)
    if df is not None:
        return df, "yfinance"

    df = _fetch_coingecko_range(start, end)
    if df is not None:
        return df, "coingecko"

    df = _fetch_binance_range(start, end)
    if df is not None:
        return df, "binance"

    return None, f"All 3 sources (yfinance, CoinGecko, Binance) failed for range {start} to {end}."


def fetch_btc_current() -> tuple[dict | None, str | None]:
    """Fetch current BTC price + 24h change + 7-day history.

    Tries yfinance first; on failure falls back to CoinGecko.

    Returns (data, source) or (None, error). The data dict has:
        price: float
        change_24h_pct: float
        history: DataFrame or None
        source: 'yfinance' or 'coingecko'
    """
    data = _fetch_yfinance_current()
    if data is not None:
        return data, "yfinance"

    data = _fetch_coingecko_current()
    if data is not None:
        return data, "coingecko"

    return None, "Both yfinance and CoinGecko failed — check network connectivity."
