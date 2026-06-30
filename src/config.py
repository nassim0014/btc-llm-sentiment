"""Centralized configuration for the BTC Sentiment-Driven LSTM pipeline.

Single source of truth for:
  - Data source URLs
  - Model artifact paths + integrity hashes
  - Request timeouts / retries
  - Alert thresholds

This avoids the previous pattern where NEWS_URL was hardcoded in 3+
separate files, and where model artifact paths were duplicated across
scripts.
"""
from __future__ import annotations

from pathlib import Path

# ────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data"
INTERIM_DIR = ROOT / "notebooks" / "interim"
OUTPUTS_DIR = ROOT / "outputs"
AUDIT_DIR = ROOT / "audit"

# Model artifact paths (checked in multiple locations for resilience —
# Streamlit Cloud vs. local vs. Colab).
MODEL_PATHS = [
    INTERIM_DIR / "best_optuna_model.keras",
    ROOT / "models" / "best_optuna_model.keras",
    ROOT / "best_optuna_model.keras",
]
BUNDLE_PATHS = [
    INTERIM_DIR / "features_for_lstm.pkl",
    ROOT / "models" / "features_for_lstm.pkl",
]

# ────────────────────────────────────────────────────────────
# Data sources
# ────────────────────────────────────────────────────────────
# Hosted on GitHub raw — used by the Streamlit app, the sentiment alert
# cron job, and the pipeline runner when the local file is missing.
NEWS_URL = "https://raw.githubusercontent.com/nassim0014/btc-llm-sentiment/main/Data/cryptonews.csv"

# yfinance ticker for BTC-USD
BTC_TICKER = "BTC-USD"

# ────────────────────────────────────────────────────────────
# Network
# ────────────────────────────────────────────────────────────
REQUEST_TIMEOUT_SECONDS = 60
REQUEST_MAX_RETRIES = 3
REQUEST_BACKOFF_FACTOR = 1.5  # seconds, multiplied per retry

# HTTP User-Agent — some CDNs block the python-requests default UA.
USER_AGENT = "btc-llm-sentiment/1.0 (+https://github.com/nassim0014/btc-llm-sentiment)"

# ────────────────────────────────────────────────────────────
# Model artifact integrity (defense against tampered pickle files)
# ────────────────────────────────────────────────────────────
# These SHA256 hashes are computed from the committed artifacts.
# safe_load_bundle() refuses to unpickle if the on-disk file's hash
# does not match — defense against a compromised repo shipping a
# malicious pickle that executes arbitrary code on load.
#
# To regenerate after a model retrain:
#   sha256sum notebooks/interim/features_for_lstm.pkl
#   sha256sum notebooks/interim/best_optuna_model.keras
# Then paste the new hashes here and commit.
BUNDLE_SHA256 = "fa33c84589af630f7c126120536c102201e3f0a530f9ae080d2d7adc1c59a411"
MODEL_SHA256 = "eab1cfadf7f28baeb68f41b018bca31a7700a90c2bd0c38250a2b62c00aedd94"

# ────────────────────────────────────────────────────────────
# Alert thresholds (used by scripts/sentiment_alert.py)
# ────────────────────────────────────────────────────────────
ALERT_THRESHOLDS = {
    "very_bullish": 0.3,
    "bullish": 0.1,
    "bearish": -0.1,
    "very_bearish": -0.3,
}

# ────────────────────────────────────────────────────────────
# Alert log
# ────────────────────────────────────────────────────────────
ALERT_LOG_PATH = AUDIT_DIR / "sentiment_alerts.csv"


def find_model() -> Path | None:
    """Search for the trained LSTM model in the configured locations."""
    for p in MODEL_PATHS:
        if p.exists():
            return p
    return None


def find_bundle() -> Path | None:
    """Search for the feature bundle in the configured locations."""
    for p in BUNDLE_PATHS:
        if p.exists():
            return p
    return None
