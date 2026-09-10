"""Tests for scripts/sentiment_alert.py's staleness guard.

The alert cron runs daily against a static news CSV. If that CSV stops
being refreshed, `evaluate_alert()` used to have no way to know — it would
keep firing confident directional (bullish/bearish) verdicts off data that
never changes, forever. These tests pin the OLD behavior directly from the
`main` branch (via `git show`, not a narrated claim) to prove the bug was
real, then prove the NEW code fixes it without changing behavior on fresh
data.
"""
from __future__ import annotations

import importlib.util
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts.sentiment_alert import evaluate_alert

ROOT = Path(__file__).resolve().parent.parent

# CI checks out the PR merge ref at shallow depth without fetching `main`
# (only the gitleaks job passes fetch-depth: 0), so `main` may not exist as a
# local ref even though `.git` is present. Try the plausible names for the
# base branch and fall back to skipping — a false-red CI run from a missing
# ref is worse than a skipped pin, and the other 5 tests still cover the new
# behavior either way.
_BASE_REV_CANDIDATES = ("main", "origin/main")


def _resolve_base_rev() -> str | None:
    for rev in _BASE_REV_CANDIDATES:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", rev],
            cwd=ROOT,
            capture_output=True,
        )
        if result.returncode == 0:
            return rev
    return None


BASE_REV = _resolve_base_rev()


def _stale_bullish_payload(now: datetime) -> dict:
    """A payload whose news is 800 days old (matches the real ~2yr freeze
    found in Data/cryptonews.csv) but whose average sentiment is
    comfortably above the very_bullish threshold (0.3)."""
    stale_date = now - timedelta(days=800)
    return {
        "latest_headline": "BTC surges on ETF optimism",
        "latest_sentiment": 0.55,
        "daily_avg_sentiment": 0.5,
        "source_date": stale_date.isoformat(),
        "headline_count_24h": 12,
        "fetch_time": now.isoformat(),
    }


def _fresh_bullish_payload(now: datetime) -> dict:
    """Same sentiment value as above, but the news is 6 hours old."""
    fresh_date = now - timedelta(hours=6)
    return {
        "latest_headline": "BTC surges on ETF optimism",
        "latest_sentiment": 0.55,
        "daily_avg_sentiment": 0.5,
        "source_date": fresh_date.isoformat(),
        "headline_count_24h": 12,
        "fetch_time": now.isoformat(),
    }


def _load_module_from_git(rev: str, relpath: str, name: str):
    """Load `relpath` exactly as it exists at git revision `rev`, without
    checking out or touching the working tree. This is a real base-commit
    pin: the executed source is what `git show` returns for that revision,
    not a copy or a paraphrase."""
    result = subprocess.run(
        ["git", "show", f"{rev}:{relpath}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    spec = importlib.util.spec_from_loader(name, loader=None, origin=str(ROOT / relpath))
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(ROOT / relpath)
    exec(compile(result.stdout, f"<git:{rev}:{relpath}>", "exec"), module.__dict__)
    return module


@pytest.mark.skipif(
    BASE_REV is None,
    reason="no local 'main'/'origin/main' ref to pin against (shallow checkout without fetch-depth: 0)",
)
def test_old_code_fires_directional_alert_on_stale_data():
    """Base-commit regression proof: the OLD evaluate_alert() (as committed
    on main, before this change) has no staleness concept at all, so
    800-day-old news scored +0.5 fires a confident VERY_BULLISH alert
    exactly as if it were live."""
    now = datetime.now(UTC)
    old_module = _load_module_from_git(BASE_REV, "scripts/sentiment_alert.py", "old_sentiment_alert")
    alert = old_module.evaluate_alert(_stale_bullish_payload(now))
    assert alert is not None
    assert alert["level"] == "VERY_BULLISH"


def test_new_code_flags_stale_data_instead_of_a_directional_alert():
    now = datetime.now(UTC)
    alert = evaluate_alert(_stale_bullish_payload(now), now=now)
    assert alert is not None
    assert alert["level"] == "STALE_DATA"


def test_new_code_still_fires_directional_alerts_on_fresh_data():
    now = datetime.now(UTC)
    alert = evaluate_alert(_fresh_bullish_payload(now), now=now)
    assert alert is not None
    assert alert["level"] == "VERY_BULLISH"


def test_default_now_matches_explicit_now_for_fresh_data():
    """Calling evaluate_alert() the old way (no now= kwarg) must still work
    and must not itself misclassify genuinely fresh data as stale."""
    now = datetime.now(UTC)
    alert = evaluate_alert(_fresh_bullish_payload(now))
    assert alert is not None
    assert alert["level"] == "VERY_BULLISH"


def test_in_range_sentiment_still_returns_none_when_fresh():
    now = datetime.now(UTC)
    payload = _fresh_bullish_payload(now)
    payload["daily_avg_sentiment"] = 0.0
    assert evaluate_alert(payload, now=now) is None


def test_stale_check_respects_custom_max_source_age_days():
    now = datetime.now(UTC)
    payload = _fresh_bullish_payload(now)  # 6 hours old
    alert = evaluate_alert(payload, now=now, max_source_age_days=0.1)  # ~2.4h window
    assert alert is not None
    assert alert["level"] == "STALE_DATA"
