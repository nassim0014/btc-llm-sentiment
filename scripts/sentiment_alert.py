"""
Sentiment Alert System — Monitors crypto news sentiment and sends notifications
when the sentiment score crosses defined thresholds.

Alert types:
  - 🔴 Very Bearish: sentiment < -0.3
  - 🟡 Bearish: sentiment < -0.1
  - 🟢 Very Bullish: sentiment > 0.3
  - 📈 Bullish: sentiment > 0.1
  - ⚠️ Stale Data: source news is older than ALERT_MAX_SOURCE_AGE_DAYS —
    fires instead of any directional alert above, regardless of the
    computed sentiment value

Notification channels (configured via environment variables):
  - Email (SMTP): SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, ALERT_EMAIL_TO
  - Slack: SLACK_WEBHOOK_URL
  - Discord: DISCORD_WEBHOOK_URL

Usage:
    python3 scripts/sentiment_alert.py
    python3 scripts/sentiment_alert.py --dry-run  # test without sending

GitHub Actions:
    Runs daily via .github/workflows/sentiment_alert.yml
"""
from __future__ import annotations

import ast
import os
import smtplib

# Use centralized config
import sys
from datetime import UTC, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import (
    ALERT_LOG_PATH,
    ALERT_MAX_SOURCE_AGE_DAYS,
    ALERT_THRESHOLDS,
    NEWS_URL,
    REQUEST_TIMEOUT_SECONDS,
    USER_AGENT,
)

# ---------------------------------------------------------------------
# Configuration (imported from src.config)
# ---------------------------------------------------------------------
# NEWS_URL, ALERT_LOG_PATH, ALERT_THRESHOLDS, USER_AGENT, REQUEST_TIMEOUT_SECONDS
# are all defined in src/config.py now.
THRESHOLDS = ALERT_THRESHOLDS  # alias for backwards-compat with the rest of this file


# ---------------------------------------------------------------------
# Sentiment fetch
# ---------------------------------------------------------------------
def fetch_latest_sentiment() -> dict | None:
    """Fetch the latest crypto news sentiment from the GitHub-hosted CSV."""
    try:
        r = requests.get(
            NEWS_URL,
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers={"User-Agent": USER_AGENT},
        )
        r.raise_for_status()
        news = pd.read_csv(StringIO(r.text))
        news["date"] = pd.to_datetime(news["date"], format="mixed", utc=True, errors="coerce")
        news = news.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        def parse(s):
            try:
                d = ast.literal_eval(s) if isinstance(s, str) else {}
                return float(d.get("polarity", 0.0))
            except Exception:
                return 0.0

        news["sentiment"] = news["sentiment"].apply(parse)
        latest = news.iloc[-1]

        # Daily average (last 24h)
        cutoff = latest["date"] - pd.Timedelta(hours=24)
        recent = news[news["date"] >= cutoff]
        daily_avg = recent["sentiment"].mean() if len(recent) > 0 else 0.0

        return {
            "latest_headline": str(latest.get("title", "N/A"))[:200],
            "latest_sentiment": float(latest["sentiment"]),
            "daily_avg_sentiment": float(daily_avg),
            "source_date": str(latest["date"]),
            "headline_count_24h": len(recent),
            "fetch_time": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        print(f"[ERROR] Could not fetch sentiment: {e}")
        return None


# ---------------------------------------------------------------------
# Alert evaluation
# ---------------------------------------------------------------------
def evaluate_alert(
    sentiment_data: dict,
    *,
    now: datetime | None = None,
    max_source_age_days: float = ALERT_MAX_SOURCE_AGE_DAYS,
) -> dict | None:
    """Determine if an alert should be fired based on sentiment thresholds.

    Before checking direction, checks whether the underlying news data is
    stale relative to `now`. The daily average is computed from whatever
    headlines the source happens to contain — if the source hasn't been
    updated in days, a "very bullish" or "very bearish" verdict is not a
    live signal, it's the same number firing again. When `source_date` is
    older than `max_source_age_days`, this returns a STALE_DATA alert
    instead of a directional one, no matter how extreme the stored
    sentiment value is.
    """
    if now is None:
        now = datetime.now(UTC)

    source_date = pd.to_datetime(sentiment_data["source_date"], utc=True, errors="coerce")
    if pd.notna(source_date):
        age_days = (now - source_date.to_pydatetime()).total_seconds() / 86400
        if age_days > max_source_age_days:
            return {
                "level": "STALE_DATA",
                "emoji": "⚠️",
                "message": (
                    f"Sentiment source data is {age_days:.1f} days old "
                    f"(source: {sentiment_data['source_date']}) — no directional "
                    f"alert fired."
                ),
                "sentiment": sentiment_data["daily_avg_sentiment"],
                "headline": sentiment_data["latest_headline"],
                "source_date": sentiment_data["source_date"],
                "headline_count": sentiment_data["headline_count_24h"],
                "timestamp": sentiment_data["fetch_time"],
            }

    avg = sentiment_data["daily_avg_sentiment"]

    if avg >= THRESHOLDS["very_bullish"]:
        level = "VERY_BULLISH"
        emoji = "🟢"
        message = f"Very bullish sentiment detected ({avg:+.3f})"
    elif avg >= THRESHOLDS["bullish"]:
        level = "BULLISH"
        emoji = "📈"
        message = f"Bullish sentiment detected ({avg:+.3f})"
    elif avg <= THRESHOLDS["very_bearish"]:
        level = "VERY_BEARISH"
        emoji = "🔴"
        message = f"Very bearish sentiment detected ({avg:+.3f})"
    elif avg <= THRESHOLDS["bearish"]:
        level = "BEARISH"
        emoji = "🟡"
        message = f"Bearish sentiment detected ({avg:+.3f})"
    else:
        return None  # No alert

    return {
        "level": level,
        "emoji": emoji,
        "message": message,
        "sentiment": avg,
        "headline": sentiment_data["latest_headline"],
        "source_date": sentiment_data["source_date"],
        "headline_count": sentiment_data["headline_count_24h"],
        "timestamp": sentiment_data["fetch_time"],
    }


# ---------------------------------------------------------------------
# Notification senders
# ---------------------------------------------------------------------
def send_email_alert(alert: dict) -> bool:
    """Send an email alert via SMTP."""
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_password = os.environ.get("SMTP_PASSWORD")
    to_email = os.environ.get("ALERT_EMAIL_TO")

    if not all([smtp_host, smtp_user, smtp_password, to_email]):
        print("  [skip] Email not configured (SMTP_HOST/SMTP_USER/SMTP_PASSWORD/ALERT_EMAIL_TO)")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["From"] = smtp_user
        msg["To"] = to_email
        msg["Subject"] = f"{alert['emoji']} BTC Sentiment Alert: {alert['level']}"

        text = f"""
BTC Sentiment Alert — {alert['level']}
========================================

{alert['message']}

Latest headline: {alert['headline']}
Source date: {alert['source_date']}
Headlines in last 24h: {alert['headline_count']}
Alert time: {alert['timestamp']}

--
BTC Sentiment-Driven LSTM Pipeline
        """

        html = f"""
<html><body style="font-family: sans-serif; color: #333;">
<h2>{alert['emoji']} BTC Sentiment Alert: {alert['level']}</h2>
<p><strong>{alert['message']}</strong></p>
<table style="border-collapse: collapse; width: 100%;">
<tr><td style="padding: 8px; border: 1px solid #ddd;">Latest headline</td><td style="padding: 8px; border: 1px solid #ddd;">{alert['headline']}</td></tr>
<tr><td style="padding: 8px; border: 1px solid #ddd;">Source date</td><td style="padding: 8px; border: 1px solid #ddd;">{alert['source_date']}</td></tr>
<tr><td style="padding: 8px; border: 1px solid #ddd;">Headlines (24h)</td><td style="padding: 8px; border: 1px solid #ddd;">{alert['headline_count']}</td></tr>
<tr><td style="padding: 8px; border: 1px solid #ddd;">Alert time</td><td style="padding: 8px; border: 1px solid #ddd;">{alert['timestamp']}</td></tr>
</table>
<hr>
<p style="color: #999; font-size: 12px;">BTC Sentiment-Driven LSTM Pipeline — Automated Alert System</p>
</body></html>
        """

        msg.attach(MIMEText(text, "plain"))
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, to_email, msg.as_string())

        print(f"  [ok] Email alert sent to {to_email}")
        return True
    except Exception as e:
        print(f"  [ERROR] Email failed: {e}")
        return False


def send_slack_alert(alert: dict) -> bool:
    """Send a Slack alert via webhook."""
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print("  [skip] Slack not configured (SLACK_WEBHOOK_URL)")
        return False

    try:
        color = {
            "VERY_BULLISH": "#16a34a",
            "BULLISH": "#22c55e",
            "BEARISH": "#f59e0b",
            "VERY_BEARISH": "#dc2626",
        }.get(alert["level"], "#999999")

        payload = {
            "attachments": [
                {
                    "color": color,
                    "title": f"{alert['emoji']} BTC Sentiment Alert: {alert['level']}",
                    "text": alert["message"],
                    "fields": [
                        {"title": "Latest Headline", "value": alert["headline"][:300], "short": False},
                        {"title": "Sentiment Score", "value": f"{alert['sentiment']:+.3f}", "short": True},
                        {"title": "Headlines (24h)", "value": str(alert["headline_count"]), "short": True},
                        {"title": "Source Date", "value": alert["source_date"][:19], "short": True},
                    ],
                    "footer": "BTC Sentiment-Driven LSTM Pipeline",
                    "ts": int(datetime.now(UTC).timestamp()),
                }
            ]
        }

        r = requests.post(webhook_url, json=payload, timeout=10)
        r.raise_for_status()
        print("  [ok] Slack alert sent")
        return True
    except Exception as e:
        print(f"  [ERROR] Slack failed: {e}")
        return False


def send_discord_alert(alert: dict) -> bool:
    """Send a Discord alert via webhook."""
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("  [skip] Discord not configured (DISCORD_WEBHOOK_URL)")
        return False

    try:
        color_map = {
            "VERY_BULLISH": 0x16A34A,
            "BULLISH": 0x22C55E,
            "BEARISH": 0xF59E0B,
            "VERY_BEARISH": 0xDC2626,
        }

        payload = {
            "embeds": [
                {
                    "title": f"{alert['emoji']} BTC Sentiment Alert: {alert['level']}",
                    "description": alert["message"],
                    "color": color_map.get(alert["level"], 0x999999),
                    "fields": [
                        {"name": "Latest Headline", "value": alert["headline"][:300], "inline": False},
                        {"name": "Sentiment Score", "value": f"{alert['sentiment']:+.3f}", "inline": True},
                        {"name": "Headlines (24h)", "value": str(alert["headline_count"]), "inline": True},
                        {"name": "Source Date", "value": alert["source_date"][:19], "inline": True},
                    ],
                    "footer": {"text": "BTC Sentiment-Driven LSTM Pipeline"},
                    "timestamp": alert["timestamp"],
                }
            ]
        }

        r = requests.post(webhook_url, json=payload, timeout=10)
        r.raise_for_status()
        print("  [ok] Discord alert sent")
        return True
    except Exception as e:
        print(f"  [ERROR] Discord failed: {e}")
        return False


# ---------------------------------------------------------------------
# Alert logging
# ---------------------------------------------------------------------
def log_alert(alert: dict, channels_sent: list[str]) -> None:
    """Log the alert to the audit CSV."""
    import csv

    row = [
        alert["timestamp"],
        alert["level"],
        f"{alert['sentiment']:+.4f}",
        alert["headline"][:100],
        alert["source_date"][:19],
        alert["headline_count"],
        ",".join(channels_sent) if channels_sent else "none",
    ]

    file_exists = ALERT_LOG_PATH.exists()
    # Ensure the audit directory exists before writing — without this, the
    # first run on a fresh clone crashes with FileNotFoundError.
    ALERT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ALERT_LOG_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "level", "sentiment", "headline",
                "source_date", "headline_count", "channels_sent",
            ])
        writer.writerow(row)
    print(f"  [ok] Alert logged to {ALERT_LOG_PATH}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(dry_run: bool = False) -> None:
    print(f"[{datetime.now(UTC).isoformat()}] Sentiment Alert System starting ...")

    # Fetch latest sentiment
    print("\n[1/3] Fetching latest crypto news sentiment ...")
    sentiment_data = fetch_latest_sentiment()
    if sentiment_data is None:
        print("  [ERROR] Could not fetch sentiment data. Aborting.")
        return

    print(f"  Latest sentiment: {sentiment_data['latest_sentiment']:+.3f}")
    print(f"  24h avg sentiment: {sentiment_data['daily_avg_sentiment']:+.3f}")
    print(f"  Headlines (24h): {sentiment_data['headline_count_24h']}")
    print(f"  Latest headline: {sentiment_data['latest_headline'][:80]}...")

    # Evaluate alert
    print("\n[2/3] Evaluating alert thresholds ...")
    alert = evaluate_alert(sentiment_data)

    if alert is None:
        print("  No alert needed — sentiment is within normal range.")
        return

    print(f"  {alert['emoji']} ALERT: {alert['level']} — {alert['message']}")

    # Send notifications
    print("\n[3/3] Sending notifications ...")
    channels_sent = []

    if dry_run:
        print("  [dry-run] Skipping actual notification sends.")
        print(f"  Would send: {alert}")
    else:
        if send_email_alert(alert):
            channels_sent.append("email")
        if send_slack_alert(alert):
            channels_sent.append("slack")
        if send_discord_alert(alert):
            channels_sent.append("discord")

        if not channels_sent:
            print("  [warn] No notification channels configured. Alert was logged but not sent.")
            print("  Configure via env vars: SMTP_HOST, SLACK_WEBHOOK_URL, DISCORD_WEBHOOK_URL")

    # Log the alert
    log_alert(alert, channels_sent)

    print(f"\n[{datetime.now(UTC).isoformat()}] Alert system complete.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="BTC Sentiment Alert System")
    ap.add_argument("--dry-run", action="store_true", help="Test without sending notifications")
    args = ap.parse_args()

    main(dry_run=args.dry_run)
