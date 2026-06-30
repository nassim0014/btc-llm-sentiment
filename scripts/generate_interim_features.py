"""Generate the interim feature bundle needed for Optuna search and SHAP.

This script:
  1. Fetches news (from local CSV)
  2. Fetches BTC prices via yfinance
  3. Uses pre-computed TextBlob sentiment (fast path)
  4. Builds the feature matrix
  5. Saves notebooks/interim/features_for_lstm.pkl and merged_with_llm_sentiment.parquet
"""
import ast
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

INTERIM = ROOT / "notebooks" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

BTC_TICKER = "BTC-USD"
HORIZON = 5


def fetch_news():
    print("[1/4] Loading news from local CSV ...")
    news = pd.read_csv(ROOT / "Data" / "cryptonews.csv")
    news["date"] = pd.to_datetime(news["date"], format="mixed", utc=True, errors="coerce")
    news = news.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    def parse(s):
        try:
            d = ast.literal_eval(s) if isinstance(s, str) else {}
            return pd.Series({
                "sentiment_class": d.get("class", "neutral"),
                "sentiment_polarity": float(d.get("polarity", 0.0)),
                "sentiment_subjectivity": float(d.get("subjectivity", 0.0)),
            })
        except Exception:
            return pd.Series({"sentiment_class": "neutral", "sentiment_polarity": 0.0, "sentiment_subjectivity": 0.0})

    sent = news["sentiment"].apply(parse)
    news = pd.concat([news.drop(columns=["sentiment"]), sent], axis=1)
    news["llm_sentiment"] = news["sentiment_polarity"].astype(float)
    print(f"  -> {len(news):,} rows")
    return news


def fetch_btc():
    print("[2/4] Fetching BTC-USD via yfinance ...")
    btc = yf.download(BTC_TICKER, start="2023-01-01", end="2024-12-31", auto_adjust=False, progress=False)
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = [" ".join(c).strip() for c in btc.columns]
    btc = btc.reset_index().rename(columns={"Date": "date"})
    btc.columns = [c.replace(" BTC-USD", "").lower() for c in btc.columns]
    btc["date"] = pd.to_datetime(btc["date"]).dt.floor("D")
    print(f"  -> {len(btc):,} daily candles")
    return btc


def build_features(news, btc):
    print("[3/4] Building features ...")
    news["date_day"] = news["date"].dt.tz_convert(None).dt.floor("D")
    daily_news = (news.groupby("date_day")
                  .agg(news_count=("title", "size"),
                       mean_polarity=("sentiment_polarity", "mean"),
                       mean_subjectivity=("sentiment_subjectivity", "mean"),
                       neg_share=("sentiment_class", lambda s: (s == "negative").mean()),
                       pos_share=("sentiment_class", lambda s: (s == "positive").mean()),
                       llm_sentiment_mean=("llm_sentiment", "mean"),
                       llm_sentiment_std=("llm_sentiment", "std"),
                       llm_headline_count=("llm_sentiment", "size"),
                       llm_pos_share=("llm_sentiment", lambda s: (s > 0.3).mean()),
                       llm_neg_share=("llm_sentiment", lambda s: (s < -0.3).mean()))
                  .reset_index().rename(columns={"date_day": "date"})
                  .fillna({"llm_sentiment_std": 0}))

    df = pd.merge(btc, daily_news, on="date", how="left")
    fill_cols = ["news_count", "mean_polarity", "neg_share", "pos_share",
                 "llm_sentiment_mean", "llm_sentiment_std", "llm_headline_count",
                 "llm_pos_share", "llm_neg_share"]
    df[fill_cols] = df[fill_cols].fillna(0)
    df = df.sort_values("date").reset_index(drop=True)

    # Technical indicators
    def rsi(close, period=14):
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / (loss + 1e-12)
        return 100 - (100 / (1 + rs))

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
    df["bb_pct_b"] = (df["close"] - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-12)
    df["bb_width"] = (4*bb_std) / (bb_mid + 1e-12)

    for lag in [1, 2, 3, 5]:
        df[f"llm_sent_lag{lag}"] = df["llm_sentiment_mean"].shift(lag)
        df[f"llm_pos_share_lag{lag}"] = df["llm_pos_share"].shift(lag)
    df["llm_sent_3d_ma"] = df["llm_sentiment_mean"].rolling(3).mean().shift(1)
    df["llm_sent_5d_ma"] = df["llm_sentiment_mean"].rolling(5).mean().shift(1)

    df["forward_ret_5d"] = df["close"].shift(-HORIZON) / df["close"] - 1
    df["target_up_5d"] = (df["forward_ret_5d"] > 0).astype(int)
    df = df.dropna(subset=["target_up_5d"]).reset_index(drop=True)

    print(f"  -> {len(df):,} rows | class balance: up={df.target_up_5d.mean():.3f}")

    # Save merged for SHAP regime analysis
    df.to_parquet(INTERIM / "merged_with_llm_sentiment.parquet", index=False)

    # Split + scale
    n = len(df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    train, val, test = df.iloc[:n_train], df.iloc[n_train:n_train+n_val], df.iloc[n_train+n_val:]

    FEATURE_COLS = [
        "ret_1d", "ret_3d", "ret_7d", "vol_7d", "vol_21d",
        "rsi_14", "macd_line", "macd_hist", "bb_pct_b", "bb_width",
        "llm_sent_lag1", "llm_sent_lag2", "llm_sent_lag3", "llm_sent_lag5",
        "llm_pos_share_lag1", "llm_pos_share_lag3", "llm_pos_share_lag5",
        "llm_sent_3d_ma", "llm_sent_5d_ma",
        "news_count", "mean_polarity", "neg_share", "pos_share",
    ]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train[FEATURE_COLS].fillna(0))
    val_x = scaler.transform(val[FEATURE_COLS].fillna(0))
    test_x = scaler.transform(test[FEATURE_COLS].fillna(0))

    train_y = train["target_up_5d"].values
    val_y = val["target_up_5d"].values
    test_y = test["target_up_5d"].values

    train_x = train_x.reshape(-1, 1, len(FEATURE_COLS))
    val_x = val_x.reshape(-1, 1, len(FEATURE_COLS))
    test_x = test_x.reshape(-1, 1, len(FEATURE_COLS))

    bundle = dict(
        train_x=train_x, train_y=train_y,
        val_x=val_x, val_y=val_y,
        test_x=test_x, test_y=test_y,
        feature_cols=FEATURE_COLS, scaler=scaler,
        train_close=train["close"].values,
        val_close=val["close"].values,
        test_close=test["close"].values,
        train_dates=train["date"].values,
        val_dates=val["date"].values,
        test_dates=test["date"].values,
        test_forward_ret_5d=test["forward_ret_5d"].values,
    )
    return bundle


def main():
    news = fetch_news()
    btc = fetch_btc()
    bundle = build_features(news, btc)

    print("[4/4] Saving feature bundle ...")
    out_path = INTERIM / "features_for_lstm.pkl"
    with out_path.open("wb") as f:
        pickle.dump(bundle, f)
    print(f"  -> {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    print(f"\nDone. Train: {bundle['train_x'].shape}, Val: {bundle['val_x'].shape}, Test: {bundle['test_x'].shape}")


if __name__ == "__main__":
    main()
