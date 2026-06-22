"""
End-to-end pipeline runner for the BTC Sentiment-Driven LSTM Trading Pipeline.

This script mirrors exactly the logic in notebooks/01-05 and writes all four
output artifacts to /outputs. It is the single-command way to reproduce the
results shown in the README.

Stages:
  1. Fetch cryptonews.csv (GitHub raw URL, with local fallback)
  2. Fetch BTC-USD via yfinance + apply MultiIndex flatten bug fix
  3. Score news headlines with HuggingFace FinBERT (fallback to DistilBERT)
  4. Engineer technical + sentiment features
  5. Train 4 LSTM configs (baseline, lightweight, regularized, bilstm) with
     class weighting + early stopping
  6. Optimize threshold per model on validation Sharpe
  7. Backtest on test window with 0.1% transaction costs
  8. Write outputs/final_model_comparison.csv
         outputs/portfolio_values_over_time.csv
         outputs/trading_metrics.csv
         outputs/complete_pipeline_summary.svg

Usage:
    python3 scripts/run_pipeline.py [--quick]
    --quick  : use a 4k-headline subset for LLM scoring (faster smoke test)
"""
from __future__ import annotations

import argparse
import ast
import os
import pickle
import sys
import warnings
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data"
INTERIM = ROOT / "notebooks" / "interim"
OUTPUTS = ROOT / "outputs"
INTERIM.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

NEWS_URL = "https://raw.githubusercontent.com/nassim0014/btc-llm-sentiment/main/Data/cryptonews.csv"
LOCAL_NEWS = DATA_DIR / "cryptonews.csv"

BTC_TICKER = "BTC-USD"
BTC_START = "2023-01-01"
BTC_END = "2024-12-31"

RANDOM_STATE = 42
HORIZON = 5
TRADING_FEE = 0.001  # 0.1% per trade
np.random.seed(RANDOM_STATE)


# ---------------------------------------------------------------------
# Stage 1 — Fetch news
# ---------------------------------------------------------------------
def fetch_news() -> pd.DataFrame:
    print(f"\n[Stage 1] Fetching news from {NEWS_URL} ...")
    try:
        r = requests.get(NEWS_URL, timeout=120)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        print(f"  -> {len(df):,} rows fetched from remote.")
    except Exception as e:
        print(f"  ! remote fetch failed ({e}); falling back to local {LOCAL_NEWS}")
        df = pd.read_csv(LOCAL_NEWS)

    # Bug Fix 2: mixed date formats
    df["date"] = pd.to_datetime(df["date"], format="mixed", utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Parse embedded sentiment dict (TextBlob-style)
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

    sent = df["sentiment"].apply(parse)
    df = pd.concat([df.drop(columns=["sentiment"]), sent], axis=1)
    print(f"  -> parsed {len(df):,} rows after date cleanup.")
    return df


# ---------------------------------------------------------------------
# Stage 2 — Fetch BTC + apply MultiIndex bug fix
# ---------------------------------------------------------------------
def fetch_btc() -> pd.DataFrame:
    print(f"\n[Stage 2] Fetching {BTC_TICKER} via yfinance ...")
    btc = yf.download(BTC_TICKER, start=BTC_START, end=BTC_END, auto_adjust=False, progress=False)
    print(f"  -> raw columns type: {type(btc.columns).__name__} | MultiIndex: {isinstance(btc.columns, pd.MultiIndex)}")
    # Bug Fix 1: flatten MultiIndex
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = [" ".join(c).strip() for c in btc.columns]
    btc = btc.reset_index().rename(columns={"Date": "date"})
    btc.columns = [c.replace(" BTC-USD", "").lower() for c in btc.columns]
    btc["date"] = pd.to_datetime(btc["date"]).dt.floor("D")
    print(f"  -> {len(btc):,} daily candles | flattened columns: {list(btc.columns)}")
    return btc


# ---------------------------------------------------------------------
# Stage 3 — LLM sentiment scoring
# ---------------------------------------------------------------------
def score_with_llm(news: pd.DataFrame, quick: bool = False, use_precomputed: bool = False) -> pd.DataFrame:
    """Score headlines with HuggingFace FinBERT.

    If `use_precomputed=True`, fall back to the TextBlob polarity already
    embedded in the CSV. This is still an NLP-derived sentiment score, but
    avoids the ~2-hour CPU runtime of running FinBERT on 31k headlines.
    The notebook 02 code path runs the full HuggingFace pipeline.
    """
    print("\n[Stage 3] Scoring headlines ...")
    if use_precomputed:
        print("  -> using pre-computed TextBlob sentiment (use_precomputed=True)")
        print("     (set use_precomputed=False to run HuggingFace FinBERT — ~2h on CPU)")
        news = news.copy()
        # Map TextBlob polarity [-1, 1] to LLM-style score [-1, 1]
        news["llm_sentiment"] = news["sentiment_polarity"].astype(float)
        print(f"  -> mean LLM sentiment: {news.llm_sentiment.mean():.4f} | std: {news.llm_sentiment.std():.4f}")
        return news

    print("  -> running HuggingFace FinBERT (this is slow on CPU) ...")
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

    DEVICE = 0 if torch.cuda.is_available() else -1
    print(f"  -> torch device: {'cuda' if DEVICE == 0 else 'cpu'}")

    candidates = [
        "ProsusAI/finbert",
        "distilbert-base-uncased-finetuned-sst-2-english",
    ]
    pipe = None
    for name in candidates:
        try:
            print(f"  -> loading {name} ...")
            tok = AutoTokenizer.from_pretrained(name)
            mdl = AutoModelForSequenceClassification.from_pretrained(name)
            pipe = pipeline("sentiment-analysis", model=mdl, tokenizer=tok, device=DEVICE,
                            truncation=True, max_length=512)
            print(f"  -> loaded {name}")
            break
        except Exception as e:
            print(f"  ! {name} failed: {e}")
    assert pipe is not None, "No sentiment model loaded"

    news = news.copy()
    news["text"] = news["title"].fillna("") + ". " + news["text"].fillna("")

    if quick:
        news = news.sample(min(4000, len(news)), random_state=RANDOM_STATE).sort_values("date").reset_index(drop=True)
        print(f"  -> QUICK MODE: scoring {len(news):,} headlines (subsample)")

    BATCH = 64
    texts = news["text"].astype(str).tolist()
    scores = []
    for i in tqdm(range(0, len(texts), BATCH), desc="LLM sentiment"):
        batch = texts[i:i + BATCH]
        try:
            results = pipe(batch)
        except Exception:
            results = [pipe(t) if len(t) > 1 else {"label": "neutral", "score": 0.5} for t in batch]
        for r in results:
            label = r["label"].lower()
            prob = float(r["score"])
            if "pos" in label:
                s = prob
            elif "neg" in label:
                s = -prob
            else:
                s = 0.0
            scores.append(s)

    news["llm_sentiment"] = scores
    print(f"  -> mean LLM sentiment: {news.llm_sentiment.mean():.4f} | std: {news.llm_sentiment.std():.4f}")
    return news


# ---------------------------------------------------------------------
# Stage 4 — Aggregate + merge + feature engineering
# ---------------------------------------------------------------------
def build_features(news: pd.DataFrame, btc: pd.DataFrame) -> tuple:
    print("\n[Stage 4] Building feature matrix ...")
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

    print(f"  -> final dataset: {len(df):,} rows | class balance: up={df.target_up_5d.mean():.3f}")

    n = len(df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    train, val, test = df.iloc[:n_train], df.iloc[n_train:n_train+n_val], df.iloc[n_train+n_val:]
    print(f"  -> train={len(train)} val={len(val)} test={len(test)}")

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
        val_close=val["close"].values,
        test_close=test["close"].values,
        val_dates=val["date"].values,
        test_dates=test["date"].values,
        test_forward_ret_5d=test["forward_ret_5d"].values,
    )
    return bundle


# ---------------------------------------------------------------------
# Stage 5 — Train LSTM configs
# ---------------------------------------------------------------------
def train_lstms(bundle: dict) -> dict:
    print("\n[Stage 5] Training 4 LSTM configs with class weighting + early stopping ...")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers, callbacks
    tf.get_logger().setLevel("ERROR")
    tf.random.set_seed(RANDOM_STATE)

    train_x, train_y = bundle["train_x"], bundle["train_y"]
    val_x, val_y = bundle["val_x"], bundle["val_y"]
    test_x = bundle["test_x"]
    n_features = train_x.shape[-1]

    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(train_y)
    weights = compute_class_weight("balanced", classes=classes, y=train_y)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"  -> class weights: {class_weight}")

    def build(config: str) -> tf.keras.Model:
        inp = layers.Input(shape=(1, n_features), name="features")
        x = inp
        if config == "baseline":
            x = layers.LSTM(64, return_sequences=False)(x)
        elif config == "lightweight":
            x = layers.LSTM(32, return_sequences=False, dropout=0.1)(x)
        elif config == "regularized":
            x = layers.LSTM(64, return_sequences=True,
                            kernel_regularizer=regularizers.l2(1e-4),
                            recurrent_dropout=0.2)(x)
            x = layers.Dropout(0.3)(x)
            x = layers.LSTM(32, return_sequences=False,
                            kernel_regularizer=regularizers.l2(1e-4))(x)
            x = layers.Dropout(0.3)(x)
        elif config == "bilstm":
            x = layers.Bidirectional(
                layers.LSTM(64, return_sequences=False, dropout=0.2,
                            kernel_regularizer=regularizers.l2(1e-5)))(x)
        x = layers.Dense(16, activation="relu")(x)
        out = layers.Dense(1, activation="sigmoid", name="prob_up")(x)
        m = models.Model(inp, out, name=config)
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
        return m

    CONFIGS = ["baseline", "lightweight", "regularized", "bilstm"]
    EPOCHS = 40
    BATCH = 32

    preds = {}
    for cfg in CONFIGS:
        print(f"\n  --- {cfg} ---")
        tf.keras.backend.clear_session()
        tf.random.set_seed(RANDOM_STATE)
        model = build(cfg)
        es = callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=7, restore_best_weights=True)
        rlrop = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
        hist = model.fit(
            train_x, train_y,
            validation_data=(val_x, val_y),
            epochs=EPOCHS, batch_size=BATCH,
            class_weight=class_weight,
            callbacks=[es, rlrop],
            verbose=0,
        )
        val_auc = max(hist.history["val_auc"])
        val_acc = max(hist.history["val_accuracy"])
        print(f"  -> val_auc={val_auc:.4f}  val_acc={val_acc:.4f}  epochs={len(hist.history['val_loss'])}")

        preds[cfg] = {
            "val_prob": model.predict(val_x, verbose=0).ravel(),
            "test_prob": model.predict(test_x, verbose=0).ravel(),
        }
    return preds


# ---------------------------------------------------------------------
# Stage 6 — Threshold optimizer + backtest
# ---------------------------------------------------------------------
def backtest(prob, close, threshold, fee=TRADING_FEE):
    signal = (prob >= threshold).astype(int)
    rets = np.diff(close) / close[:-1]
    strat_rets = signal[:-1] * rets
    trade_flags = np.abs(np.diff(signal))
    strat_rets = strat_rets - trade_flags * fee
    n_days = len(strat_rets)
    if n_days == 0:
        return dict(sharpe=0, sortino=0, max_dd=0, win_rate=0, final_value=1.0, n_trades=0,
                    returns=strat_rets, equity=np.array([1.0]))
    ann = np.sqrt(252)
    mean_r = strat_rets.mean()
    std_r = strat_rets.std() + 1e-12
    sharpe = (mean_r / std_r) * ann
    downside = strat_rets[strat_rets < 0]
    sortino = (mean_r / (downside.std() + 1e-12)) * ann if len(downside) > 0 else sharpe
    equity = np.cumprod(1 + strat_rets)
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    max_dd = drawdowns.min()
    win_rate = (strat_rets > 0).mean()
    return dict(
        sharpe=float(sharpe), sortino=float(sortino), max_dd=float(max_dd),
        win_rate=float(win_rate), final_value=float(equity[-1]),
        n_trades=int(trade_flags.sum() // 2),
        returns=strat_rets, equity=equity,
    )


def optimize_threshold(val_prob, val_close, fee=TRADING_FEE, min_trades: int = 5):
    """Scan 0.20 → 0.80 and pick the threshold that maximizes Sharpe,
    with a guard against zero-trade solutions (which always look 'safe'
    but produce no signal). Thresholds that produce fewer than
    `min_trades` on validation are penalized."""
    best_t, best_sharpe = 0.5, -np.inf
    for t in np.arange(0.20, 0.81, 0.05):
        r = backtest(val_prob, val_close, t, fee)
        # Penalty: if the threshold produces fewer than min_trades, push it down
        effective_sharpe = r["sharpe"] if r["n_trades"] >= min_trades else r["sharpe"] - 2.0
        if effective_sharpe > best_sharpe:
            best_sharpe, best_t = effective_sharpe, float(t)
    return best_t, best_sharpe


def run_backtest_and_save(preds: dict, bundle: dict) -> None:
    print("\n[Stage 6/7] Threshold optimization + test-set backtest ...")
    val_close = bundle["val_close"]
    test_close = bundle["test_close"]
    test_dates = pd.to_datetime(bundle["test_dates"])

    thresholds = {}
    for cfg, pr in preds.items():
        t, s = optimize_threshold(pr["val_prob"], val_close)
        thresholds[cfg] = t
        print(f"  -> {cfg:12s}  threshold={t:.2f}  val_sharpe={s:+.3f}")

    results = {}
    for cfg, pr in preds.items():
        t = thresholds[cfg]
        r = backtest(pr["test_prob"], test_close, t)
        results[cfg] = {k: v for k, v in r.items()}
        results[cfg]["threshold"] = t
        print(f"  -> {cfg:12s}  final={r['final_value']:.3f}  sharpe={r['sharpe']:+.3f}  sortino={r['sortino']:+.3f}  max_dd={r['max_dd']:+.3f}  win={r['win_rate']:.2f}  trades={r['n_trades']}")

    bh_rets = np.diff(test_close) / test_close[:-1]
    bh_equity = np.cumprod(1 + bh_rets)
    ann = np.sqrt(252)
    bh_sharpe = (bh_rets.mean() / (bh_rets.std()+1e-12)) * ann
    bh_downside = bh_rets[bh_rets < 0]
    bh_sortino = (bh_rets.mean() / (bh_downside.std()+1e-12)) * ann if len(bh_downside) else bh_sharpe
    bh_running_max = np.maximum.accumulate(bh_equity)
    bh_max_dd = ((bh_equity - bh_running_max) / bh_running_max).min()
    results["buy_hold"] = dict(
        sharpe=float(bh_sharpe), sortino=float(bh_sortino),
        max_dd=float(bh_max_dd), win_rate=float((bh_rets > 0).mean()),
        final_value=float(bh_equity[-1]), n_trades=1, threshold=None,
        equity=bh_equity, returns=bh_rets,
    )
    print(f"  -> {'buy_hold':12s}  final={bh_equity[-1]:.3f}  sharpe={bh_sharpe:+.3f}  sortino={bh_sortino:+.3f}  max_dd={bh_max_dd:+.3f}  win={(bh_rets>0).mean():.2f}")

    rows = []
    for k, r in results.items():
        rows.append({
            "strategy": k,
            "final_portfolio_value": round(r["final_value"], 4),
            "sharpe_ratio": round(r["sharpe"], 4),
            "sortino_ratio": round(r["sortino"], 4),
            "max_drawdown": round(r["max_dd"], 4),
            "win_rate": round(r["win_rate"], 4),
            "n_trades": r["n_trades"],
            "threshold": r.get("threshold"),
        })
    df_cmp = pd.DataFrame(rows)
    df_cmp.to_csv(OUTPUTS / "final_model_comparison.csv", index=False)
    print(f"\n  -> wrote {OUTPUTS / 'final_model_comparison.csv'}")
    print(df_cmp.to_string(index=False))

    equity_df = pd.DataFrame({"date": test_dates[1:]})
    for k, r in results.items():
        equity_df[k] = r["equity"]
    equity_df.to_csv(OUTPUTS / "portfolio_values_over_time.csv", index=False)
    print(f"  -> wrote {OUTPUTS / 'portfolio_values_over_time.csv'}")

    metrics_rows = []
    for k, r in results.items():
        rets = r["returns"]
        metrics_rows.append({
            "strategy": k,
            "total_return_pct": round((r["final_value"] - 1) * 100, 2),
            "annualized_sharpe": round(r["sharpe"], 4),
            "annualized_sortino": round(r["sortino"], 4),
            "max_drawdown_pct": round(r["max_dd"] * 100, 2),
            "win_rate_pct": round(r["win_rate"] * 100, 2),
            "n_trades": r["n_trades"],
            "avg_daily_return_pct": round(float(rets.mean()) * 100, 4) if hasattr(rets, "mean") else None,
            "daily_volatility_pct": round(float(rets.std()) * 100, 4) if hasattr(rets, "std") else None,
        })
    pd.DataFrame(metrics_rows).to_csv(OUTPUTS / "trading_metrics.csv", index=False)
    print(f"  -> wrote {OUTPUTS / 'trading_metrics.csv'}")

    save_summary_svg(results, preds, test_close, test_dates, thresholds)
    print(f"  -> wrote {OUTPUTS / 'complete_pipeline_summary.svg'}")


def save_summary_svg(results, preds, test_close, test_dates, thresholds):
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import seaborn as sns
    try:
        fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    except Exception:
        pass
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)

    ax = axes[0, 0]
    for k, r in results.items():
        ax.plot(test_dates[1:], r["equity"], label=k, lw=2)
    ax.set_title("Portfolio Value Over Time (Test Window)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Equity (start = 1.0)")
    ax.legend(loc="best")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[0, 1]
    for k, r in results.items():
        eq = r["equity"]
        rm = np.maximum.accumulate(eq)
        dd = (eq - rm) / rm
        ax.plot(test_dates[1:], dd, label=k, lw=1.5, alpha=0.85)
    ax.set_title("Drawdowns", fontsize=12, fontweight="bold")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="best")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1, 0]
    names = list(results.keys())
    sharpes = [results[k]["sharpe"] for k in names]
    colors = ["#0f766e", "#f59e0b", "#3b82f6", "#ec4899", "#8a5e21"][:len(names)]
    bars = ax.bar(names, sharpes, color=colors)
    ax.set_title("Annualized Sharpe Ratio", fontsize=12, fontweight="bold")
    ax.axhline(0, c="k", lw=0.6)
    for bar, val in zip(bars, sharpes):
        ax.text(bar.get_x() + bar.get_width()/2, val + (0.05 if val >= 0 else -0.15),
                f"{val:+.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)

    ax = axes[1, 1]
    best_cfg = max([k for k in results if k != "buy_hold"], key=lambda k: results[k]["sharpe"])
    thresholds_scan = np.arange(0.30, 0.71, 0.05)
    sharpes_scan = []
    for t in thresholds_scan:
        r = backtest(preds[best_cfg]["test_prob"], test_close, t)
        sharpes_scan.append(r["sharpe"])
    ax.plot(thresholds_scan, sharpes_scan, marker="o", color="#0f766e", lw=2)
    best_t = thresholds[best_cfg]
    ax.axvline(best_t, ls="--", c="red", alpha=0.7, label=f"Selected threshold = {best_t:.2f}")
    ax.set_title(f"Threshold Sensitivity — {best_cfg}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Sharpe ratio")
    ax.legend()

    fig.suptitle("BTC Sentiment-Driven LSTM — Pipeline Summary",
                 fontsize=15, fontweight="bold", y=1.005)
    plt.savefig(OUTPUTS / "complete_pipeline_summary.svg", bbox_inches="tight", facecolor="white")
    plt.savefig(OUTPUTS / "complete_pipeline_summary.png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()


def main(quick: bool = False, use_precomputed: bool = False) -> None:
    print("=" * 70)
    print("BTC Sentiment-Driven LSTM Trading Pipeline — end-to-end runner")
    print(f"  quick             : {quick}")
    print(f"  use_precomputed   : {use_precomputed}")
    print("=" * 70)

    news = fetch_news()
    btc = fetch_btc()
    news_scored = score_with_llm(news, quick=quick, use_precomputed=use_precomputed)
    bundle = build_features(news_scored, btc)
    preds = train_lstms(bundle)
    run_backtest_and_save(preds, bundle)

    print("\n" + "=" * 70)
    print("Pipeline complete. Outputs written to:")
    for f in sorted(OUTPUTS.iterdir()):
        print(f"  {f.name:40s}  {f.stat().st_size / 1024:.1f} KB")
    print("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Use a 4k-headline subset for faster LLM scoring.")
    ap.add_argument("--use-precomputed", action="store_true",
                    help="Use the TextBlob sentiment embedded in the CSV instead of running FinBERT "
                         "(fast; full FinBERT path takes ~2h on CPU).")
    args = ap.parse_args()
    main(quick=args.quick, use_precomputed=args.use_precomputed)
