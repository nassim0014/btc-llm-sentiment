"""Run the risk-managed backtest with the best Optuna hyperparameters.

Trains a model with the best params on the full train set, predicts on the
test set, and runs the risk-managed backtester (Kelly + vol-targeting + DD breaker).

Output: outputs/risk_managed_backtest_results.csv
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

INTERIM = ROOT / "notebooks" / "interim"
OUTPUTS = ROOT / "outputs"


def main():
    print("=" * 60)
    print("Risk-Managed Backtest (Kelly + Vol-Targeting + DD Breaker)")
    print("=" * 60)

    # Load features — use safe loader with SHA256 integrity check
    from src.utils.safe_pickle import safe_load_bundle
    bundle = safe_load_bundle()

    train_x, train_y = bundle["train_x"], bundle["train_y"]
    val_x, val_y = bundle["val_x"], bundle["val_y"]
    test_x, _test_y = bundle["test_x"], bundle["test_y"]
    test_close = bundle["test_close"]
    test_dates = pd.to_datetime(bundle["test_dates"])
    n_features = train_x.shape[-1]

    # Load best Optuna params
    with (OUTPUTS / "best_optuna_params.json").open() as f:
        best = json.load(f)
    hp = best["best_params"]
    print(f"Best params: {hp}")
    print(f"Best OOF Sharpe from search: {best['best_oof_sharpe']:+.4f}")

    # Build and train model with best params on train+val
    import tensorflow as tf
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras import callbacks

    from src.models.lstm import build_lstm_with_params

    tf.get_logger().setLevel("ERROR")
    tf.random.set_seed(42)
    np.random.seed(42)

    # Combine train+val for final model training
    X_full = np.concatenate([train_x, val_x], axis=0)
    y_full = np.concatenate([train_y, val_y], axis=0)

    classes = np.unique(y_full)
    weights = compute_class_weight("balanced", classes=classes, y=y_full)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights, strict=False)}

    model = build_lstm_with_params(
        lr=hp["lr"], units=hp["units"], dropout=hp["dropout"],
        num_layers=hp["num_layers"], n_features=n_features,
    )
    print(f"\nTraining final model on {len(X_full)} samples (train+val) ...")
    history = model.fit(
        X_full, y_full,
        validation_split=0.15,
        epochs=30, batch_size=32,
        verbose=0,
        class_weight=class_weight,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=7, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
        ],
    )
    print(f"  Trained {len(history.history['loss'])} epochs")

    # Predict on test
    test_prob = model.predict(test_x, verbose=0).ravel()
    print(f"  Test predictions: mean={test_prob.mean():.3f} std={test_prob.std():.3f}")

    # Save model for SHAP (Step 5)
    model.save(INTERIM / "best_optuna_model.keras")
    print(f"  Saved model → {INTERIM / 'best_optuna_model.keras'}")

    # Run risk-managed backtest
    from src.backtest.risk_managed import compare_strategies, risk_managed_backtest

    print("\n--- Risk-Managed Backtest ---")
    rm = risk_managed_backtest(
        prob=test_prob,
        close=test_close,
        threshold=0.5,
        fee=0.001,
        target_annual_vol=0.20,
        vol_lookback=20,
        max_drawdown_pct=0.15,
    )

    summary = rm.to_summary_dict()
    print(f"  Final value:       {summary['final_portfolio_value']:.4f}")
    print(f"  Total return:      {summary['total_return_pct']:+.2f}%")
    print(f"  Sharpe:            {summary['annualized_sharpe']:+.4f}")
    print(f"  Sortino:           {summary['annualized_sortino']:+.4f}")
    print(f"  Max DD:            {summary['max_drawdown_pct']:+.2f}%")
    print(f"  Win rate:          {summary['win_rate_pct']:.2f}%")
    print(f"  Trades:            {summary['n_trades']}")
    print(f"  Avg position:      {summary['avg_position_size']:.4f}")
    print(f"  Max position:      {summary['max_position_size']:.4f}")
    print(f"  Circuit breaker:   {'TRIGGERED' if summary['circuit_breaker_triggered'] else 'not triggered'}")
    if summary["breaker_trigger_day"] is not None:
        print(f"  Breaker day:       {summary['breaker_trigger_day']} ({test_dates[summary['breaker_trigger_day']].date()})")

    # Comparison table
    print("\n--- Strategy Comparison ---")
    comparison = compare_strategies(test_prob, test_close, threshold=0.5, fee=0.001)
    print(comparison.to_string(index=False))

    # Save results CSV
    results_df = pd.DataFrame([summary])
    results_df.to_csv(OUTPUTS / "risk_managed_backtest_results.csv", index=False)
    print(f"\n  → Saved {OUTPUTS / 'risk_managed_backtest_results.csv'}")

    # Save equity curve
    equity_df = pd.DataFrame({
        "date": test_dates,
        "equity": rm.equity,
        "position": rm.positions,
        "kelly_fraction": rm.kelly_fraction,
        "vol_target_factor": rm.vol_target_factor,
        "raw_prob": rm.raw_signal,
    })
    equity_df.to_csv(OUTPUTS / "risk_managed_equity_curve.csv", index=False)
    print(f"  → Saved {OUTPUTS / 'risk_managed_equity_curve.csv'}")

    # Save comparison
    comparison.to_csv(OUTPUTS / "strategy_comparison.csv", index=False)
    print(f"  → Saved {OUTPUTS / 'strategy_comparison.csv'}")


if __name__ == "__main__":
    main()
