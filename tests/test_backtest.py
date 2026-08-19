"""
Unit tests for the risk-managed backtester and walk-forward CV utilities.

Run with:  pytest tests/ -v
"""
import numpy as np
import pytest

from src.backtest.risk_managed import compare_strategies, risk_managed_backtest
from src.cv.walk_forward import (
    evaluate_oof_metrics,
    walk_forward_splits,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def synthetic_prob_and_close() -> tuple[np.ndarray, np.ndarray]:
    """Deterministic synthetic data: 100 days, mild uptrend with noise."""
    np.random.seed(0)
    n = 100
    # Random-walk close with positive drift
    rets = np.random.normal(0.001, 0.02, size=n)
    close = 100 * np.cumprod(1 + rets)
    # Probabilities correlated with next-day return sign
    next_ret = np.diff(close) / close[:-1]
    prob = np.zeros(n)
    prob[:-1] = 0.5 + 0.3 * np.tanh(next_ret)  # bounded in (0.2, 0.8)
    prob[-1] = 0.5
    return prob, close


@pytest.fixture
def monotonic_close() -> np.ndarray:
    """Strictly increasing close prices — no drawdowns."""
    return np.linspace(100, 200, 50)


@pytest.fixture
def drawdown_close() -> np.ndarray:
    """Close that drops 25% from the start — guaranteed equity drawdown > 15%
    when fully invested, because there's no prior gain to buffer against."""
    return np.linspace(100, 75, 40)  # -25% over 40 days, monotonic decline


# ---------------------------------------------------------------------
# Walk-Forward CV tests
# ---------------------------------------------------------------------
class TestWalkForwardSplits:
    def test_expanding_window_grows(self):
        """Train window should expand by val_size between folds."""
        folds = list(walk_forward_splits(n=730, n_folds=5, min_train=400, val_size=60))
        assert len(folds) == 5
        # Train sizes: 400, 460, 520, 580, 640
        train_sizes = [f.train_end - f.train_start for f in folds]
        assert train_sizes == [400, 460, 520, 580, 640]

    def test_strict_temporal_ordering(self):
        """val_start must equal train_end (no overlap, no gap)."""
        folds = list(walk_forward_splits(n=500, n_folds=3, min_train=200, val_size=50))
        for f in folds:
            assert f.val_start == f.train_end, f"Fold {f.fold_num} has gap/overlap"
            assert f.val_end > f.val_start

    def test_no_lookahead_leakage(self):
        """Train indices must all be < val_start."""
        folds = list(walk_forward_splits(n=400, n_folds=3, min_train=150, val_size=40))
        for f in folds:
            assert f.train_idx.max() < f.val_idx.min(), \
                f"Fold {f.fold_num}: train index leaks into val window"

    def test_fold_does_not_exceed_n(self):
        """Last fold's val_end must not exceed n."""
        n = 600
        folds = list(walk_forward_splits(n=n, n_folds=4, min_train=300, val_size=60))
        assert folds[-1].val_end <= n

    def test_invalid_params_raise(self):
        """Should raise if min_train + n_folds*val_size > n."""
        with pytest.raises(ValueError):
            list(walk_forward_splits(n=100, n_folds=5, min_train=80, val_size=10))


# ---------------------------------------------------------------------
# Backtester tests
# ---------------------------------------------------------------------
class TestRiskManagedBacktest:
    def test_no_drawdown_on_monotonic_uptrend(self, monotonic_close):
        """If price only goes up and we're always long, drawdown should be ~0."""
        n = len(monotonic_close)
        prob = np.full(n, 0.9)  # always confident long
        result = risk_managed_backtest(
            prob=prob, close=monotonic_close,
            threshold=0.5, fee=0.001,
            target_annual_vol=0.20, vol_lookback=5,
            max_drawdown_pct=0.15,
        )
        assert result.metrics["max_dd"] >= -0.01, \
            f"Expected ~0 drawdown on monotonic uptrend, got {result.metrics['max_dd']}"

    def test_circuit_breaker_triggers_on_large_drawdown(self, drawdown_close):
        """A -25% monotonic decline from day 1 should trigger the -15% circuit
        breaker, since the equity (starting at 1.0, fully invested) tracks the
        price down with no prior gain to buffer against."""
        n = len(drawdown_close)
        prob = np.full(n, 0.99)  # max confidence → Kelly caps near 1.0
        result = risk_managed_backtest(
            prob=prob, close=drawdown_close,
            threshold=0.5, fee=0.001,
            target_annual_vol=5.0,  # very high → vol factor stays 1.0 (no downscaling)
            vol_lookback=5,
            max_drawdown_pct=0.15,
        )
        assert result.circuit_breaker_triggered is True, \
            f"Circuit breaker should have triggered on -25% decline (equity max_dd={result.metrics['max_dd']:.3f})"
        assert result.breaker_trigger_day is not None

    def test_kelly_sizing_scales_with_confidence(self, synthetic_prob_and_close):
        """Higher-confidence predictions should produce larger positions on average."""
        prob, close = synthetic_prob_and_close
        # High confidence (push probs toward 1)
        prob_high = np.clip(prob + 0.3, 0.5, 0.99)
        # Low confidence (push probs toward 0.5)
        prob_low = np.clip(prob - 0.2, 0.5, 0.7)

        result_high = risk_managed_backtest(prob_high, close, threshold=0.5)
        result_low = risk_managed_backtest(prob_low, close, threshold=0.5)

        assert result_high.kelly_fraction.mean() >= result_low.kelly_fraction.mean(), \
            "High-confidence probs should produce larger Kelly positions"

    def test_position_size_capped_at_1(self, synthetic_prob_and_close):
        """Kelly + vol-target combined should never exceed 1.0."""
        prob, close = synthetic_prob_and_close
        result = risk_managed_backtest(prob, close, threshold=0.5)
        assert result.positions.max() <= 1.0 + 1e-9, \
            f"Position exceeded 1.0: max={result.positions.max()}"

    def test_equity_starts_at_1(self, synthetic_prob_and_close):
        """Equity curve should start at 1.0 (normalized)."""
        prob, close = synthetic_prob_and_close
        result = risk_managed_backtest(prob, close, threshold=0.5)
        assert abs(result.equity[0] - 1.0) < 1e-9

    def test_metrics_dict_has_required_keys(self, synthetic_prob_and_close):
        """All expected metric keys must be present."""
        prob, close = synthetic_prob_and_close
        result = risk_managed_backtest(prob, close, threshold=0.5)
        required = {"sharpe", "sortino", "max_dd", "win_rate", "n_trades", "final_value"}
        assert required.issubset(result.metrics.keys()), \
            f"Missing keys: {required - set(result.metrics.keys())}"


# ---------------------------------------------------------------------
# Strategy comparison test
# ---------------------------------------------------------------------
class TestCompareStrategies:
    def test_comparison_returns_three_strategies(self, synthetic_prob_and_close):
        """compare_strategies should return Simple, Risk-Managed, Buy & Hold."""
        prob, close = synthetic_prob_and_close
        df = compare_strategies(prob, close, threshold=0.5, fee=0.001)
        assert set(df["strategy"]) == {"Simple (all-in/out)", "Risk-Managed (Kelly+Vol+DD)", "Buy & Hold"}
        assert len(df) == 3

    def test_buy_hold_final_value_matches_close_ratio(self, synthetic_prob_and_close):
        """Buy & Hold final value should equal close[-1] / close[0]."""
        prob, close = synthetic_prob_and_close
        df = compare_strategies(prob, close, threshold=0.5, fee=0.001)
        bh = df[df["strategy"] == "Buy & Hold"].iloc[0]
        expected = close[-1] / close[0]
        assert abs(bh["final_value"] - expected) < 0.01


# ---------------------------------------------------------------------
# OOF metrics test
# ---------------------------------------------------------------------
class TestEvaluateOofMetrics:
    def test_perfect_predictions(self):
        """Perfect predictions should yield high accuracy and AUC."""
        np.random.seed(0)
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_prob = y_true.astype(float)  # perfect separation
        close = np.linspace(100, 108, len(y_true) + 1)
        metrics = evaluate_oof_metrics(y_true, y_prob, close, threshold=0.5)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0

    def test_random_predictions_around_half_accuracy(self):
        """Random predictions should hover near 0.5 accuracy."""
        np.random.seed(0)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.uniform(0.3, 0.7, 100)
        close = np.cumprod(1 + np.random.normal(0, 0.01, 101)) * 100
        metrics = evaluate_oof_metrics(y_true, y_prob, close, threshold=0.5)
        assert 0.3 < metrics["accuracy"] < 0.7

    def test_auc_nan_when_single_class(self):
        """AUC returns NaN when y_true has only one class (ValueError caught)."""
        y_true = np.ones(10, dtype=int)  # all class 1
        y_prob = np.linspace(0.1, 0.9, 10)
        close = np.linspace(100, 110, 11)
        metrics = evaluate_oof_metrics(y_true, y_prob, close)
        assert np.isnan(metrics["auc"])

    def test_empty_input_returns_zero_metrics(self):
        """Empty arrays should return zeroed trading metrics, not crash."""
        y_true = np.array([], dtype=int)
        y_prob = np.array([])
        close = np.array([100.0])
        metrics = evaluate_oof_metrics(y_true, y_prob, close)
        assert metrics["n_trades"] == 0
        assert metrics["sharpe"] == 0.0

    def test_single_element_input_does_not_crash(self):
        """Single-element input should not raise — degenerate but valid."""
        y_true = np.array([1])
        y_prob = np.array([0.8])
        close = np.array([100.0, 101.0])
        metrics = evaluate_oof_metrics(y_true, y_prob, close)
        # n_days will be 0 (len(strat_rets) == 0 since diff of 1-elem)
        assert "n_trades" in metrics

    def test_mismatched_lengths_fees_skipped(self):
        """When signal and rets lengths mismatch, fee adjustment is skipped."""
        y_true = np.array([1, 0, 1, 0, 1])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
        # close has 3 elements — len(rets) = 2, len(signal) = 5
        close = np.array([100.0, 101.0, 102.0])
        metrics = evaluate_oof_metrics(y_true, y_prob, close)
        # Should not crash, should return valid metrics
        assert "accuracy" in metrics
        assert "n_trades" in metrics

    def test_zero_std_returns_finite_sharpe(self):
        """When all returns are identical (std=0), sharpe should be finite."""
        y_true = np.array([1, 1, 1, 1])
        y_prob = np.array([0.9, 0.9, 0.9, 0.9])
        # Close that produces zero-variance returns (all same return)
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        metrics = evaluate_oof_metrics(y_true, y_prob, close)
        assert np.isfinite(metrics["sharpe"])
