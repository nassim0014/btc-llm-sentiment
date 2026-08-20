"""Tests for src/backtest/simulator.py — extracted from the Streamlit page (item 5).

Tests the run_backtest function with synthetic data to verify the core
math: position sizing, vol-targeting, circuit breaker, and metrics.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.backtest.simulator import run_backtest


class TestRunBacktestBasics:
    def test_returns_expected_keys(self):
        """The result dict has all expected keys."""
        prob = np.array([0.6, 0.7, 0.5, 0.8])
        close = np.array([100.0, 101.0, 102.0, 103.0])
        result = run_backtest(prob, close)
        expected_keys = {
            "equity", "positions", "daily_returns", "sharpe", "sortino",
            "max_dd", "win_rate", "n_trades", "final_value",
            "breaker_triggered", "breaker_trigger_day",
        }
        assert set(result.keys()) == expected_keys

    def test_equity_starts_at_one(self):
        """The equity curve starts at 1.0 (normalized)."""
        prob = np.array([0.6, 0.7])
        close = np.array([100.0, 101.0])
        result = run_backtest(prob, close)
        assert result["equity"][0] == 1.0

    def test_no_trades_when_prob_below_threshold(self):
        """When all probs are below threshold, no positions are taken."""
        prob = np.array([0.3, 0.4, 0.2])
        close = np.array([100.0, 101.0, 102.0])
        result = run_backtest(prob, close, threshold=0.5)
        # All positions should be 0
        assert np.all(result["positions"] == 0.0)
        # Equity stays at 1.0 (no returns, no costs)
        assert result["final_value"] == pytest.approx(1.0, abs=0.001)

    def test_position_taken_when_prob_above_threshold(self):
        """When prob >= threshold, a position is taken."""
        prob = np.array([0.6, 0.7, 0.8])
        close = np.array([100.0, 101.0, 102.0])
        result = run_backtest(prob, close, threshold=0.5)
        # At least one position should be > 0
        assert np.any(result["positions"] > 0)


class TestRunBacktestMetrics:
    def test_sharpe_is_float(self):
        """Sharpe ratio is a float."""
        prob = np.array([0.6, 0.7, 0.5, 0.8, 0.6])
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        result = run_backtest(prob, close)
        assert isinstance(result["sharpe"], float)

    def test_max_dd_is_non_positive(self):
        """Max drawdown is always <= 0."""
        prob = np.array([0.6, 0.7, 0.5, 0.8, 0.6])
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        result = run_backtest(prob, close)
        assert result["max_dd"] <= 0.0

    def test_win_rate_between_zero_and_one(self):
        """Win rate is in [0, 1]."""
        prob = np.array([0.6, 0.7, 0.5, 0.8, 0.6])
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        result = run_backtest(prob, close)
        assert 0.0 <= result["win_rate"] <= 1.0

    def test_n_trades_non_negative(self):
        """Number of trades is >= 0."""
        prob = np.array([0.6, 0.7, 0.5, 0.8, 0.6])
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        result = run_backtest(prob, close)
        assert result["n_trades"] >= 0


class TestRunBacktestVolTargeting:
    def test_vol_target_disabled_forces_factor_one(self):
        """When use_vol_target=False, positions are not vol-scaled."""
        prob = np.array([0.6, 0.7, 0.8, 0.9])
        close = np.array([100.0, 101.0, 102.0, 103.0])
        result_on = run_backtest(prob, close, use_vol_target=True)
        result_off = run_backtest(prob, close, use_vol_target=False)
        # With vol-targeting off, positions should generally be larger
        # (no vol-based downscaling)
        assert np.sum(result_off["positions"]) >= np.sum(result_on["positions"])


class TestRunBacktestCircuitBreaker:
    def test_circuit_breaker_not_triggered_on_normal_data(self):
        """With gently rising prices, the breaker should not trigger."""
        prob = np.array([0.6] * 10)
        close = np.array([100.0 + i for i in range(10)])
        result = run_backtest(prob, close, max_drawdown_pct=0.15)
        assert result["breaker_triggered"] is False
        assert result["breaker_trigger_day"] is None

    def test_circuit_breaker_triggered_on_large_drawdown(self):
        """A sharp price drop should trigger the circuit breaker."""
        # Prices that drop >15% rapidly
        prob = np.array([0.9] * 10)
        close = np.array([100, 99, 98, 95, 90, 85, 80, 75, 70, 65], dtype=float)
        result = run_backtest(
            prob, close, max_drawdown_pct=0.15, use_circuit_breaker=True
        )
        assert result["breaker_triggered"] is True
        assert result["breaker_trigger_day"] is not None

    def test_circuit_breaker_disabled(self):
        """When use_circuit_breaker=False, breaker never triggers."""
        prob = np.array([0.9] * 10)
        close = np.array([100, 99, 98, 95, 90, 85, 80, 75, 70, 65], dtype=float)
        result = run_backtest(
            prob, close, use_circuit_breaker=False
        )
        assert result["breaker_triggered"] is False


class TestRunBacktestEdgeCases:
    def test_single_day(self):
        """A single day of data doesn't crash."""
        prob = np.array([0.6])
        close = np.array([100.0])
        result = run_backtest(prob, close)
        assert result["equity"][0] == 1.0
        assert result["sharpe"] == 0.0  # no valid returns

    def test_two_days(self):
        """Two days of data produces a valid result."""
        prob = np.array([0.6, 0.7])
        close = np.array([100.0, 101.0])
        result = run_backtest(prob, close)
        assert len(result["equity"]) == 2
        assert result["final_value"] > 0

    def test_fees_reduce_returns(self):
        """Higher fees should reduce final equity (all else equal)."""
        prob = np.array([0.7, 0.8, 0.6, 0.9, 0.7])
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        low_fee = run_backtest(prob, close, fee=0.001)
        high_fee = run_backtest(prob, close, fee=0.01)
        # Higher fees → lower final value
        assert high_fee["final_value"] <= low_fee["final_value"]
