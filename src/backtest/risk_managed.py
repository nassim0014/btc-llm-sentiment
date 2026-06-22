"""
Risk-managed backtester with Kelly position sizing, volatility targeting,
and a drawdown circuit breaker.

This replaces the 100% all-in/all-out backtester from Phase 1 with a
production-grade risk engine.

Public API
----------
- `risk_managed_backtest`: run a full backtest with Kelly + vol-targeting + DD breaker.
- `RiskManagedResult`: dataclass holding equity curve, positions, metrics, and
  a log of circuit-breaker events.

Risk Layers
-----------
1. **Kelly Fraction Sizing**: position = (prob - threshold) / (1 - threshold),
   capped at 1.0. When the model is confident, it goes bigger; when it's
   borderline, it scales down.
2. **Volatility Targeting**: scale the Kelly-sized position inversely to the
   realized 20-day volatility so the strategy targets 20% annualized vol.
3. **Drawdown Circuit Breaker**: if the portfolio drawdown hits -15%, flatten
   all positions and halt trading for the remainder of the backtest.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------
@dataclass
class RiskManagedResult:
    """Container for risk-managed backtest results."""
    equity: np.ndarray
    positions: np.ndarray
    daily_returns: np.ndarray
    raw_signal: np.ndarray          # model probability
    kelly_fraction: np.ndarray      # (prob - threshold) / (1 - threshold)
    vol_target_factor: np.ndarray   # 20% target vol / realized 20d vol
    circuit_breaker_triggered: bool
    breaker_trigger_day: Optional[int]
    metrics: dict

    def to_summary_dict(self) -> dict:
        return {
            "final_portfolio_value": round(float(self.equity[-1]), 4),
            "total_return_pct": round((float(self.equity[-1]) - 1) * 100, 2),
            "annualized_sharpe": round(self.metrics["sharpe"], 4),
            "annualized_sortino": round(self.metrics["sortino"], 4),
            "max_drawdown_pct": round(self.metrics["max_dd"] * 100, 2),
            "win_rate_pct": round(self.metrics["win_rate"] * 100, 2),
            "n_trades": self.metrics["n_trades"],
            "circuit_breaker_triggered": self.circuit_breaker_triggered,
            "breaker_trigger_day": self.breaker_trigger_day,
            "avg_position_size": round(float(np.mean(self.positions)), 4),
            "max_position_size": round(float(np.max(self.positions)), 4),
        }


# ---------------------------------------------------------------------
# Core backtester
# ---------------------------------------------------------------------
def risk_managed_backtest(
    prob: np.ndarray,
    close: np.ndarray,
    threshold: float = 0.5,
    fee: float = 0.001,
    target_annual_vol: float = 0.20,
    vol_lookback: int = 20,
    max_drawdown_pct: float = 0.15,
    trading_days_per_year: int = 252,
) -> RiskManagedResult:
    """Run a risk-managed backtest with Kelly + vol-targeting + DD breaker.

    Parameters
    ----------
    prob : np.ndarray
        Model probability of upward 5-day move. Shape (n_days,).
    close : np.ndarray
        BTC close prices. Shape (n_days,).
    threshold : float
        Trading signal threshold. Prob >= threshold → long.
    fee : float
        Transaction cost per trade (default 0.1%).
    target_annual_vol : float
        Target annualized volatility. Default 0.20 (20%).
    vol_lookback : int
        Realized vol lookback window in days. Default 20.
    max_drawdown_pct : float
        Circuit breaker trigger level. Default 0.15 (flatten at -15% DD).
    trading_days_per_year : int
        Annualization factor. Default 252.

    Returns
    -------
    RiskManagedResult
    """
    n = len(close)
    # Daily returns
    rets = np.diff(close) / close[:-1]  # shape (n-1,)
    # Realized 20-day volatility (annualized)
    realized_vol = pd.Series(rets).rolling(vol_lookback).std().values * np.sqrt(trading_days_per_year)
    # Fill initial NaNs with the first valid value
    first_valid = np.where(~np.isnan(realized_vol))[0]
    if len(first_valid) > 0:
        realized_vol[:first_valid[0]] = realized_vol[first_valid[0]]
    else:
        realized_vol[:] = 0.20  # fallback

    # Avoid division by zero
    realized_vol = np.maximum(realized_vol, 0.01)

    # ---- Layer 1: Kelly fraction sizing ----
    # position = (prob - threshold) / (1 - threshold), capped at [0, 1]
    kelly = np.zeros(n)
    for i in range(n):
        if prob[i] >= threshold:
            k = (prob[i] - threshold) / max(1.0 - threshold, 1e-6)
            kelly[i] = min(k, 1.0)
        else:
            kelly[i] = 0.0

    # ---- Layer 2: Volatility targeting ----
    # Scale inversely to realized vol so target annual vol = 20%
    vol_factor = np.zeros(n)
    for i in range(n):
        vol_factor[i] = min(target_annual_vol / realized_vol[i], 1.0) if i < len(realized_vol) else 1.0

    # Combined position before circuit breaker
    raw_position = kelly * vol_factor  # shape (n,)

    # ---- Layer 3: Drawdown circuit breaker ----
    # Simulate day-by-day, tracking equity and drawdown
    equity = np.ones(n)         # start at 1.0
    positions = np.zeros(n)     # actual position each day
    daily_returns = np.zeros(n)
    breaker_triggered = False
    breaker_trigger_day = None

    # Position for day t is decided at end of day t-1 (using prob[t-1] etc.)
    # Return for day t = position[t-1] * rets[t-1] (since rets[t-1] = close[t]/close[t-1] - 1)
    # We'll use: positions[t] = raw_position[t] (decided at start of day t using prob up to t-1)
    # daily_returns[t] = positions[t-1] * rets[t-1] - trade_cost

    for t in range(1, n):
        # Check circuit breaker
        if breaker_triggered:
            positions[t] = 0.0
            daily_returns[t] = 0.0
            equity[t] = equity[t-1]
            continue

        # Position for today = raw_position[t] (using yesterday's signal)
        pos_today = raw_position[t]
        # Yesterday's position (for trade cost calculation)
        pos_yesterday = positions[t-1] if t > 0 else 0.0

        # Trade cost = fee * |position change|
        trade_cost = fee * abs(pos_today - pos_yesterday)

        # Daily return = position * market return - trade cost
        daily_ret = pos_yesterday * rets[t-1] - trade_cost
        daily_returns[t] = daily_ret
        equity[t] = equity[t] * (1 + daily_ret)
        positions[t] = pos_today

        # Check drawdown
        running_max = np.max(equity[:t+1])
        drawdown = (equity[t] - running_max) / running_max if running_max > 0 else 0.0

        if drawdown <= -max_drawdown_pct:
            breaker_triggered = True
            breaker_trigger_day = t
            # Flatten position immediately
            positions[t] = 0.0
            # Charge a final exit fee
            daily_returns[t] -= fee * abs(pos_today)
            equity[t] *= (1 - fee * abs(pos_today))

    # ---- Metrics ----
    valid_returns = daily_returns[1:]  # skip day 0
    n_valid = len(valid_returns)
    ann = np.sqrt(trading_days_per_year)

    if n_valid > 0 and valid_returns.std() > 0:
        sharpe = (valid_returns.mean() / valid_returns.std()) * ann
    else:
        sharpe = 0.0

    downside = valid_returns[valid_returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (valid_returns.mean() / downside.std()) * ann
    else:
        sortino = sharpe

    running_max_eq = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max_eq) / running_max_eq
    max_dd = float(drawdowns.min()) if len(drawdowns) > 0 else 0.0
    win_rate = float((valid_returns > 0).mean()) if n_valid > 0 else 0.0

    # Count trades (position changes)
    pos_changes = np.abs(np.diff(positions))
    n_trades = int(np.sum(pos_changes > 0.01))  # count changes > 1% as trades

    metrics = {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_dd": float(max_dd),
        "win_rate": float(win_rate),
        "n_trades": n_trades,
        "final_value": float(equity[-1]),
    }

    return RiskManagedResult(
        equity=equity,
        positions=positions,
        daily_returns=daily_returns,
        raw_signal=prob,
        kelly_fraction=kelly,
        vol_target_factor=vol_factor,
        circuit_breaker_triggered=breaker_triggered,
        breaker_trigger_day=breaker_trigger_day,
        metrics=metrics,
    )


# ---------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------
def compare_strategies(
    prob: np.ndarray,
    close: np.ndarray,
    threshold: float = 0.5,
    fee: float = 0.001,
) -> pd.DataFrame:
    """Run both the simple and risk-managed backtest, return a comparison."""
    # Simple backtest (all-in/all-out)
    signal = (prob >= threshold).astype(int)
    rets = np.diff(close) / close[:-1]
    simple_rets = signal[:-1] * rets
    trade_flags = np.abs(np.diff(signal))
    simple_rets = simple_rets - trade_flags * fee
    simple_equity = np.cumprod(1 + simple_rets)

    ann = np.sqrt(252)
    simple_sharpe = (simple_rets.mean() / (simple_rets.std()+1e-12)) * ann
    simple_dd = ((simple_equity - np.maximum.accumulate(simple_equity)) / np.maximum.accumulate(simple_equity)).min()

    # Risk-managed
    rm = risk_managed_backtest(prob, close, threshold, fee)

    # Buy & Hold
    bh_rets = rets
    bh_equity = np.cumprod(1 + bh_rets)
    bh_sharpe = (bh_rets.mean() / (bh_rets.std()+1e-12)) * ann
    bh_dd = ((bh_equity - np.maximum.accumulate(bh_equity)) / np.maximum.accumulate(bh_equity)).min()

    rows = [
        {
            "strategy": "Simple (all-in/out)",
            "final_value": float(simple_equity[-1]),
            "sharpe": float(simple_sharpe),
            "max_dd": float(simple_dd),
            "n_trades": int(trade_flags.sum() // 2),
            "circuit_breaker": False,
        },
        {
            "strategy": "Risk-Managed (Kelly+Vol+DD)",
            "final_value": rm.metrics["final_value"],
            "sharpe": rm.metrics["sharpe"],
            "max_dd": rm.metrics["max_dd"],
            "n_trades": rm.metrics["n_trades"],
            "circuit_breaker": rm.circuit_breaker_triggered,
        },
        {
            "strategy": "Buy & Hold",
            "final_value": float(bh_equity[-1]),
            "sharpe": float(bh_sharpe),
            "max_dd": float(bh_dd),
            "n_trades": 1,
            "circuit_breaker": False,
        },
    ]
    return pd.DataFrame(rows)
