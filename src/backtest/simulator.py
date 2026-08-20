"""Interactive backtest simulator — extracted from pages/4_🎛️_Backtest_Simulator.py (item 5).

This is the same logic as `src.backtest.risk_managed.risk_managed_backtest`
but with user-adjustable parameters for the Streamlit simulator. Extracted
to a testable module so the math can be unit-tested without running Streamlit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(
    prob: np.ndarray,
    close: np.ndarray,
    threshold: float = 0.5,
    fee: float = 0.001,
    target_annual_vol: float = 0.20,
    vol_lookback: int = 20,
    max_drawdown_pct: float = 0.15,
    kelly_cap: float = 1.0,
    use_vol_target: bool = True,
    use_circuit_breaker: bool = True,
) -> dict:
    """Run a parameterized backtest with interactive controls.

    Parameters
    ----------
    prob : np.ndarray
        Model probabilities (0-1) for each day.
    close : np.ndarray
        BTC close prices for each day.
    threshold : float
        Minimum probability to open a position.
    fee : float
        Per-trade transaction cost fraction.
    target_annual_vol : float
        Target annualized volatility for vol-targeting.
    vol_lookback : int
        Rolling window (days) for realized vol calculation.
    max_drawdown_pct : float
        Circuit-breaker trigger: stop trading if drawdown exceeds this.
    kelly_cap : float
        Maximum Kelly fraction.
    use_vol_target : bool
        If False, vol factor is always 1.0.
    use_circuit_breaker : bool
        If False, no drawdown-based stop.

    Returns
    -------
    dict with keys: equity, positions, daily_returns, sharpe, sortino,
    max_dd, win_rate, n_trades, final_value, breaker_triggered,
    breaker_trigger_day.
    """
    n = len(close)
    rets = np.diff(close) / close[:-1]  # length n-1
    # realized_vol aligned with rets (length n-1); pad to length n with a leading 0
    # so indexing matches the close/prob arrays in the backtest loop
    realized_vol = pd.Series(rets).rolling(vol_lookback).std() * np.sqrt(252)
    # Reindex to length n: prepend a NaN so vol_factor[0] corresponds to day 0
    realized_vol = pd.concat([pd.Series([np.nan]), realized_vol]).reset_index(drop=True)

    # Vectorized Volatility Scaling (replaces the buggy for-loop)
    # 1. Fill initial NaNs (from the rolling window) with target vol so factor defaults to 1.0
    realized_vol_safe = realized_vol.fillna(target_annual_vol)
    # 2. Prevent division by zero
    realized_vol_safe = realized_vol_safe.replace(0, np.inf)
    # 3. Calculate scaling factor and cap at 1.0 (max leverage)
    vol_factor = (target_annual_vol / realized_vol_safe).clip(upper=1.0)
    # 4. If vol-targeting is disabled, force factor to 1.0
    if not use_vol_target:
        vol_factor = pd.Series(1.0, index=vol_factor.index)

    # Kelly fraction sizing (vectorized)
    kelly = np.where(
        prob >= threshold,
        np.minimum((prob - threshold) / max(1.0 - threshold, 1e-6), kelly_cap),
        0.0,
    )

    # Combine Kelly + vol-target (convert vol_factor to numpy for element-wise multiply)
    raw_position = kelly * vol_factor.values

    # Backtest with optional circuit breaker
    equity = np.ones(n)
    positions = np.zeros(n)
    daily_returns = np.zeros(n)
    breaker_triggered = False
    breaker_trigger_day = None

    for t in range(1, n):
        if breaker_triggered and use_circuit_breaker:
            positions[t] = 0.0
            daily_returns[t] = 0.0
            equity[t] = equity[t - 1]
            continue

        pos_today = raw_position[t]
        pos_yesterday = positions[t - 1] if t > 0 else 0.0
        trade_cost = fee * abs(pos_today - pos_yesterday)
        daily_ret = pos_yesterday * rets[t - 1] - trade_cost
        daily_returns[t] = daily_ret
        equity[t] = equity[t - 1] * (1 + daily_ret)
        positions[t] = pos_today

        if use_circuit_breaker:
            running_max = np.max(equity[: t + 1])
            drawdown = (equity[t] - running_max) / running_max if running_max > 0 else 0.0
            if drawdown <= -max_drawdown_pct:
                breaker_triggered = True
                breaker_trigger_day = t
                positions[t] = 0.0
                daily_returns[t] -= fee * abs(pos_today)
                equity[t] *= 1 - fee * abs(pos_today)

    # Metrics
    valid_returns = daily_returns[1:]
    n_valid = len(valid_returns)
    ann = np.sqrt(252)

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
    pos_changes = np.abs(np.diff(positions))
    n_trades = int(np.sum(pos_changes > 0.01))

    return {
        "equity": equity,
        "positions": positions,
        "daily_returns": daily_returns,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_dd": float(max_dd),
        "win_rate": float(win_rate),
        "n_trades": n_trades,
        "final_value": float(equity[-1]),
        "breaker_triggered": breaker_triggered,
        "breaker_trigger_day": breaker_trigger_day,
    }
