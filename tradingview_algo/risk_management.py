"""
Risk management utilities for use with SignalOptimizer and backtesting.
Includes position sizing, stop-loss, and portfolio hedge logic.
"""
import pandas as pd
import numpy as np


def fixed_fractional_position_size(
    account_value: float, risk_per_trade: float, stop_distance: float
) -> float:
    """
    Calculate position size using fixed fractional risk model.
    account_value: total portfolio value
    risk_per_trade: fraction of account to risk per trade (e.g., 0.01)
    stop_distance: price distance from entry to stop-loss
    Returns number of shares/contracts to buy/sell.
    """
    if stop_distance <= 0:
        return 0
    return (account_value * risk_per_trade) / stop_distance


def atr_stop_loss(df: pd.DataFrame, atr_period: int = 14, atr_mult: float = 2.0) -> pd.Series:
    """
    Calculate stop-loss level using ATR (Average True Range).
    Returns stop-loss price for each row.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    return close - atr_mult * atr


def trailing_stop_loss(df: pd.DataFrame, trail_perc: float = 0.05) -> pd.Series:
    """
    Trailing stop-loss as a percentage below the highest close since entry.
    Returns stop-loss price for each row.
    """
    rolling_max = df["close"].cummax()
    return rolling_max * (1 - trail_perc)


def portfolio_hedge_signal(
    df: pd.DataFrame, market_index: pd.Series, threshold: float = -0.03
) -> pd.Series:
    """
    Generate hedge signal if market index drops below threshold (e.g., -3% in a day).
    Returns boolean Series: True if hedge should be active.
    """
    market_return = market_index.pct_change()
    return market_return < threshold


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate maximum drawdown from an equity curve.
    """
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return drawdown.min()
