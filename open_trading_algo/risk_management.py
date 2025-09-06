"""Risk management utilities for open_trading_algo.

This module provides functions for position sizing, stop-loss logic, and portfolio hedging.
"""

import pandas as pd
import numpy as np


def calculate_position_size(capital, risk_per_trade, stop_loss_pct):
    """Calculate position size based on risk.

    Args:
        capital (float): Total capital.
        risk_per_trade (float): Risk per trade as percentage.
        stop_loss_pct (float): Stop loss percentage.

    Returns:
        float: Position size.
    """
    # ...existing code...


def fixed_fractional_position_size(
    account_value: float, risk_per_trade: float, stop_distance: float
) -> float:
    """Calculate position size using fixed fractional risk model.

    Args:
        account_value (float): Total portfolio value.
        risk_per_trade (float): Fraction of account to risk per trade (e.g., 0.01).
        stop_distance (float): Price distance from entry to stop-loss.

    Returns:
        float: Number of shares/contracts to buy/sell.
    """
    if stop_distance <= 0:
        return 0
    return (account_value * risk_per_trade) / stop_distance


def atr_stop_loss(df: pd.DataFrame, atr_period: int = 14, atr_mult: float = 2.0) -> pd.Series:
    """Calculate stop-loss level using ATR (Average True Range).

    Args:
        df (pd.DataFrame): DataFrame with OHLC data.
        atr_period (int): Period for ATR calculation. Defaults to 14.
        atr_mult (float): ATR multiplier for stop distance. Defaults to 2.0.

    Returns:
        pd.Series: Stop-loss price for each row.
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
    """Calculate trailing stop-loss as a percentage below the highest close since entry.

    Args:
        df (pd.DataFrame): DataFrame with close price data.
        trail_perc (float): Trailing percentage below highest close. Defaults to 0.05.

    Returns:
        pd.Series: Stop-loss price for each row.
    """
    rolling_max = df["close"].cummax()
    return rolling_max * (1 - trail_perc)


def portfolio_hedge_signal(
    df: pd.DataFrame, market_index: pd.Series, threshold: float = -0.03
) -> pd.Series:
    """Generate hedge signal if market index drops below threshold.

    Args:
        df (pd.DataFrame): DataFrame with market data.
        market_index (pd.Series): Market index series.
        threshold (float): Threshold for hedge activation (e.g., -0.03 for -3%). Defaults to -0.03.

    Returns:
        pd.Series: Boolean Series indicating if hedge should be active.
    """
    market_return = market_index.pct_change()
    return market_return < threshold


def max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum drawdown from an equity curve.

    Args:
        equity_curve (pd.Series): Series representing the equity curve.

    Returns:
        float: Maximum drawdown as a decimal.
    """
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return drawdown.min()
