"""Comprehensive trading metrics and analytics calculations.

This module provides functions for computing various trading metrics including:
- Volume and price-based metrics
- Volatility and risk measures
- Momentum and trend indicators
- Statistical and performance analytics
- Support/resistance levels
- Seasonal and calendar effects
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _compute_column_metrics(df: pd.DataFrame, column: str, window: int) -> pd.Series:
    """Generic function to compute rolling average metrics for a DataFrame column.

    Args:
        df (pd.DataFrame): Input DataFrame containing the column to analyze.
        column (str): Name of the column to compute metrics for.
        window (int): Rolling window size in periods.

    Returns:
        pd.Series: Rolling average values for the specified column and window.
            Returns NaN for periods with insufficient data.
    """
    if column not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    return df[column].rolling(window=window, min_periods=1).mean()


def compute_volume_10d_avg(df: pd.DataFrame) -> pd.Series:
    """Compute 10-day average volume.

    Args:
        df (pd.DataFrame): DataFrame with volume data.

    Returns:
        pd.Series: 10-day rolling average volume values.
    """
    return _compute_column_metrics(df, "volume", 10)


def compute_volume_30d_avg(df: pd.DataFrame) -> pd.Series:
    """Compute 30-day average volume.

    Args:
        df (pd.DataFrame): DataFrame with volume data.

    Returns:
        pd.Series: 30-day rolling average volume values.
    """
    return _compute_column_metrics(df, "volume", 30)


def compute_volume_60d_avg(df: pd.DataFrame) -> pd.Series:
    """Compute 60-day average volume.

    Args:
        df (pd.DataFrame): DataFrame with volume data.

    Returns:
        pd.Series: 60-day rolling average volume values.
    """
    return _compute_column_metrics(df, "volume", 60)


def compute_volume_90d_avg(df: pd.DataFrame) -> pd.Series:
    """Compute 90-day average volume.

    Args:
        df (pd.DataFrame): DataFrame with volume data.

    Returns:
        pd.Series: 90-day rolling average volume values.
    """
    return _compute_column_metrics(df, "volume", 90)


# --- HIGH PRIORITY METRICS ---


def compute_volatility_ratio(
    df: pd.DataFrame, short_window: int = 10, long_window: int = 30
) -> pd.Series:
    """Compute volatility ratio comparing short-term vs long-term volatility.

    Args:
        df (pd.DataFrame): DataFrame with price data (requires 'close' column).
        short_window (int, optional): Short-term window for volatility calculation. Defaults to 10.
        long_window (int, optional): Long-term window for volatility calculation. Defaults to 30.

    Returns:
        pd.Series: Volatility ratio (short-term / long-term). Values > 1 indicate
            increasing volatility, < 1 indicate decreasing volatility.
    """
    if "close" not in df.columns:
        return pd.Series(index=df.index, dtype=float)

    returns = df["close"].pct_change()
    short_vol = returns.rolling(window=short_window, min_periods=1).std()
    long_vol = returns.rolling(window=long_window, min_periods=1).std()

    return short_vol / long_vol


def compute_trend_strength(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Compute trend strength using Average Directional Index (ADX) methodology.

    Args:
        df (pd.DataFrame): DataFrame with OHLC data (high, low, close columns).
        window (int, optional): Window for ADX calculation. Defaults to 14.

    Returns:
        pd.Series: Trend strength values. Higher values indicate stronger trends.
            Typically, values above 25 indicate strong trends.
    """
    if not all(col in df.columns for col in ["high", "low", "close"]):
        return pd.Series(index=df.index, dtype=float)

    # Calculate True Range
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

    # Calculate Directional Movement
    dm_plus = df["high"] - df["high"].shift(1)
    dm_minus = df["low"].shift(1) - df["low"]

    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)

    # Calculate Directional Indicators
    di_plus = 100 * (dm_plus.rolling(window=window).mean() / tr.rolling(window=window).mean())
    di_minus = 100 * (dm_minus.rolling(window=window).mean() / tr.rolling(window=window).mean())

    # Calculate ADX
    dx = 100 * ((di_plus - di_minus).abs() / (di_plus + di_minus))
    adx = dx.rolling(window=window).mean()

    return adx


def compute_volume_price_trend(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Compute Volume Price Trend (VPT) indicator.

    Args:
        df (pd.DataFrame): DataFrame with price and volume data.
        window (int, optional): Window for smoothing. Defaults to 20.

    Returns:
        pd.Series: Volume Price Trend values. Positive values indicate
            buying pressure, negative values indicate selling pressure.
    """
    if not all(col in df.columns for col in ["close", "volume"]):
        return pd.Series(index=df.index, dtype=float)

    price_change = df["close"].pct_change()
    vpt = (price_change * df["volume"]).cumsum()

    if window > 1:
        return vpt.rolling(window=window, min_periods=1).mean()
    return vpt


def compute_max_drawdown(df: pd.DataFrame, window: int | None = None) -> pd.Series:
    """Compute maximum drawdown over a rolling window.

    Args:
        df (pd.DataFrame): DataFrame with price data (requires 'close' column).
        window (int, optional): Rolling window size. If None, computes cumulative max drawdown.

    Returns:
        pd.Series: Maximum drawdown values as positive percentages.
            Higher values indicate larger drawdowns.
    """
    if "close" not in df.columns:
        return pd.Series(index=df.index, dtype=float)

    if window is None:
        # Cumulative max drawdown
        cumulative_max = df["close"].cummax()
        drawdown = (df["close"] - cumulative_max) / cumulative_max
        return drawdown.abs()
    else:
        # Rolling max drawdown
        rolling_max = df["close"].rolling(window=window, min_periods=1).max()
        drawdown = (df["close"] - rolling_max) / rolling_max
        return drawdown.abs()


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute Volume Weighted Average Price (VWAP).

    Args:
        df (pd.DataFrame): DataFrame with OHLC and volume data.

    Returns:
        pd.Series: VWAP values. Can be used to identify institutional activity
            and support/resistance levels.
    """
    if not all(col in df.columns for col in ["high", "low", "close", "volume"]):
        return pd.Series(index=df.index, dtype=float)

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cumulative_volume = df["volume"].cumsum()
    cumulative_price_volume = (typical_price * df["volume"]).cumsum()

    return cumulative_price_volume / cumulative_volume


# --- MEDIUM PRIORITY METRICS ---


def compute_beta(
    df: pd.DataFrame, market_df: pd.DataFrame | None = None, window: int = 252
) -> pd.Series:
    """Compute beta (market correlation coefficient).

    Args:
        df (pd.DataFrame): DataFrame with asset price data (requires 'close' column).
        market_df (pd.DataFrame, optional): Market/benchmark data. If None, uses df itself.
        window (int, optional): Rolling window for beta calculation. Defaults to 252 (1 year).

    Returns:
        pd.Series: Beta values. Values > 1 indicate higher volatility than market,
            < 1 indicate lower volatility, negative values indicate inverse correlation.
    """
    if "close" not in df.columns:
        return pd.Series(index=df.index, dtype=float)

    asset_returns = df["close"].pct_change()

    if market_df is None or "close" not in market_df.columns:
        # Use asset itself as market proxy (will give beta ≈ 1)
        market_returns = asset_returns
    else:
        market_returns = market_df["close"].pct_change()

    # Align indices
    common_index = asset_returns.index.intersection(market_returns.index)
    asset_returns = asset_returns.loc[common_index]
    market_returns = market_returns.loc[common_index]

    covariance = (asset_returns * market_returns).rolling(window=window, min_periods=30).mean()
    market_variance = (market_returns**2).rolling(window=window, min_periods=30).mean()

    return covariance / market_variance


def compute_alpha(
    df: pd.DataFrame,
    market_df: pd.DataFrame | None = None,
    risk_free_rate: float = 0.02,
    window: int = 252,
) -> pd.Series:
    """Compute alpha (risk-adjusted excess returns).

    Args:
        df (pd.DataFrame): DataFrame with asset price data (requires 'close' column).
        market_df (pd.DataFrame, optional): Market/benchmark data. If None, uses df itself.
        risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.02 (2%).
        window (int, optional): Rolling window for alpha calculation. Defaults to 252.

    Returns:
        pd.Series: Alpha values. Positive values indicate outperformance,
            negative values indicate underperformance relative to market.
    """
    if "close" not in df.columns:
        return pd.Series(index=df.index, dtype=float)

    asset_returns = df["close"].pct_change()
    daily_rf_rate = risk_free_rate / 252  # Convert to daily

    if market_df is None or "close" not in market_df.columns:
        market_returns = asset_returns
    else:
        market_returns = market_df["close"].pct_change()

    # Align indices
    common_index = asset_returns.index.intersection(market_returns.index)
    asset_returns = asset_returns.loc[common_index]
    market_returns = market_returns.loc[common_index]

    beta = compute_beta(df, market_df, window)
    expected_returns = daily_rf_rate + beta * (market_returns - daily_rf_rate)
    excess_returns = asset_returns - expected_returns

    return excess_returns.rolling(window=window, min_periods=30).mean() * 252  # Annualize


def compute_price_acceleration(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """Compute price acceleration (second derivative of price).

    Args:
        df (pd.DataFrame): DataFrame with price data (requires 'close' column).
        window (int, optional): Window for smoothing acceleration. Defaults to 10.

    Returns:
        pd.Series: Price acceleration values. Positive values indicate
            accelerating upward movement, negative values indicate accelerating downward movement.
    """
    if "close" not in df.columns:
        return pd.Series(index=df.index, dtype=float)

    # First derivative (momentum)
    momentum = df["close"].diff()

    # Second derivative (acceleration)
    acceleration = momentum.diff()

    if window > 1:
        return acceleration.rolling(window=window, min_periods=1).mean()
    return acceleration


def compute_seasonal_strength(df: pd.DataFrame, period: str = "M") -> pd.Series:
    """Compute seasonal strength based on historical patterns.

    Args:
        df (pd.DataFrame): DataFrame with price data (requires 'close' column).
        period (str, optional): Seasonal period ('M' for monthly, 'Q' for quarterly). Defaults to "M".

    Returns:
        pd.Series: Seasonal strength values. Higher values indicate stronger
            seasonal patterns at that point in time.
    """
    if "close" not in df.columns:
        return pd.Series(index=df.index, dtype=float)

    returns = df["close"].pct_change()

    if period == "M":
        seasonal_returns = returns.groupby(returns.index.month).mean()
        current_month = returns.index.month
        seasonal_expectation = pd.Series(
            [seasonal_returns[month] for month in current_month], index=returns.index
        )
    elif period == "Q":
        seasonal_returns = returns.groupby(returns.index.quarter).mean()
        current_quarter = returns.index.quarter
        seasonal_expectation = pd.Series(
            [seasonal_returns[quarter] for quarter in current_quarter], index=returns.index
        )
    else:
        return pd.Series(index=df.index, dtype=float)

    # Calculate deviation from seasonal expectation
    deviation = returns - seasonal_expectation

    # Rolling standard deviation of seasonal deviations
    seasonal_strength = deviation.rolling(window=252, min_periods=30).std()

    return seasonal_strength.abs()


def compute_monthly_returns(df: pd.DataFrame) -> pd.Series:
    """Compute average monthly returns for each calendar month.

    Args:
        df (pd.DataFrame): DataFrame with price data (requires 'close' column).

    Returns:
        pd.Series: Average monthly returns by calendar month (12 values).
            Index represents months 1-12.
    """
    if "close" not in df.columns:
        return pd.Series(dtype=float)

    returns = df["close"].pct_change()
    monthly_returns = returns.groupby(returns.index.month).mean()

    return monthly_returns


# --- LOW PRIORITY METRICS ---


def compute_correlation_matrix(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Compute rolling correlation matrix for multiple assets.

    Args:
        df (pd.DataFrame): DataFrame with multiple asset price columns.
        window (int, optional): Rolling window for correlation calculation. Defaults to 30.

    Returns:
        pd.DataFrame: Rolling correlation matrix. Shape is (n_periods, n_assets * n_assets).
    """
    if df.empty or len(df.columns) < 2:
        return pd.DataFrame()

    returns = df.pct_change()
    correlations = returns.rolling(window=window, min_periods=5).corr()

    return correlations


def compute_autocorrelation(df: pd.DataFrame, lag: int = 1, window: int = 30) -> pd.Series:
    """Compute rolling autocorrelation of price series.

    Args:
        df (pd.DataFrame): DataFrame with price data (requires 'close' column).
        lag (int, optional): Lag periods for autocorrelation. Defaults to 1.
        window (int, optional): Rolling window for calculation. Defaults to 30.

    Returns:
        pd.Series: Autocorrelation values. Values near 1 indicate strong momentum,
            values near -1 indicate strong mean reversion.
    """
    if "close" not in df.columns:
        return pd.Series(index=df.index, dtype=float)

    returns = df["close"].pct_change()
    autocorr = returns.rolling(window=window, min_periods=lag + 5).corr(returns.shift(lag))

    return autocorr


def compute_pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    """Compute traditional pivot points and support/resistance levels.

    Args:
        df (pd.DataFrame): DataFrame with OHLC data.

    Returns:
        pd.DataFrame: DataFrame with pivot point, support, and resistance levels.
            Columns: ['pivot', 's1', 's2', 'r1', 'r2']
    """
    if not all(col in df.columns for col in ["high", "low", "close"]):
        return pd.DataFrame()

    pivot = (df["high"] + df["low"] + df["close"]) / 3

    s1 = 2 * pivot - df["high"]
    s2 = pivot - (df["high"] - df["low"])

    r1 = 2 * pivot - df["low"]
    r2 = pivot + (df["high"] - df["low"])

    return pd.DataFrame({"pivot": pivot, "s1": s1, "s2": s2, "r1": r1, "r2": r2})


def compute_fibonacci_levels(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """Compute Fibonacci retracement levels.

    Args:
        df (pd.DataFrame): DataFrame with price data (requires 'high' and 'low' columns).
        lookback (int, optional): Lookback period for high/low calculation. Defaults to 50.

    Returns:
        pd.DataFrame: DataFrame with Fibonacci levels.
            Columns: ['fib_0', 'fib_0.236', 'fib_0.382', 'fib_0.5', 'fib_0.618', 'fib_1']
    """
    if not all(col in df.columns for col in ["high", "low"]):
        return pd.DataFrame()

    rolling_high = df["high"].rolling(window=lookback, min_periods=1).max()
    rolling_low = df["low"].rolling(window=lookback, min_periods=1).min()

    diff = rolling_high - rolling_low

    return pd.DataFrame(
        {
            "fib_0": rolling_low,
            "fib_0.236": rolling_low + 0.236 * diff,
            "fib_0.382": rolling_low + 0.382 * diff,
            "fib_0.5": rolling_low + 0.5 * diff,
            "fib_0.618": rolling_low + 0.618 * diff,
            "fib_1": rolling_high,
        }
    )


def compute_sharpe_ratio(
    df: pd.DataFrame, risk_free_rate: float = 0.02, window: int = 252
) -> pd.Series:
    """Compute Sharpe ratio (risk-adjusted returns).

    Args:
        df (pd.DataFrame): DataFrame with price data (requires 'close' column).
        risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.02.
        window (int, optional): Rolling window for calculation. Defaults to 252.

    Returns:
        pd.Series: Sharpe ratio values. Higher values indicate better
            risk-adjusted performance.
    """
    if "close" not in df.columns:
        return pd.Series(index=df.index, dtype=float)

    returns = df["close"].pct_change()
    daily_rf_rate = risk_free_rate / 252  # Convert to daily

    excess_returns = returns - daily_rf_rate
    rolling_mean = excess_returns.rolling(window=window, min_periods=30).mean()
    rolling_std = excess_returns.rolling(window=window, min_periods=30).std()

    sharpe = rolling_mean / rolling_std
    return sharpe * np.sqrt(252)  # Annualize


def compute_sortino_ratio(
    df: pd.DataFrame, risk_free_rate: float = 0.02, target_return: float = 0.0, window: int = 252
) -> pd.Series:
    """Compute Sortino ratio (downside risk-adjusted returns).

    Args:
        df (pd.DataFrame): DataFrame with price data (requires 'close' column).
        risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.02.
        target_return (float, optional): Minimum acceptable return. Defaults to 0.0.
        window (int, optional): Rolling window for calculation. Defaults to 252.

    Returns:
        pd.Series: Sortino ratio values. Higher values indicate better
            downside risk-adjusted performance.
    """
    if "close" not in df.columns:
        return pd.Series(index=df.index, dtype=float)

    returns = df["close"].pct_change()
    daily_rf_rate = risk_free_rate / 252
    daily_target = target_return / 252

    excess_returns = returns - daily_rf_rate
    downside_returns = excess_returns.where(excess_returns < daily_target, 0)

    rolling_mean = excess_returns.rolling(window=window, min_periods=30).mean()
    downside_std = (downside_returns**2).rolling(window=window, min_periods=30).mean() ** 0.5

    sortino = rolling_mean / downside_std
    return sortino * np.sqrt(252)  # Annualize


def compute_calmar_ratio(df: pd.DataFrame, window: int = 252) -> pd.Series:
    """Compute Calmar ratio (return vs maximum drawdown).

    Args:
        df (pd.DataFrame): DataFrame with price data (requires 'close' column).
        window (int, optional): Rolling window for calculation. Defaults to 252.

    Returns:
        pd.Series: Calmar ratio values. Higher values indicate better
            return per unit of drawdown risk.
    """
    if "close" not in df.columns:
        return pd.Series(index=df.index, dtype=float)

    returns = df["close"].pct_change()
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.rolling(window=window, min_periods=30).max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max

    max_drawdown = drawdown.rolling(window=window, min_periods=30).min().abs()
    annualized_return = cumulative_returns ** (252 / window) - 1

    return annualized_return / max_drawdown


def compute_win_rate(df: pd.DataFrame, window: int = 252) -> pd.Series:
    """Compute rolling win rate of trades/signals.

    Args:
        df (pd.DataFrame): DataFrame with returns or signal data.
        window (int, optional): Rolling window for calculation. Defaults to 252.

    Returns:
        pd.Series: Win rate values (0-1). Values closer to 1 indicate
            higher percentage of winning periods.
    """
    if df.empty:
        return pd.Series(dtype=float)

    # Assume first column contains returns or binary signals
    data = df.iloc[:, 0]

    if data.dtype in [bool, "int64", "int32"]:
        # Binary signals: count positive signals
        wins = (data > 0).rolling(window=window, min_periods=10).sum()
    else:
        # Returns: count positive returns
        wins = (data > 0).rolling(window=window, min_periods=10).sum()

    total_periods = pd.Series(1, index=data.index).rolling(window=window, min_periods=10).sum()

    return wins / total_periods


def compute_profit_factor(df: pd.DataFrame, window: int = 252) -> pd.Series:
    """Compute profit factor (gross profit / gross loss).

    Args:
        df (pd.DataFrame): DataFrame with returns data.
        window (int, optional): Rolling window for calculation. Defaults to 252.

    Returns:
        pd.Series: Profit factor values. Values > 1 indicate profitable systems,
            higher values indicate better profit/loss ratios.
    """
    if df.empty:
        return pd.Series(dtype=float)

    # Assume first column contains returns
    returns = df.iloc[:, 0]

    gross_profit = returns.where(returns > 0, 0).rolling(window=window, min_periods=10).sum()
    gross_loss = returns.where(returns < 0, 0).abs().rolling(window=window, min_periods=10).sum()

    # Avoid division by zero
    profit_factor = gross_profit / gross_loss.replace(0, np.nan)

    return profit_factor.fillna(0)
