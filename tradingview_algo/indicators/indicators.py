"""Indicator primitives extracted from `data_processing.py`.

This module groups the many small lambda-like helpers into named, documented
functions. Each function is side-effect free and easy to test.

Note: The original file contains many placeholders. Here we provide coherent
interfaces and minimal, safe behavior so the functions are usable immediately.
Where the original intent is ambiguous, we document assumptions clearly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# --- Technical Indicator Calculations ---
import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(window=window, min_periods=1).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=window, adjust=False).mean()


def wma(series: pd.Series, window: int) -> pd.Series:
    """Weighted Moving Average"""
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def dema(series: pd.Series, window: int) -> pd.Series:
    """Double Exponential Moving Average"""
    ema1 = ema(series, window)
    ema2 = ema(ema1, window)
    return 2 * ema1 - ema2


def tema(series: pd.Series, window: int) -> pd.Series:
    """Triple Exponential Moving Average"""
    ema1 = ema(series, window)
    ema2 = ema(ema1, window)
    ema3 = ema(ema2, window)
    return 3 * (ema1 - ema2) + ema3


def trima(series: pd.Series, window: int) -> pd.Series:
    """Triangular Moving Average"""
    return series.rolling(window, min_periods=1).mean().rolling(window, min_periods=1).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD, MACD Signal, MACD Histogram"""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)


def willr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Williams %R"""
    highest_high = high.rolling(window).max()
    lowest_low = low.rolling(window).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low)


def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """Commodity Channel Index"""
    tp = (high + low + close) / 3
    ma = tp.rolling(window).mean()
    md = tp.rolling(window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - ma) / (0.015 * md)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range"""
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(
        axis=1
    )
    return tr.rolling(window).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On Balance Volume"""
    direction = np.sign(close.diff()).fillna(0)
    return (volume * direction).cumsum()


def roc(series: pd.Series, window: int = 12) -> pd.Series:
    """Rate of Change"""
    return series.pct_change(periods=window) * 100


def mom(series: pd.Series, window: int = 10) -> pd.Series:
    """Momentum"""
    return series.diff(window)


def bbands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    """Bollinger Bands: returns (middle, upper, lower)"""
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return ma, upper, lower


def midpoint(series: pd.Series, window: int = 14) -> pd.Series:
    """MidPoint over period"""
    return (series.rolling(window).max() + series.rolling(window).min()) / 2


def midprice(high: pd.Series, low: pd.Series, window: int = 14) -> pd.Series:
    """Midpoint Price over period"""
    return (high.rolling(window).max() + low.rolling(window).min()) / 2


# ----------------------------- Generic helpers -----------------------------


def safe_subtract(a: float, b: float) -> float:
    """Subtract b from a with simple guards.

    - Treats None as 0.0.
    - Returns a - b.
    """

    a = 0.0 if a is None else a
    b = 0.0 if b is None else b
    return a - b


def add_many(*values: Optional[float]) -> float:
    """Sum values, ignoring None.

    Parameters
    - values: numbers (or None)

    Returns
    - float sum
    """

    return float(sum(v for v in values if v is not None))


# ------------------------------ Trend helpers ------------------------------


def trend_positive(val: float, has_data: bool) -> int:
    """Return 1 if val >= 0 and data exists, else 0.

    Mirrors the behavior of the unspecified `trend_lambda`.
    """

    if not has_data:
        return 0
    return 1 if (val is not None and val >= 0) else 0


def abs_trend_is_negative(val: float, has_data: bool) -> int:
    """Return 1 when value exists and is below 0 (absolute down-trend)."""

    if not has_data:
        return 0
    return 1 if (val is not None and val < 0) else 0


def ratio_trend(numerator: float, denominator: float, has_data: bool) -> float:
    """Compute a simple ratio-based trend when description/data is present.

    Returns 0 when missing/invalid.
    """

    if not has_data or denominator in (None, 0):
        return 0.0
    if numerator is None:
        return 0.0
    return float(numerator) / float(denominator)


# ------------------------------- CMF helpers -------------------------------


def cmf_trend(value: float, lower: float, upper: float, has_data: bool) -> int:
    """CMF trend flag is 1 when lower < value < upper and data exists."""

    if not has_data or value is None:
        return 0
    return 1 if (lower < value < upper) else 0


# ------------------------------- MACD helpers ------------------------------


def macd_cross_up(macd: float, signal: float, has_data: bool) -> int:
    """Return 1 when MACD crosses upward (macd > signal), else 0.

    The original code has two MACD_lambda placeholders. Here we use a simple,
    common interpretation. Callers can refine as needed.
    """

    if not has_data or macd is None or signal is None:
        return 0
    return 1 if macd > signal and signal != 0 else 0


def macd_tu(change_pct: float, macd_ls_su: float) -> int:
    """MACD TU flag based on change% vs MACD L>S SU strength.

    Returns 1 if change_pct > 0 and macd_ls_su < 0 and |macd_ls_su| < change_pct.
    """

    if change_pct is None or macd_ls_su is None:
        return 0
    return 1 if (change_pct > 0 and macd_ls_su < 0 and abs(macd_ls_su) < change_pct) else 0


# ------------------------------- EMA helpers -------------------------------


def ema_gap(left: float, right: float) -> float:
    """Return negative difference (right - left) with sign flipped as in v8."""

    if left is None or right is None:
        return 0.0
    return -1.0 * (left - right)


def ema_avg(a: float, b: float, c: float, has_data: bool) -> float:
    """Average of three EMA values when data exists, else 0."""

    if not has_data or a in (None, 0) or b in (None, 0) or c in (None, 0):
        return 0.0
    return (float(a) + float(b) + float(c)) / 3.0


# ---------------------------- Volatility helpers ---------------------------


def volatility_ratio(numerator: float, denominator: float, has_data: bool) -> float:
    """Compute (n/d) - 1.0 when data exists, else 0."""

    if not has_data or denominator in (None, 0) or numerator is None:
        return 0.0
    return float(numerator) / float(denominator) - 1.0


def sq_positive(col_value: float, has_data: bool) -> int:
    """Return 1 when volatility SQ column value is positive and data exists."""

    if not has_data or col_value is None:
        return 0
    return 1 if col_value > 0 else 0


# --------------------------- Percent rank helpers --------------------------


def percent_rank(value: float, has_data: bool) -> float:
    """A placeholder percent-rank contribution (0 or value).

    In the original code, percent-rank is computed relative to peers. That
    requires group-level context which lives in the pipeline. At the primitive
    level, just forward the value when available; the pipeline will normalize.
    """

    if not has_data or value is None:
        return 0.0
    return float(value)


# ------------------------------ RSI/Williams -------------------------------


def williams_flag(value: float, lower: float, upper: float, has_data: bool) -> int:
    """Return 1 when lower < value < upper and data exists."""

    if not has_data or value is None:
        return 0
    return 1 if (lower < value < upper) else 0


def rsi_band_flag(value: float, idx: float, upper: float, lower: float, has_data: bool) -> int:
    """Generic RSI band membership to support OB/OS/MD/MU variants."""

    if not has_data or value is None:
        return 0
    if upper is None or lower is None:
        return 0
    return 1 if lower <= value <= upper else 0


__all__ = [
    "safe_subtract",
    "add_many",
    "trend_positive",
    "abs_trend_is_negative",
    "ratio_trend",
    "cmf_trend",
    "macd_cross_up",
    "macd_tu",
    "ema_gap",
    "ema_avg",
    "volatility_ratio",
    "sq_positive",
    "percent_rank",
    "williams_flag",
    "rsi_band_flag",
]
