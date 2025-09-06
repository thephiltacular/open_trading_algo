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


def trix(series: pd.Series, window: int = 14) -> pd.Series:
    """1-day Rate-Of-Change (ROC) of a Triple Smooth EMA"""
    ema1 = ema(series, window)
    ema2 = ema(ema1, window)
    ema3 = ema(ema2, window)
    return ema3.pct_change() * 100


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


def natr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Normalized Average True Range"""
    atr_val = atr(high, low, close, window)
    return 100 * atr_val / close


def trange(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range"""
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(
        axis=1
    )
    return tr


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On Balance Volume"""
    direction = np.sign(close.diff()).fillna(0)
    return (volume * direction).cumsum()


def mfi(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14
) -> pd.Series:
    """Money Flow Index"""
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume

    # Positive and negative money flow
    money_flow_direction = np.sign(typical_price.diff()).fillna(0)
    positive_flow = raw_money_flow.where(money_flow_direction > 0, 0)
    negative_flow = raw_money_flow.where(money_flow_direction < 0, 0)

    # Money flow ratio
    positive_mf = positive_flow.rolling(window).sum()
    negative_mf = negative_flow.rolling(window).sum()
    money_flow_ratio = positive_mf / negative_mf.replace(0, np.nan)

    # Money Flow Index
    mfi = 100 - (100 / (1 + money_flow_ratio))
    return mfi.fillna(50)


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


def plus_dm(high: pd.Series, low: pd.Series) -> pd.Series:
    """Plus Directional Movement"""
    high_diff = high.diff()
    low_diff = low.diff()
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    return pd.Series(plus_dm, index=high.index)


def minus_dm(high: pd.Series, low: pd.Series) -> pd.Series:
    """Minus Directional Movement"""
    high_diff = high.diff()
    low_diff = low.diff()
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    return pd.Series(minus_dm, index=high.index)


def plus_di(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Plus Directional Indicator"""
    tr = trange(high, low, close)
    plus_dm_val = plus_dm(high, low)
    return 100 * plus_dm_val.rolling(window).sum() / tr.rolling(window).sum()


def minus_di(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Minus Directional Indicator"""
    tr = trange(high, low, close)
    minus_dm_val = minus_dm(high, low)
    return 100 * minus_dm_val.rolling(window).sum() / tr.rolling(window).sum()


def dx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Directional Movement Index"""
    plus_di_val = plus_di(high, low, close, window)
    minus_di_val = minus_di(high, low, close, window)
    return 100 * abs(plus_di_val - minus_di_val) / (plus_di_val + minus_di_val).replace(0, np.nan)


def ultosc(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    short_period: int = 7,
    medium_period: int = 14,
    long_period: int = 28,
) -> pd.Series:
    """Ultimate Oscillator"""
    # Calculate True Range
    tr = trange(high, low, close)

    # Calculate Buying Pressure (BP)
    bp = close - low.rolling(2).min().shift(1)

    # Calculate averages for different periods
    avg7 = bp.rolling(short_period).sum() / tr.rolling(short_period).sum()
    avg14 = bp.rolling(medium_period).sum() / tr.rolling(medium_period).sum()
    avg28 = bp.rolling(long_period).sum() / tr.rolling(long_period).sum()

    # Ultimate Oscillator
    ultosc = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
    return ultosc


def sar(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    acceleration: float = 0.02,
    max_acceleration: float = 0.2,
) -> pd.Series:
    """Parabolic SAR"""
    sar = pd.Series(index=high.index, dtype=float)

    # Initialize first SAR value
    if len(high) > 0:
        sar.iloc[0] = low.iloc[0]

    # Track trend direction
    trend = 1  # 1 = uptrend, -1 = downtrend
    ep = high.iloc[0] if trend == 1 else low.iloc[0]  # Extreme point
    af = acceleration  # Acceleration factor

    for i in range(1, len(high)):
        if trend == 1:  # Uptrend
            sar.iloc[i] = sar.iloc[i - 1] + af * (ep - sar.iloc[i - 1])
            if high.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(af + acceleration, max_acceleration)
            if sar.iloc[i] >= low.iloc[i]:
                trend = -1
                sar.iloc[i] = ep
                ep = low.iloc[i]
                af = acceleration
        else:  # Downtrend
            sar.iloc[i] = sar.iloc[i - 1] + af * (ep - sar.iloc[i - 1])
            if low.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(af + acceleration, max_acceleration)
            if sar.iloc[i] <= high.iloc[i]:
                trend = 1
                sar.iloc[i] = ep
                ep = high.iloc[i]
                af = acceleration

    return sar


# --- Stochastic Indicators ---


def stoch(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    fastk_period: int = 14,
    slowk_period: int = 3,
    slowd_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator (%K and %D).

    The Stochastic Oscillator compares a closing price to its price range over a given time period.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        fastk_period: Period for %K calculation
        slowk_period: Smoothing period for %K
        slowd_period: Smoothing period for %D

    Returns:
        Tuple of (slowk, slowd) - the %K and %D lines
    """
    # Calculate Fast %K
    lowest_low = low.rolling(fastk_period).min()
    highest_high = high.rolling(fastk_period).max()
    fastk = 100 * (close - lowest_low) / (highest_high - lowest_low)

    # Calculate Slow %K (smoothed Fast %K)
    slowk = fastk.rolling(slowk_period).mean()

    # Calculate Slow %D (smoothed Slow %K)
    slowd = slowk.rolling(slowd_period).mean()

    return slowk, slowd


def stochf(
    high: pd.Series, low: pd.Series, close: pd.Series, fastk_period: int = 14, fastd_period: int = 3
) -> tuple[pd.Series, pd.Series]:
    """Stochastic Fast (%K and %D).

    A faster version of the Stochastic Oscillator with less smoothing.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        fastk_period: Period for %K calculation
        fastd_period: Smoothing period for %D

    Returns:
        Tuple of (fastk, fastd) - the fast %K and %D lines
    """
    # Calculate Fast %K
    lowest_low = low.rolling(fastk_period).min()
    highest_high = high.rolling(fastk_period).max()
    fastk = 100 * (close - lowest_low) / (highest_high - lowest_low)

    # Calculate Fast %D (smoothed Fast %K)
    fastd = fastk.rolling(fastd_period).mean()

    return fastk, fastd


def stochrsi(
    series: pd.Series,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_period: int = 3,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic RSI.

    Applies the Stochastic Oscillator formula to RSI values instead of price.

    Args:
        series: Input price series
        rsi_period: Period for RSI calculation
        stoch_period: Period for Stochastic calculation
        k_period: Smoothing period for %K
        d_period: Smoothing period for %D

    Returns:
        Tuple of (stochrsi_k, stochrsi_d) - the Stochastic RSI %K and %D
    """
    # Calculate RSI first
    rsi_values = rsi(series, rsi_period)

    # Apply Stochastic formula to RSI
    rsi_min = rsi_values.rolling(stoch_period).min()
    rsi_max = rsi_values.rolling(stoch_period).max()

    # Handle case where min == max (no variation in RSI)
    denominator = rsi_max - rsi_min
    denominator = denominator.replace(0, np.nan)  # Avoid division by zero
    stochrsi_k = 100 * (rsi_values - rsi_min) / denominator

    # Fill NaN values with 50 (neutral value)
    stochrsi_k = stochrsi_k.fillna(50)

    # Smooth %K to get %D
    stochrsi_d = stochrsi_k.rolling(k_period).mean()

    return stochrsi_k, stochrsi_d


def aroon(high: pd.Series, low: pd.Series, window: int = 14) -> tuple[pd.Series, pd.Series]:
    """Aroon Indicator: returns (aroon_up, aroon_down)"""
    # Days since highest high
    high_max_idx = high.rolling(window).apply(lambda x: window - np.argmax(x) - 1, raw=True)
    aroon_up = 100 * (window - high_max_idx) / window

    # Days since lowest low
    low_min_idx = low.rolling(window).apply(lambda x: window - np.argmin(x) - 1, raw=True)
    aroon_down = 100 * (window - low_min_idx) / window

    return aroon_up, aroon_down


def aroonosc(high: pd.Series, low: pd.Series, window: int = 14) -> pd.Series:
    """Aroon Oscillator"""
    aroon_up, aroon_down = aroon(high, low, window)
    return aroon_up - aroon_down


# --- Accumulation/Distribution Indicators ---


def ad(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Chaikin Accumulation/Distribution Line.

    A volume-based indicator that measures the cumulative flow of money into and out of a security.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume data

    Returns:
        Accumulation/Distribution Line
    """
    # Calculate Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)

    # Calculate Money Flow Volume
    mfv = mfm * volume

    # Calculate Accumulation/Distribution Line
    ad_line = mfv.cumsum()

    return ad_line


def adosc(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    fast_period: int = 3,
    slow_period: int = 10,
) -> pd.Series:
    """Chaikin Accumulation/Distribution Oscillator.

    The MACD of the Accumulation/Distribution Line.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume data
        fast_period: Fast EMA period
        slow_period: Slow EMA period

    Returns:
        Accumulation/Distribution Oscillator
    """
    # Calculate A/D Line
    ad_line = ad(high, low, close, volume)

    # Calculate fast and slow EMAs of A/D Line
    fast_ema = ema(ad_line, fast_period)
    slow_ema = ema(ad_line, slow_period)

    # Calculate oscillator
    oscillator = fast_ema - slow_ema

    return oscillator


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
    return 1 if (val is not None and not np.isnan(val) and val >= 0) else 0


def abs_trend_is_negative(val: float, has_data: bool) -> int:
    """Return 1 when value exists and is below 0 (absolute down-trend)."""

    if not has_data:
        return 0
    return 1 if (val is not None and val < 0) else 0


def ratio_trend(numerator: float, denominator: float, has_data: bool) -> float:
    """Compute a simple ratio-based trend when description/data is present.

    Returns 0 when missing/invalid.
    """

    if not has_data or denominator in (None, 0) or np.isnan(denominator):
        return 0.0
    if numerator is None or np.isnan(numerator):
        return 0.0
    return float(numerator) / float(denominator)


# ------------------------------- CMF helpers -------------------------------


def cmf_trend(value: float, lower: float, upper: float, has_data: bool) -> int:
    """CMF trend flag is 1 when lower < value < upper and data exists."""

    if not has_data or value is None or np.isnan(value):
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


# --- Hilbert Transform Indicators ---


def hilbert_transform(series: pd.Series, window: int = 20) -> tuple[pd.Series, pd.Series]:
    """Hilbert Transform - returns in-phase (I) and quadrature (Q) components.

    The Hilbert Transform creates an analytic signal by computing the
    convolution of the input signal with the Hilbert kernel.

    Args:
        series: Input price series
        window: Window size for the transform (affects cycle detection)

    Returns:
        Tuple of (in_phase, quadrature) components
    """
    if len(series) < window:
        return pd.Series([0] * len(series), index=series.index), pd.Series(
            [0] * len(series), index=series.index
        )

    # Create Hilbert kernel (simplified approximation)
    # In practice, this would use a more sophisticated kernel
    kernel_size = min(window, len(series))
    kernel = np.zeros(kernel_size)

    # Simple Hilbert kernel approximation
    for i in range(1, kernel_size, 2):
        kernel[i] = 2 / (np.pi * i)

    # Apply convolution for in-phase component
    in_phase = pd.Series(np.convolve(series.values, kernel, mode="same"), index=series.index)

    # Quadrature component (90-degree phase shift)
    quadrature = (
        pd.Series(np.convolve(series.values, kernel, mode="same"), index=series.index)
        .shift(-1)
        .fillna(0)
    )

    return in_phase, quadrature


def hilbert_sine_wave(series: pd.Series, cycle_period: int = 20) -> tuple[pd.Series, pd.Series]:
    """Hilbert Sine Wave - returns sine wave and lead sine wave.

    Uses the Hilbert Transform to reconstruct a sine wave that follows
    the dominant cycle in the price data.

    Args:
        series: Input price series
        cycle_period: Expected cycle period for sine wave reconstruction

    Returns:
        Tuple of (sine_wave, lead_sine_wave)
    """
    if len(series) < cycle_period:
        return pd.Series([0] * len(series), index=series.index), pd.Series(
            [0] * len(series), index=series.index
        )

    # Get Hilbert components
    in_phase, quadrature = hilbert_transform(series, cycle_period)

    # Calculate amplitude and phase
    amplitude = np.sqrt(in_phase**2 + quadrature**2)
    phase = np.arctan2(quadrature, in_phase)

    # Reconstruct sine wave
    sine_wave = amplitude * np.sin(phase)

    # Lead sine wave (phase shifted by 45 degrees)
    lead_sine_wave = amplitude * np.sin(phase + np.pi / 4)

    return sine_wave, lead_sine_wave


def hilbert_cycle_period(series: pd.Series, min_period: int = 10, max_period: int = 50) -> pd.Series:
    """Hilbert Cycle Period - estimates the dominant cycle period.

    Uses the Hilbert Transform to estimate the dominant cycle length
    in the price series.

    Args:
        series: Input price series
        min_period: Minimum cycle period to consider
        max_period: Maximum cycle period to consider

    Returns:
        Series of estimated cycle periods
    """
    if len(series) < max_period:
        return pd.Series([20] * len(series), index=series.index)

    periods = []

    for i in range(len(series)):
        if i < max_period:
            periods.append(20)  # Default period
            continue

        # Calculate cycle period using autocorrelation or similar method
        # Simplified implementation - in practice would use more sophisticated cycle detection
        window = series.iloc[max(0, i - max_period) : i + 1]

        if len(window) < min_period:
            periods.append(20)
            continue

        # Simple period estimation using zero crossings of detrended data
        detrended = window - sma(window, min_period)
        zero_crossings = 0
        prev_sign = np.sign(detrended.iloc[0])

        for val in detrended.iloc[1:]:
            current_sign = np.sign(val)
            if current_sign != prev_sign and current_sign != 0:
                zero_crossings += 1
            prev_sign = current_sign

        # Estimate period from zero crossings
        if zero_crossings > 0:
            estimated_period = len(window) / (zero_crossings / 2)
            period = np.clip(estimated_period, min_period, max_period)
        else:
            period = 20

        periods.append(period)

    return pd.Series(periods, index=series.index)


def hilbert_instantaneous_trendline(
    series: pd.Series, window: int = 20, smoothing: int = 3
) -> pd.Series:
    """Hilbert Instantaneous Trendline - smoothed trend component.

    Extracts the trend component using Hilbert Transform and applies smoothing.

    Args:
        series: Input price series
        window: Window for Hilbert Transform
        smoothing: Smoothing period for the trendline

    Returns:
        Smoothed trendline series
    """
    if len(series) < window:
        return pd.Series([series.iloc[0]] * len(series), index=series.index)

    # Get Hilbert components
    in_phase, quadrature = hilbert_transform(series, window)

    # Calculate amplitude
    amplitude = np.sqrt(in_phase**2 + quadrature**2)

    # Apply smoothing to create trendline
    trendline = sma(amplitude, smoothing)

    return trendline


def hilbert_trend_vs_cycle(series: pd.Series, cycle_period: int = 20) -> tuple[pd.Series, pd.Series]:
    """Hilbert Trend vs Cycle - separates trend from cycle components.

    Uses Hilbert Transform to decompose the price series into
    trend and cycle components.

    Args:
        series: Input price series
        cycle_period: Cycle period for decomposition

    Returns:
        Tuple of (trend_component, cycle_component)
    """
    if len(series) < cycle_period:
        return pd.Series([series.iloc[0]] * len(series), index=series.index), pd.Series(
            [0] * len(series), index=series.index
        )

    # Get Hilbert components
    in_phase, quadrature = hilbert_transform(series, cycle_period)

    # Calculate amplitude (trend proxy)
    trend_component = np.sqrt(in_phase**2 + quadrature**2)

    # Calculate phase for cycle component
    phase = np.arctan2(quadrature, in_phase)

    # Cycle component as sine wave reconstruction
    cycle_component = trend_component * np.sin(phase)

    return trend_component, cycle_component


def hilbert_dcphase(series: pd.Series, window: int = 20) -> pd.Series:
    """Hilbert Transform - Dominant Cycle Phase"""
    in_phase, quadrature = hilbert_transform(series, window)
    phase = np.arctan2(quadrature, in_phase)
    return phase


def hilbert_phasor(series: pd.Series, window: int = 20) -> tuple[pd.Series, pd.Series]:
    """Hilbert Transform - Phasor Components (In-phase and Quadrature)"""
    return hilbert_transform(series, window)


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
    "hilbert_transform",
    "hilbert_sine_wave",
    "hilbert_cycle_period",
    "hilbert_instantaneous_trendline",
    "hilbert_trend_vs_cycle",
    "hilbert_dcphase",
    "hilbert_phasor",
    "stoch",
    "stochf",
    "stochrsi",
    "ad",
    "adosc",
    "obv",
    "atr",
    "natr",
    "trange",
    "mfi",
    "plus_dm",
    "minus_dm",
    "plus_di",
    "minus_di",
    "dx",
    "aroon",
    "aroonosc",
    "trix",
    "ultosc",
    "sar",
]
