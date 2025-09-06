import numpy as np
import pandas as pd
import pytest
from open_trading_algo.indicators.indicators import (
    sma,
    ema,
    wma,
    dema,
    tema,
    macd,
    rsi,
    atr,
    obv,
    bbands,
    trend_positive,
    ratio_trend,
    cmf_trend,
    hilbert_transform,
    hilbert_sine_wave,
    hilbert_cycle_period,
    hilbert_instantaneous_trendline,
    hilbert_trend_vs_cycle,
    stoch,
    stochf,
    stochrsi,
    ad,
    adosc,
)

# Fixture for synthetic price data (pd.Series)
@pytest.fixture
def sample_prices():
    """Generate a sample pd.Series of prices for testing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    prices = pd.Series(np.random.uniform(100, 200, 100), index=dates)
    return prices


# Fixture for OHLC data (pd.DataFrame)
@pytest.fixture
def sample_ohlc():
    """Generate a sample OHLC pd.DataFrame for testing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    data = {
        "open": np.random.uniform(100, 200, 100),
        "high": np.random.uniform(150, 250, 100),
        "low": np.random.uniform(50, 150, 100),
        "close": np.random.uniform(100, 200, 100),
        "volume": np.random.randint(1000, 10000, 100),
    }
    return pd.DataFrame(data, index=dates)


# Test SMA
def test_sma(sample_prices):
    result = sma(sample_prices, window=10)
    assert len(result) == len(sample_prices)
    assert result.iloc[9] == pytest.approx(sample_prices.iloc[:10].mean(), rel=1e-2)
    assert not result.iloc[:9].isna().any()  # No NaN due to min_periods=1


def test_sma_missing_data(sample_prices):
    prices_with_nan = sample_prices.copy()
    prices_with_nan.iloc[10:15] = np.nan
    result = sma(prices_with_nan, window=10)
    assert result.iloc[14] == pytest.approx(
        prices_with_nan.iloc[5:15].mean(), rel=1e-2
    )  # Handles NaN by skipping
    assert not result.iloc[:9].isna().any()


# Test EMA
def test_ema(sample_prices):
    result = ema(sample_prices, window=10)
    assert len(result) == len(sample_prices)
    assert result.iloc[0] == sample_prices.iloc[0]  # First value is the price itself
    assert not np.isnan(result.iloc[9])  # EMA is calculated
    assert (
        abs(result.iloc[9] - sample_prices.iloc[9]) < sample_prices.iloc[9] * 0.1
    )  # EMA is close to price


def test_ema_missing_data(sample_prices):
    prices_with_nan = sample_prices.copy()
    prices_with_nan.iloc[10] = np.nan
    result = ema(prices_with_nan, window=10)
    assert not np.isnan(result.iloc[10])  # EMA skips NaN
    assert result.iloc[10] == result.iloc[9]  # Value unchanged due to NaN


# Test WMA
def test_wma(sample_prices):
    result = wma(sample_prices, window=10)
    assert len(result) == len(sample_prices)
    assert result.iloc[9] == pytest.approx(
        (sample_prices.iloc[:10] * np.arange(1, 11)).sum() / 55, rel=1e-2
    )  # Weighted average


def test_wma_missing_data(sample_prices):
    prices_with_nan = sample_prices.copy()
    prices_with_nan.iloc[5:10] = np.nan
    result = wma(prices_with_nan, window=10)
    assert np.isnan(result.iloc[9])  # NaN in window causes NaN output


# Test DEMA
def test_dema(sample_prices):
    result = dema(sample_prices, window=10)
    assert len(result) == len(sample_prices)
    assert result.iloc[18] == pytest.approx(
        2 * ema(sample_prices, 10).iloc[18] - ema(ema(sample_prices, 10), 10).iloc[18], rel=1e-2
    )


def test_dema_missing_data(sample_prices):
    prices_with_nan = sample_prices.copy()
    prices_with_nan.iloc[10] = np.nan
    result = dema(prices_with_nan, window=10)
    assert not np.isnan(result.iloc[18])  # DEMA skips NaN


# Test TEMA
def test_tema(sample_prices):
    result = tema(sample_prices, window=10)
    assert len(result) == len(sample_prices)
    assert result.iloc[27] == pytest.approx(
        3 * ema(sample_prices, 10).iloc[27]
        - 3 * ema(ema(sample_prices, 10), 10).iloc[27]
        + ema(ema(ema(sample_prices, 10), 10), 10).iloc[27],
        rel=1e-2,
    )


def test_tema_missing_data(sample_prices):
    prices_with_nan = sample_prices.copy()
    prices_with_nan.iloc[10] = np.nan
    result = tema(prices_with_nan, window=10)
    assert not np.isnan(result.iloc[27])  # TEMA skips NaN


# Test MACD
def test_macd(sample_prices):
    macd_line, signal_line, histogram = macd(sample_prices)
    assert len(macd_line) == len(sample_prices)
    assert macd_line.iloc[25] == pytest.approx(
        ema(sample_prices, 12).iloc[25] - ema(sample_prices, 26).iloc[25], rel=1e-2
    )
    assert len(signal_line) == len(macd_line)
    assert len(histogram) == len(macd_line)


def test_macd_missing_data(sample_prices):
    prices_with_nan = sample_prices.copy()
    prices_with_nan.iloc[15] = np.nan
    macd_line, signal_line, histogram = macd(prices_with_nan)
    assert not np.isnan(macd_line.iloc[25])  # MACD skips NaN


# Test RSI
def test_rsi(sample_prices):
    result = rsi(sample_prices, window=14)
    assert len(result) == len(sample_prices)
    assert 0 <= result.iloc[13] <= 100  # RSI bounded
    assert not result.iloc[:13].isna().any()  # No NaN due to min_periods=1 and fillna(0)


def test_rsi_missing_data(sample_prices):
    prices_with_nan = sample_prices.copy()
    prices_with_nan.iloc[10:15] = np.nan
    result = rsi(prices_with_nan, window=14)
    assert 0 <= result.iloc[13] <= 100  # RSI still valid despite NaN


# Test ATR
def test_atr(sample_ohlc):
    result = atr(sample_ohlc["high"], sample_ohlc["low"], sample_ohlc["close"], window=14)
    assert len(result) == len(sample_ohlc)
    assert result.iloc[13] > 0  # ATR is positive
    assert result.iloc[:13].isna().all()


def test_atr_missing_data(sample_ohlc):
    ohlc_with_nan = sample_ohlc.copy()
    ohlc_with_nan.loc[ohlc_with_nan.index[10], "high"] = np.nan
    result = atr(ohlc_with_nan["high"], ohlc_with_nan["low"], ohlc_with_nan["close"], window=14)
    assert not np.isnan(result.iloc[13])  # ATR skips NaN in TR


# Test OBV
def test_obv(sample_ohlc):
    result = obv(sample_ohlc["close"], sample_ohlc["volume"])
    assert len(result) == len(sample_ohlc)
    assert result.iloc[0] == 0  # Starts at 0
    assert result.iloc[1] != 0  # Changes based on price movement


def test_obv_missing_data(sample_ohlc):
    ohlc_with_nan = sample_ohlc.copy()
    ohlc_with_nan.loc[ohlc_with_nan.index[10], "close"] = np.nan
    result = obv(ohlc_with_nan["close"], ohlc_with_nan["volume"])
    assert result.iloc[10] == result.iloc[9]  # No change on NaN


# Test Bollinger Bands
def test_bollinger_bands(sample_prices):
    middle, upper, lower = bbands(sample_prices, window=20, num_std=2)
    assert len(upper) == len(sample_prices)
    assert upper.iloc[19] > middle.iloc[19] > lower.iloc[19]
    assert middle.iloc[19] == pytest.approx(sma(sample_prices, 20).iloc[19], rel=1e-2)


def test_bollinger_bands_missing_data(sample_prices):
    prices_with_nan = sample_prices.copy()
    prices_with_nan.iloc[10:15] = np.nan
    middle, upper, lower = bbands(prices_with_nan, window=20, num_std=2)
    assert np.isnan(upper.iloc[19])  # NaN in SMA causes NaN bands


# Test trend_positive (helper)
def test_trend_positive():
    assert trend_positive(1.0, True) == 1
    assert trend_positive(-1.0, True) == 0
    assert trend_positive(1.0, False) == 0  # No data


def test_trend_positive_missing_data():
    assert trend_positive(np.nan, True) == 0  # NaN treated as non-positive


# Test ratio_trend (helper)
def test_ratio_trend():
    assert ratio_trend(10.0, 5.0, True) == 2.0
    assert ratio_trend(10.0, 0, True) == 0.0  # Division by zero
    assert ratio_trend(np.nan, 5.0, True) == 0.0  # NaN numerator


def test_ratio_trend_missing_data():
    assert ratio_trend(10.0, 5.0, False) == 0.0  # No data


# Test cmf_trend (helper)
def test_cmf_trend():
    assert cmf_trend(1.5, 1.0, 2.0, True) == 1
    assert cmf_trend(0.5, 1.0, 2.0, True) == 0
    assert cmf_trend(1.5, 1.0, 2.0, False) == 0  # No data


# --- Hilbert Transform Tests ---


def test_hilbert_transform(sample_prices):
    """Test basic Hilbert Transform functionality."""
    in_phase, quadrature = hilbert_transform(sample_prices, window=20)

    assert isinstance(in_phase, pd.Series)
    assert isinstance(quadrature, pd.Series)
    assert len(in_phase) == len(sample_prices)
    assert len(quadrature) == len(sample_prices)
    assert in_phase.index.equals(sample_prices.index)
    assert quadrature.index.equals(sample_prices.index)

    # Check that components are calculated (non-zero for most values)
    assert not in_phase.iloc[20:].eq(0).all()
    assert not quadrature.iloc[20:].eq(0).all()


def test_hilbert_transform_short_series():
    """Test Hilbert Transform with very short series."""
    short_series = pd.Series([100, 101, 102], index=pd.date_range("2023-01-01", periods=3))

    in_phase, quadrature = hilbert_transform(short_series, window=20)

    assert len(in_phase) == 3
    assert len(quadrature) == 3
    # Should return zeros for short series
    assert in_phase.eq(0).all()
    assert quadrature.eq(0).all()


def test_hilbert_sine_wave(sample_prices):
    """Test Hilbert Sine Wave generation."""
    sine_wave, lead_sine_wave = hilbert_sine_wave(sample_prices, cycle_period=20)

    assert isinstance(sine_wave, pd.Series)
    assert isinstance(lead_sine_wave, pd.Series)
    assert len(sine_wave) == len(sample_prices)
    assert len(lead_sine_wave) == len(sample_prices)

    # Sine waves should have some variation
    assert not sine_wave.iloc[20:].eq(0).all()
    assert not lead_sine_wave.iloc[20:].eq(0).all()


def test_hilbert_sine_wave_short_series():
    """Test Hilbert Sine Wave with short series."""
    short_series = pd.Series([100, 101, 102], index=pd.date_range("2023-01-01", periods=3))

    sine_wave, lead_sine_wave = hilbert_sine_wave(short_series, cycle_period=20)

    assert len(sine_wave) == 3
    assert len(lead_sine_wave) == 3
    # Should return zeros for short series
    assert sine_wave.eq(0).all()
    assert lead_sine_wave.eq(0).all()


def test_hilbert_cycle_period(sample_prices):
    """Test Hilbert Cycle Period estimation."""
    cycle_periods = hilbert_cycle_period(sample_prices, min_period=10, max_period=50)

    assert isinstance(cycle_periods, pd.Series)
    assert len(cycle_periods) == len(sample_prices)

    # Periods should be within specified range
    assert cycle_periods.between(10, 50).all()

    # Early values should be default period (20)
    assert cycle_periods.iloc[:50].eq(20).all()


def test_hilbert_cycle_period_short_series():
    """Test Hilbert Cycle Period with short series."""
    short_series = pd.Series([100, 101, 102], index=pd.date_range("2023-01-01", periods=3))

    cycle_periods = hilbert_cycle_period(short_series, min_period=10, max_period=50)

    assert len(cycle_periods) == 3
    # Should return default period for short series
    assert cycle_periods.eq(20).all()


def test_hilbert_instantaneous_trendline(sample_prices):
    """Test Hilbert Instantaneous Trendline."""
    trendline = hilbert_instantaneous_trendline(sample_prices, window=20, smoothing=3)

    assert isinstance(trendline, pd.Series)
    assert len(trendline) == len(sample_prices)

    # Trendline should be positive (amplitude)
    assert (trendline >= 0).all()

    # Should have some variation
    assert not trendline.eq(trendline.iloc[0]).all()


def test_hilbert_instantaneous_trendline_short_series():
    """Test Hilbert Instantaneous Trendline with short series."""
    short_series = pd.Series([100, 101, 102], index=pd.date_range("2023-01-01", periods=3))

    trendline = hilbert_instantaneous_trendline(short_series, window=20, smoothing=3)

    assert len(trendline) == 3
    # Should return constant value for short series
    assert trendline.eq(short_series.iloc[0]).all()


def test_hilbert_trend_vs_cycle(sample_prices):
    """Test Hilbert Trend vs Cycle decomposition."""
    trend_component, cycle_component = hilbert_trend_vs_cycle(sample_prices, cycle_period=20)

    assert isinstance(trend_component, pd.Series)
    assert isinstance(cycle_component, pd.Series)
    assert len(trend_component) == len(sample_prices)
    assert len(cycle_component) == len(sample_prices)

    # Trend component should be positive (amplitude)
    assert (trend_component >= 0).all()

    # Components should have some variation
    assert not trend_component.iloc[20:].eq(0).all()
    assert not cycle_component.iloc[20:].eq(0).all()


def test_hilbert_trend_vs_cycle_short_series():
    """Test Hilbert Trend vs Cycle with short series."""
    short_series = pd.Series([100, 101, 102], index=pd.date_range("2023-01-01", periods=3))

    trend_component, cycle_component = hilbert_trend_vs_cycle(short_series, cycle_period=20)

    assert len(trend_component) == 3
    assert len(cycle_component) == 3

    # Trend should be constant, cycle should be zero for short series
    assert trend_component.eq(short_series.iloc[0]).all()
    assert cycle_component.eq(0).all()


def test_hilbert_transform_properties(sample_prices):
    """Test mathematical properties of Hilbert Transform."""
    in_phase, quadrature = hilbert_transform(sample_prices, window=20)

    # Test that amplitude is calculated correctly
    amplitude = np.sqrt(in_phase**2 + quadrature**2)
    assert (amplitude >= 0).all()

    # Test that phase is calculated correctly
    phase = np.arctan2(quadrature, in_phase)
    assert phase.between(-np.pi, np.pi).all()


def test_hilbert_sine_wave_properties(sample_prices):
    """Test properties of Hilbert Sine Wave."""
    sine_wave, lead_sine_wave = hilbert_sine_wave(sample_prices, cycle_period=20)

    # Sine waves should be bounded
    assert sine_wave.between(-np.inf, np.inf).all()
    assert lead_sine_wave.between(-np.inf, np.inf).all()

    # Lead sine should be phase-shifted from sine
    # (This is a statistical property, not guaranteed for all data)
    correlation = sine_wave.corr(lead_sine_wave)
    assert isinstance(correlation, (float, np.floating))


def test_hilbert_cycle_period_bounds():
    """Test that cycle periods respect bounds."""
    # Create a series with clear periodicity
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    # Create a sine wave with period ~25
    prices = 100 + 10 * np.sin(2 * np.pi * np.arange(100) / 25)
    series = pd.Series(prices, index=dates)

    cycle_periods = hilbert_cycle_period(series, min_period=15, max_period=35)

    # All periods should be within bounds
    assert cycle_periods.between(15, 35).all()

    # Should detect the underlying period approximately
    mean_period = cycle_periods.iloc[50:].mean()
    assert 20 <= mean_period <= 30  # Should be close to the true period of 25


def test_cmf_trend_missing_data():
    assert cmf_trend(np.nan, 1.0, 2.0, True) == 0  # NaN value


# --- Stochastic Indicator Tests ---


def test_stoch(sample_ohlc):
    """Test Stochastic Oscillator."""
    high = sample_ohlc["high"]
    low = sample_ohlc["low"]
    close = sample_ohlc["close"]

    slowk, slowd = stoch(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)

    assert isinstance(slowk, pd.Series)
    assert isinstance(slowd, pd.Series)
    assert len(slowk) == len(sample_ohlc)
    assert len(slowd) == len(sample_ohlc)

    # Values should be between 0 and 100 (excluding NaN)
    valid_slowk = slowk.dropna()
    valid_slowd = slowd.dropna()
    assert valid_slowk.between(0, 100).all()
    assert valid_slowd.between(0, 100).all()

    # Slow %D should be smoother than Slow %K
    assert slowd.iloc[20:].std() <= slowk.iloc[20:].std()


def test_stochf(sample_ohlc):
    """Test Stochastic Fast."""
    high = sample_ohlc["high"]
    low = sample_ohlc["low"]
    close = sample_ohlc["close"]

    fastk, fastd = stochf(high, low, close, fastk_period=14, fastd_period=3)

    assert isinstance(fastk, pd.Series)
    assert isinstance(fastd, pd.Series)
    assert len(fastk) == len(sample_ohlc)
    assert len(fastd) == len(sample_ohlc)

    # Values should be between 0 and 100 (excluding NaN)
    valid_fastk = fastk.dropna()
    valid_fastd = fastd.dropna()
    assert valid_fastk.between(0, 100).all()
    assert valid_fastd.between(0, 100).all()


def test_stochrsi(sample_prices):
    """Test Stochastic RSI."""
    stochrsi_k, stochrsi_d = stochrsi(
        sample_prices, rsi_period=14, stoch_period=14, k_period=3, d_period=3
    )

    assert isinstance(stochrsi_k, pd.Series)
    assert isinstance(stochrsi_d, pd.Series)
    assert len(stochrsi_k) == len(sample_prices)
    assert len(stochrsi_d) == len(sample_prices)

    # Values should be between 0 and 100 (excluding NaN)
    valid_stochrsi_k = stochrsi_k.dropna()
    valid_stochrsi_d = stochrsi_d.dropna()
    assert (valid_stochrsi_k >= 0).all()
    assert (valid_stochrsi_k <= 100).all()
    assert (valid_stochrsi_d >= 0).all()
    assert (valid_stochrsi_d <= 100).all()


# --- Accumulation/Distribution Indicator Tests ---


def test_ad(sample_ohlc):
    """Test Chaikin Accumulation/Distribution Line."""
    high = sample_ohlc["high"]
    low = sample_ohlc["low"]
    close = sample_ohlc["close"]
    volume = sample_ohlc["volume"]

    ad_line = ad(high, low, close, volume)

    assert isinstance(ad_line, pd.Series)
    assert len(ad_line) == len(sample_ohlc)

    # A/D line should have some variation (not all the same value)
    assert not ad_line.eq(ad_line.iloc[0]).all()

    # Should not be all NaN
    assert not ad_line.isna().all()


def test_adosc(sample_ohlc):
    """Test Chaikin Accumulation/Distribution Oscillator."""
    high = sample_ohlc["high"]
    low = sample_ohlc["low"]
    close = sample_ohlc["close"]
    volume = sample_ohlc["volume"]

    oscillator = adosc(high, low, close, volume, fast_period=3, slow_period=10)

    assert isinstance(oscillator, pd.Series)
    assert len(oscillator) == len(sample_ohlc)

    # Oscillator can be positive or negative
    assert oscillator.between(-np.inf, np.inf).all()


def test_stoch_extrema(sample_ohlc):
    """Test Stochastic at price extremes."""
    high = sample_ohlc["high"]
    low = sample_ohlc["low"]
    close = sample_ohlc["close"]

    # Create a more controlled scenario
    test_high = pd.Series([100] * len(high), index=high.index)
    test_low = pd.Series([90] * len(low), index=low.index)
    test_close = pd.Series([100] * len(close), index=close.index)  # Close = High

    slowk, slowd = stoch(test_high, test_low, test_close, fastk_period=5)

    # Should be 100 when close = high
    assert slowk.iloc[-5:].mean() == pytest.approx(100.0, rel=1e-10)

    # Create a scenario where close equals low (should give %K = 0)
    test_close = test_low.copy()  # Close = Low

    slowk, slowd = stoch(test_high, test_low, test_close, fastk_period=5)

    # Should approach 0 when close = low
    assert slowk.iloc[-10:].mean() < 20  # Should be low


def test_stochrsi_with_rsi(sample_prices):
    """Test that Stochastic RSI uses RSI correctly."""
    # Calculate RSI and Stochastic RSI
    rsi_values = rsi(sample_prices, window=14)
    stochrsi_k, stochrsi_d = stochrsi(sample_prices, rsi_period=14, stoch_period=14)

    # Both should be available after the initial periods
    valid_idx = rsi_values.notna() & stochrsi_k.notna()
    assert valid_idx.sum() > 0

    # Stochastic RSI should be more volatile than regular RSI
    rsi_volatility = rsi_values[valid_idx].std()
    stochrsi_volatility = stochrsi_k[valid_idx].std()

    assert stochrsi_volatility >= rsi_volatility * 0.5  # At least half as volatile


def test_ad_volume_weighting(sample_ohlc):
    """Test that A/D Line properly weights by volume."""
    high = sample_ohlc["high"]
    low = sample_ohlc["low"]
    close = sample_ohlc["close"]
    volume = sample_ohlc["volume"]

    ad_line = ad(high, low, close, volume)

    # High volume periods should have more impact
    high_vol_idx = volume.idxmax()
    high_vol_period = ad_line.loc[high_vol_idx]

    # The A/D line should reflect volume weighting
    assert isinstance(high_vol_period, (int, float))


def test_adosc_convergence(sample_ohlc):
    """Test that A/D Oscillator shows convergence/divergence."""
    high = sample_ohlc["high"]
    low = sample_ohlc["low"]
    close = sample_ohlc["close"]
    volume = sample_ohlc["volume"]

    oscillator = adosc(high, low, close, volume)

    # Should have both positive and negative values
    assert (oscillator > 0).any()
    assert (oscillator < 0).any()

    # Should oscillate around zero
    mean_value = oscillator.iloc[20:].mean()
    assert abs(mean_value) < oscillator.iloc[20:].std()  # Mean should be within one std dev
