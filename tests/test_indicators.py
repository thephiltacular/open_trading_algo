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
    assert result.iloc[9] > sample_prices.iloc[9]  # EMA smooths upward/downward


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


def test_cmf_trend_missing_data():
    assert cmf_trend(np.nan, 1.0, 2.0, True) == 0  # NaN value
