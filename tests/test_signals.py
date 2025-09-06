import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from open_trading_algo.signal_optimizer import SignalOptimizer
from open_trading_algo.indicators.indicators import sma, ema, rsi, macd
from tests.test_data_enrichment import generate_comprehensive_test_data


@pytest.fixture
def sample_ohlc_data():
    """Generate sample OHLC data for testing."""
    return generate_comprehensive_test_data(100)


@pytest.fixture
def sample_multi_ticker_data():
    """Generate sample data for multiple tickers."""
    tickers = ["AAPL", "GOOGL", "MSFT"]
    data = {}
    for ticker in tickers:
        data[ticker] = generate_comprehensive_test_data(50)
    return data


@pytest.fixture
def sample_indicators():
    """Sample indicator functions for testing."""
    return {
        "sma_20": lambda df: sma(df["close"], window=20),
        "ema_12": lambda df: ema(df["close"], window=12),
        "rsi_14": lambda df: rsi(df["close"], window=14),
    }


@pytest.fixture
def sample_signal_generators():
    """Sample signal generator functions for testing."""

    def sma_crossover_signal(df, indicators):
        """Generate SMA crossover signals."""
        sma_short = sma(df["close"], window=10)
        sma_long = sma(df["close"], window=20)
        return pd.Series(np.where(sma_short > sma_long, 1, -1), index=df.index)

    def rsi_oversold_signal(df, indicators):
        """Generate RSI oversold signals."""
        rsi_series = rsi(df["close"], window=14)
        return pd.Series(np.where(rsi_series < 30, 1, 0), index=df.index)

    def macd_signal(df, indicators):
        """Generate MACD signals."""
        macd_line, signal_line, histogram = macd(df["close"])
        return pd.Series(np.where(macd_line > signal_line, 1, -1), index=df.index)

    return {
        "sma_crossover": sma_crossover_signal,
        "rsi_oversold": rsi_oversold_signal,
        "macd_crossover": macd_signal,
    }


class TestSignalOptimizer:
    """Test SignalOptimizer class functionality."""

    def test_signal_optimizer_initialization(
        self, sample_multi_ticker_data, sample_indicators, sample_signal_generators
    ):
        """Test SignalOptimizer initialization."""
        optimizer = SignalOptimizer(
            data=sample_multi_ticker_data,
            indicators=sample_indicators,
            signal_generators=sample_signal_generators,
        )

        assert optimizer.data == sample_multi_ticker_data
        assert optimizer.indicators == sample_indicators
        assert optimizer.signal_generators == sample_signal_generators
        assert optimizer.results == {}

    def test_compute_indicators(
        self, sample_multi_ticker_data, sample_indicators, sample_signal_generators
    ):
        """Test indicator computation for all tickers."""
        optimizer = SignalOptimizer(
            data=sample_multi_ticker_data,
            indicators=sample_indicators,
            signal_generators=sample_signal_generators,
        )

        optimizer.compute_indicators()

        assert hasattr(optimizer, "indicator_results")
        assert len(optimizer.indicator_results) == len(sample_multi_ticker_data)

        for ticker in sample_multi_ticker_data.keys():
            assert ticker in optimizer.indicator_results
            assert len(optimizer.indicator_results[ticker]) == len(sample_indicators)

            for indicator_name in sample_indicators.keys():
                assert indicator_name in optimizer.indicator_results[ticker]
                result = optimizer.indicator_results[ticker][indicator_name]
                assert isinstance(result, pd.Series)
                assert len(result) == len(sample_multi_ticker_data[ticker])

    def test_generate_signals(
        self, sample_multi_ticker_data, sample_indicators, sample_signal_generators
    ):
        """Test signal generation for all tickers."""
        optimizer = SignalOptimizer(
            data=sample_multi_ticker_data,
            indicators=sample_indicators,
            signal_generators=sample_signal_generators,
        )

        optimizer.compute_indicators()
        optimizer.generate_signals()

        assert hasattr(optimizer, "signal_results")
        assert len(optimizer.signal_results) == len(sample_multi_ticker_data)

        for ticker in sample_multi_ticker_data.keys():
            assert ticker in optimizer.signal_results
            assert len(optimizer.signal_results[ticker]) == len(sample_signal_generators)

            for signal_name in sample_signal_generators.keys():
                assert signal_name in optimizer.signal_results[ticker]
                result = optimizer.signal_results[ticker][signal_name]
                assert isinstance(result, pd.Series)
                assert len(result) == len(sample_multi_ticker_data[ticker])

    def test_backtest_signals(
        self, sample_multi_ticker_data, sample_indicators, sample_signal_generators
    ):
        """Test signal backtesting functionality."""
        optimizer = SignalOptimizer(
            data=sample_multi_ticker_data,
            indicators=sample_indicators,
            signal_generators=sample_signal_generators,
        )

        optimizer.compute_indicators()
        optimizer.generate_signals()

        # Mock backtest method if it exists
        if hasattr(optimizer, "backtest_signals"):
            results = optimizer.backtest_signals()

            assert isinstance(results, dict)
            assert len(results) == len(sample_multi_ticker_data)

    def test_signal_optimizer_with_empty_data(self):
        """Test SignalOptimizer with empty data."""
        optimizer = SignalOptimizer(data={}, indicators={}, signal_generators={})

        optimizer.compute_indicators()
        optimizer.generate_signals()

        assert optimizer.indicator_results == {}
        assert optimizer.signal_results == {}


class TestSignalGenerators:
    """Test individual signal generation functions."""

    def test_sma_crossover_signal(self, sample_ohlc_data):
        """Test SMA crossover signal generation."""

        def sma_crossover_signal(df):
            sma_short = sma(df["close"], window=10)
            sma_long = sma(df["close"], window=20)
            return pd.Series(np.where(sma_short > sma_long, 1, -1), index=df.index)

        signals = sma_crossover_signal(sample_ohlc_data)

        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_ohlc_data)
        assert signals.isin([-1, 1]).all()  # Only bullish/bearish signals

    def test_rsi_oversold_signal(self, sample_ohlc_data):
        """Test RSI oversold signal generation."""

        def rsi_oversold_signal(df):
            rsi_series = rsi(df["close"], window=14)
            return pd.Series(np.where(rsi_series < 30, 1, 0), index=df.index)

        signals = rsi_oversold_signal(sample_ohlc_data)

        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_ohlc_data)
        assert signals.isin([0, 1]).all()  # Only 0 or 1 signals

    def test_macd_signal(self, sample_ohlc_data):
        """Test MACD signal generation."""

        def macd_signal(df):
            macd_line, signal_line, histogram = macd(df["close"])
            return pd.Series(np.where(macd_line > signal_line, 1, -1), index=df.index)

        signals = macd_signal(sample_ohlc_data)

        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_ohlc_data)
        assert signals.isin([-1, 1]).all()  # Only bullish/bearish signals

    def test_signal_with_missing_data(self, sample_ohlc_data):
        """Test signal generation with missing data."""
        # Introduce NaN values
        data_with_nan = sample_ohlc_data.copy()
        data_with_nan.loc[data_with_nan.index[10:15], "close"] = np.nan

        def simple_sma_signal(df):
            sma_series = sma(df["close"], window=10)
            return pd.Series(np.where(sma_series > df["close"], 1, -1), index=df.index)

        signals = simple_sma_signal(data_with_nan)

        assert isinstance(signals, pd.Series)
        assert len(signals) == len(data_with_nan)
        # Signals should still be generated even with NaN in input


class TestSignalValidation:
    """Test signal validation and edge cases."""

    def test_signal_timing(self, sample_ohlc_data):
        """Test that signals are properly timed with market data."""

        def momentum_signal(df):
            returns = df["close"].pct_change()
            return pd.Series(np.where(returns > 0, 1, -1), index=df.index)

        signals = momentum_signal(sample_ohlc_data)

        # First signal should be -1 due to pct_change producing NaN which np.where treats as False
        assert signals.iloc[0] == -1
        assert not pd.isna(signals.iloc[1:]).any()

    def test_signal_consistency(self, sample_ohlc_data):
        """Test signal consistency across different runs."""

        def sma_signal(df):
            sma_20 = sma(df["close"], window=20)
            return pd.Series(np.where(df["close"] > sma_20, 1, -1), index=df.index)

        signals1 = sma_signal(sample_ohlc_data)
        signals2 = sma_signal(sample_ohlc_data)

        pd.testing.assert_series_equal(signals1, signals2)

    def test_signal_boundary_conditions(self):
        """Test signals at data boundaries."""
        # Create minimal dataset
        minimal_data = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "close": [102, 103, 104],
                "volume": [1000, 1100, 1200],
            },
            index=pd.date_range("2023-01-01", periods=3),
        )

        def simple_signal(df):
            return pd.Series(np.where(df["close"] > df["open"], 1, -1), index=df.index)

        signals = simple_signal(minimal_data)

        assert len(signals) == 3
        assert signals.isin([-1, 1]).all()
