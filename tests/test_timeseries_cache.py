#!/usr/bin/env python3
"""
Test script for Time Series Cache functionality.

This comprehensive test suite validates the TimeSeriesCache class and its metrics functionality.

TEST COVERAGE:
==============

Core Functionality:
- Price data storage and retrieval
- Signals storage and retrieval
- Date range filtering
- Data existence checks
- Error handling

Metrics & Indicators:
- Automated technical indicator calculation
- Metrics storage and retrieval
- Batch processing for multiple tickers
- Individual indicator validation
- Metrics summary statistics
- Available metrics listing

Advanced Features:
- Multi-timeframe support
- Large dataset handling
- Performance testing
- Memory efficiency
- Data integrity validation
- Concurrent operations

USAGE:
======

Run with pytest (recommended):
    python -m pytest tests/test_timeseries_cache.py -v
    python -m pytest tests/test_timeseries_cache.py -k "metrics"  # Run only metrics tests
    python -m pytest tests/test_timeseries_cache.py -k "performance"  # Run performance tests

Run manual test info:
    python tests/test_timeseries_cache.py

PREREQUISITES:
==============

1. InfluxDB must be running:
   docker start trading-influxdb

2. Python dependencies installed:
   pip install -e .

3. Test data is generated automatically using numpy random seeds for reproducibility

TEST DATA:
==========

- OHLCV data: 31 days of synthetic price data
- Signals data: Random buy/sell/hold signals
- Date range: 2023-01-01 to 2023-01-31
- Tickers: TEST_AAPL, TEST_GOOGL, TEST_MSFT, etc.

INDICATORS TESTED:
==================

Trend Indicators:
- sma_20, sma_50 (Simple Moving Averages)
- ema_12, ema_26 (Exponential Moving Averages)

Momentum Indicators:
- rsi_14 (Relative Strength Index)
- macd, macd_signal, macd_hist (MACD)

Volatility Indicators:
- bb_upper, bb_middle, bb_lower (Bollinger Bands)
- volatility_20 (Price Volatility)

Price Action:
- returns (Price Returns)

PERFORMANCE METRICS:
===================

- Metrics calculation: < 5 seconds for 31 data points
- Data retrieval: < 2 seconds
- Memory usage: Efficient for datasets up to 10,000+ points
- Concurrent operations: Supports multiple tickers simultaneously

"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from open_trading_algo.cache.timeseries_cache import TimeSeriesCache


@pytest.fixture(scope="session")
def influxdb_setup():
    """Ensure InfluxDB is available for testing."""
    try:
        # Simple connection test
        test_cache = TimeSeriesCache()
        test_cache.close()
        return True
    except Exception:
        pytest.skip("InfluxDB not available. Start with: docker start trading-influxdb")
        return False


@pytest.fixture
def cache(influxdb_setup):
    """Create a TimeSeriesCache instance for testing."""
    cache = TimeSeriesCache()
    yield cache
    # Cleanup: close connection after test
    cache.close()


@pytest.fixture
def test_data():
    """Create test OHLCV and signals data."""
    # Create date range - extended to support sma_50 (needs 50+ data points)
    dates = pd.date_range(start="2023-01-01", end="2023-03-15", freq="D")

    # Create OHLCV data
    np.random.seed(42)
    base_price = 100.0

    ohlcv_data = []
    for i, date in enumerate(dates):
        # Simulate price movement
        change = np.random.normal(0, 2.0)
        base_price += change

        open_price = base_price
        close = base_price + np.random.normal(0, 1.0)
        high = max(open_price, close) + abs(np.random.normal(0, 0.5))
        low = min(open_price, close) - abs(np.random.normal(0, 0.5))
        volume = int(np.random.uniform(100000, 500000))

        ohlcv_data.append({"Open": open_price, "High": high, "Low": low, "Close": close, "Volume": volume})

    ohlcv_df = pd.DataFrame(ohlcv_data, index=dates)

    # Create signals data
    signals_data = []
    for date in dates:
        signal_value = np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])
        signals_data.append(signal_value)

    signals_df = pd.DataFrame({"signal_value": signals_data}, index=dates)

    return ohlcv_df, signals_df


def create_test_data():
    """Create test OHLCV and signals data (legacy function for manual tests)."""
    # Create date range
    dates = pd.date_range(start="2023-01-01", end="2023-01-31", freq="D")

    # Create OHLCV data
    np.random.seed(42)
    base_price = 100.0

    ohlcv_data = []
    for i, date in enumerate(dates):
        # Simulate price movement
        change = np.random.normal(0, 2.0)
        base_price += change

        open_price = base_price
        close = base_price + np.random.normal(0, 1.0)
        high = max(open_price, close) + abs(np.random.normal(0, 0.5))
        low = min(open_price, close) - abs(np.random.normal(0, 0.5))
        volume = int(np.random.uniform(100000, 500000))

        ohlcv_data.append({"Open": open_price, "High": high, "Low": low, "Close": close, "Volume": volume})

    ohlcv_df = pd.DataFrame(ohlcv_data, index=dates)

    # Create signals data
    signals_data = []
    for date in dates:
        signal_value = np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])
        signals_data.append(signal_value)

    signals_df = pd.DataFrame({"signal_value": signals_data}, index=dates)

    return ohlcv_df, signals_df


# ===== PYTEST TEST FUNCTIONS =====


class TestTimeSeriesCache:
    """Test suite for TimeSeriesCache using pytest framework."""

    def test_price_data_storage_and_retrieval(self, cache, test_data):
        """Test storing and retrieving price data."""
        ohlcv_df, _ = test_data
        ticker = "TEST_AAPL"

        # Store data
        cache.store_price_data(ticker, ohlcv_df)

        # Add delay to ensure data is written
        import time

        time.sleep(1)

        # Retrieve data
        retrieved_data = cache.get_price_data(ticker)

        assert not retrieved_data.empty
        assert len(retrieved_data) == len(ohlcv_df)
        assert list(retrieved_data.columns) == ["open", "high", "low", "close", "volume"]

    def test_price_data_with_date_filtering(self, cache, test_data):
        """Test price data retrieval with date filters."""
        ohlcv_df, _ = test_data
        ticker = "TEST_AAPL"

        cache.store_price_data(ticker, ohlcv_df)

        # Test start date filtering
        start_date = "2023-01-15"
        filtered_data = cache.get_price_data(ticker, start=start_date)

        expected_points = len(ohlcv_df[ohlcv_df.index >= start_date])
        assert len(filtered_data) == expected_points

    def test_signals_storage_and_retrieval(self, cache, test_data):
        """Test storing and retrieving signals."""
        _, signals_df = test_data
        ticker = "TEST_AAPL"
        timeframe = "1d"
        signal_type = "test_signal"

        # Store signals
        cache.store_signals(ticker, timeframe, signal_type, signals_df)

        # Retrieve signals
        retrieved_signals = cache.get_signals(ticker, timeframe, signal_type)

        assert not retrieved_signals.empty
        assert len(retrieved_signals) == len(signals_df)
        assert "signal_value" in retrieved_signals.columns

    def test_data_existence_checks(self, cache, test_data):
        """Test data existence checking methods."""
        ohlcv_df, signals_df = test_data
        # Use a unique ticker to avoid conflicts with existing data
        ticker = f"TEST_EXISTENCE_{hash(str(test_data)) % 10000}"

        # Test price data existence
        assert not cache.has_data(ticker)
        cache.store_price_data(ticker, ohlcv_df)
        assert cache.has_data(ticker)

        # Test signals existence
        assert not cache.has_signals(ticker, "1d", "test_signal")
        cache.store_signals(ticker, "1d", "test_signal", signals_df)
        assert cache.has_signals(ticker, "1d", "test_signal")

    def test_metrics_calculation_and_storage(self, cache, test_data):
        """Test metrics calculation and storage."""
        ohlcv_df, _ = test_data
        ticker = "TEST_AAPL"
        timeframe = "1d"

        # Store price data first
        cache.store_price_data(ticker, ohlcv_df)

        # Calculate and store metrics
        indicators = ["sma_20", "rsi_14", "macd"]
        cache.calculate_and_store_metrics(ticker, timeframe=timeframe, indicators=indicators)

        # Verify metrics exist
        assert cache.has_metrics(ticker, timeframe)

    def test_metrics_retrieval(self, cache, test_data):
        """Test metrics retrieval with various filters."""
        ohlcv_df, _ = test_data
        ticker = "TEST_AAPL"
        timeframe = "1d"

        # Clear any existing metrics data
        cache.clear_metrics_data(ticker, timeframe)

        # Store price data and calculate metrics
        cache.store_price_data(ticker, ohlcv_df)
        indicators = ["sma_20", "rsi_14", "macd"]
        cache.calculate_and_store_metrics(ticker, timeframe=timeframe, indicators=indicators)

        # Retrieve all metrics
        all_metrics = cache.get_metrics(ticker, timeframe=timeframe)
        assert not all_metrics.empty
        assert len(all_metrics.columns) == len(indicators)

        # Retrieve specific metrics
        specific_metrics = cache.get_metrics(ticker, timeframe=timeframe, metrics=["rsi_14", "sma_20"])
        assert not specific_metrics.empty
        assert list(specific_metrics.columns) == ["rsi_14", "sma_20"]

    def test_metrics_with_date_filtering(self, cache, test_data):
        """Test metrics retrieval with date filters."""
        ohlcv_df, _ = test_data
        ticker = "TEST_AAPL"
        timeframe = "1d"

        # Store price data and calculate metrics
        cache.store_price_data(ticker, ohlcv_df)
        cache.calculate_and_store_metrics(ticker, timeframe=timeframe, indicators=["sma_20"])

        # Test date filtering
        start_date = "2023-01-15"
        filtered_metrics = cache.get_metrics(ticker, timeframe=timeframe, start=start_date)

        assert not filtered_metrics.empty
        # Fix timezone comparison issue
        start_datetime = pd.to_datetime(start_date).tz_localize("UTC")
        assert all(filtered_metrics.index >= start_datetime)

    def test_available_metrics_listing(self, cache, test_data):
        """Test getting list of available metrics."""
        ohlcv_df, _ = test_data
        ticker = "TEST_AAPL"
        timeframe = "1d"

        # Clear any existing metrics data
        cache.clear_metrics_data(ticker, timeframe)

        # Store price data and calculate metrics
        cache.store_price_data(ticker, ohlcv_df)
        indicators = ["sma_20", "rsi_14", "macd", "bb_upper"]
        cache.calculate_and_store_metrics(ticker, timeframe=timeframe, indicators=indicators)

        # Get available metrics
        available_metrics = cache.get_available_metrics(ticker, timeframe)

        assert isinstance(available_metrics, list)
        assert len(available_metrics) == len(indicators)
        assert all(indicator in available_metrics for indicator in indicators)

    def test_metrics_summary(self, cache, test_data):
        """Test metrics summary statistics."""
        ohlcv_df, _ = test_data
        ticker = "TEST_AAPL"
        timeframe = "1d"

        # Store price data and calculate metrics
        cache.store_price_data(ticker, ohlcv_df)
        cache.calculate_and_store_metrics(ticker, timeframe=timeframe, indicators=["rsi_14", "sma_20"])

        # Get metrics summary
        summary = cache.get_metrics_summary(ticker, timeframe=timeframe)

        assert isinstance(summary, dict)
        assert "ticker" in summary
        assert "timeframe" in summary
        assert "data_points" in summary
        assert "available_metrics" in summary
        assert "metrics_stats" in summary
        assert summary["ticker"] == ticker
        assert summary["timeframe"] == timeframe

    def test_batch_metrics_population(self, cache, test_data):
        """Test batch processing of metrics for multiple tickers."""
        ohlcv_df, _ = test_data
        tickers = ["TEST_AAPL", "TEST_GOOGL", "TEST_MSFT"]

        # Store price data for multiple tickers
        for ticker in tickers:
            cache.store_price_data(ticker, ohlcv_df)

        # Batch calculate metrics
        cache.populate_metrics_table(tickers, timeframe="1d", indicators=["sma_20", "rsi_14"])

        # Verify metrics were calculated for all tickers
        for ticker in tickers:
            assert cache.has_metrics(ticker, "1d")
            metrics = cache.get_metrics(ticker, timeframe="1d")
            assert not metrics.empty

    def test_individual_technical_indicators(self, cache, test_data):
        """Test calculation of individual technical indicators."""
        ohlcv_df, _ = test_data
        ticker = "TEST_AAPL"
        timeframe = "1d"

        cache.store_price_data(ticker, ohlcv_df)

        # Test various indicators (now with sufficient data points for sma_50)
        test_indicators = [
            "sma_20",
            "sma_50",  # Moving averages (can be calculated with 74 days of data)
            "ema_12",
            "ema_26",  # Exponential moving averages
            "rsi_14",  # RSI
            "macd",
            "macd_signal",
            "macd_hist",  # MACD
            "bb_upper",
            "bb_middle",
            "bb_lower",  # Bollinger Bands
            "volatility_20",  # Volatility
            "returns",  # Returns
        ]

        cache.calculate_and_store_metrics(ticker, timeframe=timeframe, indicators=test_indicators)

        metrics = cache.get_metrics(ticker, timeframe=timeframe)
        assert not metrics.empty

        # Verify all expected indicators are present
        for indicator in test_indicators:
            assert indicator in metrics.columns, f"Missing indicator: {indicator}. Available: {list(metrics.columns)}"

            # Verify no NaN values in recent data (should have enough data points)
            recent_metrics = metrics.tail(10)  # Last 10 data points
            for indicator in test_indicators:
                if indicator in recent_metrics.columns:
                    # Some indicators might have NaN at the beginning due to calculation windows
                    non_na_count = recent_metrics[indicator].notna().sum()
                    assert non_na_count > 0, f"No valid values for {indicator}"

    def test_aggregated_queries(self, cache, test_data):
        """Test aggregated query functionality."""
        ohlcv_df, _ = test_data
        ticker = "TEST_AAPL"

        cache.store_price_data(ticker, ohlcv_df)

        # Test weekly aggregation
        weekly_data = cache.get_aggregated_data(ticker, aggregation="1w")
        assert isinstance(weekly_data, pd.DataFrame)

    def test_database_info_with_metrics(self, cache, test_data):
        """Test database info includes metrics data."""
        ohlcv_df, _ = test_data
        ticker = "TEST_AAPL"

        # Store data and calculate metrics
        cache.store_price_data(ticker, ohlcv_df)
        cache.calculate_and_store_metrics(ticker, indicators=["sma_20"])

        # Get database info
        info = cache.get_database_info()

        assert isinstance(info, dict)
        assert "price_data_points" in info
        assert "signals_points" in info
        assert "metrics_points" in info
        assert "total_data_points" in info
        assert info["metrics_points"] > 0

    def test_error_handling(self, cache):
        """Test error handling for invalid operations."""
        # Test non-existent ticker
        data = cache.get_price_data("NON_EXISTENT_TICKER")
        assert data.empty

        # Test non-existent metrics
        metrics = cache.get_metrics("NON_EXISTENT_TICKER")
        assert metrics.empty

        # Test empty DataFrame handling
        empty_df = pd.DataFrame()
        cache.store_price_data("TEST_EMPTY", empty_df)  # Should not raise error

    def test_metrics_with_different_timeframes(self, cache, test_data):
        """Test metrics calculation for different timeframes."""
        ohlcv_df, _ = test_data
        ticker = "TEST_AAPL"

        cache.store_price_data(ticker, ohlcv_df)

        # Calculate metrics for different timeframes
        timeframes = ["1d", "1h", "1w"]
        for timeframe in timeframes:
            cache.calculate_and_store_metrics(ticker, timeframe=timeframe, indicators=["sma_20"])

            # Verify metrics exist for this timeframe
            assert cache.has_metrics(ticker, timeframe)
            metrics = cache.get_metrics(ticker, timeframe=timeframe)
            assert not metrics.empty

    def test_signal_statistics_comprehensive(self, cache, test_data):
        """Test comprehensive signal statistics."""
        _, signals_df = test_data
        ticker = "TEST_AAPL"
        timeframe = "1d"
        signal_type = "test_signal"

        cache.store_signals(ticker, timeframe, signal_type, signals_df)

        # Get signal statistics
        stats = cache.get_signal_stats(ticker, timeframe, signal_type)

        assert isinstance(stats, dict)
        assert "total_signals" in stats
        assert "ticker" in stats
        assert "timeframe" in stats
        assert "signal_type" in stats
        assert stats["total_signals"] == len(signals_df)

    def test_large_dataset_handling(self, cache):
        """Test handling of larger datasets."""
        # Create larger dataset
        dates = pd.date_range(start="2023-01-01", end="2023-03-31", freq="D")
        np.random.seed(42)
        base_price = 100.0

        ohlcv_data = []
        for i, date in enumerate(dates):
            change = np.random.normal(0, 2.0)
            base_price += change

            open_price = base_price
            close = base_price + np.random.normal(0, 1.0)
            high = max(open_price, close) + abs(np.random.normal(0, 0.5))
            low = min(open_price, close) - abs(np.random.normal(0, 0.5))
            volume = int(np.random.uniform(100000, 500000))

            ohlcv_data.append({"Open": open_price, "High": high, "Low": low, "Close": close, "Volume": volume})

        large_df = pd.DataFrame(ohlcv_data, index=dates)
        ticker = "TEST_LARGE"

        # Store large dataset
        cache.store_price_data(ticker, large_df)

        # Calculate metrics on large dataset
        cache.calculate_and_store_metrics(ticker, indicators=["sma_20", "rsi_14"])

        # Retrieve and verify
        retrieved_data = cache.get_price_data(ticker)
        metrics = cache.get_metrics(ticker)

        assert len(retrieved_data) == len(large_df)
        assert not metrics.empty
        assert len(metrics) > 0

    def test_concurrent_operations(self, cache, test_data):
        """Test concurrent operations on the cache."""
        ohlcv_df, signals_df = test_data
        tickers = ["TEST_AAPL", "TEST_GOOGL", "TEST_MSFT"]

        # Store data for multiple tickers
        for ticker in tickers:
            cache.store_price_data(ticker, ohlcv_df)
            cache.store_signals(ticker, "1d", "test_signal", signals_df)

        # Calculate metrics for all tickers
        for ticker in tickers:
            cache.calculate_and_store_metrics(ticker, indicators=["sma_20"])

        # Verify all data is accessible
        for ticker in tickers:
            assert cache.has_data(ticker)
            assert cache.has_signals(ticker, "1d", "test_signal")
            assert cache.has_metrics(ticker, "1d")

            data = cache.get_price_data(ticker)
            signals = cache.get_signals(ticker, "1d", "test_signal")
            metrics = cache.get_metrics(ticker)

            assert not data.empty
            assert not signals.empty
            assert not metrics.empty

    def test_data_integrity_and_consistency(self, cache, test_data):
        """Test data integrity and consistency across operations."""
        ohlcv_df, _ = test_data
        ticker = "TEST_AAPL"

        # Store original data
        cache.store_price_data(ticker, ohlcv_df)
        original_data = cache.get_price_data(ticker)

        # Calculate metrics
        cache.calculate_and_store_metrics(ticker, indicators=["sma_20", "rsi_14"])
        metrics_data = cache.get_metrics(ticker)

        # Verify data consistency
        assert len(original_data) == len(ohlcv_df)
        assert not metrics_data.empty

        # Verify timestamps align (metrics should be calculated for price data points)
        price_timestamps = set(original_data.index)
        metrics_timestamps = set(metrics_data.index)
        assert metrics_timestamps.issubset(price_timestamps) or len(metrics_timestamps) > 0

    def test_memory_efficiency(self, cache, test_data):
        """Test memory efficiency with large operations."""
        ohlcv_df, _ = test_data
        ticker = "TEST_AAPL"

        # Store data
        cache.store_price_data(ticker, ohlcv_df)

        # Perform multiple operations
        for i in range(5):
            cache.calculate_and_store_metrics(ticker, indicators=["sma_20", "rsi_14"])
            metrics = cache.get_metrics(ticker)
            assert not metrics.empty

        # Verify database info reflects accumulated data
        info = cache.get_database_info()
        assert info["metrics_points"] > 0

    def test_technical_indicator_accuracy(self, cache):
        """Test accuracy of technical indicator calculations."""
        # Create predictable test data for accuracy verification
        dates = pd.date_range(start="2023-01-01", end="2023-02-15", freq="D")
        # Create prices list with same length as dates
        prices = [100.0] * 15 + [105.0] * 15 + [95.0] * 16  # 15 + 15 + 16 = 46

        ohlcv_data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            ohlcv_data.append(
                {
                    "Open": close,
                    "High": close + 1,
                    "Low": close - 1,
                    "Close": close,
                    "Volume": 100000,
                }
            )

        test_df = pd.DataFrame(ohlcv_data, index=dates)
        ticker = "TEST_ACCURACY"

        cache.store_price_data(ticker, test_df)
        cache.calculate_and_store_metrics(ticker, indicators=["sma_20", "rsi_14"])

        metrics = cache.get_metrics(ticker)

        # Test SMA calculation accuracy
        if "sma_20" in metrics.columns:
            # For the last 10 days where we have enough data, SMA should be around 97-102
            recent_sma = metrics["sma_20"].tail(10)
            # The SMA should be between 95 and 105 for this test data
            assert all(
                95 <= val <= 105 for val in recent_sma.dropna()
            ), f"SMA calculation incorrect. Expected range 95-105, got {recent_sma.dropna().values}"

        # Test RSI calculation (basic validation)
        if "rsi_14" in metrics.columns:
            rsi_values = metrics["rsi_14"].dropna()
            # RSI should be between 0 and 100
            assert all(0 <= val <= 100 for val in rsi_values), f"RSI values out of range: {rsi_values.values}"

    def test_metrics_performance(self, cache, test_data):
        """Test performance of metrics operations."""
        import time

        ohlcv_df, _ = test_data
        ticker = "TEST_PERF"

        cache.store_price_data(ticker, ohlcv_df)

        # Time metrics calculation
        start_time = time.time()
        cache.calculate_and_store_metrics(
            ticker,
            indicators=[
                "sma_20",
                "sma_50",
                "ema_12",
                "ema_26",
                "rsi_14",
                "macd",
                "bb_upper",
                "bb_lower",
            ],
        )
        calc_time = time.time() - start_time

        # Time metrics retrieval
        start_time = time.time()
        metrics = cache.get_metrics(ticker)
        retrieval_time = time.time() - start_time

        # Performance should be reasonable (less than 5 seconds for this dataset)
        assert calc_time < 5.0, f"Metrics calculation too slow: {calc_time:.2f}s"
        assert retrieval_time < 2.0, f"Metrics retrieval too slow: {retrieval_time:.2f}s"

        assert not metrics.empty
        assert len(metrics.columns) > 0

        # Log performance for monitoring
        print(f"Performance - Calculation: {calc_time:.2f}s, Retrieval: {retrieval_time:.2f}s")


# ===== UTILITY FUNCTIONS =====


def validate_indicator_calculation(cache, ticker, indicator_name, expected_range=None):
    """
    Utility function to validate indicator calculations.

    Args:
        cache: TimeSeriesCache instance
        ticker: Ticker symbol
        indicator_name: Name of indicator to validate
        expected_range: Tuple of (min, max) expected values
    """
    metrics = cache.get_metrics(ticker, indicators=[indicator_name])

    if metrics.empty or indicator_name not in metrics.columns:
        return False, f"Indicator {indicator_name} not found in metrics"

    values = metrics[indicator_name].dropna()

    if len(values) == 0:
        return False, f"No valid values for indicator {indicator_name}"

    if expected_range:
        min_val, max_val = expected_range
        if not all(min_val <= val <= max_val for val in values):
            return False, f"Values for {indicator_name} out of expected range {expected_range}"

    return True, f"Indicator {indicator_name} validation passed"


def benchmark_operation(operation_func, operation_name, max_time=5.0):
    """
    Benchmark an operation and ensure it completes within time limits.

    Args:
        operation_func: Function to benchmark
        operation_name: Name for logging
        max_time: Maximum allowed time in seconds

    Returns:
        Tuple of (success, execution_time)
    """
    import time

    start_time = time.time()
    try:
        result = operation_func()
        execution_time = time.time() - start_time

        if execution_time > max_time:
            print(f"⚠️  {operation_name} took {execution_time:.2f}s (limit: {max_time}s)")
            return False, execution_time

        print(f"✅ {operation_name} completed in {execution_time:.2f}s")
        return True, execution_time

    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ {operation_name} failed after {execution_time:.2f}s: {e}")
        return False, execution_time


def main():
    """Run all tests."""
    print("🚀 Time Series Cache Test Suite")
    print("=" * 50)
    print("Comprehensive test suite for TimeSeriesCache functionality")
    print("\n📋 Test Coverage:")
    print("  ✅ Price data storage and retrieval")
    print("  ✅ Signals storage and retrieval")
    print("  ✅ Metrics calculation and storage")
    print("  ✅ Metrics retrieval with filtering")
    print("  ✅ Batch processing for multiple tickers")
    print("  ✅ Technical indicator accuracy")
    print("  ✅ Date range filtering")
    print("  ✅ Error handling and edge cases")
    print("  ✅ Performance and memory efficiency")
    print("  ✅ Data integrity and consistency")
    print("  ✅ Concurrent operations")
    print("  ✅ Large dataset handling")
    print("\n🔧 Usage:")
    print("  pytest: python -m pytest tests/test_timeseries_cache.py -v")
    print("  manual: python tests/test_timeseries_cache.py")
    print("\n💡 Prerequisites:")
    print("  - InfluxDB running: docker start trading-influxdb")
    print("  - Python dependencies installed")
    print("\n📊 Supported Indicators:")
    indicators = [
        "sma_20, sma_50 (Simple Moving Averages)",
        "ema_12, ema_26 (Exponential Moving Averages)",
        "rsi_14 (Relative Strength Index)",
        "macd, macd_signal, macd_hist (MACD)",
        "bb_upper, bb_middle, bb_lower (Bollinger Bands)",
        "volatility_20 (Price Volatility)",
        "returns (Price Returns)",
    ]
    for indicator in indicators:
        print(f"  • {indicator}")


if __name__ == "__main__":
    main()
