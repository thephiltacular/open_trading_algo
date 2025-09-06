#!/usr/bin/env python3
"""
Test script for Time Series Cache functionality.

This script tests the TimeSeriesCache class to ensure it works correctly
with various operations like storing, retrieving, and querying data.

Usage:
    python -m pytest tests/test_timeseries_cache.py -v
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from open_trading_algo.cache.timeseries_cache import TimeSeriesCache


def create_test_data():
    """Create test OHLCV and signals data."""
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

        ohlcv_data.append(
            {"Open": open_price, "High": high, "Low": low, "Close": close, "Volume": volume}
        )

    ohlcv_df = pd.DataFrame(ohlcv_data, index=dates)

    # Create signals data
    signals_data = []
    for date in dates:
        signal_value = np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])
        signals_data.append(signal_value)

    signals_df = pd.DataFrame({"signal_value": signals_data}, index=dates)

    return ohlcv_df, signals_df


def test_price_data_operations():
    """Test price data storage and retrieval."""
    print("🧪 Testing Price Data Operations...")

    cache = TimeSeriesCache()
    ticker = "TEST_AAPL"

    # Create test data
    ohlcv_df, _ = create_test_data()

    # Test storage
    try:
        cache.store_price_data(ticker, ohlcv_df)
        print("   ✅ Price data stored successfully")
    except Exception as e:
        print(f"   ❌ Failed to store price data: {e}")
        return False

    # Test retrieval
    try:
        retrieved_data = cache.get_price_data(ticker)
        if len(retrieved_data) == len(ohlcv_df):
            print("   ✅ Price data retrieved successfully")
        else:
            print(f"   ❌ Data length mismatch: expected {len(ohlcv_df)}, got {len(retrieved_data)}")
            return False
    except Exception as e:
        print(f"   ❌ Failed to retrieve price data: {e}")
        return False

    # Test date filtering
    try:
        start_date = "2023-01-15"
        filtered_data = cache.get_price_data(ticker, start=start_date)
        expected_points = len(ohlcv_df[ohlcv_df.index >= start_date])
        if len(filtered_data) == expected_points:
            print("   ✅ Date filtering works correctly")
        else:
            print(
                f"   ❌ Date filtering failed: expected {expected_points}, got {len(filtered_data)}"
            )
            return False
    except Exception as e:
        print(f"   ❌ Date filtering failed: {e}")
        return False

    # Test data existence check
    try:
        exists = cache.has_data(ticker)
        if exists:
            print("   ✅ Data existence check works")
        else:
            print("   ❌ Data existence check failed")
            return False
    except Exception as e:
        print(f"   ❌ Data existence check failed: {e}")
        return False

    cache.close()
    return True


def test_signals_operations():
    """Test signals storage and retrieval."""
    print("🧪 Testing Signals Operations...")

    cache = TimeSeriesCache()
    ticker = "TEST_AAPL"
    timeframe = "1d"
    signal_type = "test_signal"

    # Create test data
    _, signals_df = create_test_data()

    # Test storage
    try:
        cache.store_signals(ticker, timeframe, signal_type, signals_df)
        print("   ✅ Signals stored successfully")
    except Exception as e:
        print(f"   ❌ Failed to store signals: {e}")
        return False

    # Test retrieval
    try:
        retrieved_signals = cache.get_signals(ticker, timeframe, signal_type)
        if len(retrieved_signals) == len(signals_df):
            print("   ✅ Signals retrieved successfully")
        else:
            print(
                f"   ❌ Signals length mismatch: expected {len(signals_df)}, got {len(retrieved_signals)}"
            )
            return False
    except Exception as e:
        print(f"   ❌ Failed to retrieve signals: {e}")
        return False

    # Test signals existence check
    try:
        exists = cache.has_signals(ticker, timeframe, signal_type)
        if exists:
            print("   ✅ Signals existence check works")
        else:
            print("   ❌ Signals existence check failed")
            return False
    except Exception as e:
        print(f"   ❌ Signals existence check failed: {e}")
        return False

    cache.close()
    return True


def test_aggregated_queries():
    """Test aggregated query functionality."""
    print("🧪 Testing Aggregated Queries...")

    cache = TimeSeriesCache()
    ticker = "TEST_AAPL"

    # Test aggregated data query
    try:
        weekly_data = cache.get_aggregated_data(ticker, aggregation="1w")
        if not weekly_data.empty:
            print("   ✅ Aggregated query works")
        else:
            print("   ⚠️  No aggregated data (this may be expected if no data exists)")
    except Exception as e:
        print(f"   ❌ Aggregated query failed: {e}")
        return False

    # Test signal statistics
    try:
        stats = cache.get_signal_stats(ticker, "1d", "test_signal")
        if isinstance(stats, dict):
            print("   ✅ Signal statistics query works")
        else:
            print("   ❌ Signal statistics returned invalid data")
            return False
    except Exception as e:
        print(f"   ❌ Signal statistics failed: {e}")
        return False

    cache.close()
    return True


def test_database_info():
    """Test database information retrieval."""
    print("🧪 Testing Database Info...")

    cache = TimeSeriesCache()

    try:
        info = cache.get_database_info()
        if isinstance(info, dict) and "bucket" in info:
            print("   ✅ Database info retrieved successfully")
            print(f"      Bucket: {info.get('bucket')}")
            print(f"      Price data points: {info.get('price_data_points', 0)}")
            print(f"      Signals points: {info.get('signals_points', 0)}")
        else:
            print("   ❌ Database info returned invalid data")
            return False
    except Exception as e:
        print(f"   ❌ Database info failed: {e}")
        return False

    cache.close()
    return True


def test_error_handling():
    """Test error handling for invalid operations."""
    print("🧪 Testing Error Handling...")

    cache = TimeSeriesCache()

    # Test with non-existent ticker
    try:
        data = cache.get_price_data("NON_EXISTENT_TICKER")
        if data.empty:
            print("   ✅ Non-existent ticker handled correctly")
        else:
            print("   ❌ Non-existent ticker returned data unexpectedly")
            return False
    except Exception as e:
        print(f"   ❌ Error handling failed: {e}")
        return False

    # Test with empty DataFrame
    try:
        empty_df = pd.DataFrame()
        cache.store_price_data("TEST_EMPTY", empty_df)
        print("   ✅ Empty DataFrame handled correctly")
    except Exception as e:
        print(f"   ❌ Empty DataFrame caused error: {e}")
        return False

    cache.close()
    return True


def main():
    """Run all tests."""
    print("🚀 Time Series Cache Test Suite")
    print("=" * 40)

    tests = [
        ("Price Data Operations", test_price_data_operations),
        ("Signals Operations", test_signals_operations),
        ("Aggregated Queries", test_aggregated_queries),
        ("Database Info", test_database_info),
        ("Error Handling", test_error_handling),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔬 Running: {test_name}")
        print("-" * 30)

        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")

    print("\n" + "=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Time Series Cache is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")

    print("\n💡 Make sure InfluxDB is running before running these tests:")
    print("   docker start trading-influxdb")


if __name__ == "__main__":
    main()
