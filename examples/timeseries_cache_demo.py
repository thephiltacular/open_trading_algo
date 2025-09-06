#!/usr/bin/env python3
"""
Demo script for Time Series Cache using InfluxDB.

This script demonstrates how to use the TimeSeriesCache class
for storing and retrieving financial data and trading signals.

Usage:
    python examples/timeseries_cache_demo.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from open_trading_algo.cache.timeseries_cache import TimeSeriesCache


def generate_sample_ohlcv_data(ticker: str, days: int = 100) -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)  # For reproducible results

    # Generate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Generate realistic price data
    base_price = 100 + np.random.uniform(-20, 20)

    prices = []
    current_price = base_price

    for _ in dates:
        # Random walk with some volatility
        change = np.random.normal(0, 2.0)
        current_price += change

        # Generate OHLC with some spread
        high = current_price + abs(np.random.normal(0, 1.0))
        low = current_price - abs(np.random.normal(0, 1.0))
        open_price = current_price + np.random.normal(0, 0.5)
        close = current_price + np.random.normal(0, 0.5)

        # Ensure high >= max(open, close) and low <= min(open, close)
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        # Generate volume
        volume = int(np.random.uniform(100000, 1000000))

        prices.append(
            {"Open": open_price, "High": high, "Low": low, "Close": close, "Volume": volume}
        )

    df = pd.DataFrame(prices, index=dates)
    return df


def generate_sample_signals(ticker: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate sample trading signals."""
    np.random.seed(123)

    signals = []
    for _ in dates:
        # Generate random signals (-1, 0, 1)
        signal_value = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
        signals.append(signal_value)

    df = pd.DataFrame({"signal_value": signals}, index=dates)
    return df


def demo_price_data_operations():
    """Demonstrate price data storage and retrieval."""
    print("📊 Demonstrating Price Data Operations")
    print("-" * 40)

    # Initialize cache
    cache = TimeSeriesCache()

    # Generate sample data
    ticker = "AAPL"
    print(f"📈 Generating sample OHLCV data for {ticker}...")
    price_data = generate_sample_ohlcv_data(ticker, days=50)
    print(f"   Generated {len(price_data)} data points")

    # Store data
    print(f"💾 Storing price data for {ticker}...")
    cache.store_price_data(ticker, price_data)
    print("   ✅ Data stored successfully")

    # Retrieve data
    print(f"📖 Retrieving price data for {ticker}...")
    retrieved_data = cache.get_price_data(ticker)
    print(f"   Retrieved {len(retrieved_data)} data points")

    # Show sample of retrieved data
    print("\n📋 Sample of retrieved data:")
    print(retrieved_data.head())

    # Test date range filtering
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    print(f"\n📅 Retrieving data from {start_date}...")
    filtered_data = cache.get_price_data(ticker, start=start_date)
    print(f"   Retrieved {len(filtered_data)} data points in date range")

    # Check data existence
    exists = cache.has_data(ticker)
    print(f"🔍 Data exists for {ticker}: {exists}")

    cache.close()


def demo_signals_operations():
    """Demonstrate signals storage and retrieval."""
    print("\n📊 Demonstrating Signals Operations")
    print("-" * 40)

    # Initialize cache
    cache = TimeSeriesCache()

    # Generate sample data
    ticker = "AAPL"
    timeframe = "1d"
    signal_type = "momentum"

    # Get dates from existing price data
    price_data = cache.get_price_data(ticker)
    if price_data.empty:
        print("   No price data found, generating new signals data...")
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=30), end=datetime.now(), freq="D"
        )
    else:
        dates = price_data.index

    print(f"🎯 Generating sample signals for {ticker} ({timeframe}, {signal_type})...")
    signals_data = generate_sample_signals(ticker, dates)
    print(f"   Generated {len(signals_data)} signal points")

    # Store signals
    print(f"💾 Storing signals for {ticker}...")
    cache.store_signals(ticker, timeframe, signal_type, signals_data)
    print("   ✅ Signals stored successfully")

    # Retrieve signals
    print(f"📖 Retrieving signals for {ticker}...")
    retrieved_signals = cache.get_signals(ticker, timeframe, signal_type)
    print(f"   Retrieved {len(retrieved_signals)} signal points")

    # Show sample of retrieved signals
    print("\n📋 Sample of retrieved signals:")
    print(retrieved_signals.head())

    # Check signals existence
    exists = cache.has_signals(ticker, timeframe, signal_type)
    print(f"🔍 Signals exist for {ticker}: {exists}")

    cache.close()


def demo_aggregated_queries():
    """Demonstrate aggregated data queries."""
    print("\n📊 Demonstrating Aggregated Queries")
    print("-" * 40)

    cache = TimeSeriesCache()

    ticker = "AAPL"

    # Get aggregated data (weekly)
    print(f"📈 Getting weekly aggregated data for {ticker}...")
    weekly_data = cache.get_aggregated_data(ticker, aggregation="1w")
    print(f"   Retrieved {len(weekly_data)} weekly data points")

    if not weekly_data.empty:
        print("\n📋 Sample of weekly aggregated data:")
        print(weekly_data.head())

    # Get signal statistics
    print(f"\n📊 Getting signal statistics for {ticker}...")
    stats = cache.get_signal_stats(ticker, "1d", "momentum")
    if stats:
        print(f"   Total signals: {stats.get('total_signals', 0)}")
        print(f"   Ticker: {stats.get('ticker')}")
        print(f"   Timeframe: {stats.get('timeframe')}")
        print(f"   Signal type: {stats.get('signal_type')}")

    cache.close()


def demo_database_info():
    """Show database information and statistics."""
    print("\n📊 Database Information")
    print("-" * 40)

    cache = TimeSeriesCache()

    info = cache.get_database_info()
    if info:
        print(f"🗄️  Database URL: {info.get('database_url')}")
        print(f"🏢 Organization: {info.get('organization')}")
        print(f"🪣 Bucket: {info.get('bucket')}")
        print(f"💰 Price data points: {info.get('price_data_points', 0)}")
        print(f"🎯 Signals points: {info.get('signals_points', 0)}")
        print(f"📊 Total data points: {info.get('total_data_points', 0)}")
    else:
        print("❌ Could not retrieve database information")

    cache.close()


def main():
    """Main demo function."""
    print("🚀 Time Series Cache Demo")
    print("=" * 50)
    print("This demo shows how to use InfluxDB for caching financial time series data.")
    print()

    try:
        # Run demonstrations
        demo_price_data_operations()
        demo_signals_operations()
        demo_aggregated_queries()
        demo_database_info()

        print("\n🎉 Demo completed successfully!")
        print("\n💡 Key Features Demonstrated:")
        print("   • Efficient storage of OHLCV financial data")
        print("   • Flexible signal storage and retrieval")
        print("   • Date range filtering")
        print("   • Aggregated queries (time windows)")
        print("   • Signal statistics and analytics")
        print("   • High-performance time series queries")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Make sure InfluxDB is running and accessible.")


if __name__ == "__main__":
    main()
