#!/usr/bin/env python3
"""
Demo script for Time Series Cache Metrics functionality.

This script demonstrates how to calculate and store technical indicators
and metrics from price data using the TimeSeriesCache.

Usage:
    python examples/timeseries_cache_demo.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from open_trading_algo.cache.timeseries_cache import TimeSeriesCache


def generate_sample_price_data(ticker: str, days: int = 100) -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)  # For reproducible results

    # Generate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Generate realistic price data
    base_price = 100.0 + np.random.uniform(-20, 20)

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

    df = pd.DataFrame(ohlcv_data, index=dates)
    return df


def demo_price_data_storage():
    """Demonstrate storing price data."""
    print("📊 Storing Price Data")
    print("-" * 30)

    cache = TimeSeriesCache()
    ticker = "AAPL"

    # Generate sample data
    print(f"📈 Generating sample OHLCV data for {ticker}...")
    price_data = generate_sample_price_data(ticker, days=50)
    print(f"   Generated {len(price_data)} data points")

    # Store data
    print(f"💾 Storing price data for {ticker}...")
    cache.store_price_data(ticker, price_data)
    print("   ✅ Price data stored successfully")

    cache.close()


def demo_metrics_calculation():
    """Demonstrate calculating and storing metrics."""
    print("\n📊 Calculating and Storing Metrics")
    print("-" * 40)

    cache = TimeSeriesCache()
    ticker = "AAPL"
    timeframe = "1d"

    # Calculate and store all available metrics
    print(f"🧮 Calculating technical indicators for {ticker}...")
    cache.calculate_and_store_metrics(
        ticker=ticker,
        timeframe=timeframe,
        indicators=[
            "sma_20",
            "sma_50",
            "ema_12",
            "ema_26",
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "volatility_20",
            "returns",
        ],
    )

    # Get available metrics
    print(f"📋 Available metrics for {ticker}:")
    available_metrics = cache.get_available_metrics(ticker, timeframe)
    for metric in available_metrics:
        print(f"   • {metric}")

    cache.close()


def demo_metrics_retrieval():
    """Demonstrate retrieving stored metrics."""
    print("\n📊 Retrieving Metrics Data")
    print("-" * 30)

    cache = TimeSeriesCache()
    ticker = "AAPL"
    timeframe = "1d"

    # Retrieve specific metrics
    metrics_to_get = ["rsi_14", "macd", "sma_20", "bb_upper", "bb_lower"]
    print(f"📖 Retrieving metrics: {', '.join(metrics_to_get)}")

    metrics_df = cache.get_metrics(ticker=ticker, timeframe=timeframe, metrics=metrics_to_get, start="-30d")  # Last 30 days

    if not metrics_df.empty:
        print(f"   Retrieved {len(metrics_df)} data points")
        print("\n📋 Sample of retrieved metrics:")
        print(metrics_df.tail())

        # Show latest values
        print("\n📈 Latest metric values:")
        latest = metrics_df.iloc[-1]
        for metric, value in latest.items():
            if pd.notna(value):
                print(f"   {metric}: {value:.4f}")
    else:
        print("   No metrics data found")

    cache.close()


def demo_metrics_summary():
    """Demonstrate getting metrics summary statistics."""
    print("\n📊 Metrics Summary Statistics")
    print("-" * 35)

    cache = TimeSeriesCache()
    ticker = "AAPL"
    timeframe = "1d"

    summary = cache.get_metrics_summary(ticker, timeframe)

    if "error" not in summary:
        print(f"📊 Summary for {summary['ticker']} ({summary['timeframe']}):")
        print(f"   Data points: {summary['data_points']}")
        print(f"   Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"   Available metrics: {len(summary['available_metrics'])}")

        print("\n📈 Key metrics statistics:")
        key_metrics = ["rsi_14", "macd", "sma_20"]
        for metric in key_metrics:
            if metric in summary["metrics_stats"]:
                stats = summary["metrics_stats"][metric]
                print(f"   {metric}:")
                print(f"     Mean: {stats['mean']:.2f}")
                print(f"     Std: {stats['std']:.2f}")
                print(f"     Min: {stats['min']:.2f}")
                print(f"     Max: {stats['max']:.2f}")
                print(f"     Last: {stats['last_value']:.2f}")
    else:
        print(f"   {summary['error']}")

    cache.close()


def demo_batch_metrics_population():
    """Demonstrate populating metrics for multiple tickers."""
    print("\n📊 Batch Metrics Population")
    print("-" * 32)

    cache = TimeSeriesCache()

    # List of tickers to process
    tickers = ["AAPL", "GOOGL", "MSFT", "TSLA"]

    # First, store price data for all tickers
    print("💾 Storing price data for multiple tickers...")
    for ticker in tickers:
        price_data = generate_sample_price_data(ticker, days=30)
        cache.store_price_data(ticker, price_data)
        print(f"   ✅ Stored price data for {ticker}")

    # Calculate metrics for all tickers
    print("\n🧮 Calculating metrics for all tickers...")
    cache.populate_metrics_table(tickers, timeframe="1d", indicators=["sma_20", "rsi_14", "macd", "volatility_20"])

    # Show database statistics
    print("\n📊 Database Statistics:")
    info = cache.get_database_info()
    if info:
        print(f"   Price data points: {info.get('price_data_points', 0)}")
        print(f"   Signals points: {info.get('signals_points', 0)}")
        print(f"   Metrics points: {info.get('metrics_points', 0)}")
        print(f"   Total data points: {info.get('total_data_points', 0)}")

    cache.close()


def demo_custom_indicators():
    """Demonstrate calculating custom indicators."""
    print("\n📊 Custom Indicators Example")
    print("-" * 32)

    cache = TimeSeriesCache()
    ticker = "AAPL"
    timeframe = "1d"

    # Calculate only specific indicators
    custom_indicators = ["sma_20", "sma_50", "rsi_14", "volatility_20"]

    print(f"🎯 Calculating custom indicators for {ticker}: {custom_indicators}")

    cache.calculate_and_store_metrics(ticker=ticker, timeframe=timeframe, indicators=custom_indicators)

    # Retrieve and display the custom indicators
    metrics_df = cache.get_metrics(ticker, timeframe, metrics=custom_indicators)

    if not metrics_df.empty:
        print("\n📋 Custom indicators data:")
        print(metrics_df.tail(3))

    cache.close()


def main():
    """Run all metrics demonstrations."""
    print("🚀 Time Series Cache Metrics Demo")
    print("=" * 45)
    print("This demo shows how to calculate and store technical indicators")
    print("and metrics from price data using InfluxDB.")
    print()

    try:
        # Run demonstrations
        demo_price_data_storage()
        demo_metrics_calculation()
        demo_metrics_retrieval()
        demo_metrics_summary()
        demo_batch_metrics_population()
        demo_custom_indicators()

        print("\n🎉 Metrics demo completed successfully!")
        print("\n💡 Key Features Demonstrated:")
        print("   • Automatic calculation of technical indicators")
        print("   • Storage and retrieval of metrics data")
        print("   • Batch processing for multiple tickers")
        print("   • Custom indicator selection")
        print("   • Metrics summary statistics")
        print("   • Efficient time series queries")

        print("\n📚 Available Indicators:")
        indicators = [
            "sma_20, sma_50 - Simple Moving Averages",
            "ema_12, ema_26 - Exponential Moving Averages",
            "rsi_14 - Relative Strength Index",
            "macd, macd_signal, macd_hist - MACD indicators",
            "bb_upper, bb_middle, bb_lower - Bollinger Bands",
            "volatility_20 - 20-day volatility",
            "returns, cumulative_returns - Return calculations",
        ]
        for indicator in indicators:
            print(f"   • {indicator}")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Make sure InfluxDB is running and accessible.")


if __name__ == "__main__":
    main()
