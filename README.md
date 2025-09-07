# open_trading_algo

A comprehensive Python library for algorithmic trading, technical analysis, and financial data processing. Built for performance, reliability, and ease of use in both research and production environments.

The overall goal is to get as much value as possible without paying for api access ;)
This means managing API query rates, storing as much data as possible locally, and leveraging multiple APIs to get the data we need.

## 🚀 Quick Links

- **[📚 Complete Documentation](docs/README.md)** - Comprehensive guides and API reference
- **[⚡ Quick Start Guide](docs/quickstart.md)** - Get up and running in minutes
- **[🔧 Installation & Setup](docs/installation.md)** - Installation instructions and configuration
- **[📊 Data APIs Guide](docs/data-apis.md)** - Multi-source financial data fetching
- **[📈 Technical Indicators](docs/indicators.md)** - Complete guide to 50+ indicators with charts and accuracy data
- **[📊 Trading Metrics](docs/metrics.md)** - Comprehensive metrics for risk, performance, and analysis
- **[💾 Cache System](docs/data-cache.md)** - Local data storage and optimization

## Key Features

### 🎯 Multi-Source Data Integration
- **7+ data providers**: Yahoo Finance, Finnhub, Alpha Vantage, FMP, Twelve Data, Polygon, Tiingo
- **Automatic rate limiting** and error handling
- **Smart failover** between data sources
- **Local caching** for performance and cost reduction

### 📈 Advanced Technical Analysis
- **50+ technical indicators**: RSI, MACD, Bollinger Bands, ADX, Stochastic, Williams %R, etc.
- **Custom indicators**: Fibonacci retracements, volume profiles, market breadth
- **Multi-timeframe analysis** support
- **Signal aggregation** and optimization

### 🎯 Intelligent Signal Generation
- **Long/short equity signals** with multiple strategies
- **Options trading signals** for calls and puts
- **Sentiment-based signals** from social media and analyst ratings
- **Machine learning ensemble** methods
- **Modular trading models** combining indicators and strategies

### ⚖️ Risk Management
- **Dynamic position sizing** based on volatility
- **Automated stop-loss** and take-profit levels
- **Portfolio-level risk controls**
- **Correlation-based hedging** strategies

### 🔄 Live Trading Ready (WIP - this is a future goal)
- **Real-time data feeds** with configurable intervals
- **Event-driven processing** for low-latency signals
- **Production logging** and monitoring
- **Thread-safe operations** for concurrent processing

## Quick Start

### Installation

```bash
git clone https://github.com/thephiltacular/open_trading_algo.git
cd open_trading_algo
pip install -e .
```

### Basic Usage

```python
from open_trading_algo.fin_data_apis.fetchers import fetch_yahoo
from open_trading_algo.indicators.indicators import calculate_rsi
from open_trading_algo.indicators.long_signals import rsi_oversold_signal

# Fetch current market data
data = fetch_yahoo(["AAPL", "GOOGL"], ["price", "volume"])
print(f"AAPL: ${data['AAPL']['price']:.2f}")

# Technical analysis with historical data
import yfinance as yf

df = yf.Ticker("AAPL").history(period="6mo")

# Calculate RSI and generate signals
rsi = calculate_rsi(df["Close"])
signals = rsi_oversold_signal(df["Close"])

print(f"Current RSI: {rsi.iloc[-1]:.2f}")
print(f"Active signals: {signals.sum()}")
```

## 📊 Trading Metrics

Calculate comprehensive risk and performance metrics:

```python
from open_trading_algo.indicators.metrics import (
    compute_sharpe_ratio, compute_max_drawdown,
    compute_volatility_ratio, compute_vwap
)

# Risk and performance analysis
sharpe = compute_sharpe_ratio(returns_df)
max_dd = compute_max_drawdown(price_df)
vol_ratio = compute_volatility_ratio(price_df)
vwap = compute_vwap(price_df)

print(f"Sharpe Ratio: {sharpe.iloc[-1]:.3f}")
print(f"Max Drawdown: {max_dd.iloc[-1]:.3f}")
print(f"Volatility Ratio: {vol_ratio.iloc[-1]:.3f}")
print(f"VWAP: ${vwap.iloc[-1]:.2f}")
```

## 🏗️ Architecture

### Project Structure

```
open_trading_algo/
├── 📊 fin_data_apis/     # Multi-source data integration
│   ├── fetchers.py       # Unified data fetching interface
│   ├── rate_limit.py     # Automatic rate limiting
│   └── [7 API modules]   # Individual data source integrations
├── 📈 indicators/        # Technical analysis library
│   ├── indicators.py     # 50+ technical indicators
│   ├── metrics.py        # 24 trading metrics (Sharpe, drawdown, etc.)
│   ├── long_signals.py   # Long position signals
│   ├── short_signals.py  # Short position signals
│   └── options_signals.py # Options trading signals
├── 🤖 models/            # Trading strategy models
│   ├── base_model.py     # Abstract base class for all models
│   ├── momentum_model.py # Momentum-based strategies
│   ├── mean_reversion_model.py # Mean reversion strategies
│   └── trend_following_model.py # Trend following strategies
├── 💾 cache/            # High-performance local storage
├── 🎯 sentiment/        # Sentiment analysis integration
├── ⚖️ risk_management.py # Position sizing and risk controls
└── 🔄 signal_optimizer.py # Multi-signal optimization
```

### Core Modules

- **`fin_data_apis/`**: Multi-source financial data fetching with rate limiting
- **`indicators/`**: 50+ technical indicators and 24 trading metrics
- **`models/`**: Trading strategy models and machine learning algorithms
- **`cache/`**: High-performance time series database (InfluxDB) with automated technical indicator calculations
- **`backtest/`**: Historical strategy testing and Monte Carlo simulation
- **`sentiment/`**: Social media and analyst sentiment analysis
- **`alerts/`**: Real-time signal notifications and alerts
- **`live/`**: Real-time data streaming and event processing



### Signal Caching: Avoid Recomputing Signals

open_trading_algo caches all computed signals (long, short, options, sentiment) for each ticker, timeframe, and signal type. This means:
- Signals are only computed once per unique (ticker, timeframe, signal_type) combination.
- All signal modules (`long_signals.py`, `short_signals.py`, `options_signals.py`, `sentiment_signals.py`) are integrated with the cache.
- On repeated runs, signals are loaded instantly from the database.

### Signal Generation Pipeline

```python
# 1. Fetch and cache data
from open_trading_algo.cache.data_cache import DataCache
cache = DataCache()
cache.store_ohlcv("AAPL", "1d", df)

# 2. Generate signals
from open_trading_algo.indicators.long_signals import compute_and_cache_long_signals
signals_df = compute_and_cache_long_signals("AAPL", df, "1d")

# 3. Cache signals for reuse
cache.store_signals("AAPL", "1d", "long_trend", signals_df)

# 4. Retrieve cached signals
df = cache.get_signals("AAPL", "1d", "long_trend")
print(df)
```

### Advanced Time Series Database (InfluxDB)

For high-performance time series data storage and analytics, open_trading_algo also supports **InfluxDB** as an alternative to SQLite:

#### Key Benefits:
- **Optimized for Time Series**: Columnar storage designed specifically for financial data
- **Automated Metrics Calculation**: Automatically calculate and store 15+ technical indicators from price data
- **Multi-Table Architecture**: Separate optimized tables for price data, signals, and calculated metrics
- **High Performance**: Fast queries for OHLCV data, trading signals, and technical indicators
- **Advanced Analytics**: Built-in aggregation functions and time-based queries
- **Scalable**: Handles large volumes of high-frequency financial data
- **SQL-like Queries**: Use Flux language for complex analytical queries

#### Quick Setup:

```bash
# 1. Install and start InfluxDB
python open_trading_algo/cache/setup_influxdb.py

# 2. Use the time series cache
from open_trading_algo.cache.timeseries_cache import TimeSeriesCache

cache = TimeSeriesCache()

# Store price data
cache.store_price_data('AAPL', ohlcv_df)

# Automatically calculate and store technical indicators
cache.calculate_and_store_metrics('AAPL', indicators=['sma_20', 'rsi_14', 'macd'])

# Retrieve metrics
metrics = cache.get_metrics('AAPL', indicators=['rsi_14', 'macd'])

# 3. Advanced queries
weekly_data = cache.get_aggregated_data('AAPL', aggregation='1w')
stats = cache.get_signal_stats('AAPL', '1d', 'momentum')
summary = cache.get_metrics_summary('AAPL')
```

#### Features:
- **Automatic Compression**: Efficient storage with built-in compression
- **Retention Policies**: Configurable data retention (default: 10 years for price data)
- **Real-time Analytics**: Aggregate data by time windows (hourly, daily, weekly)
- **Concurrent Access**: Optimized for multiple concurrent queries
- **Pandas Integration**: Seamless conversion to/from DataFrames

See the [Time Series Cache Documentation](open_trading_algo/cache/README_TimeSeries.md) for complete setup and usage instructions.

### Notes

- The default cache uses SQLite for maximum portability and zero setup
- InfluxDB provides superior performance for large datasets and complex queries
- Both cache systems maintain the same API for easy switching
- For advanced users, you can point `db_path` to a remote or cloud database

## 🤝 Contributing

We welcome contributions from the community! Please read our [Contributing Guide](CONTRIBUTING.md) for instructions on how to get started, code style, testing, and submitting pull requests.

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
