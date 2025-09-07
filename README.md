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
├── 💾 cache/            # Multiple caching implementations
│   ├── data_cache.py     # SQLite-based cache (default)
│   ├── parquet_cache.py  # Parquet columnar storage
│   ├── timeseries_cache.py # InfluxDB time series database
│   ├── setup_influxdb.py # InfluxDB setup utilities
│   └── README_TimeSeries.md # Time series cache documentation
├── 🎯 sentiment/        # Sentiment analysis integration
├── ⚖️ risk_management.py # Position sizing and risk controls
└── 🔄 signal_optimizer.py # Multi-signal optimization
```

### Core Modules

- **`fin_data_apis/`**: Multi-source financial data fetching with rate limiting
- **`indicators/`**: 50+ technical indicators and 24 trading metrics
- **`models/`**: Trading strategy models and machine learning algorithms
- **`cache/`**: Multiple caching implementations (SQLite, Parquet, InfluxDB) for different performance and storage needs
- **`backtest/`**: Historical strategy testing and Monte Carlo simulation
- **`sentiment/`**: Social media and analyst sentiment analysis
- **`alerts/`**: Real-time signal notifications and alerts
- **`live/`**: Real-time data streaming and event processing



### Cache System

open_trading_algo provides **three different caching implementations** optimized for different use cases:

#### 1. **SQLite DataCache** (Default - Zero Configuration)
- **Best for**: Getting started quickly, development, small to medium datasets
- **Storage**: SQLite database with automatic table creation
- **Features**: OHLCV data, signals storage, thread-safe operations
- **Setup**: No additional dependencies required
- **Performance**: Fast for most use cases, excellent for repeated queries

```python
from open_trading_algo.cache.data_cache import DataCache

cache = DataCache()  # Uses default SQLite database
cache.store_price_data('AAPL', ohlcv_df)
cached_data = cache.get_price_data('AAPL')
```

#### 2. **Parquet Cache** (Columnar Storage)
- **Best for**: Analytical workloads, large datasets, research environments
- **Storage**: Apache Parquet files with partitioning by ticker
- **Features**: High compression, fast analytical queries, pandas integration
- **Setup**: Requires `pyarrow` package
- **Performance**: Superior for complex queries and aggregations

```python
from open_trading_algo.cache.parquet_cache import ParquetCache

cache = ParquetCache()  # Uses Parquet files
cache.store_price_data('AAPL', ohlcv_df)
cached_data = cache.get_price_data('AAPL')
```

#### 3. **InfluxDB Time Series Cache** (High Performance)
- **Best for**: Production systems, high-frequency data, real-time analytics
- **Storage**: InfluxDB time series database with automatic compression
- **Features**: Automated technical indicator calculation, advanced queries, retention policies
- **Setup**: Requires Docker and InfluxDB container
- **Performance**: Optimized for time series queries, handles millions of data points

```python
from open_trading_algo.cache.timeseries_cache import TimeSeriesCache

cache = TimeSeriesCache()  # Uses InfluxDB
cache.store_price_data('AAPL', ohlcv_df)

# Automatically calculate and store technical indicators
cache.calculate_and_store_metrics('AAPL', indicators=['sma_20', 'rsi_14', 'macd'])
metrics = cache.get_metrics('AAPL')
```

### Choosing the Right Cache

| Feature | SQLite Cache | Parquet Cache | InfluxDB Cache |
|---------|-------------|---------------|----------------|
| **Setup Complexity** | 🟢 None | 🟡 Low | 🔴 Medium |
| **Performance** | 🟢 Good | 🟡 Very Good | 🟢 Excellent |
| **Storage Efficiency** | 🟡 Good | 🟢 Excellent | 🟢 Excellent |
| **Query Flexibility** | 🟢 Good | 🟡 Very Good | 🟢 Excellent |
| **Time Series Features** | 🔴 Basic | 🟡 Good | 🟢 Excellent |
| **Technical Indicators** | 🔴 Manual | 🔴 Manual | 🟢 Automatic |
| **Best Use Case** | Development/Quick Start | Research/Analytics | Production/Real-time |

### Signal Caching: Avoid Recomputing Signals

All cache types support signal caching to avoid recomputing expensive calculations:
- Signals are only computed once per unique (ticker, timeframe, signal_type) combination
- All signal modules are integrated with the cache system
- On repeated runs, signals are loaded instantly from the database

### Signal Generation Pipeline

```python
# 1. Fetch and cache data
from open_trading_algo.cache.data_cache import DataCache
cache = DataCache()
cache.store_price_data("AAPL", df)

# 2. Generate signals
from open_trading_algo.indicators.long_signals import compute_and_cache_long_signals
signals_df = compute_and_cache_long_signals("AAPL", df, "1d")

# 3. Cache signals for reuse
cache.store_signals("AAPL", "1d", "long_trend", signals_df)

# 4. Retrieve cached signals
df = cache.get_signals("AAPL", "1d", "long_trend")
print(df)
```

### Cache Configuration

All cache types support configuration via `config/db_config.yaml`:

```yaml
# SQLite Cache Configuration
sqlite:
  db_path: "/path/to/custom/database.db"
  enable_caching: true

# Parquet Cache Configuration
parquet:
  cache_dir: "/path/to/parquet/cache"
  compression: "snappy"

# InfluxDB Cache Configuration
influxdb:
  url: "http://localhost:8086"
  token: "your-token"
  org: "trading-org"
  bucket: "trading-data"
```

### Cache Migration

You can easily switch between cache types without changing your application code:

```python
# Switch from SQLite to InfluxDB
from open_trading_algo.cache.timeseries_cache import TimeSeriesCache

# Your existing code works unchanged
cache = TimeSeriesCache()
cache.store_price_data('AAPL', ohlcv_df)
data = cache.get_price_data('AAPL')
```

See the [Cache System Documentation](docs/data-cache.md) for complete setup and usage instructions.

## 🤝 Contributing

We welcome contributions from the community! Please read our [Contributing Guide](CONTRIBUTING.md) for instructions on how to get started, code style, testing, and submitting pull requests.

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
