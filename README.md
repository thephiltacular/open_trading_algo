# TradingViewAlgoDev

A comprehensive Python library for algorithmic trading, technical analysis, and financial data processing. Built for performance, reliability, and ease of use in both research and production environments.

## 🚀 Quick Links

- **[📚 Complete Documentation](docs/README.md)** - Comprehensive guides and API reference
- **[⚡ Quick Start Guide](docs/quickstart.md)** - Get up and running in minutes
- **[🔧 Installation & Setup](docs/installation.md)** - Installation instructions and configuration
- **[📊 Data APIs Guide](docs/data-apis.md)** - Multi-source financial data fetching
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

### ⚖️ Risk Management
- **Dynamic position sizing** based on volatility
- **Automated stop-loss** and take-profit levels
- **Portfolio-level risk controls**
- **Correlation-based hedging** strategies

### 🔄 Live Trading Ready
- **Real-time data feeds** with configurable intervals
- **Event-driven processing** for low-latency signals
- **Production logging** and monitoring
- **Thread-safe operations** for concurrent processing

## Quick Start

### Installation

```bash
git clone https://github.com/thephiltacular/TradingViewAlgoDev.git
cd TradingViewAlgoDev
pip install -e .
```

### Basic Usage

```python
from tradingview_algo.fin_data_apis.fetchers import fetch_yahoo
from tradingview_algo.indicators.indicators import calculate_rsi
from tradingview_algo.indicators.long_signals import rsi_oversold_signal

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

## Architecture Overview

```
TradingViewAlgoDev/
├── 📊 fin_data_apis/     # Multi-source data integration
│   ├── fetchers.py       # Unified data fetching interface
│   ├── rate_limit.py     # Automatic rate limiting
│   └── [7 API modules]   # Individual data source integrations
├── 📈 indicators/        # Technical analysis library
│   ├── indicators.py     # 50+ technical indicators
│   ├── long_signals.py   # Long position signals
│   ├── short_signals.py  # Short position signals
│   └── options_signals.py # Options trading signals
├── 💾 cache/            # High-performance local storage
├── 🎯 sentiment/        # Sentiment analysis integration
├── ⚖️ risk_management.py # Position sizing and risk controls
└── 🔄 signal_optimizer.py # Multi-signal optimization
```

## Automated Local Database Setup

TradingViewAlgoDev uses a persistent, efficient SQLite database to cache and store all financial data locally. This ensures:
- Minimal requests to external APIs (prevents throttling)
- Fast repeated access to historical and live data
- Data is retained even if the host is shut down

### How It Works

* On first use, the database is automatically created in the project directory as `tradingview_algo/tv_data_cache.sqlite3`.
* All data fetches (live, enrichment, backtest) check the local database first. Only missing data is requested from yfinance or other sources.
* New data is automatically stored in the database for future use.
* The database is managed by the `DataCache` class in `tradingview_algo/data_cache.py`.

### Custom Database Path

If you want to use a custom database location or a remote database, create a `db_config.yaml` file in the project root:

```yaml
db_path: /path/to/your/database.sqlite3
```

All modules will use this path automatically if the config file is present.

### Automated Setup

The database is set up automatically on first use. To manually initialize or verify the database, run:

```bash
python scripts/setup_db.py
```

This will create the database and all required tables if they do not exist.


### Signal Caching: Avoid Recomputing Signals

TradingViewAlgoDev caches all computed signals (long, short, options, sentiment) for each ticker, timeframe, and signal type. This means:
- Signals are only computed once per unique (ticker, timeframe, signal_type) combination.
- All signal modules (`long_signals.py`, `short_signals.py`, `options_signals.py`, `sentiment_signals.py`) are integrated with the cache.
- On repeated runs, signals are loaded instantly from the database.

#### Example Usage

```python
from tradingview_algo.data_cache import DataCache
import pandas as pd

# Suppose you have a DataFrame `signals_df` with datetime index and a 'signal_value' column
signals_df = pd.DataFrame(
    {"signal_value": [1, 0, 1]},
    index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
)

cache = DataCache()
cache.store_signals("AAPL", "1d", "long_trend", signals_df)

# Retrieve cached signals
df = cache.get_signals("AAPL", "1d", "long_trend")
print(df)

# Check if signals are cached
exists = cache.has_signals("AAPL", "1d", "long_trend")
print("Signals cached:", exists)
```

#### In Signal Modules

Each signal module provides a `compute_and_cache_*_signals` function:

```python
from tradingview_algo.long_signals import compute_and_cache_long_signals

signals_df = compute_and_cache_long_signals("AAPL", price_df, "1d")
```

This pattern is used for all signal types (long, short, options, sentiment).

### Notes

- The database is SQLite for maximum portability and zero setup. For advanced users, you can point `db_path` to a remote or cloud SQLite file.
- If you ever want to reset the cache, simply delete the `.sqlite3` file and it will be recreated on next use.
