# Time Series Database Cache

This directory contains an alternative caching implementation using **InfluxDB**, a high-performance time series database optimized for financial data storage and analytics.

## Overview

The `TimeSeriesCache` class provides:
## Performance Benefits

Compared to SQLite:

1. **Query Performance**: InfluxDB uses columnar storage optimized for time series queries
2. **Automated Analytics**: Built-in calculation and storage of technical indicators
3. **Multi-Table Optimization**: Separate tables for raw data and derived metrics improve query performance
4. **Compression**: Automatic data compression reduces storage requirements
5. **Retention Policies**: Automatic data expiration and cleanup
6. **Concurrent Access**: Better handling of multiple concurrent queries
7. **Analytics**: Built-in aggregation and analytical functions
8. **Scalability**: Designed to handle high-frequency financial dataized Time Series Storage**: InfluxDB is specifically designed for time series data with automatic compression and efficient querying
- **Automated Metrics Calculation**: Automatically calculate and store 15+ technical indicators from price data
- **Multi-Table Architecture**: Separate tables for price data, signals, and calculated metrics for optimal performance
- **SQL-like Queries**: Use Flux query language for complex analytics and aggregations
- **High Performance**: Fast queries for OHLCV data, trading signals, and technical indicators
- **Scalable**: Handles large volumes of financial data efficiently
- **Pandas Integration**: Seamless conversion to/from pandas DataFrames and Series

## Quick Start

### 1. Setup InfluxDB

Run the setup script to start InfluxDB locally:

```bash
python setup_influxdb.py
```

This will:
- Start an InfluxDB container using Docker
- Configure the database with default settings
- Create a configuration file
- Test the connection

### 2. Basic Usage

```python
from open_trading_algo.cache.timeseries_cache import TimeSeriesCache

# Initialize cache
cache = TimeSeriesCache()

# Store OHLCV data
cache.store_price_data('AAPL', ohlcv_dataframe)

# Retrieve data
data = cache.get_price_data('AAPL', start='2023-01-01', end='2023-12-31')

# Store signals
cache.store_signals('AAPL', '1d', 'momentum', signals_dataframe)

# Retrieve signals
signals = cache.get_signals('AAPL', '1d', 'momentum')

# Close connection
cache.close()
```

### Metrics and Indicators Usage

```python
# Calculate and store technical indicators
cache.calculate_and_store_metrics('AAPL', indicators=['sma_20', 'rsi_14', 'macd'])

# Retrieve specific metrics
metrics = cache.get_metrics('AAPL', indicators=['sma_20', 'rsi_14'])

# Get all metrics for a ticker
all_metrics = cache.get_metrics('AAPL')

# Batch process metrics for multiple tickers
cache.populate_metrics_table(['AAPL', 'GOOGL', 'MSFT'])

# Get metrics summary
summary = cache.get_metrics_summary('AAPL')
```

### 3. Run Demo

See the cache in action with sample data:

```bash
python examples/timeseries_cache_demo.py
```

## Configuration

The cache can be configured via `config/timeseries_config.yaml`:

```yaml
influxdb:
  url: "http://localhost:8086"
  token: "my-token"
  org: "trading-org"
  bucket: "trading-data"

retention:
  price_data: 3650  # 10 years
  signals: 365      # 1 year

query:
  default_range: "-365d"
  max_points_per_query: 100000
```

## API Reference

### TimeSeriesCache Class

#### Constructor
```python
TimeSeriesCache(url="http://localhost:8086", token="my-token",
                org="trading-org", bucket="trading-data")
```

#### Price Data Methods

**store_price_data(ticker, df)**
- Store OHLCV data for a ticker
- `df`: DataFrame with OHLCV columns and datetime index

**get_price_data(ticker, start=None, end=None)**
- Retrieve OHLCV data for a ticker
- Returns DataFrame with datetime index

**has_data(ticker, start=None, end=None)**
- Check if data exists for a ticker
- Returns boolean

#### Signals Methods

**store_signals(ticker, timeframe, signal_type, df)**
- Store trading signals
- `timeframe`: e.g., '1d', '1h', '1w'
- `signal_type`: e.g., 'momentum', 'mean_reversion'
- `df`: DataFrame with 'signal_value' column and datetime index

**get_signals(ticker, timeframe, signal_type, start=None, end=None)**
- Retrieve signals for a ticker/timeframe/signal_type
- Returns DataFrame with datetime index and 'signal_value' column

**has_signals(ticker, timeframe, signal_type, start=None, end=None)**
- Check if signals exist
- Returns boolean

#### Metrics and Indicators Methods

**calculate_and_store_metrics(ticker, timeframe='1d', indicators=None, start=None, end=None)**
- Calculate technical indicators and store them in the metrics table
- `timeframe`: Timeframe for calculations (default: '1d')
- `indicators`: List of indicators to calculate (default: all available)
- Automatically calculates from existing price data

**get_metrics(ticker, timeframe='1d', indicators=None, start=None, end=None)**
- Retrieve calculated metrics for a ticker
- `timeframe`: Timeframe for the metrics (default: '1d')
- `indicators`: Filter by specific indicators
- Returns DataFrame with datetime index and indicator columns

**populate_metrics_table(tickers, timeframe='1d', start=None, end=None, indicators=None)**
- Batch process and populate metrics for multiple tickers
- `tickers`: List of ticker symbols to process
- `timeframe`: Timeframe for calculations (default: '1d')
- `indicators`: List of indicators to calculate

**get_metrics_summary(ticker, timeframe='1d', start=None, end=None)**
- Get summary statistics for all metrics
- `timeframe`: Timeframe for the metrics (default: '1d')
- Returns dictionary with indicator statistics

#### Advanced Queries

**get_aggregated_data(ticker, aggregation="1d", start=None, end=None)**
- Get aggregated OHLCV data
- `aggregation`: Time window (e.g., '1h', '1d', '1w')
- Returns aggregated DataFrame

**get_signal_stats(ticker, timeframe, signal_type, start=None, end=None)**
- Get statistics for signals
- Returns dictionary with signal statistics

**get_database_info()**
- Get database statistics and information
- Returns dictionary with database info

## Data Schema

### Price Data Measurement
- **Measurement**: `price_data`
- **Tags**: `ticker`
- **Fields**:
  - `open`: Opening price (float)
  - `high`: High price (float)
  - `low`: Low price (float)
  - `close`: Closing price (float)
  - `volume`: Trading volume (float)
- **Timestamp**: Date/time of the data point

### Signals Measurement
- **Measurement**: `signals`
- **Tags**:
  - `ticker`: Ticker symbol
  - `timeframe`: Timeframe (e.g., '1d', '1h')
  - `signal_type`: Type of signal (e.g., 'momentum')
- **Fields**:
  - `signal_value`: Signal value (-1, 0, 1, or continuous)
- **Timestamp**: Date/time of the signal

### Metrics Measurement
- **Measurement**: `metrics`
- **Tags**:
  - `ticker`: Ticker symbol
  - `timeframe`: Timeframe (e.g., '1d', '1h')
  - `indicator`: Indicator name (e.g., 'sma_20', 'rsi_14', 'macd')
- **Fields**:
  - `value`: Indicator value (float)
  - `signal`: Derived signal (-1, 0, 1) if applicable
- **Timestamp**: Date/time of the calculation

## Available Indicators

The system supports the following technical indicators:

### Trend Indicators
- **SMA (Simple Moving Average)**: `sma_20`, `sma_50`, `sma_200`
- **EMA (Exponential Moving Average)**: `ema_12`, `ema_26`, `ema_50`

### Momentum Indicators
- **RSI (Relative Strength Index)**: `rsi_14`
- **MACD (Moving Average Convergence Divergence)**: `macd`, `macd_signal`, `macd_histogram`

### Volatility Indicators
- **Bollinger Bands**: `bb_upper`, `bb_middle`, `bb_lower`, `bb_width`, `bb_percent`
- **Volatility**: `volatility_20`, `volatility_50`

### Price Action
- **Returns**: `daily_return`, `cumulative_return`
- **Price Changes**: `price_change`, `pct_change`

## Performance Benefits

Compared to SQLite:

1. **Query Performance**: InfluxDB uses columnar storage optimized for time series queries
2. **Compression**: Automatic data compression reduces storage requirements
3. **Retention Policies**: Automatic data expiration and cleanup
4. **Concurrent Access**: Better handling of multiple concurrent queries
5. **Analytics**: Built-in aggregation and analytical functions
6. **Scalability**: Designed to handle high-frequency financial data

## Integration with Existing Code

The `TimeSeriesCache` class maintains the same interface as the existing `DataCache` class, making it a drop-in replacement:

```python
# Existing code
from open_trading_algo.cache.data_cache import DataCache
cache = DataCache()

# New time series code
from open_trading_algo.cache.timeseries_cache import TimeSeriesCache
cache = TimeSeriesCache()
```

## Troubleshooting

### Connection Issues
- Ensure InfluxDB is running: `docker ps | grep influxdb`
- Check InfluxDB logs: `docker logs trading-influxdb`
- Verify configuration in `config/timeseries_config.yaml`

### Import Errors
- Install dependencies: `poetry install`
- Ensure Python path includes the project directory

### Performance Issues
- Check query date ranges (avoid very large ranges)
- Use appropriate aggregation windows
- Monitor InfluxDB resource usage

## Requirements

- **Docker**: For running InfluxDB locally
- **Python 3.8+**: Required for the InfluxDB client
- **Dependencies**: Listed in `pyproject.toml`
  - `influxdb-client>=1.40.0`
  - `pandas`
  - `pyarrow`

## Docker Commands

```bash
# Start InfluxDB
docker start trading-influxdb

# Stop InfluxDB
docker stop trading-influxdb

# View logs
docker logs trading-influxdb

# Access InfluxDB UI
open http://localhost:8086
```

## Next Steps

1. **Metrics Population**: Run metrics calculation for existing price data using `populate_metrics_table()`
2. **Data Migration**: Migrate existing SQLite data to InfluxDB
3. **Retention Policies**: Configure data retention based on your needs for price, signals, and metrics data
4. **Backup Strategy**: Set up regular backups of InfluxDB data
5. **Monitoring**: Monitor query performance and resource usage
6. **High Availability**: Consider clustering for production use
