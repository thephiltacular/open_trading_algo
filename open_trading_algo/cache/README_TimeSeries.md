# Time Series Database Cache

This directory contains an alternative caching implementation using **InfluxDB**, a high-performance time series database optimized for financial data storage and analytics.

## Overview

The `TimeSeriesCache` class provides:

- **Optimized Time Series Storage**: InfluxDB is specifically designed for time series data with automatic compression and efficient querying
- **SQL-like Queries**: Use Flux query language for complex analytics and aggregations
- **High Performance**: Fast queries for OHLCV data and trading signals
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

### 3. Run Demo

See the cache in action with sample data:

```bash
python examples/timeseries_cache_demo.py
```

### 4. Run Tests

Verify the implementation with comprehensive tests:

```bash
python -m pytest tests/test_timeseries_cache.py -v
```

### 5. Migrate Existing Data

Migrate data from SQLite to InfluxDB:

```bash
python scripts/migrate_to_timeseries.py
```

See the cache in action with sample data:

```bash
python demo_timeseries_cache.py
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

1. **Data Migration**: Migrate existing SQLite data to InfluxDB
2. **Retention Policies**: Configure data retention based on your needs
3. **Backup Strategy**: Set up regular backups of InfluxDB data
4. **Monitoring**: Monitor query performance and resource usage
5. **High Availability**: Consider clustering for production use
