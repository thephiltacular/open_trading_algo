# Data Cache System

The open_trading_algo cache system provides persistent local storage for financial data, signals, and computed indicators. It uses SQLite for zero-configuration setup while offering enterprise-grade performance and reliability.

## Overview

The cache system is designed to:
- **Minimize API calls** by storing data locally
- **Accelerate repeated analysis** with instant data retrieval
- **Persist signals** across application restarts
- **Handle large datasets** efficiently with indexed storage
- **Provide thread-safe access** for concurrent operations

## Basic Usage

### Initialize Cache

```python
from open_trading_algo.cache.data_cache import DataCache

# Use default database location
cache = DataCache()

# Or specify custom location
cache = DataCache(db_path="/path/to/custom/database.db")
```

### Store and Retrieve Price Data

```python
import yfinance as yf
import pandas as pd

# Get historical data
ticker = yf.Ticker("AAPL")
df = ticker.history(period="1y")

# Store in cache
cache.store_price_data("AAPL", df)

# Retrieve from cache
cached_df = cache.get_price_data("AAPL")
print(f"Cached data shape: {cached_df.shape}")
print(f"Date range: {cached_df.index[0]} to {cached_df.index[-1]}")

# Check if data exists
has_data = cache.has_price_data("AAPL")
print(f"Has AAPL data: {has_data}")
```

### Store and Retrieve Signals

```python
from open_trading_algo.indicators.indicators import calculate_rsi

# Calculate RSI signals
rsi = calculate_rsi(df["Close"])
rsi_oversold = (rsi < 30).astype(int)

# Create signals DataFrame
signals_df = pd.DataFrame({"signal": rsi_oversold, "rsi_value": rsi}, index=df.index)

# Store signals
cache.store_signals("AAPL", "1d", "rsi_oversold", signals_df)

# Retrieve signals
cached_signals = cache.get_signals("AAPL", "1d", "rsi_oversold")
print(f"Signal count: {cached_signals['signal'].sum()}")

# Check if signals exist
has_signals = cache.has_signals("AAPL", "1d", "rsi_oversold")
print(f"Has RSI signals: {has_signals}")
```

## Advanced Usage

### Smart Caching with Auto-Update

```python
import pandas as pd
from datetime import datetime, timedelta


class SmartCache:
    """Intelligent cache that auto-updates stale data"""

    def __init__(self, max_age_hours=4):
        self.cache = DataCache()
        self.max_age = timedelta(hours=max_age_hours)

    def get_fresh_price_data(self, ticker):
        """Get price data, updating if stale"""

        # Check if we have cached data
        if self.cache.has_price_data(ticker):
            df = self.cache.get_price_data(ticker)

            # Check if data is fresh enough
            latest_date = df.index[-1]
            now = pd.Timestamp.now(tz=latest_date.tz)

            if now - latest_date < self.max_age:
                print(f"Using cached data for {ticker}")
                return df
            else:
                print(f"Data for {ticker} is stale, updating...")

        # Fetch fresh data
        import yfinance as yf

        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")

        # Update cache
        self.cache.store_price_data(ticker, df)

        return df

    def get_computed_indicator(self, ticker, indicator_name, compute_func):
        """Get indicator, computing if not cached"""

        signal_key = f"{indicator_name}_computed"

        if self.cache.has_signals(ticker, "1d", signal_key):
            print(f"Using cached {indicator_name} for {ticker}")
            return self.cache.get_signals(ticker, "1d", signal_key)

        # Compute indicator
        print(f"Computing {indicator_name} for {ticker}")
        df = self.get_fresh_price_data(ticker)
        indicator_data = compute_func(df)

        # Store in cache
        self.cache.store_signals(ticker, "1d", signal_key, indicator_data)

        return indicator_data


# Usage
smart_cache = SmartCache(max_age_hours=2)

# This will fetch fresh data if cache is empty or stale
df = smart_cache.get_fresh_price_data("AAPL")

# This will compute RSI only if not cached
def compute_rsi_signals(df):
    from open_trading_algo.indicators.indicators import calculate_rsi

    rsi = calculate_rsi(df["Close"])
    return pd.DataFrame({"rsi": rsi}, index=df.index)


rsi_data = smart_cache.get_computed_indicator("AAPL", "rsi", compute_rsi_signals)
```

### Batch Operations

```python
class BatchCacheManager:
    """Efficient batch operations for multiple tickers"""

    def __init__(self):
        self.cache = DataCache()

    def batch_store_price_data(self, ticker_data_dict):
        """Store price data for multiple tickers efficiently"""

        for ticker, df in ticker_data_dict.items():
            print(f"Storing {ticker}: {len(df)} rows")
            self.cache.store_price_data(ticker, df)

        print(f"Stored data for {len(ticker_data_dict)} tickers")

    def batch_compute_signals(self, tickers, signal_name, compute_func):
        """Compute signals for multiple tickers"""

        results = {}

        for ticker in tickers:
            try:
                # Check cache first
                if self.cache.has_signals(ticker, "1d", signal_name):
                    print(f"Using cached {signal_name} for {ticker}")
                    signals = self.cache.get_signals(ticker, "1d", signal_name)
                else:
                    # Get price data
                    df = self.cache.get_price_data(ticker)
                    if df.empty:
                        print(f"No price data for {ticker}, skipping")
                        continue

                    # Compute signals
                    print(f"Computing {signal_name} for {ticker}")
                    signals = compute_func(df)

                    # Store in cache
                    self.cache.store_signals(ticker, "1d", signal_name, signals)

                results[ticker] = signals

            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue

        return results

    def get_cache_summary(self, tickers):
        """Get summary of cached data"""

        summary = {}

        for ticker in tickers:
            summary[ticker] = {
                "has_price_data": self.cache.has_price_data(ticker),
                "signals": [],
            }

            if summary[ticker]["has_price_data"]:
                df = self.cache.get_price_data(ticker)
                summary[ticker]["price_data_rows"] = len(df)
                summary[ticker]["date_range"] = f"{df.index[0]} to {df.index[-1]}"

        return summary


# Usage
batch_manager = BatchCacheManager()

# Example: Batch process a watchlist
watchlist = ["AAPL", "GOOGL", "MSFT", "TSLA"]

# Define signal computation function
def compute_macd_signals(df):
    from open_trading_algo.indicators.indicators import calculate_macd

    macd_line, macd_signal, macd_hist = calculate_macd(df["Close"])

    # Generate buy/sell signals
    buy_signal = (
        (macd_line > macd_signal) & (macd_line.shift(1) <= macd_signal.shift(1))
    ).astype(int)

    return pd.DataFrame(
        {
            "macd_line": macd_line,
            "macd_signal": macd_signal,
            "macd_histogram": macd_hist,
            "buy_signal": buy_signal,
        },
        index=df.index,
    )


# Batch compute MACD signals
macd_results = batch_manager.batch_compute_signals(
    watchlist, "macd_signals", compute_macd_signals
)

# Get cache summary
summary = batch_manager.get_cache_summary(watchlist)
for ticker, info in summary.items():
    print(f"{ticker}: {info}")
```

## Database Management

### Custom Database Configuration

```python
# Create db_config.yaml
db_config = """
db_path: /data/trading/cache.db
connection_pool_size: 20
timeout_seconds: 30
enable_wal_mode: true
"""

with open("db_config.yaml", "w") as f:
    f.write(db_config)

# Cache will automatically use this configuration
cache = DataCache()
```

### Database Maintenance

```python
class CacheMaintenanceManager:
    """Tools for database maintenance and optimization"""

    def __init__(self):
        self.cache = DataCache()

    def vacuum_database(self):
        """Optimize database storage"""

        import sqlite3

        conn = sqlite3.connect(self.cache.db_path)
        cursor = conn.cursor()

        print("Vacuuming database...")
        cursor.execute("VACUUM")

        conn.close()
        print("Database optimization complete")

    def get_database_stats(self):
        """Get database size and table statistics"""

        import sqlite3
        import os

        # File size
        db_size_mb = os.path.getsize(self.cache.db_path) / (1024 * 1024)

        conn = sqlite3.connect(self.cache.db_path)
        cursor = conn.cursor()

        # Table statistics
        tables = ["price_data", "signals"]
        stats = {"database_size_mb": db_size_mb}

        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]
            stats[f"{table}_rows"] = row_count

        conn.close()
        return stats

    def cleanup_old_data(self, days_to_keep=365):
        """Remove data older than specified days"""

        import sqlite3
        from datetime import datetime, timedelta

        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        conn = sqlite3.connect(self.cache.db_path)
        cursor = conn.cursor()

        # Clean price data
        cursor.execute(
            """
            DELETE FROM price_data
            WHERE date < ?
        """,
            (cutoff_date.isoformat(),),
        )

        price_deleted = cursor.rowcount

        # Clean signals
        cursor.execute(
            """
            DELETE FROM signals
            WHERE date < ?
        """,
            (cutoff_date.isoformat(),),
        )

        signals_deleted = cursor.rowcount

        conn.commit()
        conn.close()

        print(f"Deleted {price_deleted} price rows, {signals_deleted} signal rows")
        return price_deleted, signals_deleted


# Usage
maintenance = CacheMaintenanceManager()

# Get stats
stats = maintenance.get_database_stats()
print(f"Database size: {stats['database_size_mb']:.2f} MB")
print(f"Price data rows: {stats['price_data_rows']:,}")
print(f"Signal rows: {stats['signals_rows']:,}")

# Cleanup old data (keep last year)
maintenance.cleanup_old_data(days_to_keep=365)

# Optimize
maintenance.vacuum_database()
```

## Performance Optimization

### Indexing Strategy

```python
class OptimizedDataCache(DataCache):
    """Enhanced cache with optimized indexing"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_additional_indexes()

    def _create_additional_indexes(self):
        """Create additional indexes for performance"""

        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Additional indexes for common query patterns
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_price_ticker_date ON price_data(ticker, date DESC)",
            "CREATE INDEX IF NOT EXISTS idx_signals_ticker_type_timeframe ON signals(ticker, signal_type, timeframe)",
            "CREATE INDEX IF NOT EXISTS idx_signals_date ON signals(date DESC)",
        ]

        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
            except sqlite3.Error as e:
                print(f"Index creation warning: {e}")

        conn.commit()
        conn.close()

    def bulk_insert_price_data(self, ticker_data_list):
        """Efficient bulk insert for multiple tickers"""

        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Prepare bulk insert data
        insert_data = []
        for ticker, df in ticker_data_list:
            for date, row in df.iterrows():
                insert_data.append(
                    (
                        ticker,
                        date.isoformat(),
                        row["Open"],
                        row["High"],
                        row["Low"],
                        row["Close"],
                        row["Volume"],
                    )
                )

        # Bulk insert
        cursor.executemany(
            """
            INSERT OR REPLACE INTO price_data
            (ticker, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            insert_data,
        )

        conn.commit()
        conn.close()

        print(f"Bulk inserted {len(insert_data)} price records")


# Usage with optimized cache
opt_cache = OptimizedDataCache()

# Bulk operations are much faster
ticker_data_pairs = [
    ("AAPL", yf.Ticker("AAPL").history(period="1y")),
    ("GOOGL", yf.Ticker("GOOGL").history(period="1y")),
    ("MSFT", yf.Ticker("MSFT").history(period="1y")),
]

opt_cache.bulk_insert_price_data(ticker_data_pairs)
```

### Memory-Efficient Operations

```python
class StreamingCacheProcessor:
    """Process large datasets without loading everything into memory"""

    def __init__(self):
        self.cache = DataCache()

    def process_large_timeframe(self, ticker, start_date, end_date, chunk_months=3):
        """Process large date ranges in chunks"""

        import pandas as pd
        from dateutil.relativedelta import relativedelta

        current_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

        results = []

        while current_date < end_date:
            # Calculate chunk end date
            chunk_end = min(current_date + relativedelta(months=chunk_months), end_date)

            print(f"Processing {ticker}: {current_date.date()} to {chunk_end.date()}")

            # Get data for chunk
            df = self.cache.get_price_data(ticker)
            chunk_df = df[(df.index >= current_date) & (df.index < chunk_end)]

            if not chunk_df.empty:
                # Process chunk (example: calculate returns)
                chunk_results = self._process_chunk(chunk_df)
                results.append(chunk_results)

            current_date = chunk_end

        # Combine results
        if results:
            return pd.concat(results)
        return pd.DataFrame()

    def _process_chunk(self, df):
        """Process a single chunk of data"""

        # Example: Calculate rolling statistics
        return pd.DataFrame(
            {
                "returns": df["Close"].pct_change(),
                "volatility": df["Close"].pct_change().rolling(20).std(),
                "volume_ma": df["Volume"].rolling(20).mean(),
            },
            index=df.index,
        )


# Usage for large datasets
processor = StreamingCacheProcessor()
large_results = processor.process_large_timeframe(
    "AAPL", start_date="2020-01-01", end_date="2024-01-01", chunk_months=6
)
```

## Integration with Other Modules

### Automatic Caching in Signal Generation

```python
from open_trading_algo.indicators.long_signals import compute_and_cache_long_signals
from open_trading_algo.indicators.short_signals import compute_and_cache_short_signals

# These functions automatically use the cache
long_signals = compute_and_cache_long_signals("AAPL", df, "1d")
short_signals = compute_and_cache_short_signals("AAPL", df, "1d")

print(f"Long signals: {long_signals['signal'].sum()}")
print(f"Short signals: {short_signals['signal'].sum()}")
```

### Cache-Aware Data Fetching

```python
from open_trading_algo.fin_data_apis.fetchers import fetch_yahoo

# Fetchers automatically use cache when available
data = fetch_yahoo(["AAPL"], ["price", "volume"], cache=cache)
print(f"AAPL: ${data['AAPL']['price']:.2f}")
```

## Best Practices

### 1. Cache Warming
```python
def warm_cache_for_watchlist(tickers):
    """Pre-populate cache with essential data"""

    import yfinance as yf

    cache = DataCache()

    for ticker in tickers:
        if not cache.has_price_data(ticker):
            print(f"Warming cache for {ticker}")
            stock = yf.Ticker(ticker)
            df = stock.history(period="2y")
            cache.store_price_data(ticker, df)
        else:
            print(f"Cache already warm for {ticker}")


# Pre-populate cache
watchlist = ["AAPL", "GOOGL", "MSFT", "TSLA"]
warm_cache_for_watchlist(watchlist)
```

### 2. Cache Validation
```python
def validate_cache_integrity():
    """Check cache for data quality issues"""

    cache = DataCache()
    issues = []

    # Example validations
    tickers = ["AAPL", "GOOGL", "MSFT"]

    for ticker in tickers:
        if cache.has_price_data(ticker):
            df = cache.get_price_data(ticker)

            # Check for missing data
            if df.isnull().any().any():
                issues.append(f"{ticker}: Has null values")

            # Check for reasonable price ranges
            if (df["Close"] <= 0).any():
                issues.append(f"{ticker}: Has non-positive prices")

            # Check for data gaps
            date_diff = df.index.to_series().diff()
            max_gap = date_diff.max().days
            if max_gap > 7:  # More than a week gap
                issues.append(f"{ticker}: Has {max_gap} day gap in data")

    return issues


# Validate cache
issues = validate_cache_integrity()
if issues:
    print("Cache issues found:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("Cache integrity OK")
```

## Next Steps

- [Technical Indicators](indicators.md) - Use cached data for analysis
- [Signal Generation](signals.md) - Cache and retrieve trading signals
- [Live Data Feeds](live-data.md) - Integrate caching with real-time data
