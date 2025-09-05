## Secure API Key Storage

**Never hardcode API keys in your scripts.**

TradingViewAlgoDev supports secure API key management using a `.env` file in your project root. This keeps your keys out of source code and version control.

### How to Use

1. Copy `.env.example` to `.env` in your project root:

	```bash
	cp .env.example .env
	```

2. Edit `.env` and add your API keys:

	```env
	FINNHUB_API_KEY=your_finnhub_key
	FMP_API_KEY=your_fmp_key
	ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
	TWELVE_DATA_API_KEY=your_twelve_data_key
	```

3. The code will automatically load the correct key for each data provider. You can also set these as system environment variables if you prefer.

4. **Do not commit your `.env` file to version control.**

### Accessing API Keys in Python

Use the provided helper:

```python
from tradingview_algo.secure_api import get_api_key

key = get_api_key("finnhub")  # or "fmp", "alpha_vantage", "twelve_data"
```

The live data module will use these keys automatically if not set in your YAML config.


# TradingViewAlgoDev

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
