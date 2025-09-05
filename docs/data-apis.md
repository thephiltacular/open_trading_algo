# Data APIs & Fetchers

TradingViewAlgoDev provides unified access to multiple financial data providers through a consistent API. This module handles rate limiting, error handling, and data normalization across all supported sources.

## Supported Data Sources

| Provider | Free Tier | Strengths | Best For |
|----------|-----------|-----------|----------|
| **Yahoo Finance** | Unlimited | Real-time quotes, historical data | General market data, backtesting |
| **Finnhub** | 60/min, 1440/day | Real-time quotes, earnings | Live trading, fundamental data |
| **Alpha Vantage** | 5/min, 500/day | Technical indicators, forex | Research, alternative assets |
| **FMP** | 5/min, 250/day | Financial statements, ratios | Fundamental analysis |
| **Twelve Data** | 8/min, 800/day | Global markets, crypto | International markets |
| **Polygon** | 5/min | Options, forex, crypto | Options trading, derivatives |
| **Tiingo** | Variable | News, fundamentals | Research, news sentiment |

## Basic Usage

### Single Data Source

```python
from tradingview_algo.fin_data_apis.fetchers import fetch_yahoo

# Fetch current data
tickers = ["AAPL", "GOOGL", "MSFT"]
fields = ["price", "volume", "high", "low", "open", "previous_close"]

data = fetch_yahoo(tickers, fields)

# Access data
for ticker in tickers:
    price = data[ticker]["price"]
    volume = data[ticker]["volume"]
    print(f"{ticker}: ${price:.2f} (Vol: {volume:,})")
```

### Multiple Data Sources with Fallback

```python
from tradingview_algo.fin_data_apis.fetchers import (
    fetch_yahoo,
    fetch_finnhub,
    fetch_alpha_vantage,
)
from tradingview_algo.fin_data_apis.secure_api import get_api_key


def get_stock_data(ticker, fields):
    """Fetch data with fallback sources"""

    # Try Yahoo first (free, unlimited)
    try:
        data = fetch_yahoo([ticker], fields)
        if data[ticker]["price"] is not None:
            return data[ticker], "yahoo"
    except Exception as e:
        print(f"Yahoo failed: {e}")

    # Try Finnhub (requires API key)
    try:
        api_key = get_api_key("finnhub")
        if api_key:
            data = fetch_finnhub([ticker], fields, api_key)
            if data[ticker]["price"] is not None:
                return data[ticker], "finnhub"
    except Exception as e:
        print(f"Finnhub failed: {e}")

    # Try Alpha Vantage (requires API key)
    try:
        api_key = get_api_key("alpha_vantage")
        if api_key:
            data = fetch_alpha_vantage([ticker], fields, api_key)
            if data[ticker]["price"] is not None:
                return data[ticker], "alpha_vantage"
    except Exception as e:
        print(f"Alpha Vantage failed: {e}")

    return None, None


# Usage
stock_data, source = get_stock_data("AAPL", ["price", "volume"])
if stock_data:
    print(f"Got data from {source}: ${stock_data['price']:.2f}")
else:
    print("All sources failed")
```

## Bulk Fetching

For efficiency, use bulk fetchers when requesting data for multiple tickers:

```python
from tradingview_algo.fin_data_apis.fetchers import (
    fetch_finnhub_bulk,
    fetch_fmp_bulk,
    fetch_alpha_vantage_bulk,
)
from tradingview_algo.fin_data_apis.secure_api import get_api_key

tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
fields = ["price", "volume", "high", "low"]

# Bulk fetch from Finnhub (parallel requests)
finnhub_key = get_api_key("finnhub")
bulk_data = fetch_finnhub_bulk(tickers, fields, finnhub_key)

# Process results
for ticker, data in bulk_data.items():
    if data["price"]:
        print(f"{ticker}: ${data['price']:.2f}")
    else:
        print(f"{ticker}: No data available")
```

## Advanced Data Fetching

### Historical Data with yfinance Integration

```python
import yfinance as yf
from tradingview_algo.cache.data_cache import DataCache


def fetch_historical_with_cache(ticker, period="1y"):
    """Fetch historical data with local caching"""

    cache = DataCache()

    # Check cache first
    cached_data = cache.get_price_data(ticker)
    if not cached_data.empty:
        latest_date = cached_data.index[-1]
        print(f"Found cached data up to {latest_date}")

        # Check if we need to update
        import pandas as pd

        if pd.Timestamp.now() - latest_date < pd.Timedelta(days=1):
            return cached_data

    # Fetch fresh data
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)

    # Store in cache
    cache.store_price_data(ticker, df)

    return df


# Usage
df = fetch_historical_with_cache("AAPL", period="2y")
print(f"Data shape: {df.shape}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
```

### Real-time Data Aggregation

```python
from tradingview_algo.fin_data_apis.fetchers import (
    fetch_yahoo,
    fetch_finnhub,
    fetch_fmp,
)
import time
import pandas as pd


class MultiSourceDataAggregator:
    """Aggregate data from multiple sources for comparison"""

    def __init__(self):
        self.sources = {
            "yahoo": fetch_yahoo,
            "finnhub": lambda t, f: fetch_finnhub(t, f, get_api_key("finnhub")),
            "fmp": lambda t, f: fetch_fmp(t, f, get_api_key("fmp")),
        }
        self.data_history = []

    def fetch_comparison(self, ticker, fields=["price"]):
        """Fetch same data from all sources for comparison"""

        timestamp = pd.Timestamp.now()
        results = {"timestamp": timestamp, "ticker": ticker}

        for source_name, fetch_func in self.sources.items():
            try:
                data = fetch_func([ticker], fields)
                results[source_name] = data[ticker]
            except Exception as e:
                print(f"{source_name} error: {e}")
                results[source_name] = {field: None for field in fields}

        self.data_history.append(results)
        return results

    def get_price_comparison_df(self):
        """Convert history to DataFrame for analysis"""

        rows = []
        for entry in self.data_history:
            row = {"timestamp": entry["timestamp"], "ticker": entry["ticker"]}
            for source in ["yahoo", "finnhub", "fmp"]:
                if source in entry and "price" in entry[source]:
                    row[f"{source}_price"] = entry[source]["price"]
            rows.append(row)

        return pd.DataFrame(rows)


# Usage
aggregator = MultiSourceDataAggregator()

# Fetch data every 30 seconds for 5 minutes
for i in range(10):
    result = aggregator.fetch_comparison("AAPL")
    print(
        f"Fetch {i+1}: Yahoo=${result['yahoo']['price']:.2f}, "
        f"Finnhub=${result['finnhub']['price']:.2f}"
    )
    time.sleep(30)

# Analyze differences
df = aggregator.get_price_comparison_df()
print("\nPrice comparison statistics:")
print(df[["yahoo_price", "finnhub_price"]].describe())
```

## Rate Limiting & Error Handling

### Automatic Rate Limiting

The library automatically handles rate limits for all data sources:

```python
from tradingview_algo.fin_data_apis.rate_limit import rate_limit, RateLimiter

# Manual rate limiting
@rate_limit("finnhub")
def my_finnhub_function():
    # Your code here - automatically rate limited
    pass


# Class-based rate limiting
limiter = FinnhubRateLimiter()
limiter.check()  # Blocks if rate limit exceeded
```

### Custom Rate Limit Configuration

```yaml
# api_config.yaml
finnhub:
  free_limit_per_minute: 60
  free_limit_per_day: 1440
  paid_limit_per_minute: 300
  paid_limit_per_day: 10000

alpha_vantage:
  free_limit_per_minute: 5
  free_limit_per_day: 500
```

### Error Handling Best Practices

```python
import time
from requests.exceptions import RequestException


def robust_data_fetch(ticker, max_retries=3, backoff_factor=2):
    """Fetch data with exponential backoff retry"""

    for attempt in range(max_retries):
        try:
            data = fetch_yahoo([ticker], ["price", "volume"])
            return data[ticker]

        except RequestException as e:
            if attempt == max_retries - 1:
                raise e

            wait_time = backoff_factor ** attempt
            print(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
            time.sleep(wait_time)

        except Exception as e:
            print(f"Unexpected error: {e}")
            return None


# Usage
data = robust_data_fetch("AAPL")
if data:
    print(f"AAPL: ${data['price']:.2f}")
```

## Data Source Specific Features

### Polygon.io - Options Data

```python
from tradingview_algo.fin_data_apis.polygon_api import PolygonAPI

polygon = PolygonAPI()

# Get options chain
options = polygon.get_option_chain(
    underlying="AAPL",
    expiration="2024-01-19",  # YYYY-MM-DD
    option_type="call",  # or "put"
)

print(f"Found {len(options['results'])} call options")
for option in options["results"][:5]:  # First 5 options
    print(f"Strike: ${option['strike_price']}, " f"Expiry: {option['expiration_date']}")
```

### Tiingo - News Integration

```python
from tradingview_algo.fin_data_apis.tiingo_api import TiingoAPI

tiingo = TiingoAPI()

# Get OHLCV data
df = tiingo.get_ohlcv(
    ticker="AAPL", start_date="2024-01-01", end_date="2024-12-31", resample_freq="daily"
)

print(f"Retrieved {len(df)} daily bars")
print(df.tail())
```

### Alpha Vantage - Technical Indicators

```python
from tradingview_algo.fin_data_apis.alpha_vantage_api import AlphaVantageAPI

av = AlphaVantageAPI()

# Get pre-calculated technical indicators
rsi_data = av.get_technical_indicator(
    function="RSI", symbol="AAPL", interval="daily", time_period=14
)

print("Recent RSI values:")
for date, values in list(rsi_data.items())[-5:]:
    print(f"{date}: {float(values['RSI']):.2f}")
```

## Performance Optimization

### Caching Strategy

```python
from tradingview_algo.cache.data_cache import DataCache
import pandas as pd


class SmartDataFetcher:
    """Intelligent data fetcher with aggressive caching"""

    def __init__(self):
        self.cache = DataCache()
        self.memory_cache = {}  # In-memory cache for session

    def get_data(self, ticker, fields, max_age_minutes=5):
        """Get data with multi-level caching"""

        cache_key = f"{ticker}_{'+'.join(fields)}"
        now = pd.Timestamp.now()

        # Check memory cache first
        if cache_key in self.memory_cache:
            data, timestamp = self.memory_cache[cache_key]
            if (now - timestamp).total_seconds() < max_age_minutes * 60:
                return data

        # Check database cache
        cached_signals = self.cache.get_signals(ticker, "live", "price_data")
        if not cached_signals.empty:
            latest = cached_signals.iloc[-1]
            if (now - latest.name).total_seconds() < max_age_minutes * 60:
                data = latest.to_dict()
                self.memory_cache[cache_key] = (data, now)
                return data

        # Fetch fresh data
        fresh_data = fetch_yahoo([ticker], fields)[ticker]

        # Update all caches
        self.memory_cache[cache_key] = (fresh_data, now)

        # Store in database
        df = pd.DataFrame([fresh_data], index=[now])
        self.cache.store_signals(ticker, "live", "price_data", df)

        return fresh_data


# Usage
fetcher = SmartDataFetcher()
data = fetcher.get_data("AAPL", ["price", "volume"])
print(f"AAPL: ${data['price']:.2f}")
```

### Batch Processing

```python
def process_watchlist(tickers, batch_size=20):
    """Process large watchlists efficiently"""

    results = {}

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        print(f"Processing batch {i//batch_size + 1}: {batch}")

        try:
            batch_data = fetch_yahoo(batch, ["price", "volume"])
            results.update(batch_data)
        except Exception as e:
            print(f"Batch failed: {e}")
            # Process individually as fallback
            for ticker in batch:
                try:
                    data = fetch_yahoo([ticker], ["price", "volume"])
                    results.update(data)
                except:
                    results[ticker] = {"price": None, "volume": None}

        # Rate limiting between batches
        time.sleep(1)

    return results


# Process large watchlist
large_watchlist = [
    "AAPL",
    "GOOGL",
    "MSFT",
    "TSLA",
    "AMZN",
    "META",
    "NFLX",
    "NVDA",
    "AMD",
    "INTC",
]
results = process_watchlist(large_watchlist)

# Show results
for ticker, data in results.items():
    if data["price"]:
        print(f"{ticker}: ${data['price']:.2f}")
```

## API Configuration

### Dynamic API Selection

```python
import yaml


class ConfigurableDataFetcher:
    """Data fetcher with runtime API configuration"""

    def __init__(self, config_path="api_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.api_priorities = [
            "yahoo",  # Free, unlimited
            "finnhub",  # Good rate limits
            "alpha_vantage",  # Backup
        ]

    def fetch_with_priority(self, tickers, fields):
        """Try APIs in priority order"""

        for api_name in self.api_priorities:
            api_config = self.config.get(api_name, {})

            # Check if we have quota remaining
            if self._check_quota(api_name):
                try:
                    return self._fetch_from_api(api_name, tickers, fields)
                except Exception as e:
                    print(f"{api_name} failed: {e}")
                    continue

        raise Exception("All APIs failed or quota exceeded")

    def _check_quota(self, api_name):
        """Check if API has remaining quota"""
        # Implementation would track usage vs limits
        return True

    def _fetch_from_api(self, api_name, tickers, fields):
        """Fetch from specific API"""
        if api_name == "yahoo":
            return fetch_yahoo(tickers, fields)
        elif api_name == "finnhub":
            api_key = get_api_key("finnhub")
            return fetch_finnhub(tickers, fields, api_key)
        # Add other APIs...


# Usage
fetcher = ConfigurableDataFetcher()
data = fetcher.fetch_with_priority(["AAPL"], ["price"])
```

## Next Steps

- [Data Cache System](data-cache.md) - Learn about local data storage
- [Live Data Feeds](live-data.md) - Set up real-time data streams
- [Technical Indicators](indicators.md) - Use fetched data for analysis
