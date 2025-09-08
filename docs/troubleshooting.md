# Troubleshooting

This document provides solutions to common issues and problems that may arise when using open_trading_algo.

## Installation Issues

### Poetry Installation Problems

**Problem:** Poetry installation fails on Windows with permission errors.

**Solution:**
```bash
# Use pip to install poetry with --user flag
pip install --user poetry

# Add Poetry to PATH
export PATH="$HOME/.local/bin:$PATH"

# Configure Poetry for Windows
poetry config virtualenvs.create true
poetry config virtualenvs.in-project true
```

**Problem:** Poetry can't find Python interpreter.

**Solution:**
```bash
# Specify Python path explicitly
poetry env use /path/to/python

# Or use py launcher on Windows
poetry env use py

# Check available Python versions
poetry env list --full
```

### Dependency Installation Failures

**Problem:** Package installation fails due to missing system dependencies.

**Solution:**
```bash
# Install system dependencies first
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev build-essential

# macOS
brew install python3

# Windows
# Install Visual Studio Build Tools

# Then reinstall dependencies
poetry install --no-cache
```

## Data Loading Issues

### API Connection Problems

**Problem:** API requests fail with connection errors.

**Solution:**
```python
from open_trading_algo.cache.data_cache import DataCache
import requests

# Check API connectivity
try:
    response = requests.get('https://www.alphavantage.co')
    print(f"API Status: {response.status_code}")
except Exception as e:
    print(f"Connection Error: {e}")

# Use alternative data provider
cache = DataCache()
cache.set_fallback_provider('yahoo_finance')
```

**Problem:** API rate limits exceeded.

**Solution:**
```python
from open_trading_algo.cache.data_cache import DataCache
import time

# Implement rate limiting
cache = DataCache()

def rate_limited_request(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if 'rate limit' in str(e).lower():
            print("Rate limit hit, waiting...")
            time.sleep(60)  # Wait 1 minute
            return func(*args, **kwargs)
        raise

# Use rate limiting wrapper
data = rate_limited_request(cache.get_price_data, 'AAPL')
```

### Data Quality Issues

**Problem:** Missing or corrupted price data.

**Solution:**
```python
from open_trading_algo.cache.data_cache import DataCache
import pandas as pd

cache = DataCache()

# Validate data integrity
def validate_price_data(data):
    issues = []

    # Check for missing values
    missing_pct = data.isnull().sum() / len(data)
    if missing_pct.max() > 0.1:  # More than 10% missing
        issues.append(f"High missing data: {missing_pct.max():.1%}")

    # Check for price anomalies
    price_change = data['close'].pct_change()
    if price_change.abs().max() > 0.5:  # 50% price change
        issues.append("Extreme price movements detected")

    # Check date continuity
    date_diff = data.index.to_series().diff().dt.days
    if date_diff.max() > 7:  # Gaps larger than a week
        issues.append("Large gaps in data")

    return issues

# Validate and clean data
data = cache.get_price_data('AAPL')
issues = validate_price_data(data)

if issues:
    print("Data Issues Found:")
    for issue in issues:
        print(f"- {issue}")

    # Attempt to fix issues
    data = data.fillna(method='forward').fillna(method='backward')
    data = data[data['close'] > 0]  # Remove invalid prices
```

## Backtesting Problems

### Performance Issues

**Problem:** Backtesting runs very slowly.

**Solution:**
```python
from open_trading_algo.backtest.signal_backtester import SignalBacktester
import pandas as pd

# Optimize data types
data = data.astype({
    'open': 'float32',
    'high': 'float32',
    'low': 'float32',
    'close': 'float32',
    'volume': 'int32'
})

# Use vectorized operations
def vectorized_signal_generation(data):
    # Instead of loops, use pandas operations
    data['rsi'] = calculate_rsi_vectorized(data['close'])
    data['signal'] = np.where(data['rsi'] < 30, 1,
                            np.where(data['rsi'] > 70, -1, 0))
    return data

# Limit data range for testing
recent_data = data.last('2Y')  # Last 2 years only

# Run optimized backtest
backtester = SignalBacktester()
results = backtester.run_backtest(signals, recent_data)
```

### Memory Issues

**Problem:** Out of memory errors during backtesting.

**Solution:**
```python
import gc
from open_trading_algo.backtest.signal_backtester import SignalBacktester

# Process data in chunks
def chunked_backtest(signals, data, chunk_size=1000):
    results = []

    for i in range(0, len(data), chunk_size):
        chunk_data = data.iloc[i:i+chunk_size]
        chunk_signals = signals.iloc[i:i+chunk_size]

        backtester = SignalBacktester()
        chunk_result = backtester.run_backtest(chunk_signals, chunk_data)
        results.append(chunk_result)

        # Force garbage collection
        gc.collect()

    # Combine results
    combined_result = combine_results(results)
    return combined_result

# Use memory-efficient data structures
data = data[['open', 'high', 'low', 'close']]  # Drop unnecessary columns
data = data.astype('float32')  # Use smaller data types
```

### Signal Generation Errors

**Problem:** Signals not generating as expected.

**Solution:**
```python
from open_trading_algo.models.momentum_model import MomentumModel
import matplotlib.pyplot as plt

# Debug signal generation
model = MomentumModel(lookback_period=20, threshold=0.05)

# Add debug information
data['momentum'] = model.calculate_momentum(data)
data['signal'] = model.generate_signals(data)

# Visualize signals
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))

ax1.plot(data.index, data['close'])
ax1.set_title('Price')

ax2.plot(data.index, data['momentum'])
ax2.axhline(y=0.05, color='g', linestyle='--', label='Buy Threshold')
ax2.axhline(y=-0.05, color='r', linestyle='--', label='Sell Threshold')
ax2.set_title('Momentum')
ax2.legend()

ax3.plot(data.index, data['signal'])
ax3.set_title('Signals')

plt.tight_layout()
plt.show()

# Check signal statistics
print("Signal Statistics:")
print(data['signal'].value_counts())
print(f"Signal density: {data['signal'].ne(0).sum() / len(data):.2%}")
```

## Live Trading Issues

### Connection Problems

**Problem:** Live data streaming disconnects frequently.

**Solution:**
```python
from open_trading_algo.live_data.price_stream import PriceDataStreamer
import websocket
import time

class ResilientStreamer(PriceDataStreamer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5

    def on_connection_lost(self):
        if self.reconnect_attempts < self.max_reconnect_attempts:
            print(f"Connection lost, attempting reconnect {self.reconnect_attempts + 1}")
            time.sleep(self.reconnect_delay)
            self.reconnect_attempts += 1
            self.start_streaming()
        else:
            print("Max reconnection attempts reached")
            self.send_alert("Connection failed after max retries")

# Use resilient streamer
streamer = ResilientStreamer(provider='polygon', api_key='key')
streamer.start_streaming()
```

### Execution Delays

**Problem:** Orders execute with significant delays.

**Solution:**
```python
from open_trading_algo.live_data.live_executor import LiveTradeExecutor

# Optimize order execution
executor = LiveTradeExecutor(
    broker='alpaca',
    api_key='key',
    secret_key='secret'
)

# Use market orders for speed
fast_order = {
    'ticker': 'AAPL',
    'action': 'BUY',
    'quantity': 100,
    'order_type': 'MARKET',
    'time_in_force': 'DAY'
}

# Or use limit orders near the market
current_price = executor.get_current_price('AAPL')
limit_price = current_price * 1.001  # 0.1% above market

limit_order = {
    'ticker': 'AAPL',
    'action': 'BUY',
    'quantity': 100,
    'order_type': 'LIMIT',
    'limit_price': limit_price,
    'time_in_force': 'DAY'
}

# Execute with timeout
result = executor.execute_order(fast_order, timeout_seconds=30)
if result['status'] != 'FILLED':
    print(f"Order not filled: {result['status']}")
    # Implement alternative execution logic
```

### Position Tracking Errors

**Problem:** Position sizes don't match expectations.

**Solution:**
```python
from open_trading_algo.live_data.live_portfolio import LivePortfolioTracker

# Implement position reconciliation
tracker = LivePortfolioTracker(broker='alpaca')

def reconcile_positions():
    # Get positions from broker
    broker_positions = tracker.get_broker_positions()

    # Get positions from local tracking
    local_positions = tracker.get_local_positions()

    # Compare and reconcile
    for ticker in set(broker_positions.keys()) | set(local_positions.keys()):
        broker_qty = broker_positions.get(ticker, 0)
        local_qty = local_positions.get(ticker, 0)

        if abs(broker_qty - local_qty) > 0.01:  # Allow for rounding
            print(f"Position mismatch for {ticker}: "
                  f"Broker: {broker_qty}, Local: {local_qty}")

            # Update local tracking
            tracker.update_local_position(ticker, broker_qty)

# Run reconciliation periodically
import schedule
schedule.every(5).minutes.do(reconcile_positions)
```

## Sentiment Analysis Problems

### API Rate Limiting

**Problem:** Twitter/News API rate limits hit frequently.

**Solution:**
```python
from open_trading_algo.sentiment.twitter_sentiment import TwitterSentimentAnalyzer
import time
from functools import wraps

def rate_limit_retry(max_retries=3, delay=60):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if 'rate limit' in str(e).lower():
                        if attempt < max_retries - 1:
                            print(f"Rate limited, waiting {delay}s...")
                            time.sleep(delay)
                            delay *= 2  # Exponential backoff
                        else:
                            raise
                    else:
                        raise
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Apply rate limiting to sentiment analysis
analyzer = TwitterSentimentAnalyzer(api_key='key')

@rate_limit_retry()
def analyze_with_retry(ticker):
    return analyzer.analyze_ticker_sentiment(ticker)

# Use cached results when possible
sentiment_cache = {}
def get_cached_sentiment(ticker, max_age_hours=1):
    cache_key = f"{ticker}_{int(time.time() / 3600)}"  # Hourly cache

    if cache_key not in sentiment_cache:
        sentiment_cache[cache_key] = analyze_with_retry(ticker)

    return sentiment_cache[cache_key]
```

### Sentiment Quality Issues

**Problem:** Sentiment scores are noisy or inaccurate.

**Solution:**
```python
from open_trading_algo.sentiment.sentiment_debugger import SentimentDebugger

# Debug sentiment analysis
debugger = SentimentDebugger()

# Analyze sentiment distribution
sentiment_data = analyzer.analyze_ticker_sentiment('AAPL')
debugger.analyze_sentiment_distribution(sentiment_data)

# Check source quality
debugger.check_source_quality(sentiment_data)

# Implement sentiment filters
def filter_sentiment(sentiment_data, min_confidence=0.7, max_noise=0.3):
    # Filter by confidence
    filtered = sentiment_data[sentiment_data['confidence'] > min_confidence]

    # Remove outliers
    sentiment_scores = filtered['sentiment']
    q1, q3 = sentiment_scores.quantile([0.25, 0.75])
    iqr = q3 - q1
    filtered = filtered[
        (sentiment_scores >= q1 - 1.5 * iqr) &
        (sentiment_scores <= q3 + 1.5 * iqr)
    ]

    # Smooth noisy data
    filtered['smoothed_sentiment'] = filtered['sentiment'].rolling(5).mean()

    return filtered

# Apply filters
clean_sentiment = filter_sentiment(sentiment_data)
```

## Risk Management Issues

### Stop Loss Problems

**Problem:** Stop losses trigger too frequently or not at all.

**Solution:**
```python
from open_trading_algo.risk_management.stop_loss import DynamicStopLoss

# Implement intelligent stop loss
dynamic_stop = DynamicStopLoss(
    initial_stop_pct=0.05,
    trailing_pct=0.03,
    volatility_adjusted=True,
    time_based=True
)

# Test stop loss on historical data
def test_stop_loss(data, stop_config):
    portfolio_value = 100000
    trades = []

    for i in range(len(data)):
        current_price = data.iloc[i]['close']

        # Check if stop should be triggered
        if dynamic_stop.should_stop(current_price, portfolio_value):
            # Execute stop loss
            stop_price = dynamic_stop.get_stop_price()
            loss_pct = (current_price - stop_price) / stop_price

            trades.append({
                'date': data.index[i],
                'action': 'STOP_LOSS',
                'price': current_price,
                'loss_pct': loss_pct
            })

            # Reset stop for next trade
            dynamic_stop.reset_stop()

    return trades

# Analyze stop loss performance
stop_trades = test_stop_loss(data, stop_config)
print(f"Stop Loss Triggers: {len(stop_trades)}")
print(f"Average Loss: {np.mean([t['loss_pct'] for t in stop_trades]):.2%}")
```

### Position Sizing Errors

**Problem:** Position sizes don't match risk calculations.

**Solution:**
```python
from open_trading_algo.risk_management.position_sizer import VolatilityPositionSizer

# Debug position sizing
sizer = VolatilityPositionSizer(
    risk_per_trade=0.02,
    max_portfolio_risk=0.05
)

# Test position sizing with different scenarios
test_scenarios = [
    {'price': 100, 'stop_loss': 95, 'volatility': 0.02, 'portfolio': 100000},
    {'price': 50, 'stop_loss': 45, 'volatility': 0.05, 'portfolio': 100000},
    {'price': 200, 'stop_loss': 180, 'volatility': 0.03, 'portfolio': 100000}
]

for scenario in test_scenarios:
    size = sizer.calculate_position_size(
        current_price=scenario['price'],
        stop_loss_price=scenario['stop_loss'],
        portfolio_value=scenario['portfolio'],
        volatility=scenario['volatility']
    )

    print(f"Scenario: ${scenario['price']} stock")
    print(f"Position Size: {size['quantity']} shares")
    print(f"Risk Amount: ${size['risk_amount']:.2f}")
    print(f"Portfolio Allocation: {size['portfolio_allocation']:.2%}")
    print()
```

## Performance Optimization

### Profiling Code Performance

**Problem:** Code runs too slowly for real-time use.

**Solution:**
```python
import cProfile
import pstats
from io import StringIO

def profile_function(func, *args, **kwargs):
    pr = cProfile.Profile()
    pr.enable()

    result = func(*args, **kwargs)

    pr.disable()
    s = StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

    return result

# Profile signal generation
profile_function(model.generate_signals, data)
```

### Memory Optimization

**Problem:** High memory usage with large datasets.

**Solution:**
```python
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Memory Usage: {memory_usage:.1f} MB")

    if memory_usage > 1000:  # 1GB threshold
        print("High memory usage detected")
        # Implement cleanup
        gc.collect()

# Use memory-efficient data structures
def optimize_dataframe(df):
    # Downcast numeric types
    for col in df.select_dtypes(include=['float64']):
        df[col] = pd.to_numeric(df[col], downcast='float')

    for col in df.select_dtypes(include=['int64']):
        df[col] = pd.to_numeric(df[col], downcast='integer')

    # Use categorical for string columns
    for col in df.select_dtypes(include=['object']):
        if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique
            df[col] = df[col].astype('category')

    return df

# Optimize data
data = optimize_dataframe(data)
monitor_memory()
```

## Common Error Messages

### "Module not found" errors

**Problem:** Import errors for open_trading_algo modules.

**Solution:**
```bash
# Ensure you're in the correct environment
poetry shell

# Check if package is installed
poetry show open-trading-algo

# Reinstall if necessary
poetry install

# Check Python path
python -c "import sys; print(sys.path)"
```

### "API key not found" errors

**Problem:** Missing or invalid API keys.

**Solution:**
```python
# Check environment variables
import os
print("API Keys Status:")
print(f"ALPHA_VANTAGE_KEY: {'Set' if os.getenv('ALPHA_VANTAGE_KEY') else 'Missing'}")
print(f"POLYGON_KEY: {'Set' if os.getenv('POLYGON_KEY') else 'Missing'}")

# Load from .env file
from dotenv import load_dotenv
load_dotenv()

# Validate API keys
from open_trading_algo.utils.validation import validate_api_keys
issues = validate_api_keys()
if issues:
    print("API Key Issues:")
    for issue in issues:
        print(f"- {issue}")
```

### "Data not available" errors

**Problem:** Requested data is not available.

**Solution:**
```python
from open_trading_algo.cache.data_cache import DataCache

cache = DataCache()

# Check data availability
def check_data_availability(ticker, start_date, end_date):
    try:
        data = cache.get_price_data(ticker, start_date, end_date)
        print(f"Data available: {len(data)} rows")
        return True
    except Exception as e:
        print(f"Data not available: {e}")

        # Try alternative sources
        alternative_sources = ['yahoo_finance', 'iex']
        for source in alternative_sources:
            try:
                cache.set_provider(source)
                data = cache.get_price_data(ticker, start_date, end_date)
                print(f"Data found from {source}: {len(data)} rows")
                return True
            except:
                continue

        return False

# Check availability
available = check_data_availability('AAPL', '2020-01-01', '2024-01-01')
```

## Getting Help

### Debug Information

```python
# Collect system information
import sys
import platform

def collect_debug_info():
    info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'architecture': platform.architecture(),
        'processor': platform.processor()
    }

    # Package versions
    try:
        import open_trading_algo
        info['ota_version'] = open_trading_algo.__version__
    except:
        info['ota_version'] = 'Not installed'

    # Dependencies
    dependencies = ['pandas', 'numpy', 'requests', 'matplotlib']
    for dep in dependencies:
        try:
            module = __import__(dep)
            info[f'{dep}_version'] = getattr(module, '__version__', 'Unknown')
        except:
            info[f'{dep}_version'] = 'Not installed'

    return info

debug_info = collect_debug_info()
for key, value in debug_info.items():
    print(f"{key}: {value}")
```

### Logging Configuration

```python
import logging

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_algo_debug.log'),
        logging.StreamHandler()
    ]
)

# Log errors with context
try:
    result = some_trading_function()
except Exception as e:
    logging.error(f"Error in trading function: {e}", exc_info=True)
    # Include additional context
    logging.error(f"Input parameters: {locals()}")
```

This troubleshooting guide covers the most common issues encountered when using open_trading_algo. If you encounter an issue not covered here, please check the GitHub issues page or create a new issue with detailed information about your problem.</content>
<parameter name="filePath">/home/philipmai/repos/TradingViewAlgoDev/docs/troubleshooting.md
