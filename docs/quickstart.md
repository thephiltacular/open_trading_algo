# Quick Start Guide

This guide will get you up and running with TradingViewAlgoDev in minutes.

## Basic Data Fetching

### Fetch Current Stock Prices

```python
from open_trading_algo.fin_data_apis.fetchers import fetch_yahoo

# Get current prices for multiple stocks
tickers = ["AAPL", "GOOGL", "MSFT", "TSLA"]
fields = ["price", "volume", "high", "low", "open"]

data = fetch_yahoo(tickers, fields)

for ticker, info in data.items():
    print(f"{ticker}: ${info['price']:.2f} (Volume: {info['volume']:,})")
```

### Use Multiple Data Sources

```python
from open_trading_algo.fin_data_apis.fetchers import (
    fetch_yahoo,
    fetch_finnhub,
    fetch_alpha_vantage,
)
from open_trading_algo.fin_data_apis.secure_api import get_api_key

# Yahoo Finance (free, no API key required)
yahoo_data = fetch_yahoo(["AAPL"], ["price"])

# Finnhub (requires API key)
finnhub_key = get_api_key("finnhub")
finnhub_data = fetch_finnhub(["AAPL"], ["price"], finnhub_key)

# Alpha Vantage (requires API key)
av_key = get_api_key("alpha_vantage")
av_data = fetch_alpha_vantage(["AAPL"], ["price"], av_key)

print(f"Yahoo: ${yahoo_data['AAPL']['price']:.2f}")
print(f"Finnhub: ${finnhub_data['AAPL']['price']:.2f}")
print(f"Alpha Vantage: ${av_data['AAPL']['price']:.2f}")
```

## Technical Analysis

### Calculate Basic Indicators

```python
import yfinance as yf
from open_trading_algo.indicators.indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
)

# Get historical data
ticker = yf.Ticker("AAPL")
df = ticker.history(period="6mo")

# Calculate RSI
rsi = calculate_rsi(df["Close"])
print(f"Current RSI: {rsi.iloc[-1]:.2f}")

# Calculate MACD
macd_line, macd_signal, macd_histogram = calculate_macd(df["Close"])
print(f"MACD: {macd_line.iloc[-1]:.4f}")

# Calculate Bollinger Bands
bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df["Close"])
current_price = df["Close"].iloc[-1]
print(f"Price: ${current_price:.2f}")
print(f"BB Upper: ${bb_upper.iloc[-1]:.2f}")
print(f"BB Lower: ${bb_lower.iloc[-1]:.2f}")
```

### Advanced Technical Analysis

```python
from open_trading_algo.indicators.indicators import (
    calculate_adx,
    calculate_stochastic,
    calculate_williams_r,
)

# ADX (Trend Strength)
adx = calculate_adx(df["High"], df["Low"], df["Close"])
print(f"ADX (Trend Strength): {adx.iloc[-1]:.2f}")

# Stochastic Oscillator
stoch_k, stoch_d = calculate_stochastic(df["High"], df["Low"], df["Close"])
print(f"Stochastic %K: {stoch_k.iloc[-1]:.2f}")

# Williams %R
williams_r = calculate_williams_r(df["High"], df["Low"], df["Close"])
print(f"Williams %R: {williams_r.iloc[-1]:.2f}")
```

## Signal Generation

### Long Signals

```python
from open_trading_algo.indicators.long_signals import (
    rsi_oversold_signal,
    macd_bullish_crossover,
    bollinger_squeeze_breakout,
)

# RSI Oversold Signal
rsi_signal = rsi_oversold_signal(df["Close"])
print(f"RSI Oversold Signal: {rsi_signal.iloc[-1]}")

# MACD Bullish Crossover
macd_signal = macd_bullish_crossover(df["Close"])
print(f"MACD Bullish Signal: {macd_signal.iloc[-1]}")

# Bollinger Band Squeeze Breakout
bb_signal = bollinger_squeeze_breakout(df["Close"], df["Volume"])
print(f"BB Breakout Signal: {bb_signal.iloc[-1]}")
```

### Short Signals

```python
from open_trading_algo.indicators.short_signals import (
    rsi_overbought_signal,
    macd_bearish_crossover,
    breaking_support,
)

# RSI Overbought Signal
rsi_short = rsi_overbought_signal(df["Close"])
print(f"RSI Overbought Signal: {rsi_short.iloc[-1]}")

# MACD Bearish Crossover
macd_short = macd_bearish_crossover(df["Close"])
print(f"MACD Bearish Signal: {macd_short.iloc[-1]}")

# Breaking Support
support_break = breaking_support(df["Close"], df["Volume"])
print(f"Support Break Signal: {support_break.iloc[-1]}")
```

## Data Caching

### Using the Cache System

```python
from open_trading_algo.cache.data_cache import DataCache
import pandas as pd

# Initialize cache
cache = DataCache()

# Store price data
cache.store_price_data("AAPL", df)

# Retrieve cached data
cached_df = cache.get_price_data("AAPL")
print(f"Cached data shape: {cached_df.shape}")

# Store signals
rsi_signals = pd.DataFrame({"signal": rsi_signal}, index=df.index)

cache.store_signals("AAPL", "1d", "rsi_oversold", rsi_signals)

# Retrieve cached signals
cached_signals = cache.get_signals("AAPL", "1d", "rsi_oversold")
print(f"Signal count: {cached_signals['signal'].sum()}")
```

## Live Data Feeds

### Real-time Data Streaming

```python
from open_trading_algo.fin_data_apis.feed import LiveDataFeed
from pathlib import Path
import time

# Create live data config
config_content = """
source: "yahoo"
tickers: ["AAPL", "GOOGL", "MSFT"]
fields: ["price", "volume"]
update_rate: 30
"""

config_path = Path("live_config.yaml")
config_path.write_text(config_content)

# Callback function for data updates
def on_data_update(data):
    print("Live data update:")
    for ticker, info in data.items():
        print(f"  {ticker}: ${info['price']:.2f}")


# Start live feed
feed = LiveDataFeed(config_path, on_update=on_data_update)
feed.start()

# Let it run for 2 minutes
time.sleep(120)
feed.stop()
```

## Risk Management

### Position Sizing

```python
from open_trading_algo.risk_management import (
    percent_of_portfolio,
    fixed_dollar_amount,
    volatility_adjusted,
)
import numpy as np

portfolio_value = 100000  # $100k portfolio
current_price = 150.0  # $150 per share

# 2% of portfolio risk
shares_pct = percent_of_portfolio(portfolio_value, current_price, 0.02)
print(f"2% position: {shares_pct} shares")

# Fixed $5000 position
shares_fixed = fixed_dollar_amount(5000, current_price)
print(f"$5k position: {shares_fixed} shares")

# Volatility adjusted (using 20-day volatility)
returns = df["Close"].pct_change().dropna()
volatility = returns.rolling(20).std().iloc[-1]
shares_vol = volatility_adjusted(portfolio_value, current_price, volatility)
print(f"Volatility adjusted: {shares_vol} shares")
```

### Stop Loss Calculation

```python
from open_trading_algo.risk_management import (
    percentage_stop_loss,
    atr_stop_loss,
    support_resistance_stop,
)

entry_price = 150.0

# 5% stop loss
stop_pct = percentage_stop_loss(entry_price, 0.05)
print(f"5% stop loss: ${stop_pct:.2f}")

# ATR-based stop loss
atr = calculate_atr(df["High"], df["Low"], df["Close"])
stop_atr = atr_stop_loss(entry_price, atr.iloc[-1], multiplier=2.0)
print(f"ATR stop loss: ${stop_atr:.2f}")
```

## Signal Optimization

### Multi-Signal Analysis

```python
from open_trading_algo.signal_optimizer import SignalOptimizer

# Initialize optimizer with data
optimizer = SignalOptimizer(df)

# Add signals
long_signals = {
    "rsi_oversold": rsi_oversold_signal(df["Close"]),
    "macd_bullish": macd_bullish_crossover(df["Close"]),
    "bb_squeeze": bollinger_squeeze_breakout(df["Close"], df["Volume"]),
}

optimizer.add_long_signals(long_signals)

# Optimize signal combination
best_signals = optimizer.optimize_signals(trade_type="long", max_signals=2)
print(f"Best signal combination: {best_signals}")
```

## Backtesting

### Simple Strategy Backtest

```python
# Create a simple strategy combining RSI and MACD
combined_signal = (rsi_signal & macd_signal).astype(int)

# Backtest the strategy
results = optimizer.backtest_signals(
    trade_type="long", initial_capital=10000, commission=0.001  # 0.1% commission
)

print(f"Total Return: {results['total_return']:.2%}")
print(f"Win Rate: {results['win_rate']:.2%}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
```

## Complete Example: End-to-End Analysis

```python
from open_trading_algo.fin_data_apis.fetchers import fetch_yahoo
from open_trading_algo.indicators.indicators import *
from open_trading_algo.indicators.long_signals import *
from open_trading_algo.cache.data_cache import DataCache
from open_trading_algo.signal_optimizer import SignalOptimizer
import yfinance as yf


def analyze_stock(ticker):
    # Get historical data
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")

    # Calculate indicators
    rsi = calculate_rsi(df["Close"])
    macd_line, macd_signal, _ = calculate_macd(df["Close"])
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df["Close"])

    # Generate signals
    rsi_long = rsi_oversold_signal(df["Close"])
    macd_long = macd_bullish_crossover(df["Close"])

    # Optimize signals
    optimizer = SignalOptimizer(df)
    optimizer.add_long_signals({"rsi_oversold": rsi_long, "macd_bullish": macd_long})

    # Backtest
    results = optimizer.backtest_signals(trade_type="long")

    # Cache results
    cache = DataCache()
    cache.store_price_data(ticker, df)

    return {
        "ticker": ticker,
        "current_price": df["Close"].iloc[-1],
        "rsi": rsi.iloc[-1],
        "macd": macd_line.iloc[-1],
        "signals": {
            "rsi_oversold": rsi_long.iloc[-1],
            "macd_bullish": macd_long.iloc[-1],
        },
        "backtest": results,
    }


# Analyze multiple stocks
stocks = ["AAPL", "GOOGL", "MSFT", "TSLA"]
analyses = [analyze_stock(stock) for stock in stocks]

# Display results
for analysis in analyses:
    print(f"\n{analysis['ticker']}:")
    print(f"  Price: ${analysis['current_price']:.2f}")
    print(f"  RSI: {analysis['rsi']:.2f}")
    print(f"  Signals: {analysis['signals']}")
    print(f"  Backtest Return: {analysis['backtest']['total_return']:.2%}")
```

## Next Steps

- [Configuration Guide](configuration.md) - Customize the library settings
- [Data APIs Documentation](data-apis.md) - Learn about all data sources
- [Indicators Reference](indicators.md) - Complete technical analysis guide
- [Signal Generation](signals.md) - Advanced signal strategies
- [Backtesting Guide](backtesting.md) - Comprehensive strategy testing
