# Examples

This document provides practical examples and use cases for open_trading_algo, demonstrating how to implement various trading strategies and workflows.

## Getting Started

### Basic Setup

```python
import pandas as pd
import numpy as np
from open_trading_algo.cache.data_cache import DataCache
from open_trading_algo.indicators.technical_indicators import TechnicalIndicators
from open_trading_algo.models.momentum_model import MomentumModel

# Initialize components
cache = DataCache()
indicators = TechnicalIndicators()
model = MomentumModel()

# Load data
data = cache.get_price_data('AAPL', start='2020-01-01', end='2024-01-01')
print(f"Loaded {len(data)} days of AAPL data")
```

## Momentum Trading Strategy

### Simple Momentum Strategy

```python
from open_trading_algo.models.momentum_model import MomentumModel
from open_trading_algo.backtest.signal_backtester import SignalBacktester

# Initialize momentum model
momentum_model = MomentumModel(
    lookback_period=20,
    threshold=0.05  # 5% momentum threshold
)

# Generate signals
signals = momentum_model.generate_signals(data)

# Backtest the strategy
backtester = SignalBacktester(
    initial_capital=100000,
    commission=0.001,
    slippage=0.0005
)

results = backtester.run_backtest(signals, data)

print("Momentum Strategy Results:")
print(f"Total Return: {results['total_return']:.2%}")
print(f"Annual Return: {results['annual_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

### Advanced Momentum with Multiple Timeframes

```python
from open_trading_algo.indicators.technical_indicators import TechnicalIndicators

# Calculate multiple timeframe momentum
short_term = momentum_model.calculate_momentum(data, lookback=10)
medium_term = momentum_model.calculate_momentum(data, lookback=20)
long_term = momentum_model.calculate_momentum(data, lookback=50)

# Combine signals
combined_signals = pd.DataFrame(index=data.index)
combined_signals['short_momentum'] = short_term
combined_signals['medium_momentum'] = medium_term
combined_signals['long_momentum'] = long_term

# Generate combined signal
combined_signals['signal'] = np.where(
    (combined_signals['short_momentum'] > 0.03) &
    (combined_signals['medium_momentum'] > 0.05) &
    (combined_signals['long_momentum'] > 0.08),
    1,  # Buy signal
    np.where(
        (combined_signals['short_momentum'] < -0.03) &
        (combined_signals['medium_momentum'] < -0.05),
        -1,  # Sell signal
        0    # Hold
    )
)

print("Combined Momentum Signals:")
print(combined_signals['signal'].value_counts())
```

## Mean Reversion Strategy

### RSI-Based Mean Reversion

```python
from open_trading_algo.models.mean_reversion_model import MeanReversionModel

# Initialize mean reversion model
mr_model = MeanReversionModel(
    lookback_period=20,
    entry_threshold=2.0,  # 2 standard deviations
    exit_threshold=0.5    # 0.5 standard deviations
)

# Generate mean reversion signals
mr_signals = mr_model.generate_signals(data)

# Add RSI filter
rsi = indicators.calculate_rsi(data, period=14)
mr_signals['rsi_filter'] = rsi

# Filter signals with RSI
mr_signals['filtered_signal'] = np.where(
    (mr_signals['signal'] == 1) & (rsi < 30), 1,  # Buy when oversold
    np.where(
        (mr_signals['signal'] == -1) & (rsi > 70), -1,  # Sell when overbought
        0
    )
)

print("Mean Reversion Signals with RSI Filter:")
print(mr_signals['filtered_signal'].value_counts())
```

### Bollinger Band Squeeze Strategy

```python
# Calculate Bollinger Bands
bb = indicators.calculate_bollinger_bands(data, period=20, std_dev=2.0)

# Calculate band width (squeeze indicator)
band_width = (bb['upper'] - bb['lower']) / bb['middle']

# Generate squeeze signals
squeeze_threshold = band_width.quantile(0.2)  # Bottom 20% of band widths
squeeze_signals = pd.DataFrame(index=data.index)
squeeze_signals['band_width'] = band_width
squeeze_signals['squeeze'] = band_width < squeeze_threshold

# Calculate momentum for breakout direction
momentum = indicators.calculate_momentum(data, period=5)

# Generate breakout signals after squeeze
squeeze_signals['signal'] = np.where(
    squeeze_signals['squeeze'].shift(1) & ~squeeze_signals['squeeze'],
    np.where(momentum > 0, 1, -1),  # Breakout direction
    0
)

print("Bollinger Band Squeeze Signals:")
print(squeeze_signals['signal'].value_counts())
```

## Multi-Asset Portfolio Strategy

### Sector Rotation Strategy

```python
from open_trading_algo.models.sector_rotation import SectorRotationModel

# Define sector ETFs
sectors = {
    'technology': 'QQQ',
    'healthcare': 'XLV',
    'financials': 'XLF',
    'energy': 'XLE',
    'consumer': 'XLY'
}

# Initialize sector rotation model
sector_model = SectorRotationModel(
    sectors=sectors,
    momentum_period=3,  # 3-month momentum
    rebalance_months=3  # Quarterly rebalancing
)

# Generate sector rotation signals
sector_signals = sector_model.generate_sector_signals()

# Backtest sector rotation
sector_backtest = backtester.run_backtest(
    sector_signals,
    sector_model.get_sector_data(),
    start_date='2018-01-01'
)

print("Sector Rotation Results:")
print(f"Total Return: {sector_backtest['total_return']:.2%}")
print(f"Annual Return: {sector_backtest['annual_return']:.2%}")
print(f"Sharpe Ratio: {sector_backtest['sharpe_ratio']:.2f}")
```

### Risk Parity Portfolio

```python
from open_trading_algo.risk_management.risk_parity import RiskParityAllocator

# Define asset universe
assets = ['SPY', 'BND', 'GLD', 'VNQ', 'EFA']
asset_data = cache.get_multiple_price_data(assets, start='2015-01-01')

# Initialize risk parity allocator
risk_parity = RiskParityAllocator(
    target_volatility=0.12,  # 12% target volatility
    rebalance_threshold=0.05
)

# Calculate risk parity weights
rp_weights = risk_parity.calculate_weights(asset_data)

print("Risk Parity Weights:")
for asset, weight in rp_weights.items():
    print(f"{asset}: {weight:.2%}")

# Backtest risk parity portfolio
rp_backtest = backtester.run_portfolio_backtest(
    asset_data,
    rp_weights,
    rebalance_frequency='quarterly'
)

print("Risk Parity Portfolio Results:")
print(f"Total Return: {rp_backtest['total_return']:.2%}")
print(f"Volatility: {rp_backtest['volatility']:.2%}")
print(f"Sharpe Ratio: {rp_backtest['sharpe_ratio']:.2f}")
```

## Sentiment-Based Trading

### Twitter Sentiment Strategy

```python
from open_trading_algo.sentiment.twitter_sentiment import TwitterSentimentAnalyzer
from open_trading_algo.sentiment.sentiment_signals import SentimentSignalGenerator

# Initialize sentiment analyzer
twitter_analyzer = TwitterSentimentAnalyzer(
    api_key='your_twitter_api_key',
    api_secret='your_twitter_api_secret'
)

# Initialize signal generator
signal_generator = SentimentSignalGenerator(
    sentiment_threshold=0.2,
    confidence_threshold=0.75
)

# Analyze sentiment and generate signals
sentiment_data = twitter_analyzer.analyze_ticker_sentiment(
    'TSLA',
    lookback_hours=24,
    min_tweets=100
)

sentiment_signals = signal_generator.generate_signals(
    'TSLA',
    sentiment_data,
    data
)

print("Twitter Sentiment Signals:")
for signal in sentiment_signals[-5:]:
    print(f"{signal['date']}: {signal['signal']} "
          f"(sentiment: {signal['sentiment']:.2f})")
```

### News Sentiment Integration

```python
from open_trading_algo.sentiment.news_sentiment import NewsSentimentAnalyzer

# Initialize news analyzer
news_analyzer = NewsSentimentAnalyzer(
    api_key='your_news_api_key'
)

# Get news sentiment
news_sentiment = news_analyzer.analyze_news_sentiment(
    'AAPL',
    days_back=7
)

# Combine with technical signals
technical_signals = momentum_model.generate_signals(data)

# Weight signals
combined_signal = (
    0.6 * technical_signals['signal'] +
    0.4 * news_sentiment['sentiment_score']
)

print("Combined Technical + News Signals:")
print(f"Average Combined Signal: {combined_signal.mean():.2f}")
```

## Live Trading Examples

### Real-Time Momentum Trading

```python
from open_trading_algo.live_data.price_stream import PriceDataStreamer
from open_trading_algo.live_data.live_signal_processor import LiveSignalProcessor
from open_trading_algo.live_data.live_executor import LiveTradeExecutor

# Initialize live components
price_streamer = PriceDataStreamer(
    provider='polygon',
    api_key='your_polygon_key',
    tickers=['AAPL', 'GOOGL', 'MSFT']
)

live_processor = LiveSignalProcessor(
    signal_threshold=0.7
)

live_executor = LiveTradeExecutor(
    broker='alpaca',
    api_key='your_alpaca_key',
    secret_key='your_alpaca_secret',
    paper_trading=True
)

# Define live signal generation
def generate_live_signals(ticker, price_data, indicators):
    # Calculate momentum
    momentum = indicators.get('momentum', 0)

    if momentum > 0.05:  # Strong upward momentum
        signal = {
            'ticker': ticker,
            'action': 'BUY',
            'quantity': 100,
            'type': 'momentum'
        }
        return [signal]
    elif momentum < -0.05:  # Strong downward momentum
        signal = {
            'ticker': ticker,
            'action': 'SELL',
            'quantity': 100,
            'type': 'momentum'
        }
        return [signal]

    return []

# Set up live processing pipeline
live_processor.set_signal_callback(generate_live_signals)
live_processor.set_execution_callback(live_executor.execute_order)

# Start live trading
price_streamer.start_streaming()
live_processor.start_signal_generation()

print("Live momentum trading started...")
```

### Live Risk Management

```python
from open_trading_algo.live_data.live_risk_manager import LiveRiskManager

# Initialize live risk manager
risk_manager = LiveRiskManager(
    max_drawdown=0.05,
    max_position_size=0.1,
    max_daily_loss=0.03
)

# Define risk monitoring
def monitor_portfolio_risk():
    positions = live_executor.get_positions()
    current_prices = price_streamer.get_current_prices()

    risk_status = risk_manager.assess_risk(positions, current_prices)

    if risk_status['breached_limits']:
        print("Risk limits breached!")
        for breach in risk_status['breached_limits']:
            print(f"- {breach}")

        # Implement risk mitigation
        risk_manager.mitigate_risk(positions, current_prices)

# Set up risk monitoring
risk_manager.set_risk_callback(monitor_portfolio_risk)
risk_manager.start_risk_monitoring()

print("Live risk monitoring started...")
```

## Advanced Strategy Examples

### Machine Learning-Enhanced Strategy

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from open_trading_algo.data_enrichment.feature_engineer import FeatureEngineer

# Initialize feature engineer
feature_engineer = FeatureEngineer()

# Create features
features = feature_engineer.create_features(data)

# Create target (next day return direction)
features['target'] = np.where(data['close'].shift(-1) > data['close'], 1, 0)

# Prepare training data
X = features.drop('target', axis=1)
y = features['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Train model
ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
ml_model.fit(X_train, y_train)

# Generate ML signals
features_today = feature_engineer.create_features(data.iloc[-1:])
ml_signals = ml_model.predict_proba(features_today)

print("ML Model Prediction Probabilities:")
print(f"Up probability: {ml_signals[0][1]:.2%}")
print(f"Down probability: {ml_signals[0][0]:.2%}")
```

### Options Strategy Example

```python
from open_trading_algo.models.options_model import OptionsModel

# Initialize options model
options_model = OptionsModel(
    risk_free_rate=0.05,
    volatility_period=20
)

# Generate options signals
options_signals = options_model.generate_options_signals(
    underlying_data=data,
    option_type='call',  # or 'put'
    strike_selection='atm',  # at-the-money
    expiration_months=3
)

print("Options Strategy Signals:")
for signal in options_signals[-3:]:
    print(f"{signal['date']}: {signal['action']} "
          f"{signal['option_type']} strike=${signal['strike']:.2f}")
```

## Pipeline Examples

### Complete Trading Pipeline

```python
from open_trading_algo.pipeline.trading_pipeline import TradingPipeline

# Define pipeline configuration
pipeline_config = {
    'data_source': 'alpha_vantage',
    'cache_enabled': True,
    'indicators': ['rsi', 'macd', 'bollinger_bands'],
    'models': ['momentum', 'mean_reversion'],
    'risk_management': {
        'max_position_size': 0.1,
        'stop_loss_pct': 0.05
    },
    'execution': {
        'mode': 'paper',
        'broker': 'alpaca'
    }
}

# Initialize pipeline
pipeline = TradingPipeline(pipeline_config)

# Add custom steps
def custom_signal_filter(data, signals):
    """Custom signal filtering logic"""
    # Filter out signals with low volume
    volume_filter = data['volume'] > data['volume'].rolling(20).mean()
    filtered_signals = signals[volume_filter]
    return filtered_signals

pipeline.add_step('custom_filter', custom_signal_filter)

# Run complete pipeline
tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
results = pipeline.run_pipeline(tickers=tickers)

print("Pipeline Results:")
for ticker, result in results.items():
    print(f"{ticker}: {result['total_return']:.2%} return, "
          f"{result['sharpe_ratio']:.2f} Sharpe")
```

### Custom Strategy Pipeline

```python
# Create custom strategy class
class CustomStrategy:
    def __init__(self, params):
        self.params = params

    def generate_signals(self, data):
        # Custom signal logic
        rsi = indicators.calculate_rsi(data)
        macd = indicators.calculate_macd(data)
        bb = indicators.calculate_bollinger_bands(data)

        # Combine indicators
        signals = pd.DataFrame(index=data.index)
        signals['rsi_signal'] = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))
        signals['macd_signal'] = np.where(macd['histogram'] > 0, 1, -1)
        signals['bb_signal'] = np.where(
            data['close'] < bb['lower'], 1,
            np.where(data['close'] > bb['upper'], -1, 0)
        )

        # Majority vote
        signals['combined'] = signals.sum(axis=1)
        signals['signal'] = np.where(signals['combined'] >= 2, 1,
                                   np.where(signals['combined'] <= -2, -1, 0))

        return signals

# Use custom strategy
custom_strategy = CustomStrategy({'rsi_period': 14, 'macd_periods': (12, 26, 9)})
signals = custom_strategy.generate_signals(data)

# Backtest custom strategy
custom_results = backtester.run_backtest(signals, data)

print("Custom Strategy Results:")
print(f"Total Return: {custom_results['total_return']:.2%}")
print(f"Win Rate: {custom_results['win_rate']:.1%}")
print(f"Profit Factor: {custom_results['profit_factor']:.2f}")
```

## Performance Analysis Examples

### Strategy Comparison

```python
from open_trading_algo.backtest.performance_analyzer import PerformanceAnalyzer

# Initialize performance analyzer
performance_analyzer = PerformanceAnalyzer()

# Compare multiple strategies
strategies = {
    'momentum': momentum_model.generate_signals(data),
    'mean_reversion': mr_model.generate_signals(data),
    'combined': combined_signals
}

comparison_results = {}
for name, signals in strategies.items():
    results = backtester.run_backtest(signals, data)
    comparison_results[name] = results

# Print comparison
print("Strategy Comparison:")
print("Strategy\t\tReturn\t\tSharpe\t\tMax DD")
print("-" * 50)
for name, results in comparison_results.items():
    print(f"{name}\t\t{results['total_return']:.1%}\t\t"
          f"{results['sharpe_ratio']:.2f}\t\t{results['max_drawdown']:.1%}")
```

### Risk-Adjusted Performance

```python
# Calculate risk-adjusted metrics
for name, results in comparison_results.items():
    returns = results['returns']  # Assuming returns are available

    metrics = performance_analyzer.calculate_metrics(returns)

    print(f"\n{name} Risk Metrics:")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
    print(f"Value at Risk (95%): {metrics['var_95']:.2%}")
    print(f"Expected Shortfall: {metrics['expected_shortfall']:.2%}")
```

These examples demonstrate the versatility and power of open_trading_algo for implementing various trading strategies. Each example can be adapted and extended based on specific requirements and market conditions.</content>
<parameter name="filePath">/home/philipmai/repos/TradingViewAlgoDev/docs/examples.md
