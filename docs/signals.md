# Signal Generation

This document covers the signal generation capabilities in open_trading_algo, including long/short signals, options signals, and sentiment-based signals.

## Overview

The signal generation system provides multiple approaches to creating trading signals:

- **Technical signals** based on indicators and price action
- **Sentiment signals** from social media and analyst ratings
- **Options signals** for derivatives trading
- **Ensemble signals** combining multiple signal types

## Signal Types

### Long/Short Equity Signals

Long/short signals are generated using technical indicators and can be customized for different strategies.

#### Basic Signal Generation

```python
from open_trading_algo.models.momentum_model import MomentumModel
from open_trading_algo.cache.data_cache import DataCache

# Initialize components
cache = DataCache()
model = MomentumModel()

# Get data and generate signals
data = cache.get_price_data('AAPL', start='2023-01-01', end='2024-01-01')
signals = model.generate_signals(data)

print(signals.head())
# Output: Series with values 1 (BUY), -1 (SELL), 0 (HOLD)
```

#### Custom Signal Rules

```python
import pandas as pd
from open_trading_algo.indicators.indicators import rsi, macd, sma

def custom_signal_strategy(data: pd.DataFrame) -> pd.Series:
    """Custom signal generation strategy."""
    # Calculate indicators
    rsi_values = rsi(data['close'], window=14)
    macd_line, signal_line, histogram = macd(data['close'])
    sma_20 = sma(data['close'], window=20)
    sma_50 = sma(data['close'], window=50)

    # Generate signals based on rules
    signals = pd.Series(0, index=data.index)

    # Buy signals
    buy_condition = (
        (rsi_values < 30) &  # Oversold
        (macd_line > signal_line) &  # MACD bullish crossover
        (data['close'] > sma_20)  # Above short-term MA
    )
    signals[buy_condition] = 1

    # Sell signals
    sell_condition = (
        (rsi_values > 70) &  # Overbought
        (macd_line < signal_line) &  # MACD bearish crossover
        (data['close'] < sma_50)  # Below long-term MA
    )
    signals[sell_condition] = -1

    return signals
```

### Options Signals

Options signals are designed for call/put option trading strategies.

```python
from open_trading_algo.signals.options_signals import OptionsSignalGenerator

# Initialize options signal generator
options_signals = OptionsSignalGenerator()

# Generate options signals
underlying_data = cache.get_price_data('AAPL')
volatility_data = cache.get_price_data('VIX')  # Volatility index

options_signals_df = options_signals.generate_signals(
    underlying_data=underlying_data,
    volatility_data=volatility_data,
    expiration_days=30,
    strike_percentages=[0.95, 1.0, 1.05]  # OTM, ATM, ITM strikes
)

print(options_signals_df.head())
# Output: DataFrame with CALL_BUY, CALL_SELL, PUT_BUY, PUT_SELL columns
```

### Sentiment Signals

Sentiment signals combine social media sentiment with traditional technical analysis.

```python
from open_trading_algo.sentiment.social_sentiment import SocialSentimentAnalyzer
from open_trading_algo.sentiment.analyst_sentiment import AnalystSentimentAnalyzer

# Initialize sentiment analyzers
social_sentiment = SocialSentimentAnalyzer(api_key='your_twitter_api_key')
analyst_sentiment = AnalystSentimentAnalyzer()

# Get sentiment data
ticker = 'AAPL'
social_score = social_sentiment.get_sentiment_score(ticker)
analyst_score = analyst_sentiment.get_analyst_ratings(ticker)

# Combine with technical signals
combined_signal = technical_signal * 0.7 + social_score * 0.2 + analyst_score * 0.1
```

## Signal Optimization

### Multi-Signal Portfolio Optimization

```python
from open_trading_algo.signal_optimizer import SignalOptimizer

# Define multiple signal strategies
signal_strategies = {
    'momentum': lambda df: momentum_signal(df),
    'mean_reversion': lambda df: mean_reversion_signal(df),
    'trend_following': lambda df: trend_following_signal(df)
}

# Initialize optimizer
optimizer = SignalOptimizer(
    data={'AAPL': aapl_data, 'GOOGL': googl_data},
    indicators={'rsi': rsi, 'macd': macd},
    signal_generators=signal_strategies
)

# Optimize signal weights
optimal_weights = optimizer.optimize_weights(
    target_return=0.15,
    risk_tolerance=0.02,
    time_horizon=252  # Trading days
)

print("Optimal signal weights:", optimal_weights)
```

### Walk-Forward Analysis

```python
# Perform walk-forward optimization
walk_forward_results = optimizer.walk_forward_optimization(
    initial_train_days=252,  # 1 year training
    test_days=63,            # 3 months testing
    step_days=21             # 1 month step
)

# Analyze performance stability
stability_metrics = optimizer.analyze_stability(walk_forward_results)
print("Signal stability metrics:", stability_metrics)
```

## Signal Validation

### Backtesting Signals

```python
from open_trading_algo.backtest.signal_backtester import SignalBacktester

# Initialize backtester
backtester = SignalBacktester(
    initial_capital=100000,
    commission=0.001,  # 0.1% per trade
    slippage=0.0005    # 0.05% slippage
)

# Run backtest
backtest_results = backtester.run_backtest(
    signals=signals,
    price_data=data,
    start_date='2023-01-01',
    end_date='2024-01-01'
)

# Analyze results
performance_metrics = backtester.calculate_metrics(backtest_results)
print("Backtest performance:", performance_metrics)
```

### Signal Quality Metrics

```python
from open_trading_algo.signals.signal_metrics import SignalMetricsAnalyzer

# Initialize analyzer
metrics_analyzer = SignalMetricsAnalyzer()

# Calculate signal quality metrics
signal_metrics = metrics_analyzer.analyze_signals(
    signals=signals,
    price_data=data,
    benchmark_returns=market_returns
)

print("Signal quality metrics:")
print(f"Win Rate: {signal_metrics['win_rate']:.2%}")
print(f"Profit Factor: {signal_metrics['profit_factor']:.2f}")
print(f"Max Drawdown: {signal_metrics['max_drawdown']:.2%}")
print(f"Sharpe Ratio: {signal_metrics['sharpe_ratio']:.2f}")
```

## Advanced Signal Features

### Ensemble Methods

```python
from open_trading_algo.signals.ensemble_signals import EnsembleSignalGenerator

# Create ensemble of different signal types
ensemble = EnsembleSignalGenerator([
    ('momentum', momentum_model, 0.4),
    ('mean_reversion', mean_reversion_model, 0.3),
    ('sentiment', sentiment_model, 0.3)
])

# Generate ensemble signals
ensemble_signals = ensemble.generate_signals(data)
```

### Machine Learning Signals

```python
from open_trading_algo.signals.ml_signals import MLSignalGenerator
from sklearn.ensemble import RandomForestClassifier

# Initialize ML signal generator
ml_signals = MLSignalGenerator(
    model=RandomForestClassifier(n_estimators=100),
    features=['rsi', 'macd', 'volume_ratio', 'price_change']
)

# Train and generate signals
ml_signals.train(data, labels)
ml_predictions = ml_signals.predict(data)
```

## Configuration

### Signal Configuration

```yaml
# config/signals.yaml
signals:
  # Technical signal parameters
  rsi_oversold: 30
  rsi_overbought: 70
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9

  # Options signal parameters
  options:
    expiration_days: 30
    strike_range: 0.1  # 10% OTM to 10% ITM
    volatility_lookback: 20

  # Sentiment parameters
  sentiment:
    social_weight: 0.3
    analyst_weight: 0.2
    news_weight: 0.1

  # Ensemble parameters
  ensemble:
    momentum_weight: 0.4
    mean_reversion_weight: 0.3
    sentiment_weight: 0.3
```

## Best Practices

### Signal Development

1. **Start Simple**: Begin with basic technical indicators before complex strategies
2. **Validate Signals**: Always backtest signals before live trading
3. **Diversify**: Use multiple signal types to reduce overfitting
4. **Monitor Performance**: Regularly review signal performance and adjust as needed

### Risk Management

1. **Position Sizing**: Use proper position sizing based on signal confidence
2. **Stop Losses**: Always implement stop-loss orders
3. **Maximum Drawdown**: Monitor and limit maximum drawdown
4. **Correlation**: Ensure signals aren't overly correlated

### Performance Monitoring

1. **Track Metrics**: Monitor win rate, profit factor, and Sharpe ratio
2. **Regular Review**: Review signal performance monthly
3. **Adaptation**: Adjust signals based on changing market conditions
4. **Documentation**: Keep detailed records of signal changes and rationale

## Troubleshooting

### Common Issues

**Signals not generating**: Check that required indicators are available and data is clean
**Poor signal quality**: Review signal logic and consider adding filters
**Overfitting**: Use out-of-sample testing and walk-forward analysis
**Market changes**: Signals may need adjustment during different market regimes

### Debug Tools

```python
# Debug signal generation
from open_trading_algo.signals.debug_signals import SignalDebugger

debugger = SignalDebugger()
debugger.analyze_signal_distribution(signals)
debugger.plot_signal_timing(signals, data)
debugger.correlation_analysis(signals, market_data)
```</content>
<parameter name="filePath">/home/philipmai/repos/TradingViewAlgoDev/docs/signals.md
