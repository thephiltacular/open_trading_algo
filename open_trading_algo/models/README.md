# Trading Models

This directory contains trading models that combine technical indicators, data sources, and signals to create complete trading strategies.

## Overview

The models in this directory are designed to:

- **Combine Multiple Indicators**: Each model integrates various technical indicators for robust signal generation
- **Handle Multiple Data Sources**: Models can work with different data providers and formats
- **Implement Risk Management**: Position sizing and risk controls are built into each model
- **Provide Backtesting Integration**: Models are designed to work with the backtesting framework
- **Support Live Trading**: Models can be used in live trading environments

## Available Models

### BaseTradingModel
Abstract base class providing common functionality for all trading models.

**Features:**
- Data validation and preparation
- Indicator calculation and caching
- Signal generation framework
- Position sizing utilities
- Risk management integration

### MomentumModel
Momentum-based trading strategy using RSI, MACD, and Stochastic indicators.

**Use Cases:**
- Trending markets
- Short to medium-term trades
- High momentum environments

**Indicators Used:**
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator

### MeanReversionModel
Mean reversion strategy using Bollinger Bands and RSI for overbought/oversold signals.

**Use Cases:**
- Ranging markets
- Mean-reverting assets
- Risk-averse strategies

**Indicators Used:**
- Bollinger Bands
- RSI (Relative Strength Index)
- Price deviation from mean

### TrendFollowingModel
Trend following strategy using moving averages and directional indicators.

**Use Cases:**
- Strong trending markets
- Long-term position trading
- Trend confirmation strategies

**Indicators Used:**
- Moving Averages (SMA crossover)
- ADX (Average Directional Index)
- Directional Movement

## Usage Example

```python
from open_trading_algo.models import MomentumModel
from open_trading_algo.fin_data_apis import YahooFinanceAPI

# Initialize model
model = MomentumModel({
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'stoch_overbought': 80,
    'stoch_oversold': 20,
})

# Get market data
api = YahooFinanceAPI()
data = api.get_historical_data('AAPL', period='1y')

# Prepare data with indicators
prepared_data = model.prepare_data(data)

# Generate trading signals
signals = model.generate_signals(prepared_data)

# Calculate position size
capital = 10000
position_size = model.calculate_position_size(capital, risk_per_trade=0.02)

print(f"Generated {len(signals[signals != 0])} signals")
print(f"Position size: ${position_size}")
```

## Creating Custom Models

To create a custom trading model:

1. **Inherit from BaseTradingModel**:
```python
from .base_model import BaseTradingModel

class MyCustomModel(BaseTradingModel):
    def __init__(self, config=None):
        super().__init__(config)
        # Your custom configuration
```

2. **Implement required methods**:
```python
def generate_signals(self, data):
    # Your signal generation logic
    pass

def calculate_position_size(self, capital, risk_per_trade=0.02):
    # Your position sizing logic
    pass
```

3. **Add to __init__.py**:
```python
from .my_custom_model import MyCustomModel

__all__ = [..., "MyCustomModel"]
```

## Model Configuration

Models accept configuration dictionaries for customization:

```python
config = {
    'rsi_overbought': 75,      # Custom RSI thresholds
    'rsi_oversold': 25,
    'macd_signal_threshold': 0.5,  # Custom MACD sensitivity
    'position_size_multiplier': 1.5,  # Custom position sizing
}
```

## Integration with Backtesting

Models are designed to work seamlessly with the backtesting framework:

```python
from open_trading_algo.backtest import Backtester
from open_trading_algo.models import MomentumModel

model = MomentumModel()
backtester = Backtester(model, initial_capital=10000)
results = backtester.run(data)
```

## Best Practices

1. **Indicator Selection**: Choose indicators that complement each other
2. **Risk Management**: Always implement proper position sizing
3. **Signal Filtering**: Use multiple confirmations to reduce false signals
4. **Parameter Optimization**: Test different configurations
5. **Market Conditions**: Different models work better in different market conditions

## Contributing

When adding new models:

1. Follow the existing code structure
2. Include comprehensive docstrings
3. Add unit tests in `tests/test_models.py`
4. Update this README
5. Ensure compatibility with existing infrastructure
