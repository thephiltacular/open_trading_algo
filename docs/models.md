# Trading Models

The `models/` directory provides a comprehensive framework for implementing and managing trading strategies. Built on an extensible architecture, it allows you to create, test, and deploy various trading models that combine technical indicators with data sources.

## Architecture Overview

```
models/
├── __init__.py              # Module exports
├── base_model.py           # Abstract base class
├── momentum_model.py       # Momentum-based strategies
├── mean_reversion_model.py # Mean reversion strategies
├── trend_following_model.py # Trend following strategies
└── README.md               # Model documentation
```

## Base Trading Model

All trading models inherit from `BaseTradingModel`, which provides:

- **Data Validation**: Ensures input data meets requirements
- **Indicator Caching**: Efficiently caches computed indicators
- **Signal Generation Framework**: Standardized signal generation interface
- **Configuration Management**: Flexible model configuration
- **Error Handling**: Robust error handling and logging

### Key Methods

```python
class BaseTradingModel(ABC):
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and prepare input data"""

    def get_indicator_value(self, indicator_name: str, **params) -> pd.Series:
        """Get cached indicator values"""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals - must be implemented by subclasses"""
```

## Available Models

### Momentum Model

Implements momentum-based trading strategies using RSI, MACD, and Stochastic indicators.

```python
from open_trading_algo.models import MomentumModel

model = MomentumModel()
signals = model.generate_signals(price_data)
```

**Key Features:**
- RSI overbought/oversold signals
- MACD crossover signals
- Stochastic confirmation
- Configurable thresholds

### Mean Reversion Model

Uses Bollinger Bands and RSI for mean reversion strategies.

```python
from open_trading_algo.models import MeanReversionModel

model = MeanReversionModel()
signals = model.generate_signals(price_data)
```

**Key Features:**
- Bollinger Band squeeze detection
- RSI divergence signals
- Price deviation calculations
- Mean reversion entry/exit points

### Trend Following Model

Combines moving averages with ADX for trend-following strategies.

```python
from open_trading_algo.models import TrendFollowingModel

model = TrendFollowingModel()
signals = model.generate_signals(price_data)
```

**Key Features:**
- Moving average crossovers
- ADX trend strength confirmation
- Trend direction detection
- Dynamic stop-loss levels

## Usage Examples

### Basic Signal Generation

```python
import pandas as pd
from open_trading_algo.models import MomentumModel

# Load your price data
price_data = pd.DataFrame({
    'Open': [...],
    'High': [...],
    'Low': [...],
    'Close': [...],
    'Volume': [...]
})

# Initialize model
model = MomentumModel()

# Generate signals
signals = model.generate_signals(price_data)
print(signals.head())
```

### Custom Model Configuration

```python
from open_trading_algo.models import MeanReversionModel

# Custom configuration
config = {
    'rsi_period': 14,
    'bb_period': 20,
    'bb_std': 2.0,
    'rsi_oversold': 30,
    'rsi_overbought': 70
}

model = MeanReversionModel(config=config)
signals = model.generate_signals(price_data)
```

### Model Comparison

```python
from open_trading_algo.models import (
    MomentumModel,
    MeanReversionModel,
    TrendFollowingModel
)

models = [
    MomentumModel(),
    MeanReversionModel(),
    TrendFollowingModel()
]

results = {}
for model in models:
    signals = model.generate_signals(price_data)
    results[model.__class__.__name__] = signals
```

## Extending the Framework

### Creating Custom Models

```python
from open_trading_algo.models.base_model import BaseTradingModel
import pandas as pd

class CustomModel(BaseTradingModel):
    def __init__(self, config=None):
        super().__init__(config)
        self.required_indicators = ['sma', 'ema', 'rsi']

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # Implement your custom signal logic
        sma = self.get_indicator_value('sma', period=20)
        ema = self.get_indicator_value('ema', period=50)
        rsi = self.get_indicator_value('rsi', period=14)

        # Your signal generation logic here
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0  # Neutral
        signals.loc[(sma > ema) & (rsi < 30), 'signal'] = 1  # Buy
        signals.loc[(sma < ema) & (rsi > 70), 'signal'] = -1  # Sell

        return signals
```

### Adding New Indicators

Models automatically cache indicators for performance. To add new indicators:

```python
class AdvancedModel(BaseTradingModel):
    def __init__(self, config=None):
        super().__init__(config)
        # Add your custom indicators to the required list
        self.required_indicators = ['sma', 'ema', 'custom_indicator']
```

## Configuration

Models support flexible configuration through dictionaries:

```python
config = {
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'stochastic_k': 14,
    'stochastic_d': 3
}

model = MomentumModel(config=config)
```

## Testing

The models include comprehensive test coverage:

```bash
# Run model tests
pytest tests/test_models.py

# Run specific model tests
pytest tests/test_models.py::test_momentum_model

# Run with coverage
pytest --cov=open_trading_algo.models tests/test_models.py
```

## Performance Considerations

- **Indicator Caching**: Computed indicators are cached to avoid recalculation
- **Data Validation**: Input data is validated once and cached
- **Memory Management**: Large datasets are processed efficiently
- **Thread Safety**: Models are designed for concurrent use

## Integration with Backtesting

Models integrate seamlessly with the backtesting framework:

```python
from open_trading_algo.backtest import Backtester
from open_trading_algo.models import MomentumModel

model = MomentumModel()
backtester = Backtester(model)
results = backtester.run(price_data)
```

## Best Practices

1. **Data Quality**: Ensure input data is clean and properly formatted
2. **Parameter Tuning**: Use historical testing to optimize model parameters
3. **Risk Management**: Always combine models with proper risk controls
4. **Regular Retraining**: Periodically re-evaluate and update model parameters
5. **Documentation**: Document custom models and their expected behavior

## API Reference

For detailed API documentation, see the docstrings in each model file:

- `BaseTradingModel`: Core functionality and abstract methods
- `MomentumModel`: Momentum strategy implementation details
- `MeanReversionModel`: Mean reversion strategy specifics
- `TrendFollowingModel`: Trend following implementation</content>
<parameter name="filePath">/home/philipmai/repos/TradingViewAlgoDev/docs/models.md
