# API Reference

This document provides comprehensive API documentation for all modules in open_trading_algo.

## Core Modules

### Data Cache (`open_trading_algo.cache.data_cache`)

#### DataCache

Main class for caching price and market data.

```python
class DataCache:
    def __init__(self, cache_dir: str = './cache', max_age_days: int = 30)
```

**Methods:**

- `store_price_data(ticker: str, data: pd.DataFrame, data_type: str = 'price')` - Store price data
- `get_price_data(ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame` - Retrieve price data
- `clear_cache(ticker: str = None, data_type: str = None)` - Clear cache data
- `get_cache_info() -> dict` - Get cache statistics

**Parameters:**
- `cache_dir`: Directory for cache files
- `max_age_days`: Maximum age of cached data in days

### Indicators (`open_trading_algo.indicators`)

#### TechnicalIndicators

Calculate technical indicators for price data.

```python
class TechnicalIndicators:
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series
    def calculate_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> dict
    def calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> dict
    def calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> dict
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series
```

#### LiveTechnicalIndicators

Real-time indicator calculations for streaming data.

```python
class LiveTechnicalIndicators:
    def __init__(self, indicators: list, periods: dict = None)
    def calculate_indicators(self, ticker: str, price_data: dict) -> dict
    def check_signals(self, ticker: str, indicators: dict) -> list
```

### Models (`open_trading_algo.models`)

#### BaseModel

Abstract base class for trading models.

```python
class BaseModel(ABC):
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
```

#### MomentumModel

Momentum-based trading model.

```python
class MomentumModel(BaseModel):
    def __init__(self, lookback_period: int = 20, threshold: float = 0.05)
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame
    def calculate_momentum(self, data: pd.DataFrame) -> pd.Series
```

**Parameters:**
- `lookback_period`: Period for momentum calculation
- `threshold`: Signal threshold for momentum

#### MeanReversionModel

Mean reversion trading model.

```python
class MeanReversionModel(BaseModel):
    def __init__(self, lookback_period: int = 20, entry_threshold: float = 2.0, exit_threshold: float = 0.5)
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame
    def calculate_zscore(self, data: pd.DataFrame) -> pd.Series
```

#### LiveTradingModel

Real-time signal generation for live trading.

```python
class LiveTradingModel:
    def __init__(self, model_type: str, signal_threshold: float = 0.7, max_positions: int = 5)
    def generate_signals(self, ticker: str, price_data: dict, indicators: dict) -> list
    def validate_signal(self, signal: dict) -> bool
```

## Signal Processing

### Signal Optimizer (`open_trading_algo.signal_optimizer`)

#### SignalOptimizer

Optimize trading signals and strategies.

```python
class SignalOptimizer:
    def __init__(self, metric: str = 'sharpe_ratio', method: str = 'grid_search')
    def optimize_parameters(self, strategy_func: callable, data: pd.DataFrame, param_ranges: dict) -> dict
    def walk_forward_optimization(self, strategy_func: callable, data: pd.DataFrame, train_days: int = 252, test_days: int = 63) -> dict
    def multi_objective_optimize(self, strategy_func: callable, objectives: dict, population_size: int = 100) -> list
```

**Methods:**
- `optimize_parameters()`: Grid search or random search parameter optimization
- `walk_forward_optimization()`: Walk-forward analysis to prevent overfitting
- `multi_objective_optimize()`: Pareto optimization for multiple objectives

### Signal Backtester (`open_trading_algo.backtest.signal_backtester`)

#### SignalBacktester

Backtest trading signals with realistic conditions.

```python
class SignalBacktester:
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001, slippage: float = 0.0005)
    def run_backtest(self, signals: pd.DataFrame, price_data: pd.DataFrame, start_date: str = None, end_date: str = None) -> dict
    def calculate_metrics(self, returns: pd.Series) -> dict
    def plot_results(self, results: dict) -> matplotlib.figure.Figure
```

**Parameters:**
- `initial_capital`: Starting portfolio value
- `commission`: Commission per trade (decimal)
- `slippage`: Slippage per trade (decimal)

## Sentiment Analysis

### Twitter Sentiment (`open_trading_algo.sentiment.twitter_sentiment`)

#### TwitterSentimentAnalyzer

Analyze sentiment from Twitter data.

```python
class TwitterSentimentAnalyzer:
    def __init__(self, api_key: str, api_secret: str, access_token: str = None, access_secret: str = None)
    def analyze_ticker_sentiment(self, ticker: str, lookback_hours: int = 24, min_tweets: int = 100) -> dict
    def get_tweets(self, query: str, count: int = 100, lang: str = 'en') -> list
    def calculate_sentiment_score(self, tweets: list) -> dict
```

### News Sentiment (`open_trading_algo.sentiment.news_sentiment`)

#### NewsSentimentAnalyzer

Analyze sentiment from financial news.

```python
class NewsSentimentAnalyzer:
    def __init__(self, api_key: str, sources: list = None)
    def analyze_news_sentiment(self, ticker: str, days_back: int = 7, min_articles: int = 10) -> dict
    def get_news_articles(self, ticker: str, days_back: int = 7) -> list
    def calculate_article_sentiment(self, article: dict) -> dict
```

### Sentiment Signals (`open_trading_algo.sentiment.sentiment_signals`)

#### SentimentSignalGenerator

Generate trading signals from sentiment data.

```python
class SentimentSignalGenerator:
    def __init__(self, sentiment_threshold: float = 0.2, confidence_threshold: float = 0.75, lookback_periods: int = 5)
    def generate_signals(self, ticker: str, sentiment_data: pd.DataFrame, price_data: pd.DataFrame) -> list
    def generate_multi_factor_signals(self, ticker: str, sentiment_sources: dict, weights: dict = None) -> dict
```

## Live Data

### Price Stream (`open_trading_algo.live_data.price_stream`)

#### PriceDataStreamer

Stream real-time price data.

```python
class PriceDataStreamer:
    def __init__(self, provider: str, api_key: str, tickers: list = None)
    def start_streaming(self) -> None
    def stop_streaming() -> None
    def set_price_callback(self, callback: callable) -> None
    def get_current_prices(self) -> dict
```

**Supported Providers:**
- 'alpha_vantage'
- 'polygon'
- 'iex'
- 'yahoo_finance'

### Live Signal Processor (`open_trading_algo.live_data.live_signal_processor`)

#### LiveSignalProcessor

Process signals in real-time.

```python
class LiveSignalProcessor:
    def __init__(self, signal_threshold: float = 0.7, max_frequency_seconds: int = 60)
    def set_signal_callback(self, callback: callable) -> None
    def set_execution_callback(self, callback: callable) -> None
    def start_signal_generation() -> None
    def stop_signal_generation() -> None
```

### Live Executor (`open_trading_algo.live_data.live_executor`)

#### LiveTradeExecutor

Execute trades in live markets.

```python
class LiveTradeExecutor:
    def __init__(self, broker: str, account_id: str, api_key: str, paper_trading: bool = True)
    def execute_order(self, order_details: dict) -> dict
    def cancel_order(self, order_id: str) -> bool
    def get_order_status(self, order_id: str) -> dict
    def get_positions(self) -> list
```

**Supported Brokers:**
- 'interactive_brokers'
- 'td_ameritrade'
- 'alpaca'
- 'etrade'

## Risk Management

### Position Sizer (`open_trading_algo.risk_management.position_sizer`)

#### VolatilityPositionSizer

Size positions based on volatility.

```python
class VolatilityPositionSizer:
    def __init__(self, risk_per_trade: float = 0.02, max_portfolio_risk: float = 0.05, volatility_lookback: int = 20)
    def calculate_position_size(self, ticker: str, current_price: float, stop_loss_price: float, portfolio_value: float, volatility: float) -> dict
    def adjust_for_correlation(self, position_sizes: dict, correlation_matrix: dict) -> dict
```

### Portfolio Risk Manager (`open_trading_algo.risk_management.portfolio_risk`)

#### PortfolioRiskManager

Manage portfolio-level risk.

```python
class PortfolioRiskManager:
    def __init__(self, max_drawdown: float = 0.1, max_daily_loss: float = 0.03, max_position_size: float = 0.1)
    def analyze_portfolio_risk(self, portfolio: dict, price_data: pd.DataFrame, correlation_lookback: int = 60) -> dict
    def calculate_correlation_matrix(self, tickers: list, price_data: pd.DataFrame, lookback_days: int = 60) -> dict
    def calculate_diversification_score(self, portfolio: dict, correlation_matrix: dict) -> float
```

### Stop Loss (`open_trading_algo.risk_management.stop_loss`)

#### DynamicStopLoss

Dynamic stop-loss management.

```python
class DynamicStopLoss:
    def __init__(self, initial_stop_pct: float = 0.05, trailing_pct: float = 0.03, volatility_adjusted: bool = True)
    def manage_stop_loss(self, position: dict, price_history: pd.DataFrame, volatility: float) -> dict
    def calculate_optimal_stop(self, entry_price: float, current_price: float, volatility: float) -> float
```

### Risk Monitor (`open_trading_algo.risk_management.risk_monitor`)

#### RiskMonitor

Monitor risk in real-time.

```python
class RiskMonitor:
    def __init__(self, alert_thresholds: dict, alert_channels: list = ['log'])
    def assess_current_risk(self, portfolio: dict, price_data: pd.DataFrame) -> dict
    def check_alerts(self, current_risk: dict) -> list
    def set_monitor_callback(self, callback: callable) -> None
    def start_monitoring(self, interval_seconds: int = 60) -> None
```

## Data Enrichment

### Data Enrichment (`open_trading_algo.data_enrichment`)

#### DataEnricher

Enrich price data with additional features.

```python
class DataEnricher:
    def __init__(self, cache_dir: str = './cache')
    def add_technical_indicators(self, data: pd.DataFrame, indicators: list = None) -> pd.DataFrame
    def add_fundamental_data(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame
    def add_sentiment_data(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame
    def add_market_data(self, data: pd.DataFrame, market_index: str = 'SPY') -> pd.DataFrame
```

### Feature Engineer (`open_trading_algo.data_enrichment.feature_engineer`)

#### FeatureEngineer

Create advanced features for machine learning.

```python
class FeatureEngineer:
    def create_lag_features(self, data: pd.DataFrame, lags: list = [1, 2, 3, 5, 10]) -> pd.DataFrame
    def create_rolling_features(self, data: pd.DataFrame, windows: list = [5, 10, 20, 50]) -> pd.DataFrame
    def create_volatility_features(self, data: pd.DataFrame, windows: list = [5, 10, 20]) -> pd.DataFrame
    def create_momentum_features(self, data: pd.DataFrame, periods: list = [5, 10, 20]) -> pd.DataFrame
```

## Pipeline

### Trading Pipeline (`open_trading_algo.pipeline`)

#### TradingPipeline

End-to-end trading pipeline.

```python
class TradingPipeline:
    def __init__(self, config: dict = None)
    def add_step(self, step_name: str, step_func: callable, **kwargs) -> None
    def run_pipeline(self, data: pd.DataFrame, tickers: list = None) -> dict
    def validate_pipeline(self) -> bool
    def get_pipeline_status(self) -> dict
```

**Built-in Steps:**
- 'data_loading': Load and cache data
- 'data_enrichment': Add features and indicators
- 'signal_generation': Generate trading signals
- 'risk_management': Apply risk controls
- 'execution': Execute trades (paper or live)

### Pipeline Config

```python
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
        'mode': 'paper',  # or 'live'
        'broker': 'alpaca'
    }
}
```

## Configuration

### Configuration Classes

#### ConfigLoader

Load configuration from files.

```python
class ConfigLoader:
    def __init__(self, config_dir: str = './config')
    def load_yaml(self, filename: str) -> dict
    def load_json(self, filename: str) -> dict
    def get_nested_value(self, config: dict, key_path: str) -> any
    def validate_config(self, config: dict, schema: dict) -> bool
```

#### APIConfig

API configuration management.

```python
class APIConfig:
    def __init__(self, config_file: str = './config/api_config.yaml')
    def get_api_key(self, provider: str) -> str
    def get_api_config(self, provider: str) -> dict
    def validate_api_keys(self) -> dict
```

## Utility Functions

### Data Utilities (`open_trading_algo.utils.data_utils`)

```python
def calculate_returns(data: pd.DataFrame, method: str = 'simple') -> pd.Series
def calculate_volatility(data: pd.DataFrame, window: int = 20, method: str = 'std') -> pd.Series
def resample_data(data: pd.DataFrame, frequency: str = 'D') -> pd.DataFrame
def fill_missing_data(data: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame
def detect_outliers(data: pd.Series, method: str = 'iqr', threshold: float = 1.5) -> pd.Series
```

### Math Utilities (`open_trading_algo.utils.math_utils`)

```python
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float
def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float
def calculate_max_drawdown(returns: pd.Series) -> float
def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float
def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float
```

### Validation Utilities (`open_trading_algo.utils.validation`)

```python
def validate_ticker(ticker: str) -> bool
def validate_date_range(start_date: str, end_date: str) -> bool
def validate_price_data(data: pd.DataFrame) -> dict
def validate_signals(signals: pd.DataFrame) -> dict
def validate_portfolio(portfolio: dict) -> dict
```

## Error Handling

### Custom Exceptions

```python
class TradingAlgoError(Exception):
    """Base exception for trading algorithm errors"""
    pass

class DataError(TradingAlgoError):
    """Data loading or processing errors"""
    pass

class APIError(TradingAlgoError):
    """API-related errors"""
    pass

class ValidationError(TradingAlgoError):
    """Data validation errors"""
    pass

class RiskError(TradingAlgoError):
    """Risk management errors"""
    pass
```

### Error Handling Patterns

```python
try:
    # Trading operation
    result = trading_pipeline.run_pipeline(data)
except DataError as e:
    logger.error(f"Data error: {e}")
    # Handle data issues
except APIError as e:
    logger.error(f"API error: {e}")
    # Retry or switch providers
except RiskError as e:
    logger.error(f"Risk error: {e}")
    # Reduce position sizes
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # General error handling
```

## Type Hints

### Common Types

```python
from typing import Dict, List, Optional, Union, Tuple
from pandas import DataFrame, Series
import numpy as np

# Data types
PriceData = DataFrame
SignalData = DataFrame
PortfolioData = Dict[str, Dict[str, Union[int, float]]]
RiskMetrics = Dict[str, float]

# Configuration types
APIConfig = Dict[str, str]
ModelConfig = Dict[str, Union[str, int, float, bool]]
PipelineConfig = Dict[str, any]

# Result types
BacktestResult = Dict[str, Union[float, DataFrame]]
OptimizationResult = Dict[str, any]
SentimentResult = Dict[str, Union[float, str, List]]
```

## Constants

### Default Values

```python
# Default periods
DEFAULT_RSI_PERIOD = 14
DEFAULT_MACD_FAST = 12
DEFAULT_MACD_SLOW = 26
DEFAULT_MACD_SIGNAL = 9
DEFAULT_BB_PERIOD = 20
DEFAULT_ATR_PERIOD = 14

# Default thresholds
DEFAULT_SIGNAL_THRESHOLD = 0.7
DEFAULT_CONFIDENCE_THRESHOLD = 0.75
DEFAULT_RISK_PER_TRADE = 0.02

# Default API settings
DEFAULT_CACHE_DIR = './cache'
DEFAULT_CONFIG_DIR = './config'
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_SECONDS = 30
```

This API reference covers the main classes, methods, and functions available in open_trading_algo. For more detailed examples and usage patterns, see the individual module documentation and examples.</content>
<parameter name="filePath">/home/philipmai/repos/TradingViewAlgoDev/docs/api-reference.md
