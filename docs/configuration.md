# Configuration Guide

open_trading_algo uses YAML configuration files to customize behavior across all modules. This guide covers all configuration options and best practices.

## Configuration Files Overview

| File | Purpose | Location |
|------|---------|----------|
| `live_data_config.yaml` | Real-time data feed settings | Project root |
| `cols_model.yaml` | Technical indicator parameters | Project root |
| `cols_alerts.yaml` | Alert and signal thresholds | Project root |
| `db_config.yaml` | Database connection settings | Project root |
| `api_config.yaml` | API endpoints and rate limits | Library root |
| `.env` | Secure API key storage | Project root |

## Environment Configuration (.env)

Store all sensitive information in a `.env` file:

```env
# Financial Data API Keys
FINNHUB_API_KEY=your_finnhub_key_here
FMP_API_KEY=your_fmp_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
TWELVE_DATA_API_KEY=your_twelve_data_key_here
POLYGON_API_KEY=your_polygon_key_here
TIINGO_API_KEY=your_tiingo_key_here

# Optional: Database settings
DATABASE_PATH=./data/trading_cache.db
DATABASE_TIMEOUT=30

# Optional: Performance settings
MAX_WORKERS=10
CACHE_TTL_HOURS=4

# Optional: Risk management
MAX_POSITION_SIZE=0.05  # 5% max position
DEFAULT_STOP_LOSS=0.02  # 2% stop loss
```

## Live Data Configuration

### Basic Live Data Setup

```yaml
# live_data_config.yaml
source: "yahoo"  # yahoo, finnhub, alpha_vantage, fmp, twelve_data
tickers:
  - "AAPL"
  - "GOOGL"
  - "MSFT"
  - "TSLA"
  - "AMZN"
fields:
  - "price"
  - "volume"
  - "high"
  - "low"
  - "open"
  - "previous_close"
update_rate: 60  # seconds between updates
api_key: null    # Uses .env file if null
```

### Advanced Live Data Configuration

```yaml
# live_data_config_advanced.yaml
source: "finnhub"
tickers:
  # Large cap tech
  - "AAPL"
  - "GOOGL"
  - "MSFT"
  - "AMZN"
  - "META"

  # Growth stocks
  - "TSLA"
  - "NVDA"
  - "AMD"

  # ETFs
  - "SPY"
  - "QQQ"

fields:
  - "price"
  - "volume"
  - "high"
  - "low"
  - "open"
  - "previous_close"
  - "change"
  - "percent_change"

# Timing configuration
update_rate: 30          # 30 seconds between updates
market_hours_only: true  # Only update during market hours
timezone: "US/Eastern"   # Market timezone

# Error handling
max_retries: 3
backoff_factor: 2        # Exponential backoff multiplier
timeout_seconds: 10

# Performance settings
batch_size: 20           # Tickers per API call
enable_caching: true
cache_duration: 300      # 5 minutes cache TTL

# Failover configuration
fallback_sources:
  - "yahoo"
  - "alpha_vantage"
auto_failover: true

# Callbacks
callbacks:
  on_update: "process_live_data"      # Function name to call
  on_error: "handle_data_error"       # Error handler
  on_reconnect: "log_reconnection"    # Reconnection handler
```

### Multiple Feed Configuration

```yaml
# multi_feed_config.yaml
feeds:
  primary:
    source: "finnhub"
    tickers: ["AAPL", "GOOGL", "MSFT"]
    fields: ["price", "volume"]
    update_rate: 15

  secondary:
    source: "yahoo"
    tickers: ["TSLA", "AMZN", "META"]
    fields: ["price", "volume", "high", "low"]
    update_rate: 60

  options:
    source: "polygon"
    tickers: ["AAPL", "MSFT"]  # Options on these underlyings
    fields: ["bid", "ask", "volume", "open_interest"]
    update_rate: 30
    options_specific:
      expiration_range: 30  # Days from now
      option_types: ["call", "put"]
      min_volume: 100
```

## Technical Indicators Configuration

### Basic Indicator Settings

```yaml
# cols_model.yaml
# Moving Averages
ema_periods: [5, 10, 20, 50, 100, 200]
sma_periods: [10, 20, 50, 200]

# Oscillators
rsi:
  period: 14
  overbought: 70
  oversold: 30

stochastic:
  k_period: 14
  d_period: 3
  smooth: 3

williams_r:
  period: 14
  overbought: -20
  oversold: -80

# Trend Indicators
macd:
  fast_period: 12
  slow_period: 26
  signal_period: 9

adx:
  period: 14
  trend_threshold: 25

# Volatility Indicators
bollinger_bands:
  period: 20
  std_dev: 2

atr:
  period: 14
```

### Advanced Indicator Configuration

```yaml
# cols_model_advanced.yaml
# Multi-timeframe analysis
timeframes:
  - "1m"   # 1 minute
  - "5m"   # 5 minutes
  - "15m"  # 15 minutes
  - "1h"   # 1 hour
  - "1d"   # 1 day
  - "1w"   # 1 week

# Custom indicators
custom_indicators:
  vwap:
    enabled: true
    periods: [20, 50]

  fibonacci:
    enabled: true
    lookback_periods: [50, 100, 200]
    levels: [0.236, 0.382, 0.5, 0.618, 0.786]

  volume_profile:
    enabled: true
    bins: 20
    lookback_days: 30

# Advanced oscillators
custom_oscillators:
  rsi_divergence:
    enabled: true
    lookback_periods: 20
    divergence_threshold: 5

  macd_histogram:
    enabled: true
    signal_threshold: 0.01

  momentum:
    periods: [10, 20, 50]
    threshold: 5

# Sector/Market indicators
market_indicators:
  vix:
    enabled: true
    threshold: 20

  advance_decline:
    enabled: true
    symbols: ["$ADVN", "$DECL"]

  sector_rotation:
    enabled: true
    sectors:
      - "XLF"  # Financial
      - "XLT"  # Technology
      - "XLE"  # Energy
      - "XLV"  # Healthcare

# Performance optimization
optimization:
  parallel_calculation: true
  max_workers: 4
  chunk_size: 100
  enable_caching: true
  cache_indicators: true
```

## Alert Configuration

### Basic Alert Settings

```yaml
# cols_alerts.yaml
price_alerts:
  - ticker: "AAPL"
    alert_type: "price_above"
    threshold: 200.0
    message: "AAPL above $200"

  - ticker: "TSLA"
    alert_type: "price_below"
    threshold: 150.0
    message: "TSLA below $150"

volume_alerts:
  - ticker: "GOOGL"
    alert_type: "volume_spike"
    threshold_multiplier: 2.0  # 2x average volume
    lookback_days: 20

signal_alerts:
  - signal_type: "rsi_oversold"
    tickers: ["AAPL", "GOOGL", "MSFT"]
    message: "RSI oversold signal"

  - signal_type: "macd_bullish_crossover"
    tickers: "ALL"  # All tracked tickers
    message: "MACD bullish crossover"
```

### Advanced Alert Configuration

```yaml
# cols_alerts_advanced.yaml
# Multi-condition alerts
complex_alerts:
  momentum_breakout:
    conditions:
      - type: "price_above_ma"
        ma_period: 50
      - type: "volume_above_average"
        multiplier: 1.5
      - type: "rsi_above"
        threshold: 60
    tickers: ["AAPL", "GOOGL", "MSFT"]
    message: "Momentum breakout detected"

  reversal_setup:
    conditions:
      - type: "rsi_oversold"
        threshold: 30
      - type: "price_near_support"
        tolerance: 0.02  # 2%
      - type: "volume_spike"
        multiplier: 1.5
    tickers: ["TSLA", "AMZN"]
    message: "Potential reversal setup"

# Alert delivery methods
delivery:
  email:
    enabled: true
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    username_env: "EMAIL_USERNAME"
    password_env: "EMAIL_PASSWORD"
    recipients: ["trader@example.com"]

  webhook:
    enabled: true
    url: "https://hooks.slack.com/your/webhook/url"
    format: "slack"

  file:
    enabled: true
    path: "./alerts.log"
    format: "json"

# Alert frequency controls
frequency:
  max_alerts_per_hour: 20
  cooldown_minutes: 15      # Per ticker cooldown
  duplicate_suppression: 60 # Minutes to suppress duplicates

# Market hours filtering
market_hours:
  enabled: true
  timezone: "US/Eastern"
  start_time: "09:30"
  end_time: "16:00"
  trading_days_only: true
```

## Database Configuration

### Basic Database Setup

```yaml
# db_config.yaml
db_path: "./data/trading_cache.db"
timeout: 30
enable_wal_mode: true
```

### Advanced Database Configuration

```yaml
# db_config_advanced.yaml
# Connection settings
connection:
  db_path: "/data/trading/cache.db"
  timeout: 30
  check_same_thread: false
  enable_wal_mode: true

# Performance tuning
performance:
  cache_size: 10000        # Pages in memory
  temp_store: "memory"     # Use memory for temp tables
  synchronous: "normal"    # WAL mode synchronization
  journal_mode: "wal"      # Write-ahead logging

# Maintenance
maintenance:
  auto_vacuum: "incremental"
  page_size: 4096
  max_page_count: 1000000

# Backup settings
backup:
  enabled: true
  interval_hours: 24
  retention_days: 30
  backup_path: "/backups/trading/"

# Data retention
retention:
  price_data_days: 730     # 2 years
  signal_data_days: 365    # 1 year
  alert_data_days: 90      # 3 months
  auto_cleanup: true
  cleanup_interval_hours: 168  # Weekly
```

## API Configuration

The library includes a comprehensive API configuration file:

```yaml
# api_config.yaml (read-only, part of library)
finnhub:
  base_url: "https://finnhub.io/api/v1"
  free_limit_per_minute: 60
  free_limit_per_day: 1440
  endpoints:
    quote: "/quote"
    company_profile: "/stock/profile2"

alpha_vantage:
  base_url: "https://www.alphavantage.co/query"
  free_limit_per_minute: 5
  free_limit_per_day: 500
  endpoints:
    quote: "?function=GLOBAL_QUOTE"
    intraday: "?function=TIME_SERIES_INTRADAY"

# ... other APIs
```

## Usage in Code

### Loading Configurations

```python
import yaml
from pathlib import Path


def load_config(config_name):
    """Load a configuration file"""
    config_path = Path(f"{config_name}.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


# Load configurations
live_config = load_config("live_data_config")
model_config = load_config("cols_model")
alert_config = load_config("cols_alerts")
```

### Using Configurations

```python
from open_trading_algo.fin_data_apis.feed import LiveDataFeed
from open_trading_algo.indicators.indicators import calculate_rsi

# Use live data config
feed = LiveDataFeed(Path("live_data_config.yaml"))

# Use model config for RSI calculation
model_config = load_config("cols_model")
rsi_config = model_config.get("rsi", {})

rsi = calculate_rsi(df["Close"], period=rsi_config.get("period", 14))

# Check RSI thresholds from config
overbought = rsi_config.get("overbought", 70)
oversold = rsi_config.get("oversold", 30)

overbought_signals = rsi > overbought
oversold_signals = rsi < oversold
```

### Dynamic Configuration Updates

```python
class ConfigManager:
    """Dynamic configuration management"""

    def __init__(self):
        self.configs = {}
        self.last_modified = {}

    def get_config(self, config_name, auto_reload=True):
        """Get config with optional auto-reload"""

        config_path = Path(f"{config_name}.yaml")

        if auto_reload and config_path.exists():
            current_mtime = config_path.stat().st_mtime
            last_mtime = self.last_modified.get(config_name, 0)

            if current_mtime > last_mtime:
                print(f"Reloading {config_name} configuration")
                with open(config_path, "r") as f:
                    self.configs[config_name] = yaml.safe_load(f)
                self.last_modified[config_name] = current_mtime

        return self.configs.get(config_name, {})

    def update_config(self, config_name, updates):
        """Update configuration dynamically"""

        config = self.get_config(config_name)
        config.update(updates)

        # Save back to file
        config_path = Path(f"{config_name}.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Update cache
        self.configs[config_name] = config
        self.last_modified[config_name] = time.time()


# Usage
config_manager = ConfigManager()

# Get current config (auto-reloads if file changed)
live_config = config_manager.get_config("live_data_config")

# Update config programmatically
config_manager.update_config(
    "live_data_config",
    {
        "update_rate": 30,  # Change from 60 to 30 seconds
        "tickers": ["AAPL", "GOOGL", "MSFT", "NVDA"],  # Add NVDA
    },
)
```

## Environment-Specific Configurations

### Development Configuration

```yaml
# dev_config.yaml
environment: "development"

logging:
  level: "DEBUG"
  console: true
  file: "./logs/dev.log"

data_sources:
  - "yahoo"  # Free sources only

cache:
  enabled: true
  ttl: 300  # 5 minutes for quick testing

alerts:
  enabled: false  # Disable in dev

performance:
  parallel_processing: false
  max_workers: 2
```

### Production Configuration

```yaml
# prod_config.yaml
environment: "production"

logging:
  level: "INFO"
  console: false
  file: "/var/log/trading/app.log"
  rotation: "daily"
  retention: 30

data_sources:
  primary: "finnhub"
  fallback: ["alpha_vantage", "yahoo"]

cache:
  enabled: true
  ttl: 3600  # 1 hour

alerts:
  enabled: true
  rate_limit: 50  # Max per hour

performance:
  parallel_processing: true
  max_workers: 8

monitoring:
  health_check_interval: 60
  metrics_enabled: true

security:
  api_key_rotation: true
  request_signing: true
```

## Configuration Validation

```python
import jsonschema

# Define schema for live data config
live_data_schema = {
    "type": "object",
    "properties": {
        "source": {
            "type": "string",
            "enum": ["yahoo", "finnhub", "alpha_vantage", "fmp", "twelve_data"],
        },
        "tickers": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "fields": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "update_rate": {"type": "integer", "minimum": 1},
    },
    "required": ["source", "tickers", "fields", "update_rate"],
}


def validate_config(config_data, schema):
    """Validate configuration against schema"""
    try:
        jsonschema.validate(config_data, schema)
        return True, "Valid"
    except jsonschema.ValidationError as e:
        return False, str(e)


# Validate configuration
config = load_config("live_data_config")
is_valid, message = validate_config(config, live_data_schema)

if not is_valid:
    print(f"Configuration error: {message}")
else:
    print("Configuration is valid")
```

## Next Steps

- [Live Data Feeds](live-data.md) - Apply configuration to real-time feeds
- [Technical Indicators](indicators.md) - Use indicator configuration
- [Signal Generation](signals.md) - Configure signal parameters
