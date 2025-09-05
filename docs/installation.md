# Installation & Setup

## Requirements

- Python 3.8 or higher
- pip package manager

## Installation

### From Source (Recommended)

```bash
git clone https://github.com/thephiltacular/TradingViewAlgoDev.git
cd TradingViewAlgoDev
pip install -e .
```

### Using pip (when published)

```bash
pip install tradingview-algo-dev
```

## Initial Setup

### 1. API Keys Configuration

Create a `.env` file in your project root to store API keys securely:

```bash
cp .env.example .env
```

Edit the `.env` file with your API keys:

```env
# Financial Data APIs
FINNHUB_API_KEY=your_finnhub_key_here
FMP_API_KEY=your_fmp_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
TWELVE_DATA_API_KEY=your_twelve_data_key_here
POLYGON_API_KEY=your_polygon_key_here
TIINGO_API_KEY=your_tiingo_key_here

# Optional: Database configuration
DATABASE_PATH=./data/trading_data.db
```

### 2. Database Initialization

The library automatically creates and manages a SQLite database for caching. You can customize the location:

```bash
# Optional: Create custom database config
echo "db_path: /path/to/your/database.sqlite3" > db_config.yaml
```

### 3. Verify Installation

Test your setup with a simple script:

```python
from open_trading_algo.fin_data_apis.fetchers import fetch_yahoo
from open_trading_algo.cache.data_cache import DataCache

# Test data fetching
data = fetch_yahoo(["AAPL"], ["price", "volume"])
print(f"AAPL price: {data['AAPL']['price']}")

# Test database
cache = DataCache()
print("Database initialized successfully")
```

## API Key Setup Guide

### Getting API Keys

#### Finnhub (Recommended for real-time quotes)
1. Visit [finnhub.io](https://finnhub.io/)
2. Sign up for a free account
3. Navigate to Dashboard → API Keys
4. Copy your API key
5. Free tier: 60 calls/minute, 1,440 calls/day

#### Alpha Vantage (Good for fundamental data)
1. Visit [alphavantage.co](https://www.alphavantage.co/)
2. Click "Get your free API key today"
3. Sign up and verify your email
4. Copy your API key from the dashboard
5. Free tier: 5 calls/minute, 500 calls/day

#### Financial Modeling Prep (FMP)
1. Visit [financialmodelingprep.com](https://financialmodelingprep.com/)
2. Sign up for a free account
3. Go to Dashboard → API Keys
4. Copy your API key
5. Free tier: 5 calls/minute, 250 calls/day

#### Twelve Data
1. Visit [twelvedata.com](https://twelvedata.com/)
2. Sign up for a free account
3. Navigate to Dashboard → API Keys
4. Copy your API key
5. Free tier: 8 calls/minute, 800 calls/day

#### Polygon.io
1. Visit [polygon.io](https://polygon.io/)
2. Sign up for a free account
3. Go to Dashboard → API Keys
4. Copy your API key
5. Free tier: 5 calls/minute

#### Tiingo
1. Visit [api.tiingo.com](https://api.tiingo.com/)
2. Sign up for a free account
3. Navigate to Account → API
4. Copy your API token
5. Free tier: Varies by endpoint

### Environment Variables

Alternatively, you can set environment variables directly:

```bash
export FINNHUB_API_KEY="your_key_here"
export ALPHA_VANTAGE_API_KEY="your_key_here"
# ... etc
```

## Configuration Files

### Live Data Configuration

Create `live_data_config.yaml` for real-time feeds:

```yaml
source: "finnhub"  # or "yahoo", "alpha_vantage", etc.
tickers: ["AAPL", "GOOGL", "MSFT", "TSLA"]
fields: ["price", "volume", "high", "low", "open"]
update_rate: 60  # seconds between updates
api_key: null  # Uses .env file if null
```

### Model Configuration

Create `cols_model.yaml` to customize technical indicators:

```yaml
# EMAs to compute
ema_periods: [5, 10, 20, 50, 200]

# RSI settings
rsi_period: 14
rsi_overbought: 70
rsi_oversold: 30

# MACD settings
macd_fast: 12
macd_slow: 26
macd_signal: 9

# Bollinger Bands
bb_period: 20
bb_std: 2
```

## Docker Setup (Optional)

For containerized deployments:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["python", "your_trading_script.py"]
```

## Development Setup

For contributors and advanced users:

```bash
# Clone repository
git clone https://github.com/thephiltacular/TradingViewAlgoDev.git
cd TradingViewAlgoDev

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

## Troubleshooting Installation

### Common Issues

1. **Import errors**: Ensure you're using Python 3.8+
2. **API key errors**: Verify your `.env` file is in the correct location
3. **Database errors**: Check write permissions in your project directory
4. **Network errors**: Verify internet connection and API endpoints

### Dependency Conflicts

If you encounter dependency conflicts:

```bash
# Create fresh virtual environment
python -m venv fresh_env
source fresh_env/bin/activate
pip install --upgrade pip
pip install -e .
```

### Performance Optimization

For better performance:

```bash
# Install optional performance dependencies
pip install numba pandas[performance] sqlalchemy[postgresql]
```

Next: [Quick Start Guide](quickstart.md)
