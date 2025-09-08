# Live Data

This document covers the live data capabilities in open_trading_algo, including real-time data feeds, streaming data processing, and live trading integration.

## Overview

The live data system provides comprehensive tools for:

- **Real-time price feeds** from multiple data providers
- **Streaming data processing** with low-latency handling
- **Live signal generation** and execution
- **Data quality monitoring** and error handling
- **Multi-asset streaming** with synchronization

## Real-Time Data Feeds

### Price Data Streaming

```python
from open_trading_algo.live_data.price_stream import PriceDataStreamer
from open_trading_algo.cache.live_cache import LiveDataCache

# Initialize price streamer
price_streamer = PriceDataStreamer(
    provider='alpha_vantage',  # or 'polygon', 'iex', 'yahoo_finance'
    api_key='your_api_key',
    tickers=['AAPL', 'GOOGL', 'MSFT', 'TSLA']
)

# Initialize live cache
live_cache = LiveDataCache()

# Set up streaming callback
def on_price_update(ticker, price_data):
    print(f"{ticker}: ${price_data['price']:.2f} "
          f"({price_data['change']:.2%})")
    live_cache.store_price_data(ticker, price_data)

price_streamer.set_price_callback(on_price_update)

# Start streaming
price_streamer.start_streaming()

# Stream for 5 minutes
import time
time.sleep(300)
price_streamer.stop_streaming()
```

### Multi-Provider Data Aggregation

```python
from open_trading_algo.live_data.multi_provider_stream import MultiProviderStreamer

# Initialize multi-provider streamer
multi_streamer = MultiProviderStreamer(
    providers={
        'polygon': {
            'api_key': 'polygon_key',
            'priority': 1
        },
        'alpha_vantage': {
            'api_key': 'av_key',
            'priority': 2
        },
        'iex': {
            'api_key': 'iex_key',
            'priority': 3
        }
    },
    failover_enabled=True
)

# Start aggregated streaming
multi_streamer.start_aggregated_stream(
    tickers=['SPY', 'QQQ', 'IWM'],
    data_types=['price', 'volume', 'quotes']
)

# Monitor stream health
health_status = multi_streamer.get_stream_health()
for provider, status in health_status.items():
    print(f"{provider}: {status['status']} "
          f"(latency: {status['avg_latency_ms']:.1f}ms)")
```

## Streaming Data Processing

### Real-Time Technical Indicators

```python
from open_trading_algo.live_data.stream_processor import StreamProcessor
from open_trading_algo.indicators.live_indicators import LiveTechnicalIndicators

# Initialize stream processor
stream_processor = StreamProcessor()

# Initialize live indicators
live_indicators = LiveTechnicalIndicators(
    indicators=['rsi', 'macd', 'bollinger_bands', 'stoch'],
    periods={'rsi': 14, 'macd': (12, 26, 9), 'bb': 20}
)

# Set up indicator calculation pipeline
def process_price_data(ticker, price_data):
    # Calculate indicators
    indicators = live_indicators.calculate_indicators(ticker, price_data)

    # Check for signals
    signals = live_indicators.check_signals(ticker, indicators)

    if signals:
        print(f"Signals for {ticker}: {signals}")

# Connect to price stream
stream_processor.set_data_callback(process_price_data)
stream_processor.start_processing()
```

### Order Book Streaming

```python
from open_trading_algo.live_data.orderbook_stream import OrderBookStreamer

# Initialize order book streamer
orderbook_streamer = OrderBookStreamer(
    provider='polygon',
    api_key='polygon_key',
    depth=10  # Top 10 bids/asks
)

# Set up order book callback
def on_orderbook_update(ticker, orderbook):
    print(f"\n{ticker} Order Book:")
    print("Bids:")
    for price, size in orderbook['bids'][:5]:
        print(f"  ${price:.2f}: {size}")

    print("Asks:")
    for price, size in orderbook['asks'][:5]:
        print(f"  ${price:.2f}: {size}")

    # Calculate spread
    best_bid = orderbook['bids'][0][0]
    best_ask = orderbook['asks'][0][0]
    spread = (best_ask - best_bid) / best_bid
    print(f"Spread: {spread:.3%}")

orderbook_streamer.set_orderbook_callback(on_orderbook_update)
orderbook_streamer.start_streaming(['AAPL', 'TSLA'])
```

## Live Signal Generation

### Real-Time Signal Processing

```python
from open_trading_algo.live_data.live_signal_processor import LiveSignalProcessor
from open_trading_algo.models.live_model import LiveTradingModel

# Initialize live signal processor
signal_processor = LiveSignalProcessor()

# Initialize live trading model
live_model = LiveTradingModel(
    model_type='momentum',
    signal_threshold=0.7,
    max_positions=5
)

# Set up signal generation pipeline
def generate_live_signals(ticker, price_data, indicators):
    # Generate signals using live model
    signals = live_model.generate_signals(ticker, price_data, indicators)

    for signal in signals:
        print(f"Live Signal: {signal['type']} {ticker} "
              f"at ${signal['price']:.2f}")

        # Execute signal if confidence is high enough
        if signal['confidence'] > 0.8:
            execute_live_trade(signal)

signal_processor.set_signal_callback(generate_live_signals)
signal_processor.start_signal_generation()
```

### Live Risk Management

```python
from open_trading_algo.live_data.live_risk_manager import LiveRiskManager

# Initialize live risk manager
risk_manager = LiveRiskManager(
    max_drawdown=0.05,      # 5% max drawdown
    max_position_size=0.1,  # 10% max position
    max_daily_loss=0.03,    # 3% max daily loss
    risk_per_trade=0.01     # 1% risk per trade
)

# Monitor live positions
def monitor_positions():
    positions = risk_manager.get_current_positions()

    for ticker, position in positions.items():
        # Check risk limits
        risk_check = risk_manager.check_position_risk(ticker, position)

        if risk_check['breached_limits']:
            print(f"Risk limit breached for {ticker}: {risk_check['breached_limits']}")
            # Execute risk mitigation
            risk_manager.mitigate_risk(ticker, risk_check)

# Set up risk monitoring
risk_manager.set_risk_callback(monitor_positions)
risk_manager.start_risk_monitoring()
```

## Live Trading Integration

### Automated Execution

```python
from open_trading_algo.live_data.live_executor import LiveTradeExecutor

# Initialize live trade executor
trade_executor = LiveTradeExecutor(
    broker='interactive_brokers',  # or 'td_ameritrade', 'alpaca', etc.
    account_id='your_account_id',
    api_key='your_api_key',
    paper_trading=True  # Set to False for live trading
)

# Execute live trade
def execute_live_trade(signal):
    order_details = {
        'ticker': signal['ticker'],
        'action': signal['action'],  # 'BUY' or 'SELL'
        'quantity': signal['quantity'],
        'order_type': 'MARKET',  # or 'LIMIT', 'STOP'
        'time_in_force': 'DAY'
    }

    # Execute order
    order_result = trade_executor.execute_order(order_details)

    if order_result['status'] == 'FILLED':
        print(f"Order filled: {order_result['filled_quantity']} "
              f"at ${order_result['avg_price']:.2f}")
    else:
        print(f"Order failed: {order_result['error']}")

# Connect signal processor to executor
signal_processor.set_execution_callback(execute_live_trade)
```

### Live Portfolio Tracking

```python
from open_trading_algo.live_data.live_portfolio import LivePortfolioTracker

# Initialize live portfolio tracker
portfolio_tracker = LivePortfolioTracker(
    broker='interactive_brokers',
    account_id='your_account_id',
    update_interval_seconds=30
)

# Track live portfolio
def track_portfolio():
    portfolio = portfolio_tracker.get_portfolio_summary()

    print("Live Portfolio Summary:")
    print(f"Total Value: ${portfolio['total_value']:.2f}")
    print(f"Cash: ${portfolio['cash']:.2f}")
    print(f"Day P&L: ${portfolio['day_pnl']:.2f} ({portfolio['day_pnl_pct']:.2%})")
    print(f"Positions: {len(portfolio['positions'])}")

    for position in portfolio['positions']:
        print(f"  {position['ticker']}: {position['quantity']} "
              f"@ ${position['avg_cost']:.2f} "
              f"P&L: ${position['unrealized_pnl']:.2f}")

portfolio_tracker.set_portfolio_callback(track_portfolio)
portfolio_tracker.start_tracking()
```

## Data Quality and Monitoring

### Stream Health Monitoring

```python
from open_trading_algo.live_data.stream_monitor import StreamHealthMonitor

# Initialize stream monitor
stream_monitor = StreamHealthMonitor(
    alert_thresholds={
        'latency_ms': 1000,
        'data_gap_seconds': 30,
        'error_rate': 0.05
    }
)

# Monitor stream health
def monitor_stream_health():
    health_report = stream_monitor.generate_health_report()

    print("Stream Health Report:")
    for ticker, health in health_report.items():
        status = "✓" if health['healthy'] else "✗"
        print(f"{status} {ticker}: "
              f"Latency: {health['avg_latency_ms']:.1f}ms, "
              f"Errors: {health['error_rate']:.1%}")

    # Alert on issues
    alerts = stream_monitor.get_active_alerts()
    for alert in alerts:
        print(f"ALERT: {alert['message']}")

stream_monitor.set_monitor_callback(monitor_stream_health)
stream_monitor.start_monitoring()
```

### Data Validation

```python
from open_trading_algo.live_data.data_validator import LiveDataValidator

# Initialize data validator
data_validator = LiveDataValidator(
    validation_rules={
        'price_range': {'min': 0.01, 'max': 10000},
        'volume_spike_threshold': 5.0,  # 5x average volume
        'stale_data_threshold_seconds': 300
    }
)

# Validate incoming data
def validate_live_data(ticker, data):
    validation_result = data_validator.validate_data(ticker, data)

    if not validation_result['valid']:
        print(f"Data validation failed for {ticker}: {validation_result['errors']}")
        return False

    # Data is valid, process it
    process_valid_data(ticker, data)
    return True

# Set up validation pipeline
stream_processor.set_validation_callback(validate_live_data)
```

## Configuration

### Live Data Configuration

```yaml
# config/live_data.yaml
live_data:
  # Data Providers
  providers:
    polygon:
      api_key: ${POLYGON_API_KEY}
      priority: 1
      rate_limit: 5  # requests per second
    alpha_vantage:
      api_key: ${ALPHA_VANTAGE_API_KEY}
      priority: 2
      rate_limit: 5
    iex:
      api_key: ${IEX_API_KEY}
      priority: 3
      rate_limit: 5

  # Streaming Settings
  streaming:
    reconnect_attempts: 3
    reconnect_delay_seconds: 5
    heartbeat_interval_seconds: 30
    buffer_size: 1000

  # Data Processing
  processing:
    indicator_periods:
      rsi: 14
      macd: [12, 26, 9]
      bollinger_bands: 20
      stochastic: [14, 3, 3]
    signal_threshold: 0.7

  # Risk Management
  risk:
    max_drawdown: 0.05
    max_position_size: 0.1
    max_daily_loss: 0.03
    risk_per_trade: 0.01

  # Broker Integration
  broker:
    provider: interactive_brokers
    account_id: ${IB_ACCOUNT_ID}
    api_key: ${IB_API_KEY}
    paper_trading: true

  # Monitoring
  monitoring:
    health_check_interval_seconds: 60
    alert_thresholds:
      latency_ms: 1000
      data_gap_seconds: 30
      error_rate: 0.05
```

## Best Practices

### Performance Optimization

1. **Asynchronous Processing**: Use async/await for non-blocking operations
2. **Data Buffering**: Buffer data to handle spikes in volume
3. **Connection Pooling**: Reuse connections to reduce latency
4. **Memory Management**: Monitor memory usage with high-frequency data

### Reliability

1. **Failover Systems**: Implement automatic failover between data providers
2. **Circuit Breakers**: Stop trading if error rates exceed thresholds
3. **Data Validation**: Always validate incoming data before processing
4. **Monitoring**: Set up comprehensive monitoring and alerting

### Risk Management

1. **Position Limits**: Never exceed predefined position size limits
2. **Loss Limits**: Implement strict stop-loss and daily loss limits
3. **Circuit Breakers**: Pause trading during extreme market conditions
4. **Paper Trading**: Always test new strategies in paper trading first

## Troubleshooting

### Common Issues

**High Latency**: Check network connection and switch data providers
**Data Gaps**: Implement failover and data gap detection
**Connection Drops**: Add reconnection logic with exponential backoff
**Memory Issues**: Monitor memory usage and implement data cleanup

### Debug Tools

```python
# Debug live data streams
from open_trading_algo.live_data.live_debugger import LiveDataDebugger

debugger = LiveDataDebugger()
debugger.analyze_stream_latency(stream_data)
debugger.check_data_quality(data_samples)
debugger.plot_price_distribution(price_history)
debugger.monitor_memory_usage()
debugger.validate_trade_execution(trade_log)
```</content>
<parameter name="filePath">/home/philipmai/repos/TradingViewAlgoDev/docs/live-data.md
