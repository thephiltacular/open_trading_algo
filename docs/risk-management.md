# Risk Management

This document covers the risk management capabilities in open_trading_algo, including position sizing, portfolio risk controls, and risk monitoring systems.

## Overview

The risk management system provides comprehensive tools for:

- **Position sizing** based on volatility and risk tolerance
- **Portfolio risk controls** with diversification and correlation analysis
- **Stop-loss management** with trailing stops and dynamic levels
- **Risk monitoring** with real-time alerts and reporting
- **Stress testing** and scenario analysis

## Position Sizing

### Volatility-Based Position Sizing

```python
from open_trading_algo.risk_management.position_sizer import VolatilityPositionSizer
from open_trading_algo.indicators.volatility import VolatilityCalculator

# Initialize position sizer
position_sizer = VolatilityPositionSizer(
    risk_per_trade=0.02,      # 2% risk per trade
    max_portfolio_risk=0.05,  # 5% max portfolio risk
    volatility_lookback=20     # 20-day volatility
)

# Initialize volatility calculator
volatility_calc = VolatilityCalculator()

# Calculate position size for a trade
ticker = 'AAPL'
current_price = 150.00
stop_loss_price = 142.50  # 5% stop loss
portfolio_value = 100000

# Get volatility
volatility = volatility_calc.calculate_atr(ticker, lookback=20)

# Calculate position size
position_size = position_sizer.calculate_position_size(
    ticker=ticker,
    current_price=current_price,
    stop_loss_price=stop_loss_price,
    portfolio_value=portfolio_value,
    volatility=volatility
)

print("Position Sizing Analysis:")
print(f"Volatility (ATR): {volatility:.2f}")
print(f"Risk per Trade: ${position_size['risk_amount']:.2f}")
print(f"Position Size: {position_size['quantity']} shares")
print(f"Position Value: ${position_size['position_value']:.2f}")
print(f"Portfolio Allocation: {position_size['portfolio_allocation']:.2%}")
```

### Kelly Criterion Position Sizing

```python
from open_trading_algo.risk_management.kelly_sizer import KellyPositionSizer

# Initialize Kelly position sizer
kelly_sizer = KellyPositionSizer(
    fraction=0.5,  # Use half of Kelly for conservatism
    max_position_size=0.1  # 10% max position
)

# Calculate Kelly position size
win_probability = 0.55  # 55% win rate
win_loss_ratio = 2.0     # 2:1 reward-to-risk ratio

kelly_size = kelly_sizer.calculate_kelly_size(
    win_probability=win_probability,
    win_loss_ratio=win_loss_ratio,
    current_price=current_price,
    stop_loss_price=stop_loss_price,
    portfolio_value=portfolio_value
)

print("Kelly Criterion Sizing:")
print(f"Full Kelly Size: {kelly_size['full_kelly']:.2%}")
print(f"Conservative Size: {kelly_size['conservative_kelly']:.2%}")
print(f"Recommended Position: {kelly_size['recommended_quantity']} shares")
```

### Fixed Fractional Position Sizing

```python
from open_trading_algo.risk_management.fixed_fractional import FixedFractionalSizer

# Initialize fixed fractional sizer
fixed_sizer = FixedFractionalSizer(
    risk_fraction=0.01,  # 1% risk per trade
    max_risk_fraction=0.02  # 2% max risk
)

# Calculate fixed fractional position
fixed_position = fixed_sizer.calculate_position(
    current_price=current_price,
    stop_loss_price=stop_loss_price,
    portfolio_value=portfolio_value,
    volatility_adjustment=True
)

print("Fixed Fractional Sizing:")
print(f"Risk Amount: ${fixed_position['risk_amount']:.2f}")
print(f"Position Size: {fixed_position['quantity']} shares")
print(f"Risk Percentage: {fixed_position['risk_percentage']:.2%}")
```

## Portfolio Risk Controls

### Portfolio Diversification Analysis

```python
from open_trading_algo.risk_management.portfolio_risk import PortfolioRiskManager

# Initialize portfolio risk manager
portfolio_risk = PortfolioRiskManager()

# Define current portfolio
portfolio = {
    'AAPL': {'quantity': 100, 'avg_cost': 150.00},
    'GOOGL': {'quantity': 50, 'avg_cost': 2500.00},
    'MSFT': {'quantity': 75, 'avg_cost': 300.00},
    'TSLA': {'quantity': 25, 'avg_cost': 800.00}
}

# Analyze portfolio risk
risk_analysis = portfolio_risk.analyze_portfolio_risk(
    portfolio=portfolio,
    price_data=price_history,
    correlation_lookback=60  # 60-day correlations
)

print("Portfolio Risk Analysis:")
print(f"Portfolio Volatility: {risk_analysis['portfolio_volatility']:.2%}")
print(f"Value at Risk (95%): ${risk_analysis['var_95']:.2f}")
print(f"Expected Shortfall: ${risk_analysis['expected_shortfall']:.2f}")
print(f"Maximum Drawdown: {risk_analysis['max_drawdown']:.2%}")
print(f"Sharpe Ratio: {risk_analysis['sharpe_ratio']:.2f}")
```

### Correlation-Based Risk Control

```python
# Analyze position correlations
correlation_matrix = portfolio_risk.calculate_correlation_matrix(
    tickers=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'],
    price_data=price_history,
    lookback_days=60
)

print("Correlation Matrix:")
for ticker1 in correlation_matrix:
    for ticker2 in correlation_matrix[ticker1]:
        if ticker1 != ticker2:
            print(f"{ticker1}-{ticker2}: {correlation_matrix[ticker1][ticker2]:.3f}")

# Check diversification
diversification_score = portfolio_risk.calculate_diversification_score(
    portfolio=portfolio,
    correlation_matrix=correlation_matrix
)

print(f"Portfolio Diversification Score: {diversification_score:.2f}")
```

### Risk Parity Allocation

```python
from open_trading_algo.risk_management.risk_parity import RiskParityAllocator

# Initialize risk parity allocator
risk_parity = RiskParityAllocator(
    target_volatility=0.15,  # 15% target volatility
    rebalance_threshold=0.05  # 5% rebalance threshold
)

# Calculate risk parity weights
risk_parity_weights = risk_parity.calculate_weights(
    assets=['SPY', 'BND', 'GLD', 'VNQ'],
    price_data=etf_price_history,
    lookback_days=252
)

print("Risk Parity Allocation:")
for asset, weight in risk_parity_weights.items():
    print(f"{asset}: {weight:.2%}")

# Check if rebalancing is needed
rebalance_needed = risk_parity.check_rebalance_needed(
    current_weights=current_portfolio_weights,
    target_weights=risk_parity_weights
)

if rebalance_needed:
    print("Rebalancing recommended")
    trades = risk_parity.generate_rebalance_trades(
        current_weights=current_portfolio_weights,
        target_weights=risk_parity_weights,
        portfolio_value=portfolio_value
    )
    print("Rebalancing Trades:", trades)
```

## Stop-Loss Management

### Dynamic Stop-Loss

```python
from open_trading_algo.risk_management.stop_loss import DynamicStopLoss

# Initialize dynamic stop loss
dynamic_stop = DynamicStopLoss(
    initial_stop_pct=0.05,    # 5% initial stop
    trailing_pct=0.03,        # 3% trailing stop
    volatility_adjusted=True,
    time_based=True
)

# Manage stop loss for a position
position = {
    'ticker': 'AAPL',
    'quantity': 100,
    'entry_price': 150.00,
    'current_price': 165.00
}

stop_loss_info = dynamic_stop.manage_stop_loss(
    position=position,
    price_history=price_history,
    volatility=volatility
)

print("Dynamic Stop Loss:")
print(f"Current Stop Price: ${stop_loss_info['stop_price']:.2f}")
print(f"Stop Type: {stop_loss_info['stop_type']}")
print(f"Risk Amount: ${stop_loss_info['risk_amount']:.2f}")
print(f"Distance to Stop: {stop_loss_info['distance_to_stop']:.2%}")
```

### Multi-Level Stop Loss

```python
from open_trading_algo.risk_management.multi_stop import MultiLevelStopLoss

# Initialize multi-level stop loss
multi_stop = MultiLevelStopLoss(
    levels=[
        {'percentage': 0.03, 'quantity': 0.25},  # 3% stop, sell 25%
        {'percentage': 0.05, 'quantity': 0.50},  # 5% stop, sell 50%
        {'percentage': 0.08, 'quantity': 1.00}   # 8% stop, sell remaining
    ]
)

# Calculate multi-level stops
multi_stops = multi_stop.calculate_stops(
    entry_price=150.00,
    position_size=100
)

print("Multi-Level Stop Loss:")
for i, stop in enumerate(multi_stops):
    print(f"Level {i+1}: ${stop['price']:.2f} "
          f"({stop['percentage']:.1%}, sell {stop['quantity_to_sell']})")
```

## Risk Monitoring and Alerts

### Real-Time Risk Monitoring

```python
from open_trading_algo.risk_management.risk_monitor import RiskMonitor

# Initialize risk monitor
risk_monitor = RiskMonitor(
    alert_thresholds={
        'max_drawdown': 0.05,
        'daily_loss': 0.03,
        'position_size': 0.1,
        'portfolio_volatility': 0.2
    },
    alert_channels=['email', 'sms', 'log']
)

# Monitor portfolio risk in real-time
def monitor_portfolio_risk():
    current_risk = risk_monitor.assess_current_risk(
        portfolio=portfolio,
        price_data=current_prices
    )

    # Check for alerts
    alerts = risk_monitor.check_alerts(current_risk)

    if alerts:
        print("Risk Alerts:")
        for alert in alerts:
            print(f"⚠️ {alert['type']}: {alert['message']}")

    return current_risk

# Set up monitoring
risk_monitor.set_monitor_callback(monitor_portfolio_risk)
risk_monitor.start_monitoring(interval_seconds=60)
```

### Risk Reporting

```python
from open_trading_algo.risk_management.risk_report import RiskReporter

# Initialize risk reporter
risk_reporter = RiskReporter()

# Generate comprehensive risk report
risk_report = risk_reporter.generate_risk_report(
    portfolio=portfolio,
    price_history=price_history,
    trades=trade_history,
    report_period='monthly'
)

print("Risk Report Summary:")
print(f"Portfolio Value: ${risk_report['portfolio_value']:.2f}")
print(f"Total Risk: ${risk_report['total_risk']:.2f}")
print(f"Risk-Adjusted Return: {risk_report['risk_adjusted_return']:.2%}")
print(f"Stress Test Results: {risk_report['stress_test_results']}")

# Export report
risk_reporter.export_report(
    report=risk_report,
    format='pdf',
    filename='monthly_risk_report.pdf'
)
```

## Stress Testing and Scenario Analysis

### Historical Stress Testing

```python
from open_trading_algo.risk_management.stress_test import StressTester

# Initialize stress tester
stress_tester = StressTester()

# Define stress scenarios
scenarios = {
    'covid_crash': {
        'description': 'COVID-19 market crash scenario',
        'returns': covid_crash_returns,
        'probability': 0.05
    },
    'tech_bubble': {
        'description': 'Tech bubble burst scenario',
        'returns': tech_bubble_returns,
        'probability': 0.03
    },
    'interest_rate_hike': {
        'description': 'Fed rate hike scenario',
        'returns': rate_hike_returns,
        'probability': 0.10
    }
}

# Run stress tests
stress_results = stress_tester.run_stress_tests(
    portfolio=portfolio,
    scenarios=scenarios,
    confidence_levels=[0.95, 0.99]
)

print("Stress Test Results:")
for scenario, results in stress_results.items():
    print(f"{scenario}:")
    print(f"  Loss at 95%: ${results['loss_95']:.2f}")
    print(f"  Loss at 99%: ${results['loss_99']:.2f}")
    print(f"  Probability: {results['probability']:.1%}")
```

### Monte Carlo Simulation

```python
from open_trading_algo.risk_management.monte_carlo import MonteCarloRisk

# Initialize Monte Carlo risk analyzer
mc_risk = MonteCarloRisk(
    num_simulations=10000,
    time_horizon_days=252,
    confidence_level=0.95
)

# Run Monte Carlo analysis
mc_results = mc_risk.run_monte_carlo(
    portfolio=portfolio,
    historical_returns=historical_returns,
    num_simulations=10000
)

print("Monte Carlo Risk Analysis:")
print(f"Expected Portfolio Value: ${mc_results['expected_value']:.2f}")
print(f"Value at Risk (95%): ${mc_results['var_95']:.2f}")
print(f"Expected Shortfall: ${mc_results['expected_shortfall']:.2f}")
print(f"Worst Case (1%): ${mc_results['worst_case']:.2f}")
print(f"Probability of Loss: {mc_results['prob_loss']:.2%}")
```

## Configuration

### Risk Management Configuration

```yaml
# config/risk_management.yaml
risk_management:
  # Position Sizing
  position_sizing:
    risk_per_trade: 0.02      # 2% risk per trade
    max_portfolio_risk: 0.05  # 5% max portfolio risk
    volatility_lookback: 20
    kelly_fraction: 0.5       # Conservative Kelly fraction

  # Portfolio Risk
  portfolio:
    max_drawdown: 0.10        # 10% max drawdown
    max_daily_loss: 0.03      # 3% max daily loss
    max_position_size: 0.10   # 10% max position
    min_diversification: 0.7  # 70% minimum diversification

  # Stop Loss
  stop_loss:
    initial_stop_pct: 0.05    # 5% initial stop
    trailing_stop_pct: 0.03   # 3% trailing stop
    volatility_adjusted: true
    time_based_stops: true

  # Risk Monitoring
  monitoring:
    alert_thresholds:
      max_drawdown: 0.05
      daily_loss: 0.03
      position_size: 0.1
      portfolio_volatility: 0.2
    check_interval_seconds: 60
    alert_channels: ['email', 'log']

  # Stress Testing
  stress_testing:
    num_simulations: 10000
    confidence_levels: [0.95, 0.99]
    scenarios:
      - covid_crash
      - tech_bubble
      - interest_rate_hike
```

## Best Practices

### Risk Control Principles

1. **Risk First**: Always determine risk before calculating position size
2. **Diversification**: Never put all eggs in one basket
3. **Stop Losses**: Always use stop losses, never remove them
4. **Position Limits**: Never exceed predefined position size limits
5. **Regular Monitoring**: Monitor risk metrics continuously

### Implementation Guidelines

1. **Conservative Sizing**: Use conservative position sizing methods
2. **Multiple Risk Measures**: Use multiple risk metrics, not just one
3. **Scenario Planning**: Plan for various market scenarios
4. **Regular Review**: Review and adjust risk parameters regularly
5. **Automation**: Automate risk monitoring and alerts

### Common Pitfalls

1. **Overconfidence**: Don't increase risk after wins
2. **Ignoring Correlations**: Account for position correlations
3. **Data Mining**: Don't optimize for past data only
4. **Emotional Trading**: Stick to predefined risk rules
5. **Ignoring Black Swans**: Plan for extreme events

## Troubleshooting

### Common Issues

**High Portfolio Volatility**: Reduce position sizes or add diversification
**Frequent Stop Losses**: Adjust stop levels or improve entry timing
**Concentration Risk**: Reduce large positions or add hedges
**Alert Fatigue**: Adjust alert thresholds or reduce frequency

### Debug Tools

```python
# Debug risk management
from open_trading_algo.risk_management.risk_debugger import RiskDebugger

debugger = RiskDebugger()
debugger.analyze_position_sizing(position_log)
debugger.check_risk_limits(portfolio)
debugger.plot_risk_metrics(risk_history)
debugger.validate_stop_losses(trade_log)
debugger.simulate_stress_scenarios(portfolio)
```</content>
<parameter name="filePath">/home/philipmai/repos/TradingViewAlgoDev/docs/risk-management.md
