# Backtesting

This document covers the backtesting capabilities in open_trading_algo, including strategy testing, performance analysis, and optimization techniques.

## Overview

The backtesting system provides comprehensive tools for:

- **Historical strategy testing** with realistic trading conditions
- **Performance analysis** with detailed metrics and visualizations
- **Walk-forward optimization** to avoid overfitting
- **Monte Carlo simulation** for risk assessment
- **Multi-asset portfolio backtesting**

## Basic Backtesting

### Simple Strategy Backtest

```python
from open_trading_algo.backtest.signal_backtester import SignalBacktester
from open_trading_algo.models.momentum_model import MomentumModel
from open_trading_algo.cache.data_cache import DataCache

# Initialize components
cache = DataCache()
model = MomentumModel()
backtester = SignalBacktester(
    initial_capital=100000,
    commission=0.001,  # 0.1% per trade
    slippage=0.0005    # 0.05% slippage
)

# Get historical data
data = cache.get_price_data('AAPL', start='2020-01-01', end='2024-01-01')

# Generate signals
signals = model.generate_signals(data)

# Run backtest
results = backtester.run_backtest(
    signals=signals,
    price_data=data,
    start_date='2021-01-01',
    end_date='2023-12-31'
)

print("Backtest Results:")
print(f"Total Return: {results['total_return']:.2%}")
print(f"Annual Return: {results['annual_return']:.2%}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
```

### Multi-Asset Portfolio Backtest

```python
from open_trading_algo.backtest.portfolio_backtester import PortfolioBacktester

# Initialize portfolio backtester
portfolio_backtester = PortfolioBacktester(
    initial_capital=500000,
    max_position_size=0.1,  # 10% max per position
    rebalance_frequency='monthly'
)

# Define portfolio assets and strategies
portfolio_config = {
    'AAPL': {
        'weight': 0.3,
        'strategy': momentum_strategy
    },
    'GOOGL': {
        'weight': 0.25,
        'strategy': mean_reversion_strategy
    },
    'MSFT': {
        'weight': 0.25,
        'strategy': trend_following_strategy
    },
    'TSLA': {
        'weight': 0.2,
        'strategy': breakout_strategy
    }
}

# Run portfolio backtest
portfolio_results = portfolio_backtester.run_portfolio_backtest(
    portfolio_config=portfolio_config,
    start_date='2021-01-01',
    end_date='2023-12-31'
)

print("Portfolio Performance:")
print(f"Total Return: {portfolio_results['total_return']:.2%}")
print(f"Annual Volatility: {portfolio_results['volatility']:.2%}")
print(f"Portfolio Sharpe: {portfolio_results['sharpe_ratio']:.2f}")
```

## Advanced Backtesting Features

### Walk-Forward Analysis

Walk-forward analysis helps prevent overfitting by testing strategies on out-of-sample data.

```python
from open_trading_algo.backtest.walk_forward import WalkForwardAnalyzer

# Initialize walk-forward analyzer
wf_analyzer = WalkForwardAnalyzer(
    initial_train_days=252,  # 1 year training
    test_days=63,            # 3 months testing
    step_days=21             # 1 month step
)

# Run walk-forward analysis
wf_results = wf_analyzer.run_walk_forward(
    strategy=strategy_function,
    data=historical_data,
    start_date='2018-01-01',
    end_date='2023-12-31'
)

# Analyze walk-forward performance
stability_metrics = wf_analyzer.analyze_stability(wf_results)

print("Walk-Forward Stability:")
print(f"Average OOS Return: {stability_metrics['avg_oos_return']:.2%}")
print(f"Return Stability: {stability_metrics['return_stability']:.2f}")
print(f"Max Drawdown Stability: {stability_metrics['dd_stability']:.2f}")
```

### Monte Carlo Simulation

Monte Carlo simulation helps assess strategy robustness under different market conditions.

```python
from open_trading_algo.backtest.monte_carlo import MonteCarloSimulator

# Initialize Monte Carlo simulator
mc_simulator = MonteCarloSimulator(
    num_simulations=1000,
    confidence_level=0.95
)

# Run Monte Carlo analysis
mc_results = mc_simulator.run_simulation(
    strategy_returns=strategy_returns,
    benchmark_returns=benchmark_returns,
    time_horizon_days=252
)

# Analyze simulation results
simulation_stats = mc_simulator.calculate_statistics(mc_results)

print("Monte Carlo Results:")
print(f"Expected Return: {simulation_stats['expected_return']:.2%}")
print(f"Value at Risk (95%): {simulation_stats['var_95']:.2%}")
print(f"Expected Shortfall: {simulation_stats['expected_shortfall']:.2%}")
print(f"Probability of Loss: {simulation_stats['prob_loss']:.2%}")
```

### Transaction Cost Analysis

```python
from open_trading_algo.backtest.cost_analyzer import TransactionCostAnalyzer

# Initialize cost analyzer
cost_analyzer = TransactionCostAnalyzer(
    commission_per_share=0.005,  # $0.005 per share
    spread_cost=0.0002,          # 0.02% spread
    market_impact=0.0001         # 0.01% market impact
)

# Analyze transaction costs
cost_analysis = cost_analyzer.analyze_costs(
    trades=trade_log,
    price_data=data,
    trade_frequency='daily'
)

print("Transaction Cost Analysis:")
print(f"Total Commissions: ${cost_analysis['total_commissions']:.2f}")
print(f"Total Spread Costs: ${cost_analysis['total_spread_costs']:.2f}")
print(f"Market Impact: ${cost_analysis['total_market_impact']:.2f}")
print(f"Net Performance Impact: {cost_analysis['net_impact']:.2%}")
```

## Performance Metrics

### Risk-Adjusted Metrics

```python
from open_trading_algo.backtest.performance_metrics import PerformanceAnalyzer

# Initialize performance analyzer
performance_analyzer = PerformanceAnalyzer(
    risk_free_rate=0.02,  # 2% risk-free rate
    benchmark_returns=sp500_returns
)

# Calculate comprehensive metrics
metrics = performance_analyzer.calculate_metrics(
    strategy_returns=strategy_returns,
    benchmark_returns=benchmark_returns
)

print("Performance Metrics:")
print(f"Total Return: {metrics['total_return']:.2%}")
print(f"Annual Return: {metrics['annual_return']:.2%}")
print(f"Volatility: {metrics['volatility']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
print(f"Alpha: {metrics['alpha']:.2%}")
print(f"Beta: {metrics['beta']:.2f}")
print(f"Information Ratio: {metrics['information_ratio']:.2f}")
```

### Drawdown Analysis

```python
# Analyze drawdowns
drawdown_analysis = performance_analyzer.analyze_drawdowns(strategy_returns)

print("Drawdown Analysis:")
print(f"Max Drawdown: {drawdown_analysis['max_drawdown']:.2%}")
print(f"Average Drawdown: {drawdown_analysis['avg_drawdown']:.2%}")
print(f"Drawdown Duration (days): {drawdown_analysis['avg_duration_days']}")
print(f"Recovery Time (days): {drawdown_analysis['avg_recovery_days']}")
print(f"Drawdown Frequency: {drawdown_analysis['frequency']}")
```

## Strategy Optimization

### Parameter Optimization

```python
from open_trading_algo.backtest.optimizer import StrategyOptimizer

# Define parameter ranges
parameter_ranges = {
    'rsi_period': [10, 14, 21],
    'macd_fast': [8, 12, 16],
    'macd_slow': [21, 26, 31],
    'stop_loss': [0.05, 0.10, 0.15]
}

# Initialize optimizer
optimizer = StrategyOptimizer(
    strategy_function=momentum_strategy,
    parameter_ranges=parameter_ranges,
    optimization_metric='sharpe_ratio'
)

# Run optimization
optimal_params = optimizer.optimize_parameters(
    data=historical_data,
    start_date='2020-01-01',
    end_date='2023-12-31'
)

print("Optimal Parameters:")
for param, value in optimal_params.items():
    print(f"{param}: {value}")
```

### Multi-Objective Optimization

```python
# Optimize for multiple objectives
objectives = {
    'maximize': ['sharpe_ratio', 'total_return'],
    'minimize': ['max_drawdown', 'volatility']
}

pareto_front = optimizer.multi_objective_optimize(
    data=historical_data,
    objectives=objectives,
    population_size=100,
    generations=50
)

print("Pareto Optimal Solutions:")
for i, solution in enumerate(pareto_front):
    print(f"Solution {i+1}: {solution}")
```

## Visualization and Reporting

### Performance Charts

```python
from open_trading_algo.backtest.visualization import BacktestVisualizer

# Initialize visualizer
visualizer = BacktestVisualizer()

# Create performance dashboard
visualizer.create_dashboard(
    strategy_returns=strategy_returns,
    benchmark_returns=benchmark_returns,
    trades=trade_log,
    title="Momentum Strategy Backtest"
)

# Generate detailed report
report = visualizer.generate_report(
    results=backtest_results,
    metrics=performance_metrics,
    format='html'
)

# Save report
report.save('backtest_report.html')
```

### Risk Analysis Plots

```python
# Create risk analysis plots
visualizer.plot_drawdowns(strategy_returns)
visualizer.plot_rolling_sharpe(strategy_returns, window=60)
visualizer.plot_return_distribution(strategy_returns)
visualizer.plot_rolling_volatility(strategy_returns, window=30)
```

## Configuration

### Backtest Configuration

```yaml
# config/backtest.yaml
backtest:
  # Capital and costs
  initial_capital: 100000
  commission_per_trade: 0.001  # 0.1%
  slippage: 0.0005             # 0.05%

  # Position sizing
  max_position_size: 0.1       # 10% max per position
  min_position_size: 0.01      # 1% min per position

  # Risk management
  max_drawdown_limit: 0.2      # 20% max drawdown
  risk_per_trade: 0.02         # 2% risk per trade

  # Walk-forward analysis
  walk_forward:
    initial_train_days: 252
    test_days: 63
    step_days: 21

  # Monte Carlo simulation
  monte_carlo:
    num_simulations: 1000
    confidence_level: 0.95
    time_horizon_days: 252
```

## Best Practices

### Backtesting Guidelines

1. **Use Realistic Assumptions**: Include transaction costs, slippage, and market impact
2. **Avoid Look-Ahead Bias**: Ensure signals are based only on historical data available at the time
3. **Out-of-Sample Testing**: Always test on data not used for strategy development
4. **Walk-Forward Analysis**: Use walk-forward optimization to validate strategy robustness

### Performance Evaluation

1. **Multiple Metrics**: Don't rely on a single performance metric
2. **Risk-Adjusted Returns**: Focus on risk-adjusted metrics like Sharpe and Sortino ratios
3. **Drawdown Analysis**: Understand maximum drawdowns and recovery times
4. **Benchmark Comparison**: Compare against relevant market benchmarks

### Common Pitfalls

1. **Overfitting**: Avoid curve-fitting by using proper validation techniques
2. **Survivorship Bias**: Include delisted stocks in historical analysis
3. **Data Mining**: Be cautious of strategies that work only on specific historical periods
4. **Transaction Costs**: Always account for realistic trading costs

## Troubleshooting

### Common Issues

**Negative Sharpe Ratio**: Strategy may be taking excessive risk or have poor risk-adjusted returns
**High Drawdowns**: Review position sizing and risk management rules
**Inconsistent Results**: Check for data quality issues or look-ahead bias
**Poor Benchmark Comparison**: Ensure appropriate benchmark selection

### Debug Tools

```python
# Debug backtest execution
from open_trading_algo.backtest.debug_backtest import BacktestDebugger

debugger = BacktestDebugger()
debugger.analyze_trade_log(trade_log)
debugger.check_data_quality(data)
debugger.validate_signals(signals)
debugger.plot_trade_timing(trades, data)
```</content>
<parameter name="filePath">/home/philipmai/repos/TradingViewAlgoDev/docs/backtesting.md
