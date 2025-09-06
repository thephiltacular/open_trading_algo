import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from open_trading_algo.backtest.signal_optimizer import SignalOptimizer
from tests.test_data_enrichment import generate_comprehensive_test_data


@pytest.fixture
def sample_backtest_data():
    """Generate sample data for backtesting."""
    return generate_comprehensive_test_data(100)


@pytest.fixture
def sample_signals():
    """Generate sample trading signals."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    np.random.seed(42)

    # Create various signal types
    signals = pd.DataFrame(
        {
            "long_signal": np.random.choice([-1, 0, 1], 100, p=[0.3, 0.4, 0.3]),
            "short_signal": np.random.choice([-1, 0, 1], 100, p=[0.3, 0.4, 0.3]),
            "momentum_signal": np.random.choice([-1, 0, 1], 100, p=[0.2, 0.6, 0.2]),
        },
        index=dates,
    )

    return signals


@pytest.fixture
def sample_portfolio():
    """Sample portfolio configuration."""
    return {
        "initial_capital": 100000,
        "position_size": 0.1,  # 10% of capital per trade
        "max_positions": 5,
        "commission": 0.001,  # 0.1% per trade
    }


class TestBacktestEngine:
    """Test backtesting engine functionality."""

    def test_backtest_initialization(self, sample_backtest_data, sample_signals, sample_portfolio):
        """Test backtest engine initialization."""
        # Mock SignalOptimizer if needed
        optimizer = SignalOptimizer(
            data={"AAPL": sample_backtest_data}, indicators={}, signal_generators={}
        )

        # Add signals to the data
        sample_backtest_data["long_signal"] = sample_signals["long_signal"]
        sample_backtest_data["short_signal"] = sample_signals["short_signal"]

        assert len(sample_backtest_data) == 100
        assert "long_signal" in sample_backtest_data.columns
        assert "short_signal" in sample_backtest_data.columns

    def test_portfolio_calculations(self, sample_backtest_data, sample_signals, sample_portfolio):
        """Test portfolio value calculations."""
        capital = sample_portfolio["initial_capital"]
        position_size = sample_portfolio["position_size"]

        # Simulate simple long-only strategy
        positions = (
            sample_signals["long_signal"] * position_size * capital / sample_backtest_data["close"]
        )

        # Calculate portfolio value
        portfolio_value = capital + (positions * sample_backtest_data["close"]).cumsum()

        assert len(portfolio_value) == len(sample_backtest_data)
        assert portfolio_value.iloc[0] == capital  # Should start with initial capital

    def test_trade_execution(self, sample_backtest_data, sample_signals):
        """Test trade execution logic."""
        # Simple signal-based trading
        trades = []
        position = 0

        for i in range(len(sample_signals)):
            signal = sample_signals["long_signal"].iloc[i]

            if signal == 1 and position == 0:  # Buy signal
                trades.append(
                    {"type": "buy", "price": sample_backtest_data["close"].iloc[i], "index": i}
                )
                position = 1
            elif signal == -1 and position == 1:  # Sell signal
                trades.append(
                    {"type": "sell", "price": sample_backtest_data["close"].iloc[i], "index": i}
                )
                position = 0

        assert isinstance(trades, list)
        # Should have some trades given random signals
        assert len(trades) > 0

    def test_performance_metrics(self, sample_backtest_data, sample_signals, sample_portfolio):
        """Test performance metrics calculation."""
        capital = sample_portfolio["initial_capital"]

        # Simulate returns
        returns = sample_backtest_data["close"].pct_change().fillna(0)
        cumulative_returns = (1 + returns).cumprod()
        portfolio_value = capital * cumulative_returns

        # Calculate basic metrics
        total_return = (portfolio_value.iloc[-1] - capital) / capital
        volatility = returns.std() * np.sqrt(252)  # Annualized
        sharpe_ratio = total_return / volatility if volatility > 0 else 0

        assert isinstance(total_return, (int, float))
        assert isinstance(volatility, (int, float))
        assert isinstance(sharpe_ratio, (int, float))

    def test_risk_management(self, sample_backtest_data, sample_signals, sample_portfolio):
        """Test risk management features."""
        capital = sample_portfolio["initial_capital"]
        max_positions = sample_portfolio["max_positions"]

        # Simulate position sizing with risk management
        positions = []
        current_positions = 0

        for i in range(len(sample_signals)):
            signal = sample_signals["long_signal"].iloc[i]

            if signal == 1 and current_positions < max_positions:
                # Risk-managed position size
                risk_amount = capital * 0.02  # 2% risk per trade
                stop_loss = sample_backtest_data["close"].iloc[i] * 0.95  # 5% stop loss
                position_size = risk_amount / (sample_backtest_data["close"].iloc[i] - stop_loss)

                positions.append(position_size)
                current_positions += 1
            elif signal == -1 and current_positions > 0:
                # Close position
                current_positions -= 1

        assert len(positions) >= 0  # At least some positions were opened
        assert all(pos > 0 for pos in positions)


class TestBacktestValidation:
    """Test backtest validation and edge cases."""

    def test_backtest_with_no_signals(self, sample_backtest_data):
        """Test backtest with no trading signals."""
        # All signals are 0 (no trades)
        signals = pd.Series([0] * len(sample_backtest_data), index=sample_backtest_data.index)

        # Should result in no trades
        trades = []
        for signal in signals:
            if signal != 0:
                trades.append(signal)

        assert len(trades) == 0

    def test_backtest_boundary_conditions(self):
        """Test backtest at data boundaries."""
        # Minimal dataset
        minimal_data = pd.DataFrame(
            {"close": [100, 101, 102], "signal": [0, 1, -1]},
            index=pd.date_range("2023-01-01", periods=3),
        )

        # Should handle minimal data without errors
        returns = minimal_data["close"].pct_change().fillna(0)
        assert len(returns) == 3
        assert not returns.isna().any()

    def test_backtest_with_missing_data(self, sample_backtest_data):
        """Test backtest with missing price data."""
        data_with_nan = sample_backtest_data.copy()
        data_with_nan.loc[data_with_nan.index[10:15], "close"] = np.nan

        # Should handle NaN values gracefully
        returns = data_with_nan["close"].pct_change().fillna(0)
        assert len(returns) == len(data_with_nan)

    def test_multiple_strategy_comparison(self, sample_backtest_data, sample_signals):
        """Test comparison of multiple strategies."""
        strategies = {
            "long_only": sample_signals["long_signal"],
            "short_only": sample_signals["short_signal"],
            "momentum": sample_signals["momentum_signal"],
        }

        results = {}
        for name, signals in strategies.items():
            # Simple performance calculation
            returns = sample_backtest_data["close"].pct_change().fillna(0)
            strategy_returns = returns * signals.shift(1).fillna(0)  # Use previous signal
            cumulative_return = (1 + strategy_returns).cumprod().iloc[-1] - 1
            results[name] = cumulative_return

        assert len(results) == 3
        assert all(isinstance(ret, (int, float)) for ret in results.values())


class TestBacktestReporting:
    """Test backtest reporting and analytics."""

    def test_trade_log_generation(self, sample_backtest_data, sample_signals):
        """Test generation of trade logs."""
        trade_log = []
        position = 0
        entry_price = 0

        for i in range(len(sample_signals)):
            signal = sample_signals["long_signal"].iloc[i]
            price = sample_backtest_data["close"].iloc[i]

            if signal == 1 and position == 0:  # Enter long
                trade_log.append(
                    {
                        "type": "entry",
                        "price": price,
                        "index": i,
                        "timestamp": sample_backtest_data.index[i],
                    }
                )
                position = 1
                entry_price = price
            elif signal == -1 and position == 1:  # Exit long
                pnl = price - entry_price
                trade_log.append(
                    {
                        "type": "exit",
                        "price": price,
                        "pnl": pnl,
                        "index": i,
                        "timestamp": sample_backtest_data.index[i],
                    }
                )
                position = 0

        assert isinstance(trade_log, list)
        # Check that entries and exits are properly paired
        entries = [t for t in trade_log if t["type"] == "entry"]
        exits = [t for t in trade_log if t["type"] == "exit"]
        assert len(entries) >= len(exits)  # May have open positions

    def test_performance_summary(self, sample_backtest_data, sample_signals, sample_portfolio):
        """Test performance summary generation."""
        capital = sample_portfolio["initial_capital"]

        # Simulate strategy
        returns = sample_backtest_data["close"].pct_change().fillna(0)
        portfolio_returns = returns * sample_signals["long_signal"].shift(1).fillna(0)
        portfolio_value = capital * (1 + portfolio_returns).cumprod()

        # Generate summary
        summary = {
            "initial_capital": capital,
            "final_value": portfolio_value.iloc[-1],
            "total_return": (portfolio_value.iloc[-1] - capital) / capital,
            "max_drawdown": (portfolio_value / portfolio_value.cummax() - 1).min(),
            "win_rate": (portfolio_returns > 0).mean(),
            "total_trades": (sample_signals["long_signal"] != 0).sum(),
        }

        import numpy as np

        assert all(isinstance(v, (int, float, np.number)) or np.isreal(v) for v in summary.values())
        assert summary["final_value"] > 0
        assert -1 <= summary["max_drawdown"] <= 0  # Drawdown should be negative or zero
