"""
Tests for trading models.

Tests the trading models that combine indicators and data sources.
"""

import pytest
import pandas as pd
import numpy as np
from open_trading_algo.models import (
    BaseTradingModel,
    MomentumModel,
    MeanReversionModel,
    TrendFollowingModel,
)


@pytest.fixture
def sample_ohlc_data():
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    np.random.seed(42)

    data = {
        "open": np.random.uniform(100, 200, 100),
        "high": np.random.uniform(150, 250, 100),
        "low": np.random.uniform(50, 150, 100),
        "close": np.random.uniform(100, 200, 100),
        "volume": np.random.randint(1000, 10000, 100),
    }

    # Ensure high >= close >= low and high >= open >= low
    for i in range(len(data["close"])):
        data["high"][i] = max(data["high"][i], data["open"][i], data["close"][i])
        data["low"][i] = min(data["low"][i], data["open"][i], data["close"][i])

    return pd.DataFrame(data, index=dates)


class TestBaseTradingModel:
    """Test the base trading model functionality."""

    def test_data_validation(self, sample_ohlc_data):
        """Test data validation using a concrete model."""
        model = MomentumModel()  # Use concrete implementation
        assert model.validate_data(sample_ohlc_data)

        # Test with missing columns
        invalid_data = sample_ohlc_data.drop("volume", axis=1)
        assert not model.validate_data(invalid_data)

    def test_data_preparation(self, sample_ohlc_data):
        """Test data preparation with indicators."""
        model = MomentumModel()  # Use concrete implementation
        prepared_data = model.prepare_data(sample_ohlc_data)

        # Check that indicators were added
        expected_indicators = [
            "sma_20",
            "sma_50",
            "ema_12",
            "ema_26",
            "rsi",
            "atr",
            "macd",
            "macd_signal",
            "macd_hist",
            "bb_middle",
            "bb_upper",
            "bb_lower",
        ]
        for indicator in expected_indicators:
            assert indicator in prepared_data.columns

    def test_get_model_info(self):
        """Test model info retrieval."""
        model = MomentumModel({"test": "config"})  # Use concrete implementation
        info = model.get_model_info()

        assert "name" in info
        assert "description" in info
        assert "config" in info
        assert info["config"]["test"] == "config"


class TestMomentumModel:
    """Test the momentum trading model."""

    def test_initialization(self):
        """Test momentum model initialization."""
        model = MomentumModel()
        assert "rsi_overbought" in model.config
        assert "rsi_oversold" in model.config
        assert model.config["rsi_overbought"] == 70
        assert model.config["rsi_oversold"] == 30

    def test_signal_generation(self, sample_ohlc_data):
        """Test momentum signal generation."""
        model = MomentumModel()
        signals = model.generate_signals(sample_ohlc_data)

        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_ohlc_data)
        assert signals.index.equals(sample_ohlc_data.index)

        # Check signal values are valid (-1, 0, 1)
        assert all(signals.isin([-1, 0, 1]))

    def test_position_size_calculation(self):
        """Test position size calculation."""
        model = MomentumModel()
        capital = 10000
        risk_per_trade = 0.02

        position_size = model.calculate_position_size(capital, risk_per_trade)
        expected_size = capital * risk_per_trade

        assert position_size == expected_size

    def test_model_info(self):
        """Test momentum model info."""
        model = MomentumModel()
        info = model.get_model_info()

        assert info["strategy_type"] == "momentum"
        assert "RSI" in info["indicators"]
        assert "MACD" in info["indicators"]


class TestMeanReversionModel:
    """Test the mean reversion trading model."""

    def test_initialization(self):
        """Test mean reversion model initialization."""
        model = MeanReversionModel()
        assert "bb_std_threshold" in model.config
        assert "rsi_overbought" in model.config
        assert model.config["rsi_overbought"] == 70

    def test_signal_generation(self, sample_ohlc_data):
        """Test mean reversion signal generation."""
        model = MeanReversionModel()
        signals = model.generate_signals(sample_ohlc_data)

        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_ohlc_data)
        assert all(signals.isin([-1, 0, 1]))

    def test_position_size_calculation(self):
        """Test position size calculation."""
        model = MeanReversionModel()
        capital = 10000
        risk_per_trade = 0.02

        position_size = model.calculate_position_size(capital, risk_per_trade)
        expected_size = capital * risk_per_trade

        assert position_size == expected_size

    def test_model_info(self):
        """Test mean reversion model info."""
        model = MeanReversionModel()
        info = model.get_model_info()

        assert info["strategy_type"] == "mean_reversion"
        assert "Bollinger Bands" in info["indicators"]


class TestTrendFollowingModel:
    """Test the trend following trading model."""

    def test_initialization(self):
        """Test trend following model initialization."""
        model = TrendFollowingModel()
        assert "fast_ma" in model.config
        assert "slow_ma" in model.config
        assert model.config["fast_ma"] == 20
        assert model.config["slow_ma"] == 50

    def test_signal_generation(self, sample_ohlc_data):
        """Test trend following signal generation."""
        model = TrendFollowingModel()
        signals = model.generate_signals(sample_ohlc_data)

        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_ohlc_data)
        assert all(signals.isin([-1, 0, 1]))

    def test_position_size_calculation(self):
        """Test position size calculation."""
        model = TrendFollowingModel()
        capital = 10000
        risk_per_trade = 0.02

        position_size = model.calculate_position_size(capital, risk_per_trade)
        expected_size = capital * risk_per_trade

        assert position_size == expected_size

    def test_model_info(self):
        """Test trend following model info."""
        model = TrendFollowingModel()
        info = model.get_model_info()

        assert info["strategy_type"] == "trend_following"
        assert "Moving Averages" in info["indicators"]


class TestModelIntegration:
    """Test model integration and usage patterns."""

    def test_model_pipeline(self, sample_ohlc_data):
        """Test complete model pipeline."""
        model = MomentumModel()

        # Prepare data
        prepared_data = model.prepare_data(sample_ohlc_data)

        # Generate signals
        signals = model.generate_signals(prepared_data)

        # Calculate position size
        position_size = model.calculate_position_size(10000)

        assert len(signals) > 0
        assert position_size > 0
        assert isinstance(prepared_data, pd.DataFrame)

    def test_custom_configuration(self, sample_ohlc_data):
        """Test models with custom configuration."""
        config = {
            "rsi_overbought": 75,
            "rsi_oversold": 25,
            "macd_signal_threshold": 0.5,
        }

        model = MomentumModel(config)
        # Check that custom config values are used
        assert model.config["rsi_overbought"] == 75
        assert model.config["rsi_oversold"] == 25
        assert model.config["macd_signal_threshold"] == 0.5

        # Test that model still works with custom config
        signals = model.generate_signals(sample_ohlc_data)
        assert isinstance(signals, pd.Series)
