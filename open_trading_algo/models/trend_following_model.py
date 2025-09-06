"""
Trend Following Trading Model.

This model uses moving averages, ADX, and trend strength indicators
to identify and follow market trends.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from .base_model import BaseTradingModel


class TrendFollowingModel(BaseTradingModel):
    """
    Trend following trading model.

    Uses moving average crossovers, trend strength, and directional
    indicators to identify and follow market trends.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        # Default configuration
        self.config.update(
            {
                "fast_ma": 20,
                "slow_ma": 50,
                "adx_threshold": 25,
                "trend_confirmation_period": 5,
            }
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trend following signals.

        Args:
            data: OHLCV DataFrame with indicators

        Returns:
            Series of signals (1=BUY, -1=SELL, 0=HOLD)
        """
        signals = pd.Series(0, index=data.index)

        # Calculate indicators if not already present
        fast_ma = self.config["fast_ma"]
        slow_ma = self.config["slow_ma"]

        if f"sma_{fast_ma}" not in data.columns:
            data[f"sma_{fast_ma}"] = self.get_indicator_value(data, "sma", window=fast_ma)

        if f"sma_{slow_ma}" not in data.columns:
            data[f"sma_{slow_ma}"] = self.get_indicator_value(data, "sma", window=slow_ma)

        # Calculate trend strength (ADX approximation using directional movement)
        if "dx" not in data.columns:
            data["dx"] = self.get_indicator_value(data, "dx", window=14)

        # Moving average crossover signals
        fast_ma_values = data[f"sma_{fast_ma}"]
        slow_ma_values = data[f"sma_{slow_ma}"]
        dx_values = data["dx"]

        # Golden Cross (fast MA crosses above slow MA)
        golden_cross = (
            (fast_ma_values > slow_ma_values)
            & (fast_ma_values.shift(1) <= slow_ma_values.shift(1))
            & (dx_values > self.config["adx_threshold"])
        )

        # Death Cross (fast MA crosses below slow MA)
        death_cross = (
            (fast_ma_values < slow_ma_values)
            & (fast_ma_values.shift(1) >= slow_ma_values.shift(1))
            & (dx_values > self.config["adx_threshold"])
        )

        signals[golden_cross] = 1  # BUY
        signals[death_cross] = -1  # SELL

        return signals

    def calculate_position_size(self, capital: float, risk_per_trade: float = 0.02) -> float:
        """
        Calculate position size based on trend strength.

        Args:
            capital: Available trading capital
            risk_per_trade: Risk per trade as fraction of capital

        Returns:
            Position size
        """
        # For trend following, position size could be based on
        # trend strength and volatility
        base_risk = capital * risk_per_trade

        # Could implement trend-strength-based sizing here
        return base_risk

    def get_model_info(self) -> Dict:
        """Get trend following model information."""
        info = super().get_model_info()
        info.update(
            {
                "strategy_type": "trend_following",
                "indicators": ["Moving Averages", "ADX", "Directional Movement"],
                "signal_logic": "Moving average crossovers with trend confirmation",
            }
        )
        return info
