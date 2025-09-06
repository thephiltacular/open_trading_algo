"""
Mean Reversion Trading Model.

This model identifies overbought/oversold conditions using
Bollinger Bands, RSI, and other mean reversion indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from .base_model import BaseTradingModel


class MeanReversionModel(BaseTradingModel):
    """
    Mean reversion trading model.

    Uses Bollinger Bands, RSI, and price deviations from moving averages
    to identify mean reversion opportunities.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the mean reversion model.

        Args:
            config (Optional[Dict]): Model configuration parameters. Defaults to None.
        """
        super().__init__(config)
        # Default configuration
        self.config.update(
            {
                "bb_std_threshold": 2.0,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "mean_reversion_window": 20,
                "deviation_threshold": 2.0,
            }
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate mean reversion signals.

        Args:
            data (pd.DataFrame): OHLCV DataFrame with indicators.

        Returns:
            pd.Series: Series of signals (1=BUY, -1=SELL, 0=HOLD).
        """
        signals = pd.Series(0, index=data.index)

        # Calculate indicators if not already present
        if "rsi" not in data.columns:
            data["rsi"] = self.get_indicator_value(data, "rsi", window=14)

        if "bb_middle" not in data.columns:
            bb_middle, bb_upper, bb_lower = self.get_indicator_value(
                data,
                "bbands",
                window=self.config["mean_reversion_window"],
                num_std=self.config["bb_std_threshold"],
            )
            data["bb_middle"] = bb_middle
            data["bb_upper"] = bb_upper
            data["bb_lower"] = bb_lower

        # Calculate price deviation from mean
        price = data["close"]
        bb_middle = data["bb_middle"]
        bb_upper = data["bb_upper"]
        bb_lower = data["bb_lower"]
        rsi = data["rsi"]

        # Price deviation from Bollinger Band middle
        deviation = (price - bb_middle) / (bb_upper - bb_lower).replace(0, np.nan)

        # Mean reversion signals
        # Buy when price is below lower BB and RSI is oversold
        buy_condition = (
            (price < bb_lower)
            & (rsi < self.config["rsi_oversold"])
            & (deviation < -self.config["deviation_threshold"])
        )

        # Sell when price is above upper BB and RSI is overbought
        sell_condition = (
            (price > bb_upper)
            & (rsi > self.config["rsi_overbought"])
            & (deviation > self.config["deviation_threshold"])
        )

        signals[buy_condition] = 1  # BUY
        signals[sell_condition] = -1  # SELL

        return signals

    def calculate_position_size(self, capital: float, risk_per_trade: float = 0.02) -> float:
        """Calculate position size based on deviation from mean.

        Args:
            capital (float): Available trading capital.
            risk_per_trade (float, optional): Risk per trade as fraction of capital. Defaults to 0.02.

        Returns:
            float: Position size.
        """
        # For mean reversion, position size could be based on
        # how far price has deviated from the mean
        base_risk = capital * risk_per_trade

        # Could implement volatility-based sizing here
        return base_risk

    def get_model_info(self) -> Dict:
        """Get mean reversion model information.

        Returns:
            Dict: Dictionary with model information.
        """
        info = super().get_model_info()
        info.update(
            {
                "strategy_type": "mean_reversion",
                "indicators": ["Bollinger Bands", "RSI", "Price Deviation"],
                "signal_logic": "Mean reversion from extreme deviations",
            }
        )
        return info
