"""
Momentum Trading Model.

This model uses momentum indicators like RSI, MACD, and Stochastic
to identify trending markets and generate trading signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from .base_model import BaseTradingModel


class MomentumModel(BaseTradingModel):
    """
    Momentum-based trading model.

    Uses RSI, MACD, and Stochastic indicators to identify
    momentum shifts and generate trading signals.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        # Set default configuration first
        default_config = {
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "stoch_overbought": 80,
            "stoch_oversold": 20,
            "macd_signal_threshold": 0,
        }
        # Update with custom config
        default_config.update(self.config)
        self.config = default_config

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate momentum-based trading signals.

        Args:
            data: OHLCV DataFrame with indicators

        Returns:
            Series of signals (1=BUY, -1=SELL, 0=HOLD)
        """
        signals = pd.Series(0, index=data.index)

        # Calculate indicators if not already present
        if "rsi" not in data.columns:
            data["rsi"] = self.get_indicator_value(data, "rsi", window=14)

        if "macd" not in data.columns:
            macd_line, signal_line, hist = self.get_indicator_value(data, "macd")
            data["macd"] = macd_line
            data["macd_signal"] = signal_line
            data["macd_hist"] = hist

        if "stoch_k" not in data.columns:
            stoch_k, stoch_d = self.get_indicator_value(
                data, "stoch", fastk_period=14, slowk_period=3, slowd_period=3
            )
            data["stoch_k"] = stoch_k
            data["stoch_d"] = stoch_d

        # Generate signals based on momentum indicators
        rsi = data["rsi"]
        macd_hist = data["macd_hist"]
        stoch_k = data["stoch_k"]
        stoch_d = data["stoch_d"]

        # Bullish momentum signals
        bullish_condition = (
            (rsi < self.config["rsi_oversold"])
            & (macd_hist > self.config["macd_signal_threshold"])
            & (stoch_k < self.config["stoch_oversold"])
        )

        # Bearish momentum signals
        bearish_condition = (
            (rsi > self.config["rsi_overbought"])
            & (macd_hist < -self.config["macd_signal_threshold"])
            & (stoch_k > self.config["stoch_overbought"])
        )

        signals[bullish_condition] = 1  # BUY
        signals[bearish_condition] = -1  # SELL

        return signals

    def calculate_position_size(self, capital: float, risk_per_trade: float = 0.02) -> float:
        """
        Calculate position size based on momentum strength.

        Args:
            capital: Available trading capital
            risk_per_trade: Risk per trade as fraction of capital

        Returns:
            Position size
        """
        # For momentum models, we might want to size positions
        # based on the strength of the momentum signal
        base_risk = capital * risk_per_trade

        # Could adjust based on momentum strength
        # For now, return base risk amount
        return base_risk

    def get_model_info(self) -> Dict:
        """Get momentum model information."""
        info = super().get_model_info()
        info.update(
            {
                "strategy_type": "momentum",
                "indicators": ["RSI", "MACD", "Stochastic"],
                "signal_logic": "Combined momentum divergence signals",
            }
        )
        return info
