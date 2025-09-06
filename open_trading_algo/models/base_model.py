"""
Base Trading Model - Abstract base class for all trading models.

This module provides the foundation for building trading models that combine
indicators, data sources, and signals into complete trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from datetime import datetime, timedelta

from ..indicators.indicators import (
    sma,
    ema,
    rsi,
    macd,
    bbands,
    atr,
    stoch,
    stochf,
    stochrsi,
    ad,
    adosc,
    dx,
)
from ..types import AlertsByDay, DataByDay, DayKey


class BaseTradingModel(ABC):
    """
    Abstract base class for trading models.

    Provides common functionality for:
    - Data fetching and processing
    - Indicator calculation
    - Signal generation
    - Position sizing
    - Risk management integration
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the trading model.

        Args:
            config: Model configuration parameters
        """
        self.config = config or {}
        self.indicators_cache = {}

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from market data.

        Args:
            data: OHLCV DataFrame with indicators

        Returns:
            Series of signals (1=BUY, -1=SELL, 0=HOLD)
        """
        pass

    @abstractmethod
    def calculate_position_size(self, capital: float, risk_per_trade: float = 0.02) -> float:
        """
        Calculate position size based on risk management.

        Args:
            capital: Available trading capital
            risk_per_trade: Risk per trade as fraction of capital

        Returns:
            Position size
        """
        pass

    def prepare_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and enrich raw market data with indicators.

        Args:
            raw_data: Raw OHLCV data

        Returns:
            DataFrame with calculated indicators
        """
        data = raw_data.copy()

        # Calculate common indicators
        data["sma_20"] = sma(data["close"], 20)
        data["sma_50"] = sma(data["close"], 50)
        data["ema_12"] = ema(data["close"], 12)
        data["ema_26"] = ema(data["close"], 26)
        data["rsi"] = rsi(data["close"], 14)
        data["atr"] = atr(data["high"], data["low"], data["close"], 14)

        # MACD
        macd_line, signal_line, hist = macd(data["close"])
        data["macd"] = macd_line
        data["macd_signal"] = signal_line
        data["macd_hist"] = hist

        # Bollinger Bands
        bb_middle, bb_upper, bb_lower = bbands(data["close"])
        data["bb_middle"] = bb_middle
        data["bb_upper"] = bb_upper
        data["bb_lower"] = bb_lower

        return data

    def get_indicator_value(self, data: pd.DataFrame, indicator: str, **kwargs) -> pd.Series:
        """
        Get or calculate an indicator value.

        Args:
            data: Market data
            indicator: Indicator name
            **kwargs: Indicator parameters

        Returns:
            Indicator values
        """
        cache_key = f"{indicator}_{kwargs}"

        if cache_key not in self.indicators_cache:
            # Import the indicator function dynamically
            indicator_func = globals().get(indicator)
            if indicator_func:
                # Check function signature to determine correct arguments
                import inspect

                sig = inspect.signature(indicator_func)
                params = list(sig.parameters.keys())

                if "high" in params and "low" in params and "close" in params:
                    # Multi-series indicator (like ATR, STOCH)
                    self.indicators_cache[cache_key] = indicator_func(
                        data["high"], data["low"], data["close"], **kwargs
                    )
                elif "high" in params and "low" in params:
                    # Two-series indicator (like AD)
                    self.indicators_cache[cache_key] = indicator_func(
                        data["high"], data["low"], data["close"], data["volume"], **kwargs
                    )
                else:
                    # Single-series indicator (like RSI, SMA)
                    self.indicators_cache[cache_key] = indicator_func(data["close"], **kwargs)
            else:
                raise ValueError(f"Unknown indicator: {indicator}")

        return self.indicators_cache[cache_key]

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that data has required columns.

        Args:
            data: Market data to validate

        Returns:
            True if data is valid
        """
        required_columns = ["open", "high", "low", "close", "volume"]
        return all(col in data.columns for col in required_columns)

    def get_model_info(self) -> Dict:
        """
        Get information about the model.

        Returns:
            Dictionary with model information
        """
        return {
            "name": self.__class__.__name__,
            "description": self.__doc__ or "No description available",
            "config": self.config,
            "indicators_used": list(self.indicators_cache.keys()),
        }
