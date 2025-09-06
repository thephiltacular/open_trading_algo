"""
open_trading_algo package initialization.

This package provides tools for financial data analysis, signal generation, backtesting,
and live data ingestion from various APIs. It includes modules for indicators, alerts,
backtesting, caching, financial data APIs, and trading models.

Modules:
    - alerts: Alert utilities.
    - backtest: Backtesting and signal optimization.
    - cache: Data caching and secure API key management.
    - fin_data_apis: Interfaces to financial data APIs.
    - indicators: Technical indicators and signals.
    - models: Trading models combining indicators and data sources.
    - sentiment: Sentiment analysis from social and analyst sources.

TradingView Algo Dev – modular utilities extracted from `data_processing.py`.

This package provides a clean, typed, and documented interface for the data
processing, indicator calculations, and alert transformations that were
previously embedded in a single monolithic script.

"""

from .pipeline import ModelPipeline
from .types import AlertsByDay, DataByDay, DayKey

__all__ = [
    "DataByDay",
    "AlertsByDay",
    "DayKey",
    "ModelPipeline",
]
