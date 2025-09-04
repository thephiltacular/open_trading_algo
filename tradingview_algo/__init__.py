"""
TradingView Algo Dev – modular utilities extracted from `data_processing.py`.

This package provides a clean, typed, and documented interface for the data
processing, indicator calculations, and alert transformations that were
previously embedded in a single monolithic script.

The original file `data_processing.py` is left untouched. Use these modules
to build new pipelines or incrementally move logic into maintainable units.
"""

from .pipeline import ModelPipeline
from .types import AlertsByDay, DataByDay, DayKey

__all__ = [
    "DataByDay",
    "AlertsByDay",
    "DayKey",
    "ModelPipeline",
]
