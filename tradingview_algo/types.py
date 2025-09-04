"""Lightweight type aliases and contracts for the TradingView algo package.

The original code (`data_processing.py`) manipulates pandas DataFrames keyed by
"day" (e.g., '03-07A'). To keep imports lightweight and avoid forcing a hard
runtime dependency on pandas for simple imports, we expose conservative aliases
that describe the shapes without importing pandas at import-time.
"""

from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple, Union

# Basic key type used for a trading day partition (e.g., '03-07A', '02-10D').
DayKey = str

# Best-effort aliases that can represent pandas DataFrame/Series at type level
# without importing pandas. When pandas is available, these can be refined.
DataFrameLike = Any
SeriesLike = Any

# Collections used throughout the pipeline
DataByDay = Mapping[DayKey, DataFrameLike]
AlertsByDay = Mapping[DayKey, DataFrameLike]

# Column labels can be strings (names) or integers (positional indices)
Column = Union[str, int]
Columns = Sequence[Column]

# Simple result container for operations that add columns to DataFrames.
class ColumnAddResult(Dict[str, Any]):
    """A mapping of new column names to their computed data.

    This allows functions to return multiple new columns that can be assigned
    to a DataFrame via df.assign(**result) when using pandas. Using a plain dict
    keeps callers flexible regardless of DataFrame implementation.
    """


__all__ = [
    "DayKey",
    "DataFrameLike",
    "SeriesLike",
    "DataByDay",
    "AlertsByDay",
    "Column",
    "Columns",
    "ColumnAddResult",
]
