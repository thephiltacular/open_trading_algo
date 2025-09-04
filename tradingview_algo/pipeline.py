"""High-level orchestration pipeline.

This `ModelPipeline` is a clean, documented, and testable façade that replaces
the monolithic class structure in `data_processing.py`. It doesn't mutate
external state on construction and avoids printing; callers control execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from . import indicators as I
from . import percent_rank as PR
from .types import AlertsByDay, ColumnAddResult, DataByDay, DayKey


@dataclass
class ModelPipeline:
    """Orchestrates indicator calculations over day-partitioned data.

    Attributes
    - data: mapping of day key -> DataFrame-like object containing columns
    - alerts: mapping of day key -> DataFrame-like alerts data

    Note: This class is library-friendly: it does not print or perform I/O.
    """

    data: DataByDay
    alerts: AlertsByDay

    # ----------------------------- Utility helpers ----------------------------
    def _has_columns(self, day: DayKey, columns: list[str]) -> bool:
        df = self.data[day]
        try:
            cols = set(df.columns)  # pandas-like
        except Exception:
            # best-effort for dict-of-lists style
            cols = set(getattr(df, "_cols", {}).keys())
        return all(c in cols for c in columns)

    # ------------------------------- Examples -------------------------------
    def compute_ema_gap(self, day: DayKey, left_col: str, right_col: str, out: str) -> None:
        """Compute EMA gap for a single day and assign to `out` column.

        Requires the day's frame to support `__getitem__` dict-like access to
        columns by name and item assignment for new columns (pandas-compatible).
        """

        df = self.data[day]
        left = df[left_col]
        right = df[right_col]
        # vectorized-friendly: assume Series-like; fall back to element-wise map
        try:
            df[out] = (right - left) * -1.0
        except Exception:
            df[out] = [I.ema_gap(l, r) for l, r in zip(left, right)]

    def compute_percent_rank(self, day: DayKey, in_col: str, out_col: str) -> None:
        """Compute percent-rank for a column within the day's data."""

        df = self.data[day]
        values = df[in_col].tolist() if hasattr(df[in_col], "tolist") else list(df[in_col])
        df[out_col] = PR.percent_rank(values)

    def compute_ema_averages_and_gap(self, day: DayKey) -> None:
        """Compute Fast/Slow EMA averages and the gap when constituent columns exist.

        Expected inputs (by column name):
        - Fast:  D-Exponential Moving Average (5|10|20)
        - Slow:  D-Exponential Moving Average (30|50|100)
        Outputs:
        - Fast EMA Avg, Slow EMA Avg, EMA Gap Slow-Fast
        """

        fast_cols = [
            "D-Exponential Moving Average (5)",
            "D-Exponential Moving Average (10)",
            "D-Exponential Moving Average (20)",
        ]
        slow_cols = [
            "D-Exponential Moving Average (30)",
            "D-Exponential Moving Average (50)",
            "D-Exponential Moving Average (100)",
        ]
        if self._has_columns(day, fast_cols):
            self.average_columns(day, "Fast EMA Avg", fast_cols)
        if self._has_columns(day, slow_cols):
            self.average_columns(day, "Slow EMA Avg", slow_cols)
        # If both were produced, compute gap
        if self._has_columns(day, ["Slow EMA Avg", "Fast EMA Avg"]):
            self.compute_ema_gap(day, "Slow EMA Avg", "Fast EMA Avg", "EMA Gap Slow-Fast")

    # ------------------------ Batch/utility operations -----------------------
    def add_from_binary_flag(self, day: DayKey, out_col: str, in_col: str) -> None:
        """Copy a binary flag (0/1) from in_col to out_col with safe casting."""

        df = self.data[day]
        src = df[in_col]
        try:
            df[out_col] = src.astype(int)
        except Exception:
            df[out_col] = [int(x or 0) for x in src]

    def average_columns(self, day: DayKey, out_col: str, columns: list[str]) -> None:
        """Average a set of numeric columns into `out_col`.

        Falls back to python averaging when vectorization fails.
        """

        df = self.data[day]
        try:
            df[out_col] = sum(df[c] for c in columns) / float(len(columns))
        except Exception:
            rows = zip(*(df[c] for c in columns))
            df[out_col] = [sum(r) / float(len(columns)) for r in rows]

    # ---------------------------- Alerts handling ---------------------------
    def attach_alert_counts(self, day: DayKey, pos_col: str, neg_col: str, latest_col: str) -> None:
        """Example of enriching data with alert counts from `self.alerts`.

        This is a placeholder showing how alerts can be merged; the concrete
        joining logic depends on the original schema, which is not fully known.
        """

        # Implementation would align on ticker and assign counts per row.
        pass


__all__ = ["ModelPipeline"]
