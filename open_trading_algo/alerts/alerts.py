"""Alert utilities for open_trading_algo.

This module provides functions for parsing and aggregating alerts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from dateutil import parser


def extract_ticker(cell: str) -> str:
    """Extract the ticker symbol from an alerts cell string.

    The original code used slicing up to the first comma after skipping 5 chars.
    Example cell: "ALRT:TSLA, some text" -> "TSLA"
    """

    if not cell:
        return ""
    try:
        return cell[5 : cell.find(",")] if "," in cell and len(cell) > 5 else cell.strip()
    except Exception:
        return cell.strip()


def is_positive_alert(description: Optional[str]) -> bool:
    """Return True if description mentions a Positive signal."""

    if description is None:
        return False
    return "positive" in description.lower()


def is_negative_alert(description: Optional[str]) -> bool:
    """Return True if description mentions a Negative signal."""

    if description is None:
        return False
    return "negative" in description.lower()


@dataclass
class TickerAlertCounts:
    positive: int = 0
    negative: int = 0
    latest_time: Optional[str] = None
    latest_dt: Optional[object] = None  # datetime, but avoid hard dependency

    def add(self, when: Optional[str], description: Optional[str]) -> None:
        if description is None:
            return
        if is_positive_alert(description):
            self.positive += 1
        if is_negative_alert(description):
            self.negative += 1
        if when:
            try:
                dt = parser.parse(when)
                if self.latest_dt is None or dt > self.latest_dt:
                    self.latest_dt = dt
                    self.latest_time = when
            except Exception:
                # Ignore parse errors; retain previous latest
                pass


def summarize_alerts(
    events: Tuple[Tuple[str, Optional[str], Optional[str]], ...]
) -> Dict[str, TickerAlertCounts]:
    """Compute per-ticker positive/negative counts and latest timestamp.

    Parameters
    - events: iterable of (ticker, time_str, description)

    Returns
    - dict[ticker] -> TickerAlertCounts
    """

    summary: Dict[str, TickerAlertCounts] = {}
    for ticker, time_str, description in events:
        bucket = summary.setdefault(ticker, TickerAlertCounts())
        bucket.add(time_str, description)
    return summary


def parse_alerts(alert_data):
    """Parse alert data.

    Args:
        alert_data (dict): Raw alert data.

    Returns:
        list: Parsed alerts.
    """
    # ...existing code...


__all__ = [
    "extract_ticker",
    "is_positive_alert",
    "is_negative_alert",
    "TickerAlertCounts",
    "summarize_alerts",
]
