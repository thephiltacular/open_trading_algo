"""Alert utilities for open_trading_algo.

This module provides functions for parsing and aggregating alerts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from dateutil import parser


def extract_ticker(cell: str) -> str:
    """Extract the ticker symbol from an alerts cell string.

    Args:
        cell (str): Alert cell string (e.g., "ALRT:TSLA, some text").

    Returns:
        str: Extracted ticker symbol.
    """

    if not cell:
        return ""
    try:
        if "," in cell and len(cell) > 5:
            ticker = cell[5 : cell.find(",")]
            return ticker if ticker else ""
        elif len(cell) > 5 and cell.startswith("ALRT:"):
            ticker = cell[5:].strip()
            return ticker if ticker else ""
        elif cell.startswith("ALRT:"):
            return ""  # Handle "ALRT:" case
        else:
            return cell.strip()
    except Exception:
        return cell.strip()


def is_positive_alert(description: Optional[str]) -> bool:
    """Return True if description mentions a Positive signal.

    Args:
        description (Optional[str]): Alert description text.

    Returns:
        bool: True if the description indicates a positive signal.
    """
    if description is None:
        return False
    desc_lower = description.lower()
    # Check for negative keywords first - if negative, it's not positive
    negative_keywords = ["negative", "sell", "bearish", "strong sell", "bear", "divergence"]
    if any(keyword in desc_lower for keyword in negative_keywords):
        return False
    positive_keywords = ["positive", "buy", "bullish", "strong buy", "bull", "momentum"]
    return any(keyword in desc_lower for keyword in positive_keywords)


def is_negative_alert(description: Optional[str]) -> bool:
    """Return True if description mentions a Negative signal.

    Args:
        description (Optional[str]): Alert description text.

    Returns:
        bool: True if the description indicates a negative signal.
    """
    if description is None:
        return False
    negative_keywords = ["negative", "sell", "bearish", "strong sell", "divergence", "bear", "down"]
    return any(keyword in description.lower() for keyword in negative_keywords)


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

    Args:
        events: Iterable of (ticker, time_str, description) tuples.

    Returns:
        Dict[str, TickerAlertCounts]: Dictionary mapping tickers to alert counts.
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
