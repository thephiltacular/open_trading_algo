"""Percent-rank and normalization helpers.

These utilities operate on simple sequences to avoid a hard dependency on
specific DataFrame libraries. Pipelines can adapt them for pandas or other
tabular structures.
"""

from __future__ import annotations

from typing import Iterable, List


def percent_rank(values: Iterable[float]) -> List[float]:
    """Compute percent-rank for each value among the values list.

    Args:
        values (Iterable[float]): Input values to rank.

    Returns:
        List[float]: Percent-rank values in [0, 1]. If all values are equal, returns 0.5 for all.
    """

    vals = list(values)
    if not vals:
        return []
    sorted_vals = sorted(vals)
    n = len(vals)
    if sorted_vals[0] == sorted_vals[-1]:
        return [0.5] * n

    def pr(v: float) -> float:
        # proportion of values less than v
        less = 0
        for s in sorted_vals:
            if s < v:
                less += 1
        return less / (n - 1) if n > 1 else 0.0

    return [pr(v) for v in vals]


def minmax(values: Iterable[float]) -> List[float]:
    """Min-max normalize to [0, 1].

    Args:
        values (Iterable[float]): Input values to normalize.

    Returns:
        List[float]: Normalized values in [0, 1]. If constant, returns 0.5 for all.
    """
    vals = list(values)
    if not vals:
        return []
    vmin = min(vals)
    vmax = max(vals)
    if vmin == vmax:
        return [0.5] * len(vals)
    rng = vmax - vmin
    return [(v - vmin) / rng for v in vals]


__all__ = ["percent_rank", "minmax"]
