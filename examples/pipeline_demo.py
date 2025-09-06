"""Minimal demo of the modular pipeline (no external I/O).

Run this to see how to call the new interfaces with a toy DataFrame-like.
"""

from __future__ import annotations

from typing import Dict

try:
    import pandas as pd
except Exception:  # pragma: no cover - only for environments without pandas
    pd = None  # type: ignore

from open_trading_algo import ModelPipeline


def make_df_lib():
    if pd is None:
        # Fallback tiny DF-like with dict-of-lists behavior
        class Tiny:
            def __init__(self, **cols):
                self._cols = dict(cols)

            def __getitem__(self, key):
                return self._cols[key]

            def __setitem__(self, key, value):
                self._cols[key] = value

            def __repr__(self):
                return f"Tiny({self._cols})"

        return Tiny
    else:
        return pd.DataFrame


def main() -> None:
    DF = make_df_lib()
    day = "03-07A"
    df = DF(
        **{
            "Slow EMA Avg": [10.0, 10.5, 11.0],
            "Fast EMA Avg": [11.0, 11.2, 10.8],
            "Volume": [100, 200, 150],
        }
    )

    pipeline = ModelPipeline(data={day: df}, alerts={})
    pipeline.compute_ema_gap(day, "Slow EMA Avg", "Fast EMA Avg", "EMA Gap Slow-Fast")
    pipeline.compute_percent_rank(day, "Volume", "PR-Volume")

    print(pipeline.data[day])


if __name__ == "__main__":
    main()
