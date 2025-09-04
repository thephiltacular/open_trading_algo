# TradingView Algo modular package

This package organizes functions and helpers previously embedded in `data_processing.py` into small, documented modules.

- `types.py`: type aliases and lightweight contracts
- `indicators.py`: pure helper functions (trend flags, EMA gap, volatility, MACD, RSI, Williams, etc.)
- `percent_rank.py`: percent-rank/minmax helpers that work on sequences
- `alerts.py`: alert parsing and aggregation utilities
- `pipeline.py`: `ModelPipeline` façade for day-partitioned DataFrame operations

The original `data_processing.py` is not modified. You can progressively migrate logic by importing from this package.

## Example

```python
from tradingview_algo import ModelPipeline

# Suppose you already loaded day-partitioned pandas DataFrames
pipeline = ModelPipeline(data=my_data_by_day, alerts=my_alerts_by_day)

# Compute an EMA gap for one day
pipeline.compute_ema_gap(
    day="03-07A",
    left_col="Slow EMA Avg",
    right_col="Fast EMA Avg",
    out="EMA Gap Slow-Fast",
)

# Percent-rank within the day
pipeline.compute_percent_rank(day="03-07A", in_col="Volume", out_col="PR-Volume")
```

This code is library-friendly (no prints/side effects) and is easy to unit test.
