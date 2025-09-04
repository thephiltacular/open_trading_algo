#!/usr/bin/env python3
"""CLI runner for the modular TradingView Algo pipeline.

This runner:
- Reads CSV files under ./data (matching the existing repo structure)
- Groups them by day key (basename without trailing A/D letter where applicable)
- Computes a subset of model outputs that are unambiguous from the original
  script (EMA averages, EMA gap, percent-rank examples)
- Writes enriched CSVs to ./data_out

The original `data_processing.py` is not modified.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    print(
        "This runner requires pandas. Please install it (e.g., pip install pandas).", file=sys.stderr
    )
    raise

from tradingview_algo import ModelPipeline
from tradingview_algo.v8_model import V8Model


def discover_day_files(data_dir: Path) -> Dict[str, list[Path]]:
    """Collect CSV files under data_dir and group by day.

    For this repo, input files are already day-based (e.g., 03-07A.csv, 03-07D.csv).
    We'll group by the date portion without the A/D suffix for aggregation or simply
    keep per-file processing, depending on needs. For now, we keep per-file keys.
    """

    files = sorted(p for p in data_dir.glob("*.csv") if p.is_file())
    groups: Dict[str, list[Path]] = {}
    for p in files:
        key = p.stem  # e.g., "03-07A"
        groups.setdefault(key, []).append(p)
    return groups


def run(
    input_dir: Path, output_dir: Path, use_v8: bool = False, cols_model_path: Path | None = None
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    day_files = discover_day_files(input_dir)

    for day, paths in day_files.items():
        # Expect one CSV per key; if multiple, concatenate naively
        frames = [pd.read_csv(p) for p in paths]
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

        pipeline = ModelPipeline(data={day: df}, alerts={})
        if use_v8:
            if cols_model_path is None:
                cols_model_path = Path("cols_model.yaml").resolve()
            v8 = V8Model.from_yaml(pipeline, cols_model_path)
            v8.compute_all()
        else:
            # Compute EMA averages and gap when inputs exist
            pipeline.compute_ema_averages_and_gap(day)
            # Example: if a "Volume" column exists, produce intra-day percent-rank
            if "Volume" in df.columns:
                pipeline.compute_percent_rank(day, "Volume", "PR-Volume")

        # Write out
        out_path = output_dir / f"{day}.csv"
        pipeline.data[day].to_csv(out_path, index=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run TradingView Algo modular pipeline")
    parser.add_argument("--data-dir", default="data", help="Input directory with CSVs")
    parser.add_argument("--out-dir", default="data_out", help="Output directory for CSVs")
    parser.add_argument("--v8", action="store_true", help="Run the v8 model pipeline")
    parser.add_argument(
        "--cols-model", default="cols_model.yaml", help="Path to cols_model.yaml for v8"
    )
    args = parser.parse_args(argv)

    input_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.out_dir).resolve()
    run(input_dir, output_dir, use_v8=args.v8, cols_model_path=Path(args.cols_model).resolve())


if __name__ == "__main__":
    main()
