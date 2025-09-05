"""V8 model orchestration over the modular pipeline.

Implements a practical, testable subset of the v8 calculation order:
- EMA averages and EMA gap
- EMA difference pairs (e.g., 5/10, 10/20, ..., 100/200) and "EMA Gap Slow-Fast"
- MACD TU/TD trend flags based on available inputs
- Generic pass to compute all "PR-" columns from their base columns within each day

This file uses `cols_model.yaml` to discover column names and ensure we only
create outputs when inputs exist. The original script contains many domain-
specific transforms that are not fully specified; the generic PR pass captures
most PR-* outputs where their base columns are present.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

import yaml

from .pipeline import ModelPipeline


def _load_cols_yaml(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # YAML maps code -> name; we only need the names set
    return {str(k): str(v) for k, v in data.items()}


@dataclass
class V8Model:
    pipeline: ModelPipeline
    cols: Dict[str, str]

    @classmethod
    def from_yaml(cls, pipeline: ModelPipeline, cols_model_path: Path) -> "V8Model":
        return cls(pipeline=pipeline, cols=_load_cols_yaml(cols_model_path))

    # ------------------------------- Core runner ------------------------------
    def compute_all(self) -> None:
        for day in list(self.pipeline.data.keys()):
            # Volume-derived metrics and splits
            self._compute_volume_metrics(day)
            # Band distances and widths (Bollinger/Keltner)
            self._compute_band_distances(day)
            # Price high/low distances divided by ATR
            self._compute_price_high_low_div_atr(day)
            # ADX > 25 flag
            self._compute_adx_filtered_flags(day)
            # Fibonacci minimum from pivot levels
            self._compute_fibonacci_min(day)
            # EMA averages and slow-fast gap
            self.pipeline.compute_ema_averages_and_gap(day)
            # EMA pairwise differences
            self._compute_ema_diffs(day)
            # MACD TU/TD flags
            self._compute_macd_flags(day)
            # MACD cross families (L>S SU / L<S SD)
            self._compute_macd_crosses(day)
            # Trend Up/Down families (TU/TD)
            self._compute_trends_TU(day)
            self._compute_trends_TD(day)
            # Mid-range MD/MU families
            self._compute_mu_family(day)
            self._compute_md_family(day)
            # SD/SU families
            self._compute_trends_SU(day)
            self._compute_avg_trends_SU(day)
            self._compute_avg_PR_trends_SU(day)
            self._compute_trends_SD(day)
            self._compute_avg_trends_SD(day)
            self._compute_avg_PR_trends_SD(day)
            # OS/OB families
            self._compute_os_family(day)
            self._compute_ob_family(day)
            # Reversal aggregates
            self._compute_reversals(day)
            # Trigger score aggregates
            self._compute_trigger_scores(day)
            # Generic pass to compute PR- columns from their base columns
            self._compute_all_percent_ranks(day)

    # ----------------------------- Step utilities ----------------------------
    def _compute_ema_diffs(self, day: str) -> None:
        pairs = [
            (
                "D-Exponential Moving Average (100/200)",
                "D-Exponential Moving Average (100)",
                "D-Exponential Moving Average (200)",
            ),
            (
                "D-Exponential Moving Average (50/100)",
                "D-Exponential Moving Average (50)",
                "D-Exponential Moving Average (100)",
            ),
            (
                "D-Exponential Moving Average (20/50)",
                "D-Exponential Moving Average (20)",
                "D-Exponential Moving Average (50)",
            ),
            (
                "D-Exponential Moving Average (20/30)",
                "D-Exponential Moving Average (20)",
                "D-Exponential Moving Average (30)",
            ),
            (
                "D-Exponential Moving Average (10/20)",
                "D-Exponential Moving Average (10)",
                "D-Exponential Moving Average (20)",
            ),
            (
                "D-Exponential Moving Average (5/10)",
                "D-Exponential Moving Average (5)",
                "D-Exponential Moving Average (10)",
            ),
            (
                "D-Exponential Moving Average (30/50)",
                "D-Exponential Moving Average (30)",
                "D-Exponential Moving Average (50)",
            ),
            ("EMA Gap Slow-Fast", "Slow EMA Avg", "Fast EMA Avg"),
        ]
        df = self.pipeline.data[day]
        for out, left, right in pairs:
            if left in getattr(df, "columns", []) and right in getattr(df, "columns", []):
                # compute negative difference (right-left) * -1
                try:
                    df[out] = (df[right] - df[left]) * -1.0
                except Exception:
                    # fallback row-wise
                    df[out] = [(r - l) * -1.0 for l, r in zip(df[left], df[right])]

    def _compute_macd_flags(self, day: str) -> None:
        # TD: in original, ML "MACD TD" from AV (Change %) vs FM (MACD L>S SU)
        df = self.pipeline.data[day]
        if "Change %" in getattr(df, "columns", []) and "MACD L>S SU" in getattr(df, "columns", []):
            left = df["Change %"]
            right = df["MACD L>S SU"]
            try:
                df["MACD TD"] = (left > 0) & (right < 0) & (right.abs() < left)
                df["MACD TD"] = df["MACD TD"].astype(int)
            except Exception:
                df["MACD TD"] = [
                    int((l > 0) and (r < 0) and (abs(r) < l)) for l, r in zip(left, right)
                ]

        # TU reuses the same condition but may be named "MACD TU" elsewhere
        if "Change %" in getattr(df, "columns", []) and "MACD L>S SU" in getattr(df, "columns", []):
            left = df["Change %"]
            right = df["MACD L>S SU"]
            try:
                df["MACD TU"] = (left > 0) & (right < 0) & (right.abs() < left)
                df["MACD TU"] = df["MACD TU"].astype(int)
            except Exception:
                df["MACD TU"] = [
                    int((l > 0) and (r < 0) and (abs(r) < l)) for l, r in zip(left, right)
                ]

    def _compute_all_percent_ranks(self, day: str) -> None:
        """Compute PR-<X> columns for all base columns <X> present in the day's DF.

        Rule: For any desired output column name that starts with "PR-", if the
        base column name (after stripping the prefix) exists in the DataFrame,
        compute the percent-rank across the day and assign the result.
        """

        df = self.pipeline.data[day]
        if not hasattr(df, "columns"):
            return
        desired_pr_outputs = {name for name in self.cols.values() if name.startswith("PR-")}
        for pr_out in sorted(desired_pr_outputs):
            base = pr_out[3:]  # strip "PR-"
            if base in df.columns:
                values = df[base].tolist() if hasattr(df[base], "tolist") else list(df[base])
                # percent rank within the day
                from .indicators.percent_rank import percent_rank as pr

                df[pr_out] = pr(values)

    def _compute_volume_metrics(self, day: str) -> None:
        """Compute volume ratios and relative-volume-based splits (VU/VD).

        Requires presence of:
        - Volume ("Volume") and Average Volume (10/30/60/90 day)
        - Relative Volume (optional) to compute D-Relative Volume ~ (RV - 1)
        """

        df = self.pipeline.data[day]
        cols = getattr(df, "columns", [])
        has = lambda c: c in cols

        # Base columns
        VOL = "Volume"
        AV10 = "Average Volume (10 day)"
        AV30 = "Average Volume (30 day)"
        AV60 = "Average Volume (60 day)"
        AV90 = "Average Volume (90 day)"
        RV = "Relative Volume"

        # Compute Volume/x ratios where possible
        if all(has(c) for c in (VOL, AV10)):
            try:
                df["Volume/10"] = df[VOL] / df[AV10]
            except Exception:
                df["Volume/10"] = [v / a if a else 0.0 for v, a in zip(df[VOL], df[AV10])]
        if all(has(c) for c in (VOL, AV30)):
            try:
                df["Volume/30"] = df[VOL] / df[AV30]
            except Exception:
                df["Volume/30"] = [v / a if a else 0.0 for v, a in zip(df[VOL], df[AV30])]
        if all(has(c) for c in (VOL, AV60)):
            try:
                df["Volume/60"] = df[VOL] / df[AV60]
            except Exception:
                df["Volume/60"] = [v / a if a else 0.0 for v, a in zip(df[VOL], df[AV60])]
        if all(has(c) for c in (VOL, AV90)):
            try:
                df["Volume/90"] = df[VOL] / df[AV90]
            except Exception:
                df["Volume/90"] = [v / a if a else 0.0 for v, a in zip(df[VOL], df[AV90])]

        # Pairwise averages ratios (e.g., 10/30, 10/60 ... 60/90)
        def _safe_div(a, b):
            try:
                return a / b
            except Exception:
                return 0.0

        if all(has(c) for c in (AV10, AV30)):
            try:
                df["Volume 10/30"] = df[AV10] / df[AV30]
            except Exception:
                df["Volume 10/30"] = [_safe_div(a, b) for a, b in zip(df[AV10], df[AV30])]
        if all(has(c) for c in (AV10, AV60)):
            try:
                df["Volume 10/60"] = df[AV10] / df[AV60]
            except Exception:
                df["Volume 10/60"] = [_safe_div(a, b) for a, b in zip(df[AV10], df[AV60])]
        if all(has(c) for c in (AV10, AV90)):
            try:
                df["Volume 10/90"] = df[AV10] / df[AV90]
            except Exception:
                df["Volume 10/90"] = [_safe_div(a, b) for a, b in zip(df[AV10], df[AV90])]
        if all(has(c) for c in (AV30, AV60)):
            try:
                df["Volume 30/60"] = df[AV30] / df[AV60]
            except Exception:
                df["Volume 30/60"] = [_safe_div(a, b) for a, b in zip(df[AV30], df[AV60])]
        if all(has(c) for c in (AV30, AV90)):
            try:
                df["Volume 30/90"] = df[AV30] / df[AV90]
            except Exception:
                df["Volume 30/90"] = [_safe_div(a, b) for a, b in zip(df[AV30], df[AV90])]
        if all(has(c) for c in (AV60, AV90)):
            try:
                df["Volume 60/90"] = df[AV60] / df[AV90]
            except Exception:
                df["Volume 60/90"] = [_safe_div(a, b) for a, b in zip(df[AV60], df[AV90])]

        # D-Relative Volume approximation from Relative Volume (RV - 1.0)
        if has(RV):
            try:
                df["D-Relative Volume"] = df[RV] - 1.0
            except Exception:
                df["D-Relative Volume"] = [(rv - 1.0) for rv in df[RV]]

            # Split into VU (>=0) and VD (<0) buckets for key ratios
            base_cols = [
                "Volume/10",
                "Volume/30",
                "Volume/60",
                "Volume/90",
                "Volume 10/30",
                "Volume 10/60",
                "Volume 10/90",
                "Volume 30/60",
                "Volume 30/90",
                "Volume 60/90",
            ]
            for base in base_cols:
                if has(base):
                    try:
                        vu = (df["D-Relative Volume"] >= 0).astype(int)
                        vd = (df["D-Relative Volume"] < 0).astype(int)
                        df[f"{base} VU"] = df[base] * vu
                        df[f"{base} VD"] = df[base] * vd
                    except Exception:
                        df[f"{base} VU"] = [
                            b if d >= 0 else 0.0 for b, d in zip(df[base], df["D-Relative Volume"])
                        ]
                        df[f"{base} VD"] = [
                            b if d < 0 else 0.0 for b, d in zip(df[base], df["D-Relative Volume"])
                        ]

    def _compute_band_distances(self, day: str) -> None:
        df = self.pipeline.data[day]
        cols = getattr(df, "columns", [])
        price_col = "Price"
        bl = "Bollinger Lower Band (20)"
        bu = "Bollinger Upper Band (20)"
        kl = "Keltner Channels Lower Band (20)"
        ku = "Keltner Channels Upper Band (20)"
        if price_col in cols and bl in cols:
            try:
                df["Bollinger Price/Lower-1"] = df[price_col] - df[bl]
            except Exception:
                df["Bollinger Price/Lower-1"] = [p - l for p, l in zip(df[price_col], df[bl])]
        if price_col in cols and bu in cols:
            try:
                df["Bollinger Price/Upper-1"] = df[price_col] - df[bu]
            except Exception:
                df["Bollinger Price/Upper-1"] = [p - u for p, u in zip(df[price_col], df[bu])]
        if price_col in cols and kl in cols:
            try:
                df["Keltner Price/Lower-1"] = df[price_col] - df[kl]
            except Exception:
                df["Keltner Price/Lower-1"] = [p - l for p, l in zip(df[price_col], df[kl])]
        if price_col in cols and ku in cols:
            try:
                df["Keltner Price/Upper-1"] = df[price_col] - df[ku]
            except Exception:
                df["Keltner Price/Upper-1"] = [p - u for p, u in zip(df[price_col], df[ku])]

    def _compute_price_high_low_div_atr(self, day: str) -> None:
        df = self.pipeline.data[day]
        cols = getattr(df, "columns", [])
        atr = "Average True Range (14)"
        price = "Price"
        lows = [
            ("D-1-Month Low/ATR", "D-1-Month Low"),
            ("D-3-Month Low/ATR", "D-3-Month Low"),
            ("D-6-Month Low/ATR", "D-6-Month Low"),
            ("D-52 Week Low/ATR", "D-52 Week Low"),
        ]
        highs = [
            ("D-1-Month High/ATR", "D-1-Month High"),
            ("D-3-Month High/ATR", "D-3-Month High"),
            ("D-6-Month High/ATR", "D-6-Month High"),
            ("D-52 Week High/ATR", "D-52 Week High"),
        ]
        if atr in cols and price in cols:
            for out, col in lows:
                if col in cols:
                    try:
                        df[out] = (df[price] - df[col]) / df[atr]
                    except Exception:
                        df[out] = [
                            ((p - c) / a if a else 0.0)
                            for p, c, a in zip(df[price], df[col], df[atr])
                        ]
            for out, col in highs:
                if col in cols:
                    try:
                        df[out] = (df[col] - df[price]) / df[atr]
                    except Exception:
                        df[out] = [
                            ((c - p) / a if a else 0.0)
                            for p, c, a in zip(df[price], df[col], df[atr])
                        ]

    def _compute_adx_filtered_flags(self, day: str) -> None:
        df = self.pipeline.data[day]
        cols = getattr(df, "columns", [])
        if "Average Directional Index (14)" in cols:
            try:
                df["ADX > 25"] = (df["Average Directional Index (14)"] > 25).astype(int)
            except Exception:
                df["ADX > 25"] = [int(v > 25) for v in df["Average Directional Index (14)"]]

    def _compute_fibonacci_min(self, day: str) -> None:
        df = self.pipeline.data[day]
        cols = getattr(df, "columns", [])
        pivots = [
            "D-Pivot Fibonacci R3",
            "D-Pivot Fibonacci R2",
            "D-Pivot Fibonacci R1",
            "D-Pivot Fibonacci P",
            "D-Pivot Fibonacci S1",
            "D-Pivot Fibonacci S2",
            "D-Pivot Fibonacci S3",
        ]
        present = [c for c in pivots if c in cols]
        if present:
            try:
                df["Fibonacci Minimum"] = df[present].min(axis=1)
            except Exception:
                rows = zip(*(df[c] for c in present))
                df["Fibonacci Minimum"] = [min(r) for r in rows]

    # ------------------------------ SU/SD families ----------------------------
    def _compute_trends_SU(self, day: str) -> None:
        df = self.pipeline.data[day]
        cols = getattr(df, "columns", [])
        AV = "Change %"
        pairs = [
            ("Hull MA SU", "D-Hull Moving Average (9)", AV),
            ("EMA-5 SU", "D-Exponential Moving Average (5)", AV),
            ("EMA-10 SU", "D-Exponential Moving Average (10)", AV),
            ("EMA-20 SU", "D-Exponential Moving Average (20)", AV),
            ("EMA-30 SU", "D-Exponential Moving Average (30)", AV),
            ("EMA-50 SU", "D-Exponential Moving Average (50)", AV),
            ("EMA-100 SU", "D-Exponential Moving Average (100)", AV),
            ("EMA-200 SU", "D-Exponential Moving Average (200)", AV),
            ("Ichi Line C/B-1 SU", "Ichi Line C/B-1", AV),
            ("P/Ichi Line C-1 SU", "P/Ichi Line C-1", AV),
            ("P/Ichi Line B-1 SU", "P/Ichi Line B-1", AV),
            ("Ichi Span A/B-1 SU", "Ichi Span A/B-1", AV),
            ("P/Ichi Span A-1 SU", "P/Ichi Span A-1", AV),
            ("P/Ichi Span B-1 SU", "P/Ichi Span B-1", AV),
        ]
        for out, left, right in pairs:
            if left in cols and right in cols:
                try:
                    df[out] = (df[left] > df[right]).astype(int)
                except Exception:
                    df[out] = [int(l > r) for l, r in zip(df[left], df[right])]

        # Price above Upper bands flags
        price = "Price"
        if price in cols and "Bollinger Upper Band (20)" in cols:
            try:
                df["Bollinger Price>Upper SU"] = (
                    df[price] > df["Bollinger Upper Band (20)"]
                ).astype(int)
            except Exception:
                df["Bollinger Price>Upper SU"] = [
                    int(p > u) for p, u in zip(df[price], df["Bollinger Upper Band (20)"])
                ]
        if price in cols and "Keltner Channels Upper Band (20)" in cols:
            try:
                df["Keltner CH Price>Upper SU"] = (
                    df[price] > df["Keltner Channels Upper Band (20)"]
                ).astype(int)
            except Exception:
                df["Keltner CH Price>Upper SU"] = [
                    int(p > u) for p, u in zip(df[price], df["Keltner Channels Upper Band (20)"])
                ]

    def _compute_avg_trends_SU(self, day: str) -> None:
        df = self.pipeline.data[day]
        cols = [
            "# SU",
            "Hull MA SU",
            "EMA-5 SU",
            "EMA-10 SU",
            "EMA-20 SU",
            "EMA-30 SU",
            "EMA-50 SU",
            "EMA-100 SU",
            "EMA-200 SU",
            "Ichi Line C/B-1 SU",
            "P/Ichi Line C-1 SU",
            "P/Ichi Line B-1 SU",
            "Ichi Span A/B-1 SU",
            "P/Ichi Span A-1 SU",
            "P/Ichi Span B-1 SU",
            "Bollinger Price>Upper SU",
            "Keltner CH Price>Upper SU",
        ]
        present = [c for c in cols if c in getattr(df, "columns", [])]
        if present:
            try:
                df["Avg SU"] = sum(df[c] for c in present) / float(len(present))
            except Exception:
                rows = zip(*(df[c] for c in present))
                df["Avg SU"] = [sum(r) / float(len(present)) for r in rows]

    def _compute_avg_PR_trends_SU(self, day: str) -> None:
        df = self.pipeline.data[day]
        pr_cols = [
            "PR-# SU",
            "PR-Avg SU",
            "PR-Hull MA SU",
            "PR-EMA-5 SU",
            "PR-EMA-10 SU",
            "PR-EMA-20 SU",
            "PR-EMA-30 SU",
            "PR-EMA-50 SU",
            "PR-EMA-100 SU",
            "PR-EMA-200 SU",
            "PR-Ichi Line C/B-1 SU",
            "PR-P/Ichi Line C-1 SU",
            "PR-P/Ichi Line B-1 SU",
            "PR-Ichi Span A/B-1 SU",
            "PR-P/Ichi Span A-1 SU",
            "PR-P/Ichi Span B-1 SU",
            "PR-Bollinger Price>Upper SU",
            "PR-Keltner CH Price>Upper SU",
            "PR-SU",
        ]
        present = [c for c in pr_cols if c in getattr(df, "columns", [])]
        if present:
            try:
                df["Avg-# SU"] = sum(df[c] for c in present) / float(len(present))
            except Exception:
                rows = zip(*(df[c] for c in present))
                df["Avg-# SU"] = [sum(r) / float(len(present)) for r in rows]

    def _compute_trends_SD(self, day: str) -> None:
        df = self.pipeline.data[day]
        cols = getattr(df, "columns", [])
        AV = "Change %"
        pairs = [
            ("Hull MA SD", "D-Hull Moving Average (9)", AV),
            ("EMA-5 SD", "D-Exponential Moving Average (5)", AV),
            ("EMA-10 SD", "D-Exponential Moving Average (10)", AV),
            ("EMA-20 SD", "D-Exponential Moving Average (20)", AV),
            ("EMA-30 SD", "D-Exponential Moving Average (30)", AV),
            ("EMA-50 SD", "D-Exponential Moving Average (50)", AV),
            ("EMA-100 SD", "D-Exponential Moving Average (100)", AV),
            ("EMA-200 SD", "D-Exponential Moving Average (200)", AV),
            ("Ichi Line C/B-1 SD", "Ichi Line C/B-1", AV),
            ("P/Ichi Line C-1 SD", "P/Ichi Line C-1", AV),
            ("P/Ichi Line B-1 SD", "P/Ichi Line B-1", AV),
            ("Ichi Span A/B-1 SD", "Ichi Span A/B-1", AV),
            ("P/Ichi Span A-1 SD", "P/Ichi Span A-1", AV),
            ("P/Ichi Span B-1 SD", "P/Ichi Span B-1", AV),
        ]
        for out, left, right in pairs:
            if left in cols and right in cols:
                try:
                    df[out] = (df[left] < df[right]).astype(int)
                except Exception:
                    df[out] = [int(l < r) for l, r in zip(df[left], df[right])]

        price = "Price"
        if price in cols and "Bollinger Lower Band (20)" in cols:
            try:
                df["Bollinger Price<Lower SD"] = (
                    df[price] < df["Bollinger Lower Band (20)"]
                ).astype(int)
            except Exception:
                df["Bollinger Price<Lower SD"] = [
                    int(p < l) for p, l in zip(df[price], df["Bollinger Lower Band (20)"])
                ]
        if price in cols and "Keltner Channels Lower Band (20)" in cols:
            try:
                df["Keltner CH Price<Lower SD"] = (
                    df[price] < df["Keltner Channels Lower Band (20)"]
                ).astype(int)
            except Exception:
                df["Keltner CH Price<Lower SD"] = [
                    int(p < l) for p, l in zip(df[price], df["Keltner Channels Lower Band (20)"])
                ]

    def _compute_macd_crosses(self, day: str) -> None:
        """Compute MACD cross flags used by SU/SD families.

        - FM: MACD L>S SU (MACD Level > MACD Signal)
        - HA: MACD L<S SD (MACD Level < MACD Signal)
        """

        df = self.pipeline.data[day]
        cols = getattr(df, "columns", [])
        if "MACD Level (12, 26)" in cols and "MACD Signal (12, 26)" in cols:
            try:
                df["MACD L>S SU"] = (df["MACD Level (12, 26)"] > df["MACD Signal (12, 26)"]).astype(
                    int
                )
                df["MACD L<S SD"] = (df["MACD Level (12, 26)"] < df["MACD Signal (12, 26)"]).astype(
                    int
                )
            except Exception:
                ml = df["MACD Level (12, 26)"]
                ms = df["MACD Signal (12, 26)"]
                df["MACD L>S SU"] = [int(a > b) for a, b in zip(ml, ms)]
                df["MACD L<S SD"] = [int(a < b) for a, b in zip(ml, ms)]

    def _compute_os_family(self, day: str) -> None:
        df = self.pipeline.data[day]
        cols = getattr(df, "columns", [])
        # Flags
        mapping = {
            "RSI (7) OS": ("Relative Strength Index (7)", "le", 30),
            "RSI (14) OS": ("Relative Strength Index (14)", "le", 30),
            "St-RSI Fast OS": ("Stochastic RSI Fast (3, 3, 14, 14)", "le", 0.2),
            "St-RSI Slow OS": ("Stochastic RSI Slow (3, 3, 14, 14)", "le", 0.2),
            "Stoch K% OS": ("Stochastic %K (14, 3, 3)", "le", 20),
            "Stoch D% OS": ("Stochastic %D (14, 3, 3)", "le", 20),
            "Williams% OS": ("Williams Percent Range (14)", "le", -80),
            "UO OS": ("Ultimate Oscillator (7, 14, 28)", "le", 30),
        }
        made = []
        for out, (src, op, thr) in mapping.items():
            if src in cols:
                if op == "le":
                    try:
                        df[out] = (df[src] <= thr).astype(int)
                    except Exception:
                        df[out] = [int(v <= thr) for v in df[src]]
                made.append(out)
        # Aggregates
        if made:
            try:
                df["# OS"] = sum(df[c] for c in made)
                df["Avg OS"] = sum(df[c] for c in made) / float(len(made))
            except Exception:
                rows = zip(*(df[c] for c in made))
                sums = [sum(r) for r in rows]
                df["# OS"] = sums
                df["Avg OS"] = [s / float(len(made)) for s in sums]

    def _compute_ob_family(self, day: str) -> None:
        df = self.pipeline.data[day]
        cols = getattr(df, "columns", [])
        mapping = {
            "RSI (7) OB": ("Relative Strength Index (7)", "ge", 70),
            "RSI (14) OB": ("Relative Strength Index (14)", "ge", 70),
            "St-RSI Fast OB": ("Stochastic RSI Fast (3, 3, 14, 14)", "ge", 0.8),
            "St-RSI Slow OB": ("Stochastic RSI Slow (3, 3, 14, 14)", "ge", 0.8),
            "Stoch K% OB": ("Stochastic %K (14, 3, 3)", "ge", 80),
            "Stoch D% OB": ("Stochastic %D (14, 3, 3)", "ge", 80),
            "Willams% OB": ("Williams Percent Range (14)", "ge", -20),
            "CCI OB": ("Commodity Channel Index (20)", "ge", 100),
            "CMF OB": ("Chaikin Money Flow (20)", "ge", 0.1),
            "MFI OB": ("Money Flow (14)", "ge", 80),
            "UO OB": ("Ultimate Oscillator (7, 14, 28)", "ge", 70),
        }
        made = []
        for out, (src, op, thr) in mapping.items():
            if src in cols:
                if op == "ge":
                    try:
                        df[out] = (df[src] >= thr).astype(int)
                    except Exception:
                        df[out] = [int(v >= thr) for v in df[src]]
                made.append(out)
        if made:
            try:
                df["# OB"] = sum(df[c] for c in made)
                df["Avg OB"] = sum(df[c] for c in made) / float(len(made))
            except Exception:
                rows = zip(*(df[c] for c in made))
                sums = [sum(r) for r in rows]
                df["# OB"] = sums
                df["Avg OB"] = [s / float(len(made)) for s in sums]

    # ------------------------------ MU/MD families ----------------------------
    def _compute_mu_family(self, day: str) -> None:
        """Mid-up range flags (e.g., RSI in 50-70)."""
        df = self.pipeline.data[day]
        cols = getattr(df, "columns", [])
        made = []

        def between(series, low, high):
            try:
                return ((series >= low) & (series <= high)).astype(int)
            except Exception:
                return [int((v is not None) and (v >= low) and (v <= high)) for v in series]

        mapping = {
            "RSI (7) 50-70 MU": ("Relative Strength Index (7)", 50, 70),
            "RSI (14) 50-70 MU": ("Relative Strength Index (14)", 50, 70),
            "St-RSI Fast MU": ("Stochastic RSI Fast (3, 3, 14, 14)", 0.4, 0.6),
            "St-RSI Slow MU": ("Stochastic RSI Slow (3, 3, 14, 14)", 0.4, 0.6),
        }
        for out, (src, lo, hi) in mapping.items():
            if src in cols:
                df[out] = between(df[src], lo, hi)
                made.append(out)
        if made:
            try:
                df["# MU"] = sum(df[c] for c in made)
                df["Avg MU"] = sum(df[c] for c in made) / float(len(made))
            except Exception:
                rows = zip(*(df[c] for c in made))
                sums = [sum(r) for r in rows]
                df["# MU"] = sums
                df["Avg MU"] = [s / float(len(made)) for s in sums]

    def _compute_md_family(self, day: str) -> None:
        """Mid-down range flags (e.g., RSI in 30-50)."""
        df = self.pipeline.data[day]
        cols = getattr(df, "columns", [])
        made = []

        def between(series, low, high):
            try:
                return ((series >= low) & (series <= high)).astype(int)
            except Exception:
                return [int((v is not None) and (v >= low) and (v <= high)) for v in series]

        mapping = {
            "RSI (14) 30-50 MD": ("Relative Strength Index (14)", 30, 50),
            "RSI (7) 30-50 MD": ("Relative Strength Index (7)", 30, 50),
            "St-RSI Fast MD": ("Stochastic RSI Fast (3, 3, 14, 14)", 0.2, 0.5),
            "St-RSI Slow MD": ("Stochastic RSI Slow (3, 3, 14, 14)", 0.2, 0.5),
        }
        for out, (src, lo, hi) in mapping.items():
            if src in cols:
                df[out] = between(df[src], lo, hi)
                made.append(out)
        if made:
            try:
                df["# MD"] = sum(df[c] for c in made)
                df["Avg MD"] = sum(df[c] for c in made) / float(len(made))
            except Exception:
                rows = zip(*(df[c] for c in made))
                sums = [sum(r) for r in rows]
                df["# MD"] = sums
                df["Avg MD"] = [s / float(len(made)) for s in sums]

    def _compute_avg_trends_SD(self, day: str) -> None:
        df = self.pipeline.data[day]
        cols = [
            "# SD",
            "Hull MA SD",
            "EMA-5 SD",
            "EMA-10 SD",
            "EMA-20 SD",
            "EMA-30 SD",
            "EMA-50 SD",
            "EMA-100 SD",
            "EMA-200 SD",
            "Ichi Line C/B-1 SD",
            "P/Ichi Line C-1 SD",
            "P/Ichi Line B-1 SD",
            "Ichi Span A/B-1 SD",
            "P/Ichi Span A-1 SD",
            "P/Ichi Span B-1 SD",
            "Bollinger Price<Lower SD",
            "Keltner CH Price<Lower SD",
        ]
        present = [c for c in cols if c in getattr(df, "columns", [])]
        if present:
            try:
                df["Avg SD"] = sum(df[c] for c in present) / float(len(present))
            except Exception:
                rows = zip(*(df[c] for c in present))
                df["Avg SD"] = [sum(r) / float(len(present)) for r in rows]

    def _compute_avg_PR_trends_SD(self, day: str) -> None:
        df = self.pipeline.data[day]
        pr_cols = [
            "PR-# SD",
            "PR-Avg SD",
            "PR-Hull MA SD",
            "PR-EMA-5 SD",
            "PR-EMA-10 SD",
            "PR-EMA-20 SD",
            "PR-EMA-30 SD",
            "PR-EMA-50 SD",
            "PR-EMA-100 SD",
            "PR-EMA-200 SD",
            "PR-Ichi Line C/B-1 SD",
            "PR-P/Ichi Line C-1 SD",
            "PR-P/Ichi Line B-1 SD",
            "PR-Ichi Span A/B-1 SD",
            "PR-P/Ichi Span A-1 SD",
            "PR-P/Ichi Span B-1 SD",
            "PR-Bollinger Price<Lower SD",
            "PR-Keltner CH Price<Lower SD",
            "PR-SD",
        ]
        present = [c for c in pr_cols if c in getattr(df, "columns", [])]
        if present:
            try:
                df["Avg-# SD"] = sum(df[c] for c in present) / float(len(present))
            except Exception:
                rows = zip(*(df[c] for c in present))
                df["Avg-# SD"] = [sum(r) / float(len(present)) for r in rows]

    # ------------------------------ TU/TD families ----------------------------
    def _compute_trends_TU(self, day: str) -> None:
        df = self.pipeline.data[day]
        cols = getattr(df, "columns", [])
        AV = "Change %"
        pairs = [
            ("P/Ichi Span A-1 TU", "P/Ichi Span A-1", AV),
            ("P/Ichi Span B-1 TU", "P/Ichi Span B-1", AV),
            ("P/Ichi Line B-1 TU", "P/Ichi Line B-1", AV),
            ("P/Ichi Line C-1 TU", "P/Ichi Line C-1", AV),
            ("Ichi Line C/B-1 TU", "Ichi Line C/B-1", AV),
            ("Ichi Span A/B-1 TU", "Ichi Span A/B-1", AV),
        ]
        for out, left, right in pairs:
            if left in cols and right in cols:
                try:
                    df[out] = (df[left] > df[right]).astype(int)
                except Exception:
                    df[out] = [int(l > r) for l, r in zip(df[left], df[right])]

        # Aggregates
        ichi_cols = [
            "Ichi Line C/B-1 TU",
            "P/Ichi Line C-1 TU",
            "P/Ichi Line B-1 TU",
            "Ichi Span A/B-1 TU",
            "P/Ichi Span A-1 TU",
            "P/Ichi Span B-1 TU",
        ]
        present = [c for c in ichi_cols if c in getattr(df, "columns", [])]
        if present:
            try:
                df["Avg Ichi TU"] = sum(df[c] for c in present) / float(len(present))
            except Exception:
                rows = zip(*(df[c] for c in present))
                df["Avg Ichi TU"] = [sum(r) / float(len(present)) for r in rows]

        # MA TU average across Hull and EMAs if present
        ma_cols = [
            "Hull MA TU",
            "EMA (5) TU",
            "EMA (10) TU",
            "EMA (20) TU",
            "EMA (30) TU",
            "EMA (50) TU",
            "EMA (100) TU",
            "EMA (200) TU",
        ]
        present_ma = [c for c in ma_cols if c in getattr(df, "columns", [])]
        if present_ma:
            try:
                df["Avg MA TU"] = sum(df[c] for c in present_ma) / float(len(present_ma))
            except Exception:
                rows = zip(*(df[c] for c in present_ma))
                df["Avg MA TU"] = [sum(r) / float(len(present_ma)) for r in rows]

        # Avg TU from summaries when available
        sum_cols = [
            "PR-Avg-# MA TU",
            "PR-Avg-# Ichi TU",
            "PR-Avg-# Vol TU",
            "PR-MACD TU",
            "PR-UO TU",
        ]
        present_sum = [c for c in sum_cols if c in getattr(df, "columns", [])]
        if present_sum:
            try:
                df["Avg TU"] = sum(df[c] for c in present_sum) / float(len(present_sum))
            except Exception:
                rows = zip(*(df[c] for c in present_sum))
                df["Avg TU"] = [sum(r) / float(len(present_sum)) for r in rows]

        # Avg-# Ichi TU
        avg_hash_cols = [
            c
            for c in ("PR-Avg Ichi TU", "PR-#-Ichi TU", "PR-Avg-# Ichi TU")
            if c in getattr(df, "columns", [])
        ]
        if avg_hash_cols:
            try:
                df["Avg-# Ichi TU"] = sum(df[c] for c in avg_hash_cols) / float(len(avg_hash_cols))
            except Exception:
                rows = zip(*(df[c] for c in avg_hash_cols))
                df["Avg-# Ichi TU"] = [sum(r) / float(len(avg_hash_cols)) for r in rows]

        # Avg-# TU from PR-# TU, PR-Avg TU and PR-TU
        cols_tu = [c for c in ("PR-# TU", "PR-Avg TU", "PR-TU") if c in getattr(df, "columns", [])]
        if cols_tu:
            try:
                df["Avg-# TU"] = sum(df[c] for c in cols_tu) / float(len(cols_tu))
            except Exception:
                rows = zip(*(df[c] for c in cols_tu))
                df["Avg-# TU"] = [sum(r) / float(len(cols_tu)) for r in rows]

    def _compute_trends_TD(self, day: str) -> None:
        df = self.pipeline.data[day]
        cols = getattr(df, "columns", [])
        AV = "Change %"
        pairs = [
            ("Ichi Line C/B TD", "Ichi Line C/B-1", AV),
            ("P/Ichi Line C-1 TD", "P/Ichi Line C-1", AV),
            ("P/Ichi Line B-1 TD", "P/Ichi Line B-1", AV),
            ("Ichi Span A/B TD", "Ichi Span A/B-1", AV),
            ("P/Ichi Span A-1 TD", "P/Ichi Span A-1", AV),
            ("P/Ichi Span B-1 TD", "P/Ichi Span B-1", AV),
        ]
        for out, left, right in pairs:
            if left in cols and right in cols:
                try:
                    df[out] = (df[left] > df[right]).astype(int)
                except Exception:
                    df[out] = [int(l > r) for l, r in zip(df[left], df[right])]

        # Averages
        ma_cols = [
            c
            for c in (
                "Hull MA TD",
                "EMA (5) TD",
                "EMA (10) TD",
                "EMA (20) TD",
                "EMA (30) TD",
                "EMA (50) TD",
                "EMA (100) TD",
                "EMA (200) TD",
            )
            if c in getattr(df, "columns", [])
        ]
        if ma_cols:
            try:
                df["Avg MA TD"] = sum(df[c] for c in ma_cols) / float(len(ma_cols))
            except Exception:
                rows = zip(*(df[c] for c in ma_cols))
                df["Avg MA TD"] = [sum(r) / float(len(ma_cols)) for r in rows]

        ichi_cols = [
            c
            for c in (
                "Ichi Line C/B TD",
                "P/Ichi Line C-1 TD",
                "P/Ichi Line B-1 TD",
                "Ichi Span A/B TD",
                "P/Ichi Span A-1 TD",
                "P/Ichi Span B-1 TD",
            )
            if c in getattr(df, "columns", [])
        ]
        if ichi_cols:
            try:
                df["Avg Ichi TD"] = sum(df[c] for c in ichi_cols) / float(len(ichi_cols))
            except Exception:
                rows = zip(*(df[c] for c in ichi_cols))
                df["Avg Ichi TD"] = [sum(r) / float(len(ichi_cols)) for r in rows]

        # Avg TD from summary PRs when available
        sum_cols = [
            c
            for c in ("PR-Avg-# MA TD", "PR-Avg-# Vol TD", "PR-MACD TD", "PR-UO TD")
            if c in getattr(df, "columns", [])
        ]
        if sum_cols:
            try:
                df["Avg TD"] = sum(df[c] for c in sum_cols) / float(len(sum_cols))
            except Exception:
                rows = zip(*(df[c] for c in sum_cols))
                df["Avg TD"] = [sum(r) / float(len(sum_cols)) for r in rows]

        # Avg-# Ichi TD
        avg_hash_cols = [
            c
            for c in ("PR-Avg Ichi TD", "PR-# Ichi TD", "PR-Avg-# Ichi TD")
            if c in getattr(df, "columns", [])
        ]
        if avg_hash_cols:
            try:
                df["Avg-# Ichi TD"] = sum(df[c] for c in avg_hash_cols) / float(len(avg_hash_cols))
            except Exception:
                rows = zip(*(df[c] for c in avg_hash_cols))
                df["Avg-# Ichi TD"] = [sum(r) / float(len(avg_hash_cols)) for r in rows]

        # Avg-# TD from PR-# TD, PR-Avg TD and PR-TD
        cols_td = [c for c in ("PR-# TD", "PR-Avg TD", "PR-TD") if c in getattr(df, "columns", [])]
        if cols_td:
            try:
                df["Avg-# TD"] = sum(df[c] for c in cols_td) / float(len(cols_td))
            except Exception:
                rows = zip(*(df[c] for c in cols_td))
                df["Avg-# TD"] = [sum(r) / float(len(cols_td)) for r in rows]

    # ------------------------------ Reversals --------------------------------
    def _compute_reversals(self, day: str) -> None:
        df = self.pipeline.data[day]
        cols = getattr(df, "columns", [])
        # Bottom reversal average DY from CV, DD, DH, DP when present
        bottom_cols = [c for c in ("PR-Avg VD", "PR-Avg OS", "PR-Avg SU", "PR-Avg SQ") if c in cols]
        if bottom_cols:
            try:
                df["Avg Bottom Reversal"] = sum(df[c] for c in bottom_cols) / float(len(bottom_cols))
            except Exception:
                rows = zip(*(df[c] for c in bottom_cols))
                df["Avg Bottom Reversal"] = [sum(r) / float(len(bottom_cols)) for r in rows]
        # Top reversal average DV from CV, CZ, DL, DP
        top_cols = [c for c in ("PR-Avg VD", "PR-Avg OB", "PR-Avg SD", "PR-Avg SQ") if c in cols]
        if top_cols:
            try:
                df["Avg Top Reversal"] = sum(df[c] for c in top_cols) / float(len(top_cols))
            except Exception:
                rows = zip(*(df[c] for c in top_cols))
                df["Avg Top Reversal"] = [sum(r) / float(len(top_cols)) for r in rows]
        # Avg-# Bottom/Top reversal from PR pairs if present
        br_cols = [c for c in ("PR-# BR", "PR-Avg BR") if c in cols]
        if br_cols:
            try:
                df["Avg-# Bottom Reversal"] = sum(df[c] for c in br_cols) / float(len(br_cols))
            except Exception:
                rows = zip(*(df[c] for c in br_cols))
                df["Avg-# Bottom Reversal"] = [sum(r) / float(len(br_cols)) for r in rows]
        tr_cols = [c for c in ("PR-# TR", "PR-Avg TR") if c in cols]
        if tr_cols:
            try:
                df["Avg-# Top Reversal"] = sum(df[c] for c in tr_cols) / float(len(tr_cols))
            except Exception:
                rows = zip(*(df[c] for c in tr_cols))
                df["Avg-# Top Reversal"] = [sum(r) / float(len(tr_cols)) for r in rows]

    # ----------------------------- Trigger scores ----------------------------
    def _compute_trigger_scores(self, day: str) -> None:
        df = self.pipeline.data[day]
        cols = getattr(df, "columns", [])
        # DS Avg Trigger Score from a basket of PR metrics if present
        basket = [
            "PR-TR",
            "PR-BR",
            "PR-TU",
            "PR-TD",
            "PR-MU",
            "PR-MD",
            "PR-VU",
            "PR-VD",
            "PR-OB",
            "PR-OS",
            "PR-SU",
            "PR-SD",
            "PR-SQ",
        ]
        present = [b for b in basket if b in cols]
        if present:
            try:
                df["Avg Trigger Score"] = sum(df[c] for c in present) / float(len(present))
            except Exception:
                rows = zip(*(df[c] for c in present))
                df["Avg Trigger Score"] = [sum(r) / float(len(present)) for r in rows]
        # DQ Avg-# TS = mean(PR(# TS), Avg Trigger Score) if # Trigger Score present
        if "# Trigger Score" in cols:
            try:
                # normalize count via rank within-day
                series = df["# Trigger Score"]
                vals = series.tolist() if hasattr(series, "tolist") else list(series)
                from .indicators.percent_rank import percent_rank as pr

                pr_counts = pr(vals)
                if "Avg Trigger Score" in cols:
                    df["Avg-# TS"] = (df["Avg Trigger Score"] + pr_counts) / 2.0
                else:
                    df["Avg-# TS"] = pr_counts
            except Exception:
                pass


__all__ = ["V8Model", "_load_cols_yaml"]
