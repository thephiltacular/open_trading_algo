"""Live data interface for TradingViewAlgoDev.

This module loads configuration from a YAML file and periodically fetches
financial data for a list of tickers/ETFs from a free source (default: Yahoo Finance).
It exposes a clean interface for downstream analysis and alert generation.
"""

import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

try:
    import yfinance as yf
except ImportError:
    raise ImportError("Please install yfinance: pip install yfinance")


class LiveDataConfig:
    def __init__(self, config_path: Path):
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.update_rate = int(cfg.get("update_rate", 300))
        self.tickers = list(cfg.get("tickers", []))
        self.source = str(cfg.get("source", "yahoo"))
        self.api_key = str(cfg.get("api_key", ""))
        self.fields = list(cfg.get("fields", ["price", "volume"]))


def fetch_yahoo(tickers: List[str], fields: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch latest data for tickers from Yahoo Finance."""
    data = {}
    tickers_str = " ".join(tickers)
    df = yf.download(tickers_str, period="1d", interval="1m", group_by="ticker", progress=False)
    for ticker in tickers:
        try:
            tdf = df[ticker] if isinstance(df, dict) else df.xs(ticker, axis=1, level=1)
            latest = tdf.iloc[-1]
            out = {}
            for field in fields:
                if field == "price":
                    out[field] = float(latest["Close"])
                elif field == "open":
                    out[field] = float(latest["Open"])
                elif field == "high":
                    out[field] = float(latest["High"])
                elif field == "low":
                    out[field] = float(latest["Low"])
                elif field == "close":
                    out[field] = float(latest["Close"])
                elif field == "volume":
                    out[field] = float(latest["Volume"])
                elif field == "previous_close":
                    out[field] = float(tdf.iloc[-2]["Close"]) if len(tdf) > 1 else None
                elif field == "change":
                    prev = float(tdf.iloc[-2]["Close"]) if len(tdf) > 1 else None
                    out[field] = float(latest["Close"] - prev) if prev is not None else None
                elif field == "percent_change":
                    prev = float(tdf.iloc[-2]["Close"]) if len(tdf) > 1 else None
                    out[field] = float((latest["Close"] - prev) / prev * 100) if prev else None
                elif field == "timestamp":
                    out[field] = str(latest.name)
            data[ticker] = out
        except Exception:
            data[ticker] = {f: None for f in fields}
    return data


class LiveDataFeed:
    """Live data feed that periodically fetches ticker data and exposes it for analysis."""

    def __init__(
        self,
        config_path: Path,
        on_update: Optional[Callable[[Dict[str, Dict[str, Any]]], None]] = None,
    ):
        self.config = LiveDataConfig(config_path)
        self.on_update = on_update
        self._stop = threading.Event()
        self._thread = None
        self.latest_data: Dict[str, Dict[str, Any]] = {}

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()

    def _run(self):
        while not self._stop.is_set():
            if self.config.source == "yahoo":
                self.latest_data = fetch_yahoo(self.config.tickers, self.config.fields)
            # Add more sources here as needed
            if self.on_update:
                self.on_update(self.latest_data)
            time.sleep(self.config.update_rate)

    def get_latest(self) -> Dict[str, Dict[str, Any]]:
        return self.latest_data
