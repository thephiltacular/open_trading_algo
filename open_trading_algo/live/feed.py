from open_trading_algo.fin_data_apis.fetchers import (
    fetch_yahoo,
    fetch_finnhub,
    fetch_fmp,
    fetch_alpha_vantage,
    fetch_twelve_data,
    fetch_finnhub_bulk,
    fetch_fmp_bulk,
    fetch_alpha_vantage_bulk,
    fetch_twelve_data_bulk,
)
from open_trading_algo.fin_data_apis.config import LiveDataConfig
from open_trading_algo.cache.data_cache import DataCache
from open_trading_algo.fin_data_apis.secure_api import get_api_key
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import yaml


class LiveDataFeed:
    """Live data feed that periodically fetches ticker data and exposes it for analysis."""

    def __init__(
        self,
        config_path: Path,
        on_update: Optional[Callable[[Dict[str, Dict[str, Any]]], None]] = None,
        cache: Optional[DataCache] = None,
    ):
        self.config = LiveDataConfig(config_path)
        # If API key is not set in config, try to load from .env or environment
        if not self.config.api_key and self.config.source.lower() in {
            "finnhub",
            "fmp",
            "alpha_vantage",
            "twelve_data",
        }:
            self.config.api_key = get_api_key(self.config.source)
        self.on_update = on_update
        self._stop = threading.Event()
        self._thread = None
        self.latest_data: Dict[str, Dict[str, Any]] = {}
        self.cache = cache or DataCache()

    def start(self):
        """Start the live data feed thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the live data feed thread."""
        self._stop.set()
        if self._thread:
            self._thread.join()

    def _run(self):
        """Run the main data fetching loop."""
        while not self._stop.is_set():
            src = self.config.source.lower()
            if src == "yahoo":
                self.latest_data = fetch_yahoo(
                    self.config.tickers, self.config.fields, cache=self.cache
                )
            elif src == "finnhub":
                self.latest_data = fetch_finnhub_bulk(
                    self.config.tickers, self.config.fields, self.config.api_key
                )
            elif src == "fmp":
                self.latest_data = fetch_fmp_bulk(
                    self.config.tickers, self.config.fields, self.config.api_key
                )
            elif src == "alpha_vantage":
                self.latest_data = fetch_alpha_vantage_bulk(
                    self.config.tickers, self.config.fields, self.config.api_key
                )
            elif src == "twelve_data":
                self.latest_data = fetch_twelve_data_bulk(
                    self.config.tickers, self.config.fields, self.config.api_key
                )
            else:
                self.latest_data = {
                    t: {f: None for f in self.config.fields} for t in self.config.tickers
                }
            if self.on_update:
                self.on_update(self.latest_data)
            time.sleep(self.config.update_rate)

    def get_latest(self) -> Dict[str, Dict[str, Any]]:
        """Get the latest fetched data.

        Returns:
            Dict[str, Dict[str, Any]]: Latest data for all tickers.
        """
        return self.latest_data
