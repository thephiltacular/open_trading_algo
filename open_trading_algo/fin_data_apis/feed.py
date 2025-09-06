"""
feed.py

This module provides a LiveDataFeed class for periodically fetching and caching live ticker data
from various financial data APIs. It supports multiple sources like Yahoo, Finnhub, FMP, Alpha Vantage,
and Twelve Data, with configurable update rates and callback functions for data updates.
"""

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
        """Initialize the LiveDataFeed with configuration and optional callback.

        Args:
            config_path (Path): Path to the YAML configuration file.
            on_update (Optional[Callable[[Dict[str, Dict[str, Any]]], None]]):
                Callback function called when new data is fetched. Defaults to None.
            cache (Optional[DataCache]): Data cache instance. Defaults to None.
        """
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
        """Start the background thread to run the data feed.

        If the thread is already alive, this method does nothing. Otherwise, it clears
        the stop event, creates a new daemon thread targeting the internal _run method,
        and starts the thread.
        """
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background data feed thread.

        Sets the stop event to signal the thread to terminate and waits for the thread to join.
        """
        self._stop.set()
        if self._thread:
            self._thread.join()

    def _run(self):
        """Run the data fetching loop in the background thread.

        Continuously fetches data from the configured source at the specified update rate,
        updates the latest_data dictionary, and calls the on_update callback if provided.
        Stops when the stop event is set.
        """
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
        """Return the most recent data fetched by the feed.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary mapping ticker symbols to their latest field data.
        """
        return self.latest_data
