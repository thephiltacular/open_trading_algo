"""Live data interface for TradingViewAlgoDev.

This module loads configuration from a YAML file and periodically fetches
financial data for a list of tickers/ETFs from a free source (default: Yahoo Finance).
It exposes a clean interface for downstream analysis and alert generation.
"""
from tradingview_algo.fin_data_apis.fetchers import (
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
from tradingview_algo.fin_data_apis.config import LiveDataConfig
from tradingview_algo.cache.data_cache import DataCache
from tradingview_algo.fin_data_apis.secure_api import get_api_key
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import yaml


import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


import yaml

from tradingview_algo.fin_data_apis.secure_api import get_api_key

try:
    import yfinance as yf
except ImportError:
    raise ImportError("Please install yfinance: pip install yfinance")
# --- Bulk fetchers for each provider ---
import concurrent.futures


def fetch_finnhub_bulk(
    tickers: List[str], fields: List[str], api_key: str
) -> Dict[str, Dict[str, Any]]:
    """Fetch latest data for tickers from Finnhub using parallel requests (no true bulk endpoint)."""

    def fetch_one(ticker):
        return ticker, fetch_finnhub([ticker], fields, api_key).get(ticker, {})

    data = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(fetch_one, tickers)
        for ticker, out in results:
            data[ticker] = out
    return data


def fetch_fmp_bulk(tickers: List[str], fields: List[str], api_key: str) -> Dict[str, Dict[str, Any]]:
    """Fetch latest data for tickers from FMP using a single bulk request."""
    return fetch_fmp(tickers, fields, api_key)


def fetch_alpha_vantage_bulk(
    tickers: List[str], fields: List[str], api_key: str
) -> Dict[str, Dict[str, Any]]:
    """Fetch latest data for tickers from Alpha Vantage using parallel requests (no true bulk endpoint)."""

    def fetch_one(ticker):
        return ticker, fetch_alpha_vantage([ticker], fields, api_key).get(ticker, {})

    data = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(fetch_one, tickers)
        for ticker, out in results:
            data[ticker] = out
    return data


def fetch_twelve_data_bulk(
    tickers: List[str], fields: List[str], api_key: str
) -> Dict[str, Dict[str, Any]]:
    """Fetch latest data for tickers from Twelve Data using parallel requests (no true bulk endpoint)."""

    def fetch_one(ticker):
        return ticker, fetch_twelve_data([ticker], fields, api_key).get(ticker, {})

    data = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(fetch_one, tickers)
        for ticker, out in results:
            data[ticker] = out
    return data


class LiveDataConfig:
    def __init__(self, config_path: Path):
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.update_rate = int(cfg.get("update_rate", 300))
        self.tickers = list(cfg.get("tickers", []))
        self.source = str(cfg.get("source", "yahoo"))
        self.api_key = str(cfg.get("api_key", ""))
        self.fields = list(cfg.get("fields", ["price", "volume"]))


def fetch_yahoo(
    tickers: List[str], fields: List[str], batch_size: int = 80, cache: DataCache = None
) -> Dict[str, Dict[str, Any]]:
    """Fetch latest data for tickers from Yahoo Finance, using cache to minimize requests."""
    import math

    data = {}
    uncached = []
    # Check cache first
    if cache is not None:
        for ticker in tickers:
            df = cache.get_price_data(ticker)
            if not df.empty:
                latest = df.iloc[-1]
                out = {}
                for field in fields:
                    if field == "price":
                        out[field] = float(latest["close"])
                    elif field == "open":
                        out[field] = float(latest["open"])
                    elif field == "high":
                        out[field] = float(latest["high"])
                    elif field == "low":
                        out[field] = float(latest["low"])
                    elif field == "close":
                        out[field] = float(latest["close"])
                    elif field == "volume":
                        out[field] = float(latest["volume"])
                    elif field == "previous_close":
                        out[field] = float(df.iloc[-2]["close"]) if len(df) > 1 else None
                    elif field == "change":
                        prev = float(df.iloc[-2]["close"]) if len(df) > 1 else None
                        out[field] = float(latest["close"] - prev) if prev is not None else None
                    elif field == "percent_change":
                        prev = float(df.iloc[-2]["close"]) if len(df) > 1 else None
                        out[field] = float((latest["close"] - prev) / prev * 100) if prev else None
                    elif field == "timestamp":
                        out[field] = str(latest.name)
                data[ticker] = out
            else:
                uncached.append(ticker)
    else:
        uncached = tickers
    # Fetch uncached from yfinance in batches
    n = len(uncached)
    for i in range(0, n, batch_size):
        batch = uncached[i : i + batch_size]
        tickers_str = " ".join(batch)
        try:
            df = yf.download(
                tickers_str,
                period="1d",
                interval="1m",
                group_by="ticker",
                progress=False,
                threads=True,
            )
        except Exception:
            for ticker in batch:
                data[ticker] = {f: None for f in fields}
            continue
        for ticker in batch:
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
                # Store in cache
                if cache is not None:
                    cache.store_price_data(ticker, tdf)
            except Exception:
                data[ticker] = {f: None for f in fields}
    return data


# --- Additional live data fetchers ---
import requests


def fetch_finnhub(tickers: List[str], fields: List[str], api_key: str) -> Dict[str, Dict[str, Any]]:
    """Fetch latest data for tickers from Finnhub."""
    url = "https://finnhub.io/api/v1/quote"
    data = {}
    for ticker in tickers:
        params = {"symbol": ticker, "token": api_key}
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            q = resp.json()
            out = {}
            for field in fields:
                if field == "price":
                    out[field] = q.get("c")
                elif field == "open":
                    out[field] = q.get("o")
                elif field == "high":
                    out[field] = q.get("h")
                elif field == "low":
                    out[field] = q.get("l")
                elif field == "previous_close":
                    out[field] = q.get("pc")
                elif field == "timestamp":
                    out[field] = q.get("t")
            data[ticker] = out
        except Exception:
            data[ticker] = {f: None for f in fields}
    return data


def fetch_fmp(tickers: List[str], fields: List[str], api_key: str) -> Dict[str, Dict[str, Any]]:
    """Fetch latest data for tickers from Financial Modeling Prep (FMP)."""
    url = "https://financialmodelingprep.com/api/v3/quote/{}"
    data = {}
    try:
        tickers_str = ",".join(tickers)
        resp = requests.get(url.format(tickers_str), params={"apikey": api_key}, timeout=10)
        resp.raise_for_status()
        quotes = resp.json()
        for q in quotes:
            ticker = q.get("symbol")
            out = {}
            for field in fields:
                if field == "price":
                    out[field] = q.get("price")
                elif field == "open":
                    out[field] = q.get("open")
                elif field == "high":
                    out[field] = q.get("dayHigh")
                elif field == "low":
                    out[field] = q.get("dayLow")
                elif field == "previous_close":
                    out[field] = q.get("previousClose")
                elif field == "volume":
                    out[field] = q.get("volume")
                elif field == "timestamp":
                    out[field] = q.get("timestamp")
            data[ticker] = out
        # Fill missing tickers
        for ticker in tickers:
            if ticker not in data:
                data[ticker] = {f: None for f in fields}
    except Exception:
        for ticker in tickers:
            data[ticker] = {f: None for f in fields}
    return data


def fetch_alpha_vantage(
    tickers: List[str], fields: List[str], api_key: str
) -> Dict[str, Dict[str, Any]]:
    """Fetch latest data for tickers from Alpha Vantage (batch quotes)."""
    url = "https://www.alphavantage.co/query"
    data = {}
    for ticker in tickers:
        params = {"function": "GLOBAL_QUOTE", "symbol": ticker, "apikey": api_key}
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            q = resp.json().get("Global Quote", {})
            out = {}
            for field in fields:
                if field == "price":
                    out[field] = float(q.get("05. price", 0))
                elif field == "open":
                    out[field] = float(q.get("02. open", 0))
                elif field == "high":
                    out[field] = float(q.get("03. high", 0))
                elif field == "low":
                    out[field] = float(q.get("04. low", 0))
                elif field == "previous_close":
                    out[field] = float(q.get("08. previous close", 0))
                elif field == "volume":
                    out[field] = float(q.get("06. volume", 0))
                elif field == "timestamp":
                    out[field] = q.get("07. latest trading day")
            data[ticker] = out
        except Exception:
            data[ticker] = {f: None for f in fields}
    return data


def fetch_twelve_data(
    tickers: List[str], fields: List[str], api_key: str
) -> Dict[str, Dict[str, Any]]:
    """Fetch latest data for tickers from Twelve Data."""
    url = "https://api.twelvedata.com/quote"
    data = {}
    for ticker in tickers:
        params = {"symbol": ticker, "apikey": api_key}
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            q = resp.json()
            out = {}
            for field in fields:
                if field == "price":
                    out[field] = float(q.get("close", 0))
                elif field == "open":
                    out[field] = float(q.get("open", 0))
                elif field == "high":
                    out[field] = float(q.get("high", 0))
                elif field == "low":
                    out[field] = float(q.get("low", 0))
                elif field == "previous_close":
                    out[field] = float(q.get("previous_close", 0))
                elif field == "volume":
                    out[field] = float(q.get("volume", 0))
                elif field == "timestamp":
                    out[field] = q.get("datetime")
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
        return self.latest_data
