"""
fetchers.py

Bulk and single-ticker data fetchers for various financial data APIs (Yahoo, Finnhub, FMP, Alpha Vantage, Twelve Data).
Handles rate limiting, caching, and concurrent requests where appropriate.
"""
from pathlib import Path
from typing import Any, Dict, List
from open_trading_algo.fin_data_apis.secure_api import get_api_key
import concurrent.futures
import requests
import yfinance as yf
from open_trading_algo.cache.data_cache import DataCache

# ...existing fetch_yahoo, fetch_finnhub, fetch_fmp, fetch_alpha_vantage, fetch_twelve_data, and their bulk variants...
# (To be filled in next step)

# --- Live data fetchers and bulk fetchers ---
from open_trading_algo.fin_data_apis.rate_limit import rate_limit_check


def fetch_finnhub_bulk(
    tickers: List[str], fields: List[str], api_key: str
) -> Dict[str, Dict[str, Any]]:
    """
    Fetch data for multiple tickers from Finnhub concurrently.
    Args:
        tickers: List of ticker symbols.
        fields: List of fields to fetch.
        api_key: Finnhub API key.
    Returns:
        Dictionary mapping ticker to field-value dict.
    """
    rate_limit_check("finnhub")

    def fetch_one(ticker):
        return ticker, fetch_finnhub([ticker], fields, api_key).get(ticker, {})

    data = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(fetch_one, tickers)
        for ticker, out in results:
            data[ticker] = out
    return data


def fetch_fmp_bulk(tickers: List[str], fields: List[str], api_key: str) -> Dict[str, Dict[str, Any]]:
    """
    Fetch data for multiple tickers from Financial Modeling Prep (FMP).
    Args:
        tickers: List of ticker symbols.
        fields: List of fields to fetch.
        api_key: FMP API key.
    Returns:
        Dictionary mapping ticker to field-value dict.
    """
    rate_limit_check("fmp")
    return fetch_fmp(tickers, fields, api_key)


def fetch_alpha_vantage_bulk(
    tickers: List[str], fields: List[str], api_key: str
) -> Dict[str, Dict[str, Any]]:
    """
    Fetch data for multiple tickers from Alpha Vantage concurrently.
    Args:
        tickers: List of ticker symbols.
        fields: List of fields to fetch.
        api_key: Alpha Vantage API key.
    Returns:
        Dictionary mapping ticker to field-value dict.
    """
    rate_limit_check("alpha_vantage")

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
    """
    Fetch data for multiple tickers from Twelve Data concurrently.
    Args:
        tickers: List of ticker symbols.
        fields: List of fields to fetch.
        api_key: Twelve Data API key.
    Returns:
        Dictionary mapping ticker to field-value dict.
    """
    rate_limit_check("twelve_data")

    def fetch_one(ticker):
        return ticker, fetch_twelve_data([ticker], fields, api_key).get(ticker, {})

    data = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(fetch_one, tickers)
        for ticker, out in results:
            data[ticker] = out
    return data


def fetch_yahoo(
    tickers: List[str], fields: List[str], batch_size: int = 80, cache: DataCache = None
) -> Dict[str, Dict[str, Any]]:
    """
    Fetch data for multiple tickers from Yahoo Finance, using yfinance and optional cache.
    Args:
        tickers: List of ticker symbols.
        fields: List of fields to fetch.
        batch_size: Number of tickers to fetch per batch.
        cache: Optional DataCache instance for local caching.
    Returns:
        Dictionary mapping ticker to field-value dict.
    """
    rate_limit_check("yahoo")
    import math

    data = {}
    uncached = []
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
                if cache is not None:
                    cache.store_price_data(ticker, tdf)
            except Exception:
                data[ticker] = {f: None for f in fields}
    return data


def fetch_finnhub(tickers: List[str], fields: List[str], api_key: str) -> Dict[str, Dict[str, Any]]:
    """
    Fetch quote data for tickers from Finnhub.
    Args:
        tickers: List of ticker symbols.
        fields: List of fields to fetch.
        api_key: Finnhub API key.
    Returns:
        Dictionary mapping ticker to field-value dict.
    """
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
    """
    Fetch quote data for tickers from Financial Modeling Prep (FMP).
    Args:
        tickers: List of ticker symbols.
        fields: List of fields to fetch.
        api_key: FMP API key.
    Returns:
        Dictionary mapping ticker to field-value dict.
    """
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
    """
    Fetch quote data for tickers from Alpha Vantage.
    Args:
        tickers: List of ticker symbols.
        fields: List of fields to fetch.
        api_key: Alpha Vantage API key.
    Returns:
        Dictionary mapping ticker to field-value dict.
    """
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
    """
    Fetch quote data for tickers from Twelve Data.
    Args:
        tickers: List of ticker symbols.
        fields: List of fields to fetch.
        api_key: Twelve Data API key.
    Returns:
        Dictionary mapping ticker to field-value dict.
    """
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
