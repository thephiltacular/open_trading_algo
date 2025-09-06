"""
Methods to fetch and analyze analyst sentiment for specific tickers and ETFs.
Best practices: Use multiple sources (Yahoo Finance, Finnhub, FMP, etc.), normalize, and track changes over time.
"""

import requests
from typing import Dict, Any
import pandas as pd
from open_trading_algo.cache.data_cache import DataCache, is_caching_enabled
from open_trading_algo.fin_data_apis.secure_api import get_api_key
from open_trading_algo.fin_data_apis.rate_limit import rate_limit


def fetch_bulk_finnhub_analyst_sentiment(tickers: list, api_key: str) -> pd.DataFrame:
    """Fetch Finnhub analyst sentiment for multiple tickers.

    Args:
        tickers (list): List of ticker symbols to fetch data for.
        api_key (str): Finnhub API key for authentication.

    Returns:
        pd.DataFrame: DataFrame indexed by [date, ticker] with analyst sentiment data.
    """
    cache = DataCache() if is_caching_enabled() else None
    signal_type = "analyst_finnhub_sentiment"
    timeframe = "1d"
    dfs = []
    for ticker in tickers:
        if cache and cache.has_signals(ticker, timeframe, signal_type):
            df = cache.get_signals(ticker, timeframe, signal_type).copy()
        else:
            df = fetch_finnhub_analyst_sentiment(ticker, api_key).copy()
        df["ticker"] = ticker
        dfs.append(df)
    if dfs:
        result = pd.concat(dfs)
        result = result.set_index([result.index, "ticker"])
        result.index.names = ["date", "ticker"]
        return result
    return pd.DataFrame(columns=["buy"]).set_index(
        [pd.Index([], name="date"), pd.Index([], name="ticker")]
    )


def fetch_bulk_fmp_analyst_price_targets(tickers: list, api_key: str) -> pd.DataFrame:
    """Fetch FMP analyst price targets for multiple tickers.

    Args:
        tickers (list): List of ticker symbols to fetch data for.
        api_key (str): FMP API key for authentication.

    Returns:
        pd.DataFrame: DataFrame indexed by [date, ticker] with price target data.
    """
    cache = DataCache() if is_caching_enabled() else None
    signal_type = "analyst_fmp_price_targets"
    timeframe = "1d"
    dfs = []
    for ticker in tickers:
        if cache and cache.has_signals(ticker, timeframe, signal_type):
            df = cache.get_signals(ticker, timeframe, signal_type).copy()
        else:
            df = fetch_fmp_analyst_price_targets(ticker, api_key).copy()
        df["ticker"] = ticker
        dfs.append(df)
    if dfs:
        result = pd.concat(dfs)
        result = result.set_index([result.index, "ticker"])
        result.index.names = ["date", "ticker"]
        return result
    return pd.DataFrame(columns=["targetMean"]).set_index(
        [pd.Index([], name="date"), pd.Index([], name="ticker")]
    )


# Example 1: Fetch analyst recommendations from Finnhub (requires API key)
def fetch_finnhub_analyst_sentiment(ticker: str, api_key: str) -> Dict[str, Any]:
    """Fetch analyst recommendations and target price from Finnhub.

    Args:
        ticker (str): Ticker symbol to fetch data for.
        api_key (str): Finnhub API key for authentication.

    Returns:
        Dict[str, Any]: Dictionary with buy/hold/sell counts and consensus.
    """
    if not api_key:
        api_key = get_api_key("finnhub")
    cache = DataCache() if is_caching_enabled() else None
    signal_type = "analyst_finnhub_sentiment"
    timeframe = "1d"
    if cache and cache.has_signals(ticker, timeframe, signal_type):
        df = cache.get_signals(ticker, timeframe, signal_type)
        return df

    @rate_limit("finnhub")
    def _call():
        url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={ticker}&token={api_key}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data:
                latest = data[0]
                dt = pd.Timestamp.utcnow().normalize()
                df = pd.DataFrame(
                    {
                        "buy": [latest.get("buy")],
                        "hold": [latest.get("hold")],
                        "sell": [latest.get("sell")],
                        "strongBuy": [latest.get("strongBuy")],
                        "strongSell": [latest.get("strongSell")],
                        "period": [latest.get("period")],
                    },
                    index=[dt],
                )
                if cache:
                    cache.store_signals(
                        ticker,
                        timeframe,
                        signal_type,
                        df[["buy"]].rename(columns={"buy": "signal_value"}),
                    )
                return df
        except Exception as e:
            return pd.DataFrame({"buy": [None]}, index=[pd.Timestamp.utcnow().normalize()])
        return pd.DataFrame({"buy": [None]}, index=[pd.Timestamp.utcnow().normalize()])

    return _call()


# Example 2: Fetch analyst price targets from Financial Modeling Prep (FMP)
def fetch_fmp_analyst_price_targets(ticker: str, api_key: str) -> Dict[str, Any]:
    """Fetch analyst price targets from FMP.

    Args:
        ticker (str): Ticker symbol to fetch data for.
        api_key (str): FMP API key for authentication.

    Returns:
        Dict[str, Any]: Dictionary with target price and details.
    """
    if not api_key:
        api_key = get_api_key("fmp")
    cache = DataCache() if is_caching_enabled() else None
    signal_type = "analyst_fmp_price_targets"
    timeframe = "1d"
    if cache and cache.has_signals(ticker, timeframe, signal_type):
        df = cache.get_signals(ticker, timeframe, signal_type)
        return df

    @rate_limit("fmp")
    def _call():
        url = f"https://financialmodelingprep.com/api/v3/price-target/{ticker}?apikey={api_key}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data:
                latest = data[0]
                dt = pd.Timestamp.utcnow().normalize()
                df = pd.DataFrame(
                    {
                        "targetHigh": [latest.get("targetHigh")],
                        "targetLow": [latest.get("targetLow")],
                        "targetMean": [latest.get("targetMean")],
                        "targetMedian": [latest.get("targetMedian")],
                    },
                    index=[dt],
                )
                if cache:
                    cache.store_signals(
                        ticker,
                        timeframe,
                        signal_type,
                        df[["targetMean"]].rename(columns={"targetMean": "signal_value"}),
                    )
                return df
        except Exception as e:
            return pd.DataFrame({"targetMean": [None]}, index=[pd.Timestamp.utcnow().normalize()])
        return pd.DataFrame({"targetMean": [None]}, index=[pd.Timestamp.utcnow().normalize()])

    return _call()


# Example 3: Aggregate analyst sentiment from multiple sources
def aggregate_analyst_sentiment(ticker: str, finnhub_key: str, fmp_key: str) -> Dict[str, Any]:
    """Aggregate analyst sentiment from Finnhub and FMP.

    Args:
        ticker (str): Ticker symbol to fetch data for.
        finnhub_key (str): Finnhub API key for authentication.
        fmp_key (str): FMP API key for authentication.

    Returns:
        Dict[str, Any]: Dictionary with consensus and price targets.
    """
    finnhub_df = fetch_finnhub_analyst_sentiment(ticker, finnhub_key)
    fmp_df = fetch_fmp_analyst_price_targets(ticker, fmp_key)
    # Combine as new columns for indicator DataFrame usage
    df = finnhub_df.join(fmp_df, how="outer")
    return df
