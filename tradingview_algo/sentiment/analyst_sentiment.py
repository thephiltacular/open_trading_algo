"""
Methods to fetch and analyze analyst sentiment for specific tickers and ETFs.
Best practices: Use multiple sources (Yahoo Finance, Finnhub, FMP, etc.), normalize, and track changes over time.
"""

import requests
from typing import Dict, Any
from tradingview_algo.fin_data_apis.secure_api import get_api_key
from tradingview_algo.fin_data_apis.rate_limit import rate_limit


# Example 1: Fetch analyst recommendations from Finnhub (requires API key)
def fetch_finnhub_analyst_sentiment(ticker: str, api_key: str) -> Dict[str, Any]:
    """
    Fetch analyst recommendations and target price from Finnhub.
    Returns a dict with buy/hold/sell counts and consensus.
    """
    if not api_key:
        api_key = get_api_key("finnhub")

    @rate_limit("finnhub")
    def _call():
        url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={ticker}&token={api_key}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data:
                latest = data[0]
                return {
                    "buy": latest.get("buy"),
                    "hold": latest.get("hold"),
                    "sell": latest.get("sell"),
                    "strongBuy": latest.get("strongBuy"),
                    "strongSell": latest.get("strongSell"),
                    "period": latest.get("period"),
                    "url": url,
                }
        except Exception as e:
            return {"error": str(e), "url": url}
        return {"buy": None, "url": url}

    return _call()


# Example 2: Fetch analyst price targets from Financial Modeling Prep (FMP)
def fetch_fmp_analyst_price_targets(ticker: str, api_key: str) -> Dict[str, Any]:
    """
    Fetch analyst price targets from FMP.
    Returns a dict with target price and details.
    """
    if not api_key:
        api_key = get_api_key("fmp")

    @rate_limit("fmp")
    def _call():
        url = f"https://financialmodelingprep.com/api/v3/price-target/{ticker}?apikey={api_key}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data:
                latest = data[0]
                return {
                    "targetHigh": latest.get("targetHigh"),
                    "targetLow": latest.get("targetLow"),
                    "targetMean": latest.get("targetMean"),
                    "targetMedian": latest.get("targetMedian"),
                    "url": url,
                }
        except Exception as e:
            return {"error": str(e), "url": url}
        return {"targetMean": None, "url": url}

    return _call()


# Example 3: Aggregate analyst sentiment from multiple sources
def aggregate_analyst_sentiment(ticker: str, finnhub_key: str, fmp_key: str) -> Dict[str, Any]:
    """
    Aggregate analyst sentiment from Finnhub and FMP.
    Returns a dict with consensus and price targets.
    """
    finnhub = fetch_finnhub_analyst_sentiment(ticker, finnhub_key)
    fmp = fetch_fmp_analyst_price_targets(ticker, fmp_key)
    return {
        "finnhub": finnhub,
        "fmp": fmp,
    }
