"""
Methods to fetch and analyze social sentiment for specific tickers and ETFs.
Best practices: Use multiple sources, aggregate, and normalize scores.
"""
import requests
from typing import List, Dict, Any
import pandas as pd
from open_trading_algo.cache.data_cache import DataCache, is_caching_enabled
from open_trading_algo.fin_data_apis.secure_api import get_api_key
from open_trading_algo.fin_data_apis.rate_limit import rate_limit


def fetch_bulk_twitter_sentiment(tickers: list, api_key: str = None) -> pd.DataFrame:
    """Fetch Twitter/X sentiment for multiple tickers using LunarCrush bulk API.

    Args:
        tickers (list): List of ticker symbols to fetch data for.
        api_key (str, optional): LunarCrush API key for authentication.

    Returns:
        pd.DataFrame: DataFrame indexed by [date, ticker] with sentiment data.
    """
    if not api_key:
        api_key = get_api_key("lunarcrush")
    cache = DataCache() if is_caching_enabled() else None
    signal_type = "social_twitter_sentiment"
    timeframe = "1d"
    # Check cache for each ticker
    cached = {}
    to_fetch = []
    for ticker in tickers:
        if cache and cache.has_signals(ticker, timeframe, signal_type):
            cached[ticker] = cache.get_signals(ticker, timeframe, signal_type)
        else:
            to_fetch.append(ticker)
    dfs = []
    # Add cached data
    for ticker, df in cached.items():
        df = df.copy()
        df["ticker"] = ticker
        dfs.append(df)
    # Fetch remaining tickers in one call
    if to_fetch:

        @rate_limit("lunarcrush")
        def _call():
            url = f"https://api.lunarcrush.com/v2?data=assets&key={api_key or 'demo'}&symbol={','.join(to_fetch)}"
            try:
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                if data.get("data"):
                    dt = pd.Timestamp.utcnow().normalize()
                    for asset in data["data"]:
                        ticker = asset.get("symbol")
                        score = asset.get("galaxy_score")
                        df = pd.DataFrame({"signal_value": [score], "ticker": [ticker]}, index=[dt])
                        if cache:
                            cache.store_signals(ticker, timeframe, signal_type, df[["signal_value"]])
                        dfs.append(df)
            except Exception:
                for ticker in to_fetch:
                    df = pd.DataFrame(
                        {"signal_value": [None], "ticker": [ticker]},
                        index=[pd.Timestamp.utcnow().normalize()],
                    )
                    dfs.append(df)

        _call()
    if dfs:
        result = pd.concat(dfs)
        result = result.set_index([result.index, "ticker"])
        result.index.names = ["date", "ticker"]
        return result
    # If nothing, return empty DataFrame
    return pd.DataFrame(columns=["signal_value"]).set_index(
        [pd.Index([], name="date"), pd.Index([], name="ticker")]
    )


# Example 1: Fetch Twitter/X sentiment using a third-party API (e.g., Twitter API, or a service like StockTwits)
def fetch_twitter_sentiment(ticker: str, api_key: str = None) -> Dict[str, Any]:
    """Fetch sentiment for a ticker from Twitter/X using a third-party API.

    Args:
        ticker (str): Ticker symbol to fetch data for.
        api_key (str, optional): API key for authentication.

    Returns:
        Dict[str, Any]: Dictionary with sentiment score and sample data.
    """
    # Placeholder: Replace with real API call or use a service like StockTwits, LunarCrush, or Twitter API
    # Example: Use LunarCrush public API (no key required for basic usage)
    if not api_key:
        api_key = get_api_key("lunarcrush")
    cache = DataCache() if is_caching_enabled() else None
    signal_type = "social_twitter_sentiment"
    timeframe = "1d"  # or configurable
    if cache and cache.has_signals(ticker, timeframe, signal_type):
        df = cache.get_signals(ticker, timeframe, signal_type)
        return df

    @rate_limit("lunarcrush")
    def _call():
        url = f"https://api.lunarcrush.com/v2?data=assets&key={api_key or 'demo'}&symbol={ticker}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("data"):
                asset = data["data"][0]
                # Use current date as index for caching
                dt = pd.Timestamp.utcnow().normalize()
                df = pd.DataFrame({"signal_value": [asset.get("galaxy_score")]}, index=[dt])
                if cache:
                    cache.store_signals(ticker, timeframe, signal_type, df)
                return df
        except Exception as e:
            return pd.DataFrame({"signal_value": [None]}, index=[pd.Timestamp.utcnow().normalize()])
        return pd.DataFrame({"signal_value": [None]}, index=[pd.Timestamp.utcnow().normalize()])

    return _call()


# Example 2: Fetch Reddit sentiment using a public API or Pushshift
# (Pushshift is often used for Reddit data, but may require a proxy or paid API)
def fetch_reddit_sentiment(ticker: str) -> Dict[str, Any]:
    """Fetch Reddit sentiment for a ticker using a public API.

    Args:
        ticker (str): Ticker symbol to fetch data for.

    Returns:
        Dict[str, Any]: Dictionary with mention count and sentiment score.
    """
    cache = DataCache() if is_caching_enabled() else None
    signal_type = "social_reddit_sentiment"
    timeframe = "1d"
    if cache and cache.has_signals(ticker, timeframe, signal_type):
        df = cache.get_signals(ticker, timeframe, signal_type)
        return df

    @rate_limit("pushshift")
    def _call():
        url = f"https://api.pushshift.io/reddit/search/comment/?q={ticker}&size=100"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            comments = data.get("data", [])
            pos_words = ["moon", "bull", "buy", "rocket", "win"]
            neg_words = ["bag", "bear", "sell", "crash", "loss"]
            pos = sum(any(w in c.get("body", "").lower() for w in pos_words) for c in comments)
            neg = sum(any(w in c.get("body", "").lower() for w in neg_words) for c in comments)
            score = pos - neg
            dt = pd.Timestamp.utcnow().normalize()
            df = pd.DataFrame({"signal_value": [score]}, index=[dt])
            if cache:
                cache.store_signals(ticker, timeframe, signal_type, df)
            return df
        except Exception as e:
            return pd.DataFrame({"signal_value": [None]}, index=[pd.Timestamp.utcnow().normalize()])
        return pd.DataFrame({"signal_value": [None]}, index=[pd.Timestamp.utcnow().normalize()])

    return _call()


# Example 3: Aggregate social sentiment from multiple sources
def aggregate_social_sentiment(ticker: str, api_key: str = None) -> Dict[str, Any]:
    """Aggregate social sentiment from Twitter/X and Reddit.

    Args:
        ticker (str): Ticker symbol to fetch data for.
        api_key (str, optional): API key for authentication.

    Returns:
        Dict[str, Any]: Dictionary with combined score and details.
    """
    twitter_df = fetch_twitter_sentiment(ticker, api_key)
    reddit_df = fetch_reddit_sentiment(ticker)
    # Combine as new columns for indicator DataFrame usage
    df = twitter_df.rename(columns={"signal_value": "twitter_score"}).join(
        reddit_df.rename(columns={"signal_value": "reddit_score"}), how="outer"
    )
    df["combined_score"] = df.mean(axis=1, skipna=True)
    return df
