"""
Methods to fetch and analyze social sentiment for specific tickers and ETFs.
Best practices: Use multiple sources, aggregate, and normalize scores.
"""

import requests
from typing import List, Dict, Any
from tradingview_algo.fin_data_apis.secure_api import get_api_key
from tradingview_algo.fin_data_apis.rate_limit import rate_limit

# Example 1: Fetch Twitter/X sentiment using a third-party API (e.g., Twitter API, or a service like StockTwits)
def fetch_twitter_sentiment(ticker: str, api_key: str = None) -> Dict[str, Any]:
    """
    Fetch sentiment for a ticker from Twitter/X using a third-party API.
    Returns a dict with sentiment score and sample data.
    """
    # Placeholder: Replace with real API call or use a service like StockTwits, LunarCrush, or Twitter API
    # Example: Use LunarCrush public API (no key required for basic usage)
    if not api_key:
        api_key = get_api_key("lunarcrush")

    @rate_limit("lunarcrush")
    def _call():
        url = f"https://api.lunarcrush.com/v2?data=assets&key={api_key or 'demo'}&symbol={ticker}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("data"):
                asset = data["data"][0]
                return {
                    "score": asset.get("galaxy_score"),
                    "alt_rank": asset.get("alt_rank"),
                    "tweet_volume": asset.get("tweet_volume"),
                    "url": url,
                }
        except Exception as e:
            return {"error": str(e), "url": url}
        return {"score": None, "url": url}

    return _call()


# Example 2: Fetch Reddit sentiment using a public API or Pushshift
# (Pushshift is often used for Reddit data, but may require a proxy or paid API)
def fetch_reddit_sentiment(ticker: str) -> Dict[str, Any]:
    """
    Fetch Reddit sentiment for a ticker using a public API (Pushshift or similar).
    Returns a dict with mention count and placeholder sentiment.
    """

    @rate_limit("pushshift")
    def _call():
        url = f"https://api.pushshift.io/reddit/search/comment/?q={ticker}&size=100"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            comments = data.get("data", [])
            mention_count = len(comments)
            # Placeholder: Use simple positive/negative word count for sentiment
            pos_words = ["moon", "bull", "buy", "rocket", "win"]
            neg_words = ["bag", "bear", "sell", "crash", "loss"]
            pos = sum(any(w in c.get("body", "").lower() for w in pos_words) for c in comments)
            neg = sum(any(w in c.get("body", "").lower() for w in neg_words) for c in comments)
            score = pos - neg
            return {"mention_count": mention_count, "score": score, "url": url}
        except Exception as e:
            return {"error": str(e), "url": url}

    return _call()


# Example 3: Aggregate social sentiment from multiple sources
def aggregate_social_sentiment(ticker: str, api_key: str = None) -> Dict[str, Any]:
    """
    Aggregate social sentiment from Twitter/X and Reddit.
    Returns a dict with combined score and details.
    """
    twitter = fetch_twitter_sentiment(ticker, api_key)
    reddit = fetch_reddit_sentiment(ticker)
    # Normalize and combine scores (simple average for demo)
    scores = [s for s in [twitter.get("score"), reddit.get("score")] if s is not None]
    combined = sum(scores) / len(scores) if scores else None
    return {
        "combined_score": combined,
        "twitter": twitter,
        "reddit": reddit,
    }
