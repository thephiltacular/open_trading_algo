"""
Data enrichment utilities for signal suites.
Gathers and adds required columns to a DataFrame for selected signals using yfinance and sentiment sources.
"""
import pandas as pd

import yfinance as yf
from typing import List, Dict, Optional
from tradingview_algo.fin_data_apis import fetchers


def enrich_dataframe_for_signals(
    df: pd.DataFrame,
    ticker: str,
    signals: List[str],
    hist: pd.DataFrame = None,
    source: str = "yahoo",
    api_key: Optional[str] = None,
    tickers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Adds required columns for the given signals to the DataFrame using the selected API (yahoo, finnhub, fmp, alpha_vantage, twelve_data).
    Only missing columns are fetched/added. Optionally accepts pre-fetched yfinance info and history.
    If tickers is provided, will fetch for all tickers in one bulk call (recommended for efficiency).
    """
    # Map signal names to required columns
    signal_requirements = {
        # Long signals
        "undervalued": ["pe_ratio", "earnings_growth"],
        "high_roe": ["roe"],
        "positive_earnings_revision": ["earnings_revision"],
        "sma_trend": ["close"],
        "positive_momentum": ["close"],
        "rsi_macd": ["close"],
        "breakout": ["close", "high"],
        "positive_news": ["news_sentiment"],
        "analyst_upgrades": ["analyst_upgrades"],
        "social_sentiment": ["social_sentiment"],
        # Short signals
        "overvalued": ["pe_ratio", "pb_ratio"],
        "deteriorating_financials": ["gross_margin", "debt_to_equity"],
        "negative_earnings_revision": ["earnings_revision"],
        "negative_momentum": ["close"],
        "support_breakdown": ["close"],
        "overbought_rsi": ["close"],
        "rising_short_interest": ["short_interest"],
        "negative_news": ["news_sentiment"],
        "bearish_social_sentiment": ["social_sentiment"],
        # Sentiment signals
        "news_nlp_sentiment": ["news_sentiment"],
        "news_event_sentiment": ["news_volume"],
        "social_media_trend": ["social_sentiment"],
        "social_media_influencer_impact": ["influencer_sentiment"],
        "analyst_consensus_change": ["analyst_consensus"],
        "analyst_rating_change": ["analyst_upgrades", "analyst_downgrades"],
        "options_put_call_ratio": ["put_call_ratio"],
        "options_unusual_activity": ["options_oi", "options_volume"],
        "short_interest_crowding": ["short_interest", "float_shares"],
        "volatility_sentiment": ["vix"],
        # Options signals
        "iv_vs_rv": ["implied_volatility", "realized_volatility"],
        "volatility_breakout": ["realized_volatility"],
        "support_resistance": ["close", "low", "high"],
        "earnings_event": ["earnings_event"],
        "macro_event": ["macro_event"],
        "mna_event": ["mna_event"],
        "unusual_options_activity": ["options_oi", "options_volume"],
        "order_flow": ["order_flow_imbalance"],
        "signal_news_sentiment": ["news_sentiment"],
        "signal_social_sentiment": ["social_sentiment"],
    }

    # yfinance info fields mapping
    yfinance_fields = {
        "pe_ratio": "trailingPE",
        "pb_ratio": "priceToBook",
        "earnings_growth": "earningsQuarterlyGrowth",
        "roe": "returnOnEquity",
        "gross_margin": "grossMargins",
        "debt_to_equity": "debtToEquity",
        "close": "Close",
        "high": "High",
        "low": "Low",
        "float_shares": "floatShares",
        "vix": None,  # Not available from yfinance, placeholder
        # Add more as needed
    }
    # Determine tickers to fetch
    tickers = tickers or [ticker]

    # Determine which columns are missing for each ticker
    needed_fields = set()
    for sig in signals:
        needed_fields.update(signal_requirements.get(sig, []))
    missing_fields = [col for col in needed_fields if col not in df.columns]

    price_fields = [
        f
        for f in missing_fields
        if f
        in [
            "price",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "previous_close",
            "change",
            "percent_change",
            "timestamp",
        ]
    ]

    # Fetch from selected API if not yahoo
    api_data = None
    if source != "yahoo" and price_fields:
        fetch_map = {
            "finnhub": fetchers.fetch_finnhub_bulk,
            "fmp": fetchers.fetch_fmp_bulk,
            "alpha_vantage": fetchers.fetch_alpha_vantage_bulk,
            "twelve_data": fetchers.fetch_twelve_data_bulk,
        }
        fetch_func = fetch_map.get(source.lower())
        if fetch_func and api_key:
            api_data = fetch_func(tickers, price_fields, api_key)

    # Fetch yfinance info if needed (fallback)
    yf_info = None
    yf_hist = hist
    for sig in signals:
        for col in signal_requirements.get(sig, []):
            if col not in df.columns:
                # Try API data first
                if api_data and ticker in api_data and col in api_data[ticker]:
                    df[col] = api_data[ticker][col]
                elif col in yfinance_fields and yfinance_fields[col] is not None:
                    if yf_info is None:
                        yf_info = yf.Ticker(ticker).info
                    val = yf_info.get(yfinance_fields[col], None)
                    if val is not None:
                        df[col] = val
                elif col in ["close", "high", "low"]:
                    # Try to get from yfinance history if not present
                    if yf_hist is None:
                        yf_hist = yf.Ticker(ticker).history(period="max")
                    if col.capitalize() in yf_hist.columns:
                        df[col] = yf_hist[col.capitalize()]
                else:
                    # Placeholders for external/sentiment/option data
                    df[col] = 0  # Replace with real data from APIs or calculations
    return df
