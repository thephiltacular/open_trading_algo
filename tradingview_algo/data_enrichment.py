"""
Data enrichment utilities for signal suites.
Gathers and adds required columns to a DataFrame for selected signals using yfinance and sentiment sources.
"""
import pandas as pd
import yfinance as yf
from typing import List, Dict


def enrich_dataframe_for_signals(
    df: pd.DataFrame, ticker: str, signals: List[str], info: dict = None, hist: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Adds required columns for the given signals to the DataFrame using yfinance and sentiment APIs.
    Only missing columns are fetched/added. Optionally accepts pre-fetched yfinance info and history.
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
    # Fetch yfinance info if needed
    yf_info = info
    yf_hist = hist
    for sig in signals:
        for col in signal_requirements.get(sig, []):
            if col not in df.columns:
                if col in yfinance_fields and yfinance_fields[col] is not None:
                    if yf_info is None:
                        yf_info = yf.Ticker(ticker).info
                    val = yf_info.get(yfinance_fields[col], None)
                    if val is not None:
                        df[col] = val
                elif col == "close" or col == "high" or col == "low":
                    # Try to get from yfinance history if not present
                    if yf_hist is None:
                        yf_hist = yf.Ticker(ticker).history(period="max")
                    if col.capitalize() in yf_hist.columns:
                        df[col] = yf_hist[col.capitalize()]
                else:
                    # Placeholders for external/sentiment/option data
                    df[col] = 0  # Replace with real data from APIs or calculations
    return df
