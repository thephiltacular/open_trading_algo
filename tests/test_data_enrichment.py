import pandas as pd
import pytest
from open_trading_algo.data_enrichment import enrich_dataframe_for_signals

# Example ticker for testing (liquid, lots of data)
TICKER = "AAPL"

# Signal groups
LONG_SIGNALS = [
    "undervalued",
    "high_roe",
    "positive_earnings_revision",
    "sma_trend",
    "positive_momentum",
    "rsi_macd",
    "breakout",
    "positive_news",
    "analyst_upgrades",
    "social_sentiment",
]
SHORT_SIGNALS = [
    "overvalued",
    "deteriorating_financials",
    "negative_earnings_revision",
    "negative_momentum",
    "support_breakdown",
    "overbought_rsi",
    "rising_short_interest",
    "negative_news",
    "bearish_social_sentiment",
]
SENTIMENT_SIGNALS = [
    "news_nlp_sentiment",
    "news_event_sentiment",
    "social_media_trend",
    "social_media_influencer_impact",
    "analyst_consensus_change",
    "analyst_rating_change",
    "options_put_call_ratio",
    "options_unusual_activity",
    "short_interest_crowding",
    "volatility_sentiment",
]
OPTIONS_SIGNALS = [
    "iv_vs_rv",
    "volatility_breakout",
    "support_resistance",
    "earnings_event",
    "macro_event",
    "mna_event",
    "unusual_options_activity",
    "order_flow",
    "signal_news_sentiment",
    "signal_social_sentiment",
]


import yfinance as yf

# Fetch yfinance data once for all tests
yf_ticker = yf.Ticker(TICKER)
YF_INFO = yf_ticker.info
YF_HIST = yf_ticker.history(period="max")
YF_HIST_1Y = YF_HIST.tail(252).copy()  # 1 year of trading days

SIGNAL_REQUIREMENTS = {
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


@pytest.mark.parametrize(
    "signals", [LONG_SIGNALS, SHORT_SIGNALS, SENTIMENT_SIGNALS, OPTIONS_SIGNALS]
)
def test_enrich_dataframe_for_signals(signals):
    df = YF_HIST_1Y.copy()
    df = enrich_dataframe_for_signals(df, TICKER, signals, info=YF_INFO, hist=YF_HIST)
    required_cols = set()
    for sig in signals:
        required_cols.update(SIGNAL_REQUIREMENTS.get(sig, []))
    missing = [col for col in required_cols if col not in df.columns]
    assert not missing, f"Missing columns for signals {signals}: {missing}"
