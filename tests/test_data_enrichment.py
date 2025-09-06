import pandas as pd
import pytest
import numpy as np
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


def generate_comprehensive_test_data(days: int = 252) -> pd.DataFrame:
    """Generate comprehensive test data with all required columns for signals."""
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    np.random.seed(42)  # For reproducible results

    # Generate OHLCV data
    base_price = 150
    price_changes = np.random.normal(0, 0.02, days)
    prices = base_price * np.cumprod(1 + price_changes)

    # Create OHLC data with some volatility
    high_mult = 1 + np.random.uniform(0, 0.03, days)
    low_mult = 1 - np.random.uniform(0, 0.03, days)
    open_prices = prices * (1 + np.random.normal(0, 0.01, days))
    close_prices = prices

    df = pd.DataFrame(
        {
            "open": open_prices,
            "high": close_prices * high_mult,
            "low": close_prices * low_mult,
            "close": close_prices,
            "volume": np.random.randint(1000000, 10000000, days),
        },
        index=dates,
    )

    # Add fundamental data (static or slowly changing)
    df["pe_ratio"] = np.random.uniform(10, 30, days)
    df["pb_ratio"] = np.random.uniform(1, 5, days)
    df["earnings_growth"] = np.random.uniform(-0.2, 0.3, days)
    df["roe"] = np.random.uniform(0.05, 0.25, days)
    df["gross_margin"] = np.random.uniform(0.3, 0.6, days)
    df["debt_to_equity"] = np.random.uniform(0.1, 2.0, days)
    df["float_shares"] = 5000000000  # Static for simplicity

    # Add earnings revision data
    df["earnings_revision"] = np.random.choice([-1, 0, 1], days, p=[0.3, 0.4, 0.3])

    # Add sentiment data
    df["news_sentiment"] = np.random.uniform(-1, 1, days)
    df["news_volume"] = np.random.randint(0, 100, days)
    df["social_sentiment"] = np.random.uniform(-1, 1, days)
    df["influencer_sentiment"] = np.random.uniform(-1, 1, days)
    df["analyst_consensus"] = np.random.choice(["buy", "hold", "sell"], days)
    df["analyst_upgrades"] = np.random.randint(0, 10, days)
    df["analyst_downgrades"] = np.random.randint(0, 10, days)

    # Add options data
    df["put_call_ratio"] = np.random.uniform(0.5, 1.5, days)
    df["options_oi"] = np.random.randint(100000, 1000000, days)
    df["options_volume"] = np.random.randint(10000, 100000, days)
    df["short_interest"] = np.random.uniform(0.01, 0.1, days)
    df["vix"] = np.random.uniform(10, 30, days)
    df["implied_volatility"] = np.random.uniform(0.1, 0.5, days)
    df["realized_volatility"] = np.random.uniform(0.1, 0.4, days)
    df["order_flow_imbalance"] = np.random.uniform(-0.5, 0.5, days)

    # Add event flags (sparse events)
    df["earnings_event"] = np.random.choice([0, 1], days, p=[0.95, 0.05])
    df["macro_event"] = np.random.choice([0, 1], days, p=[0.98, 0.02])
    df["mna_event"] = np.random.choice([0, 1], days, p=[0.99, 0.01])

    return df


# Generate test data once for all tests
TEST_DF = generate_comprehensive_test_data()

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
    df = TEST_DF.copy()
    df = enrich_dataframe_for_signals(df, TICKER, signals)
    required_cols = set()
    for sig in signals:
        required_cols.update(SIGNAL_REQUIREMENTS.get(sig, []))
    missing = [col for col in required_cols if col not in df.columns]
    assert not missing, f"Missing columns for signals {signals}: {missing}"


def test_all_signals_have_required_data():
    """Test that our generated test data includes all required columns for all signals."""
    all_signals = LONG_SIGNALS + SHORT_SIGNALS + SENTIMENT_SIGNALS + OPTIONS_SIGNALS
    required_cols = set()
    for sig in all_signals:
        required_cols.update(SIGNAL_REQUIREMENTS.get(sig, []))

    missing = [col for col in required_cols if col not in TEST_DF.columns]
    assert not missing, f"Test data missing columns: {missing}"


def test_signal_data_types():
    """Test that generated data has appropriate types and ranges."""
    df = TEST_DF.copy()

    # Price data should be positive
    assert (df[["open", "high", "low", "close"]] > 0).all().all()

    # Volume should be positive
    assert (df["volume"] > 0).all()

    # Ratios should be in reasonable ranges
    assert (df["pe_ratio"] > 0).all()
    assert (df["pb_ratio"] > 0).all()

    # Sentiment scores should be between -1 and 1
    assert (df["news_sentiment"].between(-1, 1)).all()
    assert (df["social_sentiment"].between(-1, 1)).all()

    # Volatility should be positive
    assert (df["implied_volatility"] > 0).all()
    assert (df["realized_volatility"] > 0).all()


def test_signal_data_completeness():
    """Test that all required data is present and non-null."""
    df = TEST_DF.copy()

    # Check that all columns exist and have no NaN values
    required_cols = []
    for signals in [LONG_SIGNALS, SHORT_SIGNALS, SENTIMENT_SIGNALS, OPTIONS_SIGNALS]:
        for sig in signals:
            required_cols.extend(SIGNAL_REQUIREMENTS.get(sig, []))

    for col in set(required_cols):
        assert col in df.columns, f"Missing column: {col}"
        assert not df[col].isna().any(), f"Column {col} has NaN values"
