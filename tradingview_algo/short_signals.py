"""
Short position signal suite for use with SignalOptimizer.
Includes fundamental, technical, and sentiment-based signals.
Each function takes a DataFrame (with required columns) and returns a boolean Series.
"""
import pandas as pd


def signal_overvalued(df: pd.DataFrame) -> pd.Series:
    """High P/E or P/B indicates overvaluation."""
    return (df.get("pe_ratio", pd.Series(False, index=df.index)) > 30) | (
        df.get("pb_ratio", pd.Series(False, index=df.index)) > 5
    )


def signal_deteriorating_financials(df: pd.DataFrame) -> pd.Series:
    """Declining margins or rising debt."""
    margin_decline = df["gross_margin"].diff() < 0
    debt_rising = df["debt_to_equity"].diff() > 0
    return margin_decline | debt_rising


def signal_negative_earnings_revision(df: pd.DataFrame) -> pd.Series:
    """Negative analyst earnings revisions."""
    return df.get("earnings_revision", pd.Series(0, index=df.index)) < 0


def signal_negative_momentum(df: pd.DataFrame) -> pd.Series:
    """Negative price momentum (e.g., 20-day return < 0)."""
    return df["close"].pct_change(20) < 0


def signal_support_breakdown(df: pd.DataFrame) -> pd.Series:
    """Breakdown below 50-day moving average (support)."""
    ma50 = df["close"].rolling(50).mean()
    return df["close"] < ma50


def signal_overbought_rsi(df: pd.DataFrame) -> pd.Series:
    """RSI > 70 signals overbought, possible reversal."""
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi > 70


def signal_rising_short_interest(df: pd.DataFrame) -> pd.Series:
    """Short interest rising over 10 days."""
    return df.get("short_interest", pd.Series(0, index=df.index)).diff(10) > 0


def signal_negative_news(df: pd.DataFrame) -> pd.Series:
    """Negative news flow score (e.g., NLP sentiment < 0)."""
    return df.get("news_sentiment", pd.Series(0, index=df.index)) < 0


def signal_bearish_social_sentiment(df: pd.DataFrame) -> pd.Series:
    """Bearish social media sentiment (e.g., Twitter, StockTwits)."""
    return df.get("social_sentiment", pd.Series(0, index=df.index)) < 0


# Dictionary of all short signals
short_signals = {
    "overvalued": signal_overvalued,
    "deteriorating_financials": signal_deteriorating_financials,
    "negative_earnings_revision": signal_negative_earnings_revision,
    "negative_momentum": signal_negative_momentum,
    "support_breakdown": signal_support_breakdown,
    "overbought_rsi": signal_overbought_rsi,
    "rising_short_interest": signal_rising_short_interest,
    "negative_news": signal_negative_news,
    "bearish_social_sentiment": signal_bearish_social_sentiment,
}
