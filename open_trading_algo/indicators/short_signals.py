"""
Short position signal suite for use with SignalOptimizer.
Includes fundamental, technical, and sentiment-based signals.
Each function takes a DataFrame (with required columns) and returns a boolean Series.
"""
import pandas as pd
from open_trading_algo.cache.data_cache import DataCache, is_caching_enabled


def compute_and_cache_short_signals(ticker: str, df: pd.DataFrame, timeframe: str):
    """Compute and cache combined short signals for a ticker.

    Args:
        ticker (str): Stock ticker symbol.
        df (pd.DataFrame): DataFrame with required columns for signal computation.
        timeframe (str): Timeframe identifier for caching.

    Returns:
        pd.DataFrame: DataFrame with combined signal values.
    """
    signal_type = "short_trend"
    if is_caching_enabled():
        cache = DataCache()
        if cache.has_signals(ticker, timeframe, signal_type):
            return cache.get_signals(ticker, timeframe, signal_type)
    combined = (
        signal_overvalued(df)
        | signal_deteriorating_financials(df)
        | signal_negative_earnings_revision(df)
        | signal_negative_momentum(df)
        | signal_support_breakdown(df)
        | signal_overbought_rsi(df)
        | signal_rising_short_interest(df)
        | signal_negative_news(df)
        | signal_bearish_social_sentiment(df)
    )
    signals_df = pd.DataFrame({"signal_value": combined.astype(int)}, index=df.index)
    if is_caching_enabled():
        cache.store_signals(ticker, timeframe, signal_type, signals_df)
    return signals_df


def signal_overvalued(df: pd.DataFrame) -> pd.Series:
    """Generate overvalued signal based on valuation metrics.

    Args:
        df (pd.DataFrame): DataFrame with valuation data columns.

    Returns:
        pd.Series: Boolean series indicating overvalued signals.
    """
    return (df.get("pe_ratio", pd.Series(False, index=df.index)) > 30) | (
        df.get("pb_ratio", pd.Series(False, index=df.index)) > 5
    )


def signal_deteriorating_financials(df: pd.DataFrame) -> pd.Series:
    """Generate signal for deteriorating financial health.

    Args:
        df (pd.DataFrame): DataFrame with financial metrics.

    Returns:
        pd.Series: Boolean series indicating deteriorating financials.
    """
    return (df.get("debt_to_equity", pd.Series(False, index=df.index)) > 2) | (
        df.get("current_ratio", pd.Series(False, index=df.index)) < 1
    )


def signal_negative_earnings_revision(df: pd.DataFrame) -> pd.Series:
    """Generate signal for negative earnings revisions.

    Args:
        df (pd.DataFrame): DataFrame with earnings revision data.

    Returns:
        pd.Series: Boolean series indicating negative earnings revisions.
    """
    return df.get("earnings_revision", pd.Series(False, index=df.index)) < -0.1


def signal_negative_momentum(df: pd.DataFrame) -> pd.Series:
    """Generate signal for negative price momentum.

    Args:
        df (pd.DataFrame): DataFrame with price momentum data.

    Returns:
        pd.Series: Boolean series indicating negative momentum.
    """
    return df.get("momentum_1m", pd.Series(False, index=df.index)) < -0.05


def signal_support_breakdown(df: pd.DataFrame) -> pd.Series:
    """Generate signal for support level breakdown.

    Args:
        df (pd.DataFrame): DataFrame with price and support level data.

    Returns:
        pd.Series: Boolean series indicating support breakdown.
    """
    return df.get("close", pd.Series(False, index=df.index)) < df.get(
        "support_level", pd.Series(float("inf"), index=df.index)
    )


def signal_overbought_rsi(df: pd.DataFrame) -> pd.Series:
    """Generate signal for overbought RSI conditions.

    Args:
        df (pd.DataFrame): DataFrame with RSI data.

    Returns:
        pd.Series: Boolean series indicating overbought RSI.
    """
    return df.get("rsi", pd.Series(False, index=df.index)) > 70


def signal_rising_short_interest(df: pd.DataFrame) -> pd.Series:
    """Generate signal for rising short interest.

    Args:
        df (pd.DataFrame): DataFrame with short interest data.

    Returns:
        pd.Series: Boolean series indicating rising short interest.
    """
    return df.get("short_interest_change", pd.Series(False, index=df.index)) > 0.1


def signal_negative_news(df: pd.DataFrame) -> pd.Series:
    """Generate signal for negative news sentiment.

    Args:
        df (pd.DataFrame): DataFrame with news sentiment data.

    Returns:
        pd.Series: Boolean series indicating negative news.
    """
    return df.get("news_sentiment", pd.Series(False, index=df.index)) < -0.5


def signal_bearish_social_sentiment(df: pd.DataFrame) -> pd.Series:
    """Generate signal for bearish social media sentiment.

    Args:
        df (pd.DataFrame): DataFrame with social sentiment data.

    Returns:
        pd.Series: Boolean series indicating bearish social sentiment.
    """
    return df.get("social_sentiment", pd.Series(False, index=df.index)) < -0.3


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
