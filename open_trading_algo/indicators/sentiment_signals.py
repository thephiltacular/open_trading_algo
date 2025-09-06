"""
Sentiment signal suite for all trade types (long, short, options).
Best practices from investment banking and hedge funds.
Each function takes a DataFrame (with required columns) and returns a boolean or numeric Series.
"""
import pandas as pd
from open_trading_algo.cache.data_cache import DataCache, is_caching_enabled


def compute_and_cache_sentiment_signals(ticker: str, df: pd.DataFrame, timeframe: str):
    """Compute and cache combined sentiment signals for a ticker.

    Args:
        ticker (str): Stock ticker symbol.
        df (pd.DataFrame): DataFrame with required columns for signal computation.
        timeframe (str): Timeframe identifier for caching.

    Returns:
        pd.DataFrame: DataFrame with combined signal values.
    """
    signal_type = "sentiment_trend"
    if is_caching_enabled():
        cache = DataCache()
        if cache.has_signals(ticker, timeframe, signal_type):
            return cache.get_signals(ticker, timeframe, signal_type)
    combined = news_nlp_sentiment(df).abs() > 0.2 | news_event_sentiment(df) | (
        social_media_trend(df) > 0
    ) | social_media_influencer_impact(df) | (analyst_consensus_change(df) > 0) | (
        analyst_rating_change(df) > 0
    ) | (
        options_put_call_ratio(df) > 1.2
    ) | options_unusual_activity(
        df
    ) | (
        short_interest_crowding(df) > 0.1
    ) | (
        volatility_sentiment(df) > 20
    )
    signals_df = pd.DataFrame({"signal_value": combined.astype(int)}, index=df.index)
    if is_caching_enabled():
        cache.store_signals(ticker, timeframe, signal_type, signals_df)
    return signals_df


def news_nlp_sentiment(df: pd.DataFrame, threshold: float = 0.2) -> pd.Series:
    """Generate news NLP sentiment signal.

    Args:
        df (pd.DataFrame): DataFrame with news sentiment data.
        threshold (float, optional): Sentiment threshold. Defaults to 0.2.

    Returns:
        pd.Series: Series with news sentiment values.
    """
    sentiment = df.get("news_sentiment", pd.Series(0, index=df.index))
    return sentiment


def news_event_sentiment(df: pd.DataFrame) -> pd.Series:
    """Generate news event sentiment signal based on volume spikes.

    Args:
        df (pd.DataFrame): DataFrame with news volume data.

    Returns:
        pd.Series: Boolean series indicating news event signals.
    """
    news_volume = df.get("news_volume", pd.Series(0, index=df.index))
    return news_volume > news_volume.rolling(20).mean() * 2


def social_media_trend(df: pd.DataFrame) -> pd.Series:
    """Generate social media trend signal based on sentiment changes.

    Args:
        df (pd.DataFrame): DataFrame with social sentiment data.

    Returns:
        pd.Series: Series with social media trend values.
    """
    sentiment = df.get("social_sentiment", pd.Series(0, index=df.index))
    return sentiment.diff()


def social_media_influencer_impact(df: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
    """Generate social media influencer impact signal.

    Args:
        df (pd.DataFrame): DataFrame with influencer sentiment data.
        threshold (float, optional): Impact threshold. Defaults to 0.5.

    Returns:
        pd.Series: Boolean series indicating influencer impact signals.
    """
    influencer_score = df.get("influencer_sentiment", pd.Series(0, index=df.index))
    return influencer_score.abs() > threshold


def analyst_consensus_change(df: pd.DataFrame) -> pd.Series:
    """Generate analyst consensus change signal.

    Args:
        df (pd.DataFrame): DataFrame with analyst consensus data.

    Returns:
        pd.Series: Series with analyst consensus changes.
    """
    consensus = df.get("analyst_consensus", pd.Series(0, index=df.index))
    return consensus.diff()


def analyst_rating_change(df: pd.DataFrame) -> pd.Series:
    """Generate analyst rating change signal.

    Args:
        df (pd.DataFrame): DataFrame with analyst rating data.

    Returns:
        pd.Series: Series with net analyst rating changes.
    """
    upgrades = df.get("analyst_upgrades", pd.Series(0, index=df.index))
    downgrades = df.get("analyst_downgrades", pd.Series(0, index=df.index))
    return upgrades - downgrades


def options_put_call_ratio(df: pd.DataFrame, threshold: float = 1.2) -> pd.Series:
    """Generate options put/call ratio signal.

    Args:
        df (pd.DataFrame): DataFrame with put/call ratio data.
        threshold (float, optional): PCR threshold. Defaults to 1.2.

    Returns:
        pd.Series: Series with put/call ratio values.
    """
    pcr = df.get("put_call_ratio", pd.Series(1, index=df.index))
    return pcr


def options_unusual_activity(df: pd.DataFrame) -> pd.Series:
    """Generate unusual options activity signal.

    Args:
        df (pd.DataFrame): DataFrame with options OI and volume data.

    Returns:
        pd.Series: Boolean series indicating unusual options activity.
    """
    avg_oi = df.get("options_oi", pd.Series(0, index=df.index)).rolling(20).mean()
    avg_vol = df.get("options_volume", pd.Series(0, index=df.index)).rolling(20).mean()
    unusual_oi = df.get("options_oi", pd.Series(0, index=df.index)) > 2 * avg_oi
    unusual_vol = df.get("options_volume", pd.Series(0, index=df.index)) > 2 * avg_vol
    return unusual_oi | unusual_vol


def short_interest_crowding(df: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
    """Generate short interest crowding signal.

    Args:
        df (pd.DataFrame): DataFrame with short interest and float data.
        threshold (float, optional): Short interest ratio threshold. Defaults to 0.1.

    Returns:
        pd.Series: Series with short interest ratios.
    """
    short_interest = df.get("short_interest", pd.Series(0, index=df.index))
    float_shares = df.get("float_shares", pd.Series(1, index=df.index))
    ratio = short_interest / float_shares
    return ratio


def volatility_sentiment(df: pd.DataFrame, vix_threshold: float = 20) -> pd.Series:
    """Generate volatility sentiment signal based on VIX levels.

    Args:
        df (pd.DataFrame): DataFrame with VIX data.
        vix_threshold (float, optional): VIX threshold. Defaults to 20.

    Returns:
        pd.Series: Series with VIX values.
    """
    vix = df.get("vix", pd.Series(0, index=df.index))
    return vix


sentiment_signals = {
    "news_nlp_sentiment": news_nlp_sentiment,
    "news_event_sentiment": news_event_sentiment,
    "social_media_trend": social_media_trend,
    "social_media_influencer_impact": social_media_influencer_impact,
    "analyst_consensus_change": analyst_consensus_change,
    "analyst_rating_change": analyst_rating_change,
    "options_put_call_ratio": options_put_call_ratio,
    "options_unusual_activity": options_unusual_activity,
    "short_interest_crowding": short_interest_crowding,
    "volatility_sentiment": volatility_sentiment,
}
# moved from open_trading_algo/sentiment_signals.py
