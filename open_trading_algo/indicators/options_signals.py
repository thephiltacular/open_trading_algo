"""
Options trading signal suite for use with SignalOptimizer.
Includes technical (volatility), fundamental (event-driven), and sentiment-based signals.
Each function takes a DataFrame (with required columns) and returns a boolean Series.
"""
from open_trading_algo.cache.data_cache import DataCache, is_caching_enabled
import pandas as pd


def signal_iv_vs_rv(df: pd.DataFrame) -> pd.Series:
    """Generate implied vs realized volatility signal.

    Args:
        df (pd.DataFrame): DataFrame with volatility data columns.

    Returns:
        pd.Series: Boolean series indicating IV vs RV signals.
    """
    iv = df.get("implied_volatility", pd.Series(0, index=df.index))
    rv = df.get("realized_volatility", pd.Series(0, index=df.index))
    return (iv - rv) > 0.1


def signal_volatility_breakout(df: pd.DataFrame) -> pd.Series:
    """Generate volatility breakout signal.

    Args:
        df (pd.DataFrame): DataFrame with volatility data columns.

    Returns:
        pd.Series: Boolean series indicating volatility breakout signals.
    """
    rv = df.get("realized_volatility", pd.Series(0, index=df.index))
    rv_max = rv.rolling(20).max().shift(1)
    return rv > rv_max


def signal_support_resistance(df: pd.DataFrame) -> pd.Series:
    """Generate support/resistance level signal.

    Args:
        df (pd.DataFrame): DataFrame with 'close', 'high', 'low' price columns.

    Returns:
        pd.Series: Boolean series indicating support/resistance signals.
    """
    support = df["low"].rolling(50).min()
    resistance = df["high"].rolling(50).max()
    near_support = (df["close"] - support).abs() < 0.01 * df["close"]
    near_resistance = (df["close"] - resistance).abs() < 0.01 * df["close"]
    return near_support | near_resistance


def signal_earnings_event(df: pd.DataFrame) -> pd.Series:
    """Generate earnings event signal.

    Args:
        df (pd.DataFrame): DataFrame with earnings event data.

    Returns:
        pd.Series: Boolean series indicating earnings event signals.
    """
    return df.get("earnings_event", pd.Series(False, index=df.index)).astype(bool)


def signal_macro_event(df: pd.DataFrame) -> pd.Series:
    """Generate macroeconomic event signal.

    Args:
        df (pd.DataFrame): DataFrame with macro event data.

    Returns:
        pd.Series: Boolean series indicating macro event signals.
    """
    return df.get("macro_event", pd.Series(False, index=df.index)).astype(bool)


def signal_mna_event(df: pd.DataFrame) -> pd.Series:
    """Generate mergers and acquisitions event signal.

    Args:
        df (pd.DataFrame): DataFrame with M&A event data.

    Returns:
        pd.Series: Boolean series indicating M&A event signals.
    """
    return df.get("mna_event", pd.Series(False, index=df.index)).astype(bool)


def signal_unusual_options_activity(df: pd.DataFrame) -> pd.Series:
    """Generate unusual options activity signal.

    Args:
        df (pd.DataFrame): DataFrame with options OI and volume data.

    Returns:
        pd.Series: Boolean series indicating unusual options activity signals.
    """
    avg_oi = df.get("options_oi", pd.Series(0, index=df.index)).rolling(20).mean()
    avg_vol = df.get("options_volume", pd.Series(0, index=df.index)).rolling(20).mean()
    unusual_oi = df.get("options_oi", pd.Series(0, index=df.index)) > 2 * avg_oi
    unusual_vol = df.get("options_volume", pd.Series(0, index=df.index)) > 2 * avg_vol
    return unusual_oi | unusual_vol


def signal_order_flow(df: pd.DataFrame) -> pd.Series:
    """Generate order flow imbalance signal.

    Args:
        df (pd.DataFrame): DataFrame with order flow data.

    Returns:
        pd.Series: Boolean series indicating order flow signals.
    """
    return df.get("order_flow_imbalance", pd.Series(0, index=df.index)).abs() > 1.5


def signal_news_sentiment(df: pd.DataFrame) -> pd.Series:
    """Generate news sentiment signal.

    Args:
        df (pd.DataFrame): DataFrame with news sentiment data.

    Returns:
        pd.Series: Boolean series indicating news sentiment signals.
    """
    return df.get("news_sentiment", pd.Series(0, index=df.index)).abs() > 0.7


def signal_social_sentiment(df: pd.DataFrame) -> pd.Series:
    """Generate social sentiment signal.

    Args:
        df (pd.DataFrame): DataFrame with social sentiment data.

    Returns:
        pd.Series: Boolean series indicating social sentiment signals.
    """
    return df.get("social_sentiment", pd.Series(0, index=df.index)).abs() > 0.7


def compute_and_cache_options_signals(ticker: str, df: pd.DataFrame, timeframe: str):
    """Compute and cache combined options signals for a ticker.

    Args:
        ticker (str): Stock ticker symbol.
        df (pd.DataFrame): DataFrame with required columns for signal computation.
        timeframe (str): Timeframe identifier for caching.

    Returns:
        pd.DataFrame: DataFrame with combined signal values.
    """
    signal_type = "options_trend"
    if is_caching_enabled():
        cache = DataCache()
        if cache.has_signals(ticker, timeframe, signal_type):
            return cache.get_signals(ticker, timeframe, signal_type)
    combined = (
        signal_iv_vs_rv(df) | signal_volatility_breakout(df) | signal_unusual_options_activity(df)
    )
    signals_df = pd.DataFrame({"signal_value": combined.astype(int)}, index=df.index)
    if is_caching_enabled():
        cache.store_signals(ticker, timeframe, signal_type, signals_df)
    return signals_df


options_signals = {
    "iv_vs_rv": signal_iv_vs_rv,
    "volatility_breakout": signal_volatility_breakout,
    "support_resistance": signal_support_resistance,
    "earnings_event": signal_earnings_event,
    "macro_event": signal_macro_event,
    "mna_event": signal_mna_event,
    "unusual_options_activity": signal_unusual_options_activity,
    "order_flow": signal_order_flow,
    "news_sentiment": signal_news_sentiment,
    "social_sentiment": signal_social_sentiment,
}
