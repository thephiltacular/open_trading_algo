"""
Options trading signal suite for use with SignalOptimizer.
Includes technical (volatility), fundamental (event-driven), and sentiment-based signals.
Each function takes a DataFrame (with required columns) and returns a boolean Series.
"""
from tradingview_algo.data_cache import DataCache, is_caching_enabled
import pandas as pd


def signal_iv_vs_rv(df: pd.DataFrame) -> pd.Series:
    """Implied volatility (IV) much higher than realized volatility (RV) signals potential premium selling."""
    iv = df.get("implied_volatility", pd.Series(0, index=df.index))
    rv = df.get("realized_volatility", pd.Series(0, index=df.index))
    return (iv - rv) > 0.1


def signal_volatility_breakout(df: pd.DataFrame) -> pd.Series:
    """Volatility breakout: realized volatility spikes above recent range."""
    rv = df.get("realized_volatility", pd.Series(0, index=df.index))
    rv_max = rv.rolling(20).max().shift(1)
    return rv > rv_max


def signal_support_resistance(df: pd.DataFrame) -> pd.Series:
    """Price near support or resistance for timing straddles/strangles."""
    support = df["low"].rolling(50).min()
    resistance = df["high"].rolling(50).max()
    near_support = (df["close"] - support).abs() < 0.01 * df["close"]
    near_resistance = (df["close"] - resistance).abs() < 0.01 * df["close"]
    return near_support | near_resistance


def signal_earnings_event(df: pd.DataFrame) -> pd.Series:
    """Earnings announcement window (e.g., event flag in data)."""
    return df.get("earnings_event", pd.Series(False, index=df.index)).astype(bool)


def signal_macro_event(df: pd.DataFrame) -> pd.Series:
    """Macroeconomic release window (e.g., event flag in data)."""
    return df.get("macro_event", pd.Series(False, index=df.index)).astype(bool)


def signal_mna_event(df: pd.DataFrame) -> pd.Series:
    """M&A activity window (e.g., event flag in data)."""
    return df.get("mna_event", pd.Series(False, index=df.index)).astype(bool)


def signal_unusual_options_activity(df: pd.DataFrame) -> pd.Series:
    """Unusual options volume/open interest."""
    avg_oi = df.get("options_oi", pd.Series(0, index=df.index)).rolling(20).mean()
    avg_vol = df.get("options_volume", pd.Series(0, index=df.index)).rolling(20).mean()
    unusual_oi = df.get("options_oi", pd.Series(0, index=df.index)) > 2 * avg_oi
    unusual_vol = df.get("options_volume", pd.Series(0, index=df.index)) > 2 * avg_vol
    return unusual_oi | unusual_vol


def signal_order_flow(df: pd.DataFrame) -> pd.Series:
    """Order flow analysis: large buy/sell imbalance in options trades."""
    return df.get("order_flow_imbalance", pd.Series(0, index=df.index)).abs() > 1.5


def signal_news_sentiment(df: pd.DataFrame) -> pd.Series:
    """News sentiment anticipating volatility spike (e.g., abs(sentiment) > threshold)."""
    return df.get("news_sentiment", pd.Series(0, index=df.index)).abs() > 0.7


def signal_social_sentiment(df: pd.DataFrame) -> pd.Series:
    """Social media sentiment anticipating volatility spike (e.g., abs(sentiment) > threshold)."""
    return df.get("social_sentiment", pd.Series(0, index=df.index)).abs() > 0.7


def compute_and_cache_options_signals(ticker: str, df: pd.DataFrame, timeframe: str):
    """
    Compute and cache options signals for a ticker and timeframe.
    Uses the enable_caching config variable in db_config.yaml to control caching.
    If caching is enabled and signals are present, loads from cache. Otherwise computes and (if enabled) stores signals.
    """
    signal_type = "options_trend"  # or use a more specific name per signal
    if is_caching_enabled():
        cache = DataCache()
        if cache.has_signals(ticker, timeframe, signal_type):
            return cache.get_signals(ticker, timeframe, signal_type)
    # Example: combine multiple signals into one
    combined = (
        signal_iv_vs_rv(df) | signal_volatility_breakout(df) | signal_unusual_options_activity(df)
    )
    signals_df = pd.DataFrame({"signal_value": combined.astype(int)}, index=df.index)
    if is_caching_enabled():
        cache.store_signals(ticker, timeframe, signal_type, signals_df)
    return signals_df


import pandas as pd


def signal_iv_vs_rv(df: pd.DataFrame) -> pd.Series:
    """Implied volatility (IV) much higher than realized volatility (RV) signals potential premium selling."""
    iv = df.get("implied_volatility", pd.Series(0, index=df.index))
    rv = df.get("realized_volatility", pd.Series(0, index=df.index))
    return (iv - rv) > 0.1


def signal_volatility_breakout(df: pd.DataFrame) -> pd.Series:
    """Volatility breakout: realized volatility spikes above recent range."""
    rv = df.get("realized_volatility", pd.Series(0, index=df.index))
    rv_max = rv.rolling(20).max().shift(1)
    return rv > rv_max


def signal_support_resistance(df: pd.DataFrame) -> pd.Series:
    """Price near support or resistance for timing straddles/strangles."""
    support = df["low"].rolling(50).min()
    resistance = df["high"].rolling(50).max()
    near_support = (df["close"] - support).abs() < 0.01 * df["close"]
    near_resistance = (df["close"] - resistance).abs() < 0.01 * df["close"]
    return near_support | near_resistance


def signal_earnings_event(df: pd.DataFrame) -> pd.Series:
    """Earnings announcement window (e.g., event flag in data)."""
    return df.get("earnings_event", pd.Series(False, index=df.index)).astype(bool)


def signal_macro_event(df: pd.DataFrame) -> pd.Series:
    """Macroeconomic release window (e.g., event flag in data)."""
    return df.get("macro_event", pd.Series(False, index=df.index)).astype(bool)


def signal_mna_event(df: pd.DataFrame) -> pd.Series:
    """M&A activity window (e.g., event flag in data)."""
    return df.get("mna_event", pd.Series(False, index=df.index)).astype(bool)


def signal_unusual_options_activity(df: pd.DataFrame) -> pd.Series:
    """Unusual options volume/open interest."""
    avg_oi = df.get("options_oi", pd.Series(0, index=df.index)).rolling(20).mean()
    avg_vol = df.get("options_volume", pd.Series(0, index=df.index)).rolling(20).mean()
    unusual_oi = df.get("options_oi", pd.Series(0, index=df.index)) > 2 * avg_oi
    unusual_vol = df.get("options_volume", pd.Series(0, index=df.index)) > 2 * avg_vol
    return unusual_oi | unusual_vol


def signal_order_flow(df: pd.DataFrame) -> pd.Series:
    """Order flow analysis: large buy/sell imbalance in options trades."""
    return df.get("order_flow_imbalance", pd.Series(0, index=df.index)).abs() > 1.5


def signal_news_sentiment(df: pd.DataFrame) -> pd.Series:
    """News sentiment anticipating volatility spike (e.g., abs(sentiment) > threshold)."""
    return df.get("news_sentiment", pd.Series(0, index=df.index)).abs() > 0.7


def signal_social_sentiment(df: pd.DataFrame) -> pd.Series:
    """Social media sentiment anticipating volatility spike (e.g., abs(sentiment) > threshold)."""
    return df.get("social_sentiment", pd.Series(0, index=df.index)).abs() > 0.7


# Dictionary of all options signals
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
