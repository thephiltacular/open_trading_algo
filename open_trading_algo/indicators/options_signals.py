"""
Options trading signal suite for use with SignalOptimizer.
Includes technical (volatility), fundamental (event-driven), and sentiment-based signals.
Each function takes a DataFrame (with required columns) and returns a boolean Series.
"""
from open_trading_algo.cache.data_cache import DataCache, is_caching_enabled
import pandas as pd


def signal_iv_vs_rv(df: pd.DataFrame) -> pd.Series:
    iv = df.get("implied_volatility", pd.Series(0, index=df.index))
    rv = df.get("realized_volatility", pd.Series(0, index=df.index))
    return (iv - rv) > 0.1


def signal_volatility_breakout(df: pd.DataFrame) -> pd.Series:
    rv = df.get("realized_volatility", pd.Series(0, index=df.index))
    rv_max = rv.rolling(20).max().shift(1)
    return rv > rv_max


def signal_support_resistance(df: pd.DataFrame) -> pd.Series:
    support = df["low"].rolling(50).min()
    resistance = df["high"].rolling(50).max()
    near_support = (df["close"] - support).abs() < 0.01 * df["close"]
    near_resistance = (df["close"] - resistance).abs() < 0.01 * df["close"]
    return near_support | near_resistance


def signal_earnings_event(df: pd.DataFrame) -> pd.Series:
    return df.get("earnings_event", pd.Series(False, index=df.index)).astype(bool)


def signal_macro_event(df: pd.DataFrame) -> pd.Series:
    return df.get("macro_event", pd.Series(False, index=df.index)).astype(bool)


def signal_mna_event(df: pd.DataFrame) -> pd.Series:
    return df.get("mna_event", pd.Series(False, index=df.index)).astype(bool)


def signal_unusual_options_activity(df: pd.DataFrame) -> pd.Series:
    avg_oi = df.get("options_oi", pd.Series(0, index=df.index)).rolling(20).mean()
    avg_vol = df.get("options_volume", pd.Series(0, index=df.index)).rolling(20).mean()
    unusual_oi = df.get("options_oi", pd.Series(0, index=df.index)) > 2 * avg_oi
    unusual_vol = df.get("options_volume", pd.Series(0, index=df.index)) > 2 * avg_vol
    return unusual_oi | unusual_vol


def signal_order_flow(df: pd.DataFrame) -> pd.Series:
    return df.get("order_flow_imbalance", pd.Series(0, index=df.index)).abs() > 1.5


def signal_news_sentiment(df: pd.DataFrame) -> pd.Series:
    return df.get("news_sentiment", pd.Series(0, index=df.index)).abs() > 0.7


def signal_social_sentiment(df: pd.DataFrame) -> pd.Series:
    return df.get("social_sentiment", pd.Series(0, index=df.index)).abs() > 0.7


def compute_and_cache_options_signals(ticker: str, df: pd.DataFrame, timeframe: str):
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
