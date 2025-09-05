"""
Short position signal suite for use with SignalOptimizer.
Includes fundamental, technical, and sentiment-based signals.
Each function takes a DataFrame (with required columns) and returns a boolean Series.
"""
import pandas as pd
from tradingview_algo.data_cache import DataCache, is_caching_enabled


def compute_and_cache_short_signals(ticker: str, df: pd.DataFrame, timeframe: str):
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
    return (df.get("pe_ratio", pd.Series(False, index=df.index)) > 30) | (
        df.get("pb_ratio", pd.Series(False, index=df.index)) > 5
    )


def signal_deteriorating_financials(df: pd.DataFrame) -> pd.Series:
    margin_decline = df["gross_margin"].diff() < 0
    debt_rising = df["debt_to_equity"].diff() > 0
    return margin_decline | debt_rising


def signal_negative_earnings_revision(df: pd.DataFrame) -> pd.Series:
    return df.get("earnings_revision", pd.Series(0, index=df.index)) < 0


def signal_negative_momentum(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change(20) < 0


def signal_support_breakdown(df: pd.DataFrame) -> pd.Series:
    ma50 = df["close"].rolling(50).mean()
    return df["close"] < ma50


def signal_overbought_rsi(df: pd.DataFrame) -> pd.Series:
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi > 70


def signal_rising_short_interest(df: pd.DataFrame) -> pd.Series:
    return df.get("short_interest", pd.Series(0, index=df.index)).diff(10) > 0


def signal_negative_news(df: pd.DataFrame) -> pd.Series:
    return df.get("news_sentiment", pd.Series(0, index=df.index)) < 0


def signal_bearish_social_sentiment(df: pd.DataFrame) -> pd.Series:
    return df.get("social_sentiment", pd.Series(0, index=df.index)) < 0


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
