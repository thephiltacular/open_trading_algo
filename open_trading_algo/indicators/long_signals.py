"""
Long position signal suite for use with SignalOptimizer.
Includes fundamental, technical, and sentiment-based signals.
Each function takes a DataFrame (with required columns) and returns a boolean Series.
"""
import pandas as pd
from open_trading_algo.cache.data_cache import DataCache, is_caching_enabled


def signal_undervalued(df: pd.DataFrame) -> pd.Series:
    pe_ok = df.get("pe_ratio", pd.Series(100, index=df.index)) < 15
    earnings_growth = df.get("earnings_growth", pd.Series(0, index=df.index)) > 0.10
    return pe_ok & earnings_growth


def signal_high_roe(df: pd.DataFrame) -> pd.Series:
    return df.get("roe", pd.Series(0, index=df.index)) > 0.15


def signal_positive_earnings_revision(df: pd.DataFrame) -> pd.Series:
    return df.get("earnings_revision", pd.Series(0, index=df.index)) > 0


def signal_sma_trend(df: pd.DataFrame) -> pd.Series:
    sma10 = df["close"].rolling(10).mean()
    sma50 = df["close"].rolling(50).mean()
    sma200 = df["close"].rolling(200).mean()
    return (df["close"] > sma10) & (df["close"] > sma50) & (df["close"] > sma200)


def signal_positive_momentum(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change(20) > 0


def signal_rsi_macd(df: pd.DataFrame) -> pd.Series:
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_cross = (macd > signal) & (macd.shift(1) <= signal.shift(1))
    return (rsi < 30) & macd_cross


def signal_breakout(df: pd.DataFrame) -> pd.Series:
    high50 = df["high"].rolling(50).max()
    return df["close"] > high50.shift(1)


def signal_positive_news(df: pd.DataFrame) -> pd.Series:
    return df.get("news_sentiment", pd.Series(0, index=df.index)) > 0


def signal_analyst_upgrades(df: pd.DataFrame) -> pd.Series:
    return df.get("analyst_upgrades", pd.Series(0, index=df.index)) > 0


def signal_social_sentiment(df: pd.DataFrame) -> pd.Series:
    sentiment = df.get("social_sentiment", pd.Series(0, index=df.index))
    return sentiment.diff() > 0


long_signals = {
    "undervalued": signal_undervalued,
    "high_roe": signal_high_roe,
    "positive_earnings_revision": signal_positive_earnings_revision,
    "sma_trend": signal_sma_trend,
    "positive_momentum": signal_positive_momentum,
    "rsi_macd": signal_rsi_macd,
    "breakout": signal_breakout,
    "positive_news": signal_positive_news,
    "analyst_upgrades": signal_analyst_upgrades,
    "social_sentiment": signal_social_sentiment,
}


def compute_and_cache_long_signals(ticker: str, df: pd.DataFrame, timeframe: str):
    signal_type = "long_trend"
    if is_caching_enabled():
        cache = DataCache()
        if cache.has_signals(ticker, timeframe, signal_type):
            return cache.get_signals(ticker, timeframe, signal_type)
    combined = (
        signal_undervalued(df)
        | signal_high_roe(df)
        | signal_positive_earnings_revision(df)
        | signal_sma_trend(df)
        | signal_positive_momentum(df)
        | signal_rsi_macd(df)
        | signal_breakout(df)
        | signal_positive_news(df)
        | signal_analyst_upgrades(df)
        | signal_social_sentiment(df)
    )
    signals_df = pd.DataFrame({"signal_value": combined.astype(int)}, index=df.index)
    if is_caching_enabled():
        cache.store_signals(ticker, timeframe, signal_type, signals_df)
    return signals_df
