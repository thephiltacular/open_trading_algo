"""
Data cache module for open_trading_algo.
Efficiently stores and retrieves financial data locally using SQLite.
Prevents redundant requests and persists data across sessions.

Signal Caching Example:
----------------------
from open_trading_algo.cache.data_cache import DataCache
import pandas as pd

# Suppose you have a DataFrame `signals_df` with datetime index and a 'signal_value' column
signals_df = pd.DataFrame({
    'signal_value': [1, 0, 1]
}, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))

cache = DataCache()
cache.store_signals('AAPL', '1d', 'long_trend', signals_df)

# Retrieve cached signals
df = cache.get_signals('AAPL', '1d', 'long_trend')
print(df)

# Check if signals are cached
exists = cache.has_signals('AAPL', '1d', 'long_trend')
print('Signals cached:', exists)
"""
import sqlite3
import pandas as pd
import os
from typing import List, Optional


import yaml


def get_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db_config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg or {}
    return {}


def is_caching_enabled():
    cfg = get_config()
    return cfg.get("enable_caching", True)


def get_db_path():
    cfg = get_config()
    db_path = cfg.get("db_path")
    if db_path:
        return db_path
    return os.path.join(os.path.dirname(__file__), "tv_data_cache.sqlite3")


DB_PATH = get_db_path()


class DataCache:
    def __init__(self, db_path: Optional[str] = None):
        # Always use config if present, else default
        self.db_path = db_path or get_db_path()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._setup()  # Always sets up DB/tables on first use

    def _setup(self):
        c = self.conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS price_data (
                ticker TEXT,
                datetime TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (ticker, datetime)
            )
        """
        )
        c.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ticker_datetime ON price_data (ticker, datetime)
        """
        )
        # Add signals table: stores signals for each ticker, timeframe, and signal_type
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS signals (
                ticker TEXT,
                datetime TEXT,
                timeframe TEXT,
                signal_type TEXT,
                signal_value REAL,
                PRIMARY KEY (ticker, datetime, timeframe, signal_type)
            )
        """
        )
        c.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_signals_ticker_timeframe_type ON signals (ticker, timeframe, signal_type, datetime)
        """
        )
        self.conn.commit()

    def store_signals(self, ticker: str, timeframe: str, signal_type: str, df: pd.DataFrame):
        """Store signals for a ticker, timeframe, and signal_type.

        Args:
            ticker (str): Ticker symbol.
            timeframe (str): Timeframe for the signals.
            signal_type (str): Type of signal.
            df (pd.DataFrame): DataFrame with datetime index and 'signal_value' column.
        """
        if df.empty:
            return
        records = [
            (ticker, str(idx), timeframe, signal_type, row["signal_value"])
            for idx, row in df.iterrows()
        ]
        c = self.conn.cursor()
        c.executemany(
            """
            INSERT OR REPLACE INTO signals (ticker, datetime, timeframe, signal_type, signal_value)
            VALUES (?, ?, ?, ?, ?)
        """,
            records,
        )
        self.conn.commit()

    def get_signals(
        self,
        ticker: str,
        timeframe: str,
        signal_type: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Retrieve signals for a ticker, timeframe, and signal_type.

        Args:
            ticker (str): Ticker symbol.
            timeframe (str): Timeframe for the signals.
            signal_type (str): Type of signal.
            start (Optional[str]): Start date for filtering.
            end (Optional[str]): End date for filtering.

        Returns:
            pd.DataFrame: DataFrame with datetime index and 'signal_value' column.
        """
        c = self.conn.cursor()
        query = "SELECT datetime, signal_value FROM signals WHERE ticker = ? AND timeframe = ? AND signal_type = ?"
        params = [ticker, timeframe, signal_type]
        if start:
            query += " AND datetime >= ?"
            params.append(start)
        if end:
            query += " AND datetime <= ?"
            params.append(end)
        query += " ORDER BY datetime ASC"
        rows = c.execute(query, params).fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["datetime", "signal_value"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        return df

    def has_signals(
        self,
        ticker: str,
        timeframe: str,
        signal_type: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> bool:
        """Check if signals exist for a ticker, timeframe, and signal_type.

        Args:
            ticker (str): Ticker symbol.
            timeframe (str): Timeframe for the signals.
            signal_type (str): Type of signal.
            start (Optional[str]): Start date for filtering.
            end (Optional[str]): End date for filtering.

        Returns:
            bool: True if signals exist, False otherwise.
        """
        c = self.conn.cursor()
        query = "SELECT 1 FROM signals WHERE ticker = ? AND timeframe = ? AND signal_type = ?"
        params = [ticker, timeframe, signal_type]
        if start:
            query += " AND datetime >= ?"
            params.append(start)
        if end:
            query += " AND datetime <= ?"
            params.append(end)
        query += " LIMIT 1"
        return c.execute(query, params).fetchone() is not None

    def store_price_data(self, ticker: str, df: pd.DataFrame):
        """Store price data for a ticker.

        Args:
            ticker (str): Ticker symbol.
            df (pd.DataFrame): DataFrame with OHLCV data.
        """
        if df.empty:
            return
        records = [
            (ticker, str(idx), row["Open"], row["High"], row["Low"], row["Close"], row["Volume"])
            for idx, row in df.iterrows()
        ]
        c = self.conn.cursor()
        c.executemany(
            """
            INSERT OR REPLACE INTO price_data (ticker, datetime, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            records,
        )
        self.conn.commit()

    def get_price_data(
        self, ticker: str, start: Optional[str] = None, end: Optional[str] = None
    ) -> pd.DataFrame:
        """Retrieve price data for a ticker.

        Args:
            ticker (str): Ticker symbol.
            start (Optional[str]): Start date for filtering.
            end (Optional[str]): End date for filtering.

        Returns:
            pd.DataFrame: DataFrame with OHLCV data.
        """
        c = self.conn.cursor()
        query = "SELECT datetime, open, high, low, close, volume FROM price_data WHERE ticker = ?"
        params = [ticker]
        if start:
            query += " AND datetime >= ?"
            params.append(start)
        if end:
            query += " AND datetime <= ?"
            params.append(end)
        query += " ORDER BY datetime ASC"
        rows = c.execute(query, params).fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        return df

    def has_data(self, ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> bool:
        """Check if price data exists for a ticker.

        Args:
            ticker (str): Ticker symbol.
            start (Optional[str]): Start date for filtering.
            end (Optional[str]): End date for filtering.

        Returns:
            bool: True if data exists, False otherwise.
        """
        c = self.conn.cursor()
        query = "SELECT 1 FROM price_data WHERE ticker = ?"
        params = [ticker]
        if start:
            query += " AND datetime >= ?"
            params.append(start)
        if end:
            query += " AND datetime <= ?"
            params.append(end)
        query += " LIMIT 1"
        return c.execute(query, params).fetchone() is not None

    def close(self):
        """Close the database connection."""
        self.conn.close()
