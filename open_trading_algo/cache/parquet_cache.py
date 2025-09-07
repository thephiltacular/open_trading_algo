import os
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


def get_config():
    """Load configuration from db_config.yaml if it exists."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "db_config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg
    return {}


def get_cache_dir():
    """Get the cache directory path from config or use default."""
    cfg = get_config()
    cache_dir = cfg.get("parquet_cache_dir")
    if cache_dir:
        return Path(cache_dir)
    return Path(os.path.dirname(__file__)) / "parquet_cache"


class ParquetCache:
    """
    Parquet-based cache implementation for time series financial data.

    This class provides an alternative to SQLite-based caching using Apache Parquet,
    which offers better performance for analytical queries and columnar storage
    optimized for time series data.

    Data is partitioned by ticker for efficient querying and storage.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize Parquet cache.

        Args:
            cache_dir: Directory to store Parquet files. Uses config or default if None.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else get_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different data types
        self.price_data_dir = self.cache_dir / "price_data"
        self.signals_dir = self.cache_dir / "signals"
        self.price_data_dir.mkdir(exist_ok=True)
        self.signals_dir.mkdir(exist_ok=True)

    def _get_price_data_path(self, ticker: str) -> Path:
        """Get the Parquet file path for price data of a ticker."""
        return self.price_data_dir / f"{ticker}.parquet"

    def _get_signals_path(self, ticker: str, timeframe: str, signal_type: str) -> Path:
        """Get the Parquet file path for signals of a ticker/timeframe/signal_type."""
        # Create a safe filename from the combination
        safe_name = f"{ticker}_{timeframe}_{signal_type}".replace("/", "_").replace("\\", "_")
        return self.signals_dir / f"{safe_name}.parquet"

    def store_price_data(self, ticker: str, df: pd.DataFrame):
        """Store price data for a ticker in Parquet format.

        Args:
            ticker: Ticker symbol
            df: DataFrame with OHLCV data and datetime index
        """
        if df.empty:
            return

        # Prepare data for Parquet storage
        df_copy = df.copy()
        df_copy = df_copy.reset_index()

        # Ensure datetime is in string format for Parquet
        if "datetime" in df_copy.columns:
            df_copy["datetime"] = df_copy["datetime"].astype(str)
        elif df_copy.index.name == "datetime" or isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy["datetime"] = df_copy.index.astype(str)
            df_copy = df_copy.reset_index(drop=True)

        # Rename columns to match expected format
        column_mapping = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        df_copy = df_copy.rename(columns=column_mapping)

        # Ensure required columns exist
        required_cols = ["datetime", "open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df_copy.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")

        # Select only required columns
        df_final = df_copy[required_cols].copy()

        # Convert to PyArrow table for efficient storage
        table = pa.Table.from_pandas(df_final)

        # Write to Parquet with optimized settings for time series
        pq.write_table(
            table,
            self._get_price_data_path(ticker),
            compression="snappy",  # Good balance of speed and compression
            use_dictionary=True,
            row_group_size=50000,  # Optimize for time series queries
        )

    def get_price_data(self, ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """Retrieve price data for a ticker.

        Args:
            ticker: Ticker symbol
            start: Start date filter (ISO format string)
            end: End date filter (ISO format string)

        Returns:
            DataFrame with OHLCV data and datetime index
        """
        file_path = self._get_price_data_path(ticker)
        if not file_path.exists():
            return pd.DataFrame()

        # Read Parquet file
        table = pq.read_table(file_path)
        df = table.to_pandas()

        if df.empty:
            return df

        # Convert datetime back to datetime index
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")

        # Apply date filters if provided
        if start:
            start_dt = pd.to_datetime(start)
            df = df[df.index >= start_dt]
        if end:
            end_dt = pd.to_datetime(end)
            df = df[df.index <= end_dt]

        # Sort by datetime
        df = df.sort_index()

        return df

    def has_data(self, ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> bool:
        """Check if price data exists for a ticker.

        Args:
            ticker: Ticker symbol
            start: Start date filter
            end: End date filter

        Returns:
            True if data exists, False otherwise
        """
        df = self.get_price_data(ticker, start, end)
        return not df.empty

    def store_signals(self, ticker: str, timeframe: str, signal_type: str, df: pd.DataFrame):
        """Store signals for a ticker, timeframe, and signal_type.

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe for the signals
            signal_type: Type of signal
            df: DataFrame with datetime index and 'signal_value' column
        """
        if df.empty:
            return

        # Prepare data for storage
        df_copy = df.copy()
        df_copy = df_copy.reset_index()

        # Ensure datetime is in string format
        if "datetime" in df_copy.columns:
            df_copy["datetime"] = df_copy["datetime"].astype(str)
        elif df_copy.index.name == "datetime" or isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy["datetime"] = df_copy.index.astype(str)
            df_copy = df_copy.reset_index(drop=True)

        # Ensure signal_value column exists
        if "signal_value" not in df_copy.columns:
            raise ValueError("DataFrame must contain 'signal_value' column")

        # Add metadata columns
        df_copy["ticker"] = ticker
        df_copy["timeframe"] = timeframe
        df_copy["signal_type"] = signal_type

        # Select required columns
        required_cols = ["datetime", "ticker", "timeframe", "signal_type", "signal_value"]
        df_final = df_copy[required_cols].copy()

        # Convert to PyArrow table
        table = pa.Table.from_pandas(df_final)

        # Write to Parquet
        pq.write_table(
            table,
            self._get_signals_path(ticker, timeframe, signal_type),
            compression="snappy",
            use_dictionary=True,
            row_group_size=50000,
        )

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
            ticker: Ticker symbol
            timeframe: Timeframe for the signals
            signal_type: Type of signal
            start: Start date filter
            end: End date filter

        Returns:
            DataFrame with datetime index and 'signal_value' column
        """
        file_path = self._get_signals_path(ticker, timeframe, signal_type)
        if not file_path.exists():
            return pd.DataFrame()

        # Read Parquet file
        table = pq.read_table(file_path)
        df = table.to_pandas()

        if df.empty:
            return df

        # Convert datetime back to datetime index
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")

        # Apply date filters if provided
        if start:
            start_dt = pd.to_datetime(start)
            df = df[df.index >= start_dt]
        if end:
            end_dt = pd.to_datetime(end)
            df = df[df.index <= end_dt]

        # Sort by datetime and return only signal_value column
        df = df.sort_index()
        return df[["signal_value"]]

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
            ticker: Ticker symbol
            timeframe: Timeframe for the signals
            signal_type: Type of signal
            start: Start date filter
            end: End date filter

        Returns:
            True if signals exist, False otherwise
        """
        df = self.get_signals(ticker, timeframe, signal_type, start, end)
        return not df.empty

    def close(self):
        """Close the cache (no-op for Parquet implementation)."""
        pass  # Parquet files don't require explicit closing

    def get_cache_info(self) -> dict:
        """Get information about the cache contents.

        Returns:
            Dictionary with cache statistics
        """
        info = {
            "cache_dir": str(self.cache_dir),
            "price_data_files": len(list(self.price_data_dir.glob("*.parquet"))),
            "signals_files": len(list(self.signals_dir.glob("*.parquet"))),
            "total_size_mb": 0.0,
        }

        # Calculate total size
        total_size = 0
        for file_path in self.price_data_dir.glob("*.parquet"):
            total_size += file_path.stat().st_size
        for file_path in self.signals_dir.glob("*.parquet"):
            total_size += file_path.stat().st_size

        info["total_size_mb"] = total_size / (1024 * 1024)
        return info
