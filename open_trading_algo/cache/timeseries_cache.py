"""
Time Series Database Cache using InfluxDB.

This module provides an alternative caching solution using InfluxDB,
a high-performance time series database optimized for financial data storage and retrieval.

Key Features:
- Optimized for time series queries
- Automatic data compression and retention
- SQL-like query language (Flux)
- High-performance for OHLCV and signal data
- Seamless pandas DataFrame integration

Usage:
    from open_trading_algo.cache.timeseries_cache import TimeSeriesCache

    cache = TimeSeriesCache()

    # Store OHLCV data
    cache.store_price_data('AAPL', ohlcv_df)

    # Retrieve data with time range
    data = cache.get_price_data('AAPL', start='2023-01-01', end='2023-12-31')

    # Store signals
    cache.store_signals('AAPL', '1d', 'momentum', signals_df)

    # Query signals
    signals = cache.get_signals('AAPL', '1d', 'momentum')
"""

import os
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client.client.query_api import QueryApi
import yaml


def get_config():
    """Load configuration from db_config.yaml if it exists."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db_config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg
    return {}


class TimeSeriesCache:
    """
    InfluxDB-based cache for time series financial data.

    This class provides efficient storage and retrieval of OHLCV data and trading signals
    using InfluxDB's optimized time series database engine.
    """

    def __init__(
        self,
        url: str = "http://localhost:8086",
        token: str = "my-token",
        org: str = "trading-org",
        bucket: str = "trading-data",
    ):
        """
        Initialize InfluxDB connection.

        Args:
            url: InfluxDB server URL
            token: Authentication token
            org: Organization name
            bucket: Bucket name for data storage
        """
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket

        # Initialize client
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()

        # Create bucket if it doesn't exist
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        """Ensure the bucket exists, create it if necessary."""
        try:
            buckets_api = self.client.buckets_api()
            buckets = buckets_api.find_buckets().buckets

            bucket_exists = any(bucket.name == self.bucket for bucket in buckets)
            if not bucket_exists:
                buckets_api.create_bucket(bucket_name=self.bucket, org=self.org)
        except Exception as e:
            print(f"Warning: Could not verify/create bucket: {e}")

    def store_price_data(self, ticker: str, df: pd.DataFrame):
        """
        Store OHLCV price data for a ticker.

        Args:
            ticker: Ticker symbol
            df: DataFrame with OHLCV data and datetime index
        """
        if df.empty:
            return

        points = []

        for timestamp, row in df.iterrows():
            # Convert timestamp to nanoseconds for InfluxDB
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)

            point = (
                Point("price_data")
                .tag("ticker", ticker)
                .field("open", float(row.get("Open", row.get("open", 0))))
                .field("high", float(row.get("High", row.get("high", 0))))
                .field("low", float(row.get("Low", row.get("low", 0))))
                .field("close", float(row.get("Close", row.get("close", 0))))
                .field("volume", float(row.get("Volume", row.get("volume", 0))))
                .time(timestamp, WritePrecision.NS)
            )

            points.append(point)

        if points:
            self.write_api.write(bucket=self.bucket, org=self.org, record=points)

    def get_price_data(
        self, ticker: str, start: Optional[str] = None, end: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieve OHLCV price data for a ticker.

        Args:
            ticker: Ticker symbol
            start: Start date (ISO format)
            end: End date (ISO format)

        Returns:
            DataFrame with OHLCV data and datetime index
        """
        # Build Flux query
        query = f"""
        from(bucket: "{self.bucket}")
        |> range(start: {start or "-365d"}, stop: {end or "now()"})
        |> filter(fn: (r) => r["_measurement"] == "price_data")
        |> filter(fn: (r) => r["ticker"] == "{ticker}")
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> sort(columns: ["_time"])
        """

        try:
            result = self.query_api.query(query, org=self.org)
            records = []

            for table in result:
                for record in table.records:
                    records.append(
                        {
                            "datetime": record.get_time(),
                            "open": record.values.get("open"),
                            "high": record.values.get("high"),
                            "low": record.values.get("low"),
                            "close": record.values.get("close"),
                            "volume": record.values.get("volume"),
                        }
                    )

            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
            df = df.sort_index()

            return df

        except Exception as e:
            print(f"Error querying price data: {e}")
            return pd.DataFrame()

    def has_data(self, ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> bool:
        """
        Check if price data exists for a ticker.

        Args:
            ticker: Ticker symbol
            start: Start date filter
            end: End date filter

        Returns:
            True if data exists
        """
        df = self.get_price_data(ticker, start, end)
        return not df.empty

    def store_signals(self, ticker: str, timeframe: str, signal_type: str, df: pd.DataFrame):
        """
        Store trading signals for a ticker.

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe (e.g., '1d', '1h')
            signal_type: Type of signal (e.g., 'momentum', 'mean_reversion')
            df: DataFrame with datetime index and 'signal_value' column
        """
        if df.empty:
            return

        points = []

        for timestamp, row in df.iterrows():
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)

            signal_value = row.get("signal_value", row.get("signal", 0))

            point = (
                Point("signals")
                .tag("ticker", ticker)
                .tag("timeframe", timeframe)
                .tag("signal_type", signal_type)
                .field("signal_value", float(signal_value))
                .time(timestamp, WritePrecision.NS)
            )

            points.append(point)

        if points:
            self.write_api.write(bucket=self.bucket, org=self.org, record=points)

    def get_signals(
        self,
        ticker: str,
        timeframe: str,
        signal_type: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Retrieve trading signals for a ticker.

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe
            signal_type: Type of signal
            start: Start date filter
            end: End date filter

        Returns:
            DataFrame with datetime index and 'signal_value' column
        """
        query = f"""
        from(bucket: "{self.bucket}")
        |> range(start: {start or "-365d"}, stop: {end or "now()"})
        |> filter(fn: (r) => r["_measurement"] == "signals")
        |> filter(fn: (r) => r["ticker"] == "{ticker}")
        |> filter(fn: (r) => r["timeframe"] == "{timeframe}")
        |> filter(fn: (r) => r["signal_type"] == "{signal_type}")
        |> sort(columns: ["_time"])
        """

        try:
            result = self.query_api.query(query, org=self.org)
            records = []

            for table in result:
                for record in table.records:
                    records.append(
                        {
                            "datetime": record.get_time(),
                            "signal_value": record.values.get("_value", 0),
                        }
                    )

            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
            df = df.sort_index()

            return df

        except Exception as e:
            print(f"Error querying signals: {e}")
            return pd.DataFrame()

    def has_signals(
        self,
        ticker: str,
        timeframe: str,
        signal_type: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> bool:
        """
        Check if signals exist for a ticker.

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe
            signal_type: Type of signal
            start: Start date filter
            end: End date filter

        Returns:
            True if signals exist
        """
        df = self.get_signals(ticker, timeframe, signal_type, start, end)
        return not df.empty

    def get_aggregated_data(
        self,
        ticker: str,
        aggregation: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get aggregated OHLCV data with custom time windows.

        Args:
            ticker: Ticker symbol
            aggregation: Time aggregation window (e.g., '1h', '1d', '1w')
            start: Start date
            end: End date

        Returns:
            Aggregated OHLCV DataFrame
        """
        query = f"""
        from(bucket: "{self.bucket}")
        |> range(start: {start or "-365d"}, stop: {end or "now()"})
        |> filter(fn: (r) => r["_measurement"] == "price_data")
        |> filter(fn: (r) => r["ticker"] == "{ticker}")
        |> aggregateWindow(every: {aggregation}, fn: mean, createEmpty: false)
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> sort(columns: ["_time"])
        """

        try:
            result = self.query_api.query(query, org=self.org)
            records = []

            for table in result:
                for record in table.records:
                    records.append(
                        {
                            "datetime": record.get_time(),
                            "open": record.values.get("open"),
                            "high": record.values.get("high"),
                            "low": record.values.get("low"),
                            "close": record.values.get("close"),
                            "volume": record.values.get("volume"),
                        }
                    )

            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
            df = df.sort_index()

            return df

        except Exception as e:
            print(f"Error querying aggregated data: {e}")
            return pd.DataFrame()

    def get_signal_stats(
        self,
        ticker: str,
        timeframe: str,
        signal_type: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get statistics for signals.

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe
            signal_type: Type of signal
            start: Start date
            end: End date

        Returns:
            Dictionary with signal statistics
        """
        query = f"""
        from(bucket: "{self.bucket}")
        |> range(start: {start or "-365d"}, stop: {end or "now()"})
        |> filter(fn: (r) => r["_measurement"] == "signals")
        |> filter(fn: (r) => r["ticker"] == "{ticker}")
        |> filter(fn: (r) => r["timeframe"] == "{timeframe}")
        |> filter(fn: (r) => r["signal_type"] == "{signal_type}")
        |> group()
        |> count()
        |> yield(name: "count")
        """

        try:
            result = self.query_api.query(query, org=self.org)
            count = 0

            for table in result:
                for record in table.records:
                    count = record.values.get("_value", 0)
                    break

            return {
                "total_signals": count,
                "ticker": ticker,
                "timeframe": timeframe,
                "signal_type": signal_type,
            }

        except Exception as e:
            print(f"Error getting signal stats: {e}")
            return {}

    def close(self):
        """Close the InfluxDB client connection."""
        self.client.close()

    def get_database_info(self) -> Dict[str, Any]:
        """
        Get information about the database and cached data.

        Returns:
            Dictionary with database statistics
        """
        try:
            # Query for basic stats
            price_query = f"""
            from(bucket: "{self.bucket}")
            |> range(start: -365d)
            |> filter(fn: (r) => r["_measurement"] == "price_data")
            |> group()
            |> count()
            |> yield(name: "price_count")
            """

            signals_query = f"""
            from(bucket: "{self.bucket}")
            |> range(start: -365d)
            |> filter(fn: (r) => r["_measurement"] == "signals")
            |> group()
            |> count()
            |> yield(name: "signals_count")
            """

            price_result = self.query_api.query(price_query, org=self.org)
            signals_result = self.query_api.query(signals_query, org=self.org)

            price_count = 0
            signals_count = 0

            for table in price_result:
                for record in table.records:
                    price_count = record.values.get("_value", 0)
                    break

            for table in signals_result:
                for record in table.records:
                    signals_count = record.values.get("_value", 0)
                    break

            return {
                "database_url": self.url,
                "organization": self.org,
                "bucket": self.bucket,
                "price_data_points": price_count,
                "signals_points": signals_count,
                "total_data_points": price_count + signals_count,
            }

        except Exception as e:
            print(f"Error getting database info: {e}")
            return {}
