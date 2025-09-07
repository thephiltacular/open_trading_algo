"""
Time Series Database Cache using InfluxDB.

This module provides an alternative caching solution using InfluxDB,
a high-performance time series database optimized for financial data storage and retrieval.

Key Features:
- Optimized for time series queries
- Automatic data compression and retention
- SQL-like query language (Flux)
- High-performance for OHLCV, signal, and metrics data
- Seamless pandas DataFrame integration
- Automated technical indicator calculations

Measurements/Tables:
- price_data: OHLCV price data for tickers
- signals: Trading signals with metadata
- metrics: Calculated technical indicators and metrics

Usage:
    from open_trading_algo.cache.timeseries_cache import TimeSeriesCache

    cache = TimeSeriesCache()

    # Store OHLCV data
    cache.store_price_data('AAPL', ohlcv_df)

    # Calculate and store technical indicators
    cache.calculate_and_store_metrics('AAPL', timeframe='1d')

    # Retrieve metrics
    metrics = cache.get_metrics('AAPL', timeframe='1d')

    # Get available metrics
    available = cache.get_available_metrics('AAPL')
"""

import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import pandas as pd
import yaml
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.query_api import QueryApi
from influxdb_client.client.write_api import ASYNCHRONOUS, SYNCHRONOUS


def get_config():
    """Load configuration from db_config.yaml if it exists."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "db_config.yaml")
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

        lines = []

        for timestamp, row in df.iterrows():
            # Convert timestamp to nanoseconds for InfluxDB
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)

            # Convert to nanoseconds since epoch
            timestamp_ns = int(timestamp.timestamp() * 1e9)

            # Create line protocol string
            line = f"price_data,ticker={ticker} "
            fields = []
            fields.append(f"open={float(row.get('Open', row.get('open', 0)))}")
            fields.append(f"high={float(row.get('High', row.get('high', 0)))}")
            fields.append(f"low={float(row.get('Low', row.get('low', 0)))}")
            fields.append(f"close={float(row.get('Close', row.get('close', 0)))}")
            fields.append(f"volume={float(row.get('Volume', row.get('volume', 0)))}")
            line += ",".join(fields)
            line += f" {timestamp_ns}"

            lines.append(line)

        if lines:
            # Write all lines at once
            self.write_api.write(bucket=self.bucket, org=self.org, record="\n".join(lines))
            self.write_api.flush()
            import time

            time.sleep(0.5)  # Longer delay to ensure write completes

    def get_price_data(self, ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve OHLCV price data for a ticker.

        Args:
            ticker: Ticker symbol
            start: Start date (ISO format)
            end: End date (ISO format)

        Returns:
            DataFrame with OHLCV data and datetime index
        """
        # Build Flux query - use a wider default range if no dates provided
        if start is None and end is None:
            # Query all data if no date range specified
            query = f"""
            from(bucket: "{self.bucket}")
            |> range(start: 0, stop: now())
            |> filter(fn: (r) => r["_measurement"] == "price_data")
            |> filter(fn: (r) => r["ticker"] == "{ticker}")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> sort(columns: ["_time"])
            """
        else:
            query = f"""
            from(bucket: "{self.bucket}")
            |> range(start: {start or "-10y"}, stop: {end or "now()"})
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
            import time

            time.sleep(0.1)  # Small delay to ensure write completes

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
        # Build Flux query - use a wider default range if no dates provided
        if start is None and end is None:
            # Query all data if no date range specified
            query = f"""
            from(bucket: "{self.bucket}")
            |> range(start: 0, stop: now())
            |> filter(fn: (r) => r["_measurement"] == "signals")
            |> filter(fn: (r) => r["ticker"] == "{ticker}")
            |> filter(fn: (r) => r["timeframe"] == "{timeframe}")
            |> filter(fn: (r) => r["signal_type"] == "{signal_type}")
            |> sort(columns: ["_time"])
            """
        else:
            query = f"""
            from(bucket: "{self.bucket}")
            |> range(start: {start or "-10y"}, stop: {end or "now()"})
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
        # Build Flux query - use a wider default range if no dates provided
        if start is None and end is None:
            # Query all data if no date range specified
            query = f"""
            from(bucket: "{self.bucket}")
            |> range(start: 0, stop: now())
            |> filter(fn: (r) => r["_measurement"] == "price_data")
            |> filter(fn: (r) => r["ticker"] == "{ticker}")
            |> aggregateWindow(every: {aggregation}, fn: mean, createEmpty: false)
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> sort(columns: ["_time"])
            """
        else:
            query = f"""
            from(bucket: "{self.bucket}")
            |> range(start: {start or "-10y"}, stop: {end or "now()"})
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
        # Build Flux query - use a wider default range if no dates provided
        if start is None and end is None:
            # Query all data if no date range specified
            query = f"""
            from(bucket: "{self.bucket}")
            |> range(start: 0, stop: now())
            |> filter(fn: (r) => r["_measurement"] == "signals")
            |> filter(fn: (r) => r["ticker"] == "{ticker}")
            |> filter(fn: (r) => r["timeframe"] == "{timeframe}")
            |> filter(fn: (r) => r["signal_type"] == "{signal_type}")
            |> group()
            |> count()
            |> yield(name: "count")
            """
        else:
            query = f"""
            from(bucket: "{self.bucket}")
            |> range(start: {start or "-10y"}, stop: {end or "now()"})
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
        if hasattr(self, "write_api") and self.write_api:
            self.write_api.close()
        if hasattr(self, "client") and self.client:
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
            |> range(start: 0, stop: now())
            |> filter(fn: (r) => r["_measurement"] == "price_data")
            |> group()
            |> count()
            |> yield(name: "price_count")
            """

            signals_query = f"""
            from(bucket: "{self.bucket}")
            |> range(start: 0, stop: now())
            |> filter(fn: (r) => r["_measurement"] == "signals")
            |> group()
            |> count()
            |> yield(name: "signals_count")
            """

            metrics_query = f"""
            from(bucket: "{self.bucket}")
            |> range(start: 0, stop: now())
            |> filter(fn: (r) => r["_measurement"] == "metrics")
            |> group()
            |> count()
            |> yield(name: "metrics_count")
            """

            price_result = self.query_api.query(price_query, org=self.org)
            signals_result = self.query_api.query(signals_query, org=self.org)
            metrics_result = self.query_api.query(metrics_query, org=self.org)

            price_count = 0
            signals_count = 0
            metrics_count = 0

            for table in price_result:
                for record in table.records:
                    price_count = record.values.get("_value", 0)
                    break

            for table in signals_result:
                for record in table.records:
                    signals_count = record.values.get("_value", 0)
                    break

            for table in metrics_result:
                for record in table.records:
                    metrics_count = record.values.get("_value", 0)
                    break

            return {
                "database_url": self.url,
                "organization": self.org,
                "bucket": self.bucket,
                "price_data_points": price_count,
                "signals_points": signals_count,
                "metrics_points": metrics_count,
                "total_data_points": price_count + signals_count + metrics_count,
            }

        except Exception as e:
            print(f"Error getting database info: {e}")
            return {}

    def calculate_and_store_metrics(
        self,
        ticker: str,
        timeframe: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        indicators: Optional[list] = None,
    ):
        """
        Calculate technical indicators and metrics from price data and store them.

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe for calculations (e.g., '1d', '1h')
            start: Start date for calculation period
            end: End date for calculation period
            indicators: List of indicators to calculate. If None, calculates all available.
        """
        # Get price data
        price_data = self.get_price_data(ticker, start, end)
        if price_data.empty:
            print(f"No price data available for {ticker}")
            return

        # Default indicators if none specified
        if indicators is None:
            indicators = [
                "sma_20",
                "sma_50",
                "ema_12",
                "ema_26",
                "rsi_14",
                "macd",
                "macd_signal",
                "macd_hist",
                "bb_upper",
                "bb_middle",
                "bb_lower",
                "bb_width",
                "volatility_20",
                "returns",
                "cumulative_returns",
            ]

        # Calculate indicators
        metrics_df = self._calculate_technical_indicators(price_data, indicators)

        if metrics_df.empty:
            print(f"No metrics calculated for {ticker}")
            return

        # Store metrics
        self._store_metrics_data(ticker, timeframe, metrics_df)

        print(f"✅ Stored {len(metrics_df)} metric data points for {ticker}")

    def _calculate_technical_indicators(self, price_data: pd.DataFrame, indicators: list) -> pd.DataFrame:
        """
        Calculate technical indicators from price data.

        Args:
            price_data: DataFrame with OHLCV data
            indicators: List of indicators to calculate

        Returns:
            DataFrame with calculated indicators
        """
        df = price_data.copy()
        calculated_indicators = {}

        # Simple Moving Averages
        if "sma_20" in indicators:
            calculated_indicators["sma_20"] = df["close"].rolling(window=20).mean()

        if "sma_50" in indicators:
            calculated_indicators["sma_50"] = df["close"].rolling(window=50).mean()

        # Exponential Moving Averages
        if "ema_12" in indicators:
            calculated_indicators["ema_12"] = df["close"].ewm(span=12).mean()

        if "ema_26" in indicators:
            calculated_indicators["ema_26"] = df["close"].ewm(span=26).mean()

        # RSI
        if "rsi_14" in indicators:
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            calculated_indicators["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD
        if any(ind in indicators for ind in ["macd", "macd_signal", "macd_hist"]):
            ema_12 = df["close"].ewm(span=12).mean()
            ema_26 = df["close"].ewm(span=26).mean()

            if "macd" in indicators:
                calculated_indicators["macd"] = ema_12 - ema_26

            if "macd_signal" in indicators:
                calculated_indicators["macd_signal"] = (ema_12 - ema_26).ewm(span=9).mean()

            if "macd_hist" in indicators:
                macd_line = ema_12 - ema_26
                signal_line = macd_line.ewm(span=9).mean()
                calculated_indicators["macd_hist"] = macd_line - signal_line

        # Bollinger Bands
        if any(ind in indicators for ind in ["bb_upper", "bb_middle", "bb_lower", "bb_width"]):
            sma_20 = df["close"].rolling(window=20).mean()
            std_20 = df["close"].rolling(window=20).std()

            if "bb_middle" in indicators:
                calculated_indicators["bb_middle"] = sma_20

            if "bb_upper" in indicators:
                calculated_indicators["bb_upper"] = sma_20 + (std_20 * 2)

            if "bb_lower" in indicators:
                calculated_indicators["bb_lower"] = sma_20 - (std_20 * 2)

            if "bb_width" in indicators:
                calculated_indicators["bb_width"] = (sma_20 + (std_20 * 2) - (sma_20 - (std_20 * 2))) / sma_20

        # Volatility
        if "volatility_20" in indicators:
            calculated_indicators["volatility_20"] = df["close"].pct_change().rolling(window=20).std() * (252**0.5)

        # Returns
        if "returns" in indicators:
            calculated_indicators["returns"] = df["close"].pct_change()

        if "cumulative_returns" in indicators:
            calculated_indicators["cumulative_returns"] = (1 + df["close"].pct_change()).cumprod() - 1

        # Combine all calculated indicators into a DataFrame
        if calculated_indicators:
            result_df = pd.DataFrame(calculated_indicators, index=df.index)
            return result_df

        return pd.DataFrame()

    def _store_metrics_data(self, ticker: str, timeframe: str, metrics_df: pd.DataFrame):
        """
        Store calculated metrics data in InfluxDB.

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe for the metrics
            metrics_df: DataFrame with calculated metrics
        """
        if metrics_df.empty:
            return

        points = []

        for timestamp, row in metrics_df.iterrows():
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)

            point = Point("metrics").tag("ticker", ticker).tag("timeframe", timeframe)

            # Add all metric fields
            for metric_name, value in row.items():
                if pd.notna(value):  # Only store non-NaN values
                    point = point.field(metric_name, float(value))

            point = point.time(timestamp, WritePrecision.NS)
            points.append(point)

        if points:
            self.write_api.write(bucket=self.bucket, org=self.org, record=points)
            import time

            time.sleep(0.1)  # Small delay to ensure write completes

    def get_metrics(
        self,
        ticker: str,
        timeframe: str = "1d",
        metrics: Optional[list] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Retrieve calculated metrics for a ticker.

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe for the metrics
            metrics: List of specific metrics to retrieve. If None, retrieves all.
            start: Start date filter
            end: End date filter

        Returns:
            DataFrame with metrics data
        """
        # Build field selection
        field_filter = ""
        if metrics:
            field_filters = [f'r["_field"] == "{metric}"' for metric in metrics]
            field_filter = f'|> filter(fn: (r) => {" or ".join(field_filters)})'

        # Build Flux query - use a wider default range if no dates provided
        if start is None and end is None:
            # Query all data if no date range specified
            query = f"""
            from(bucket: "{self.bucket}")
            |> range(start: 0, stop: now())
            |> filter(fn: (r) => r["_measurement"] == "metrics")
            |> filter(fn: (r) => r["ticker"] == "{ticker}")
            |> filter(fn: (r) => r["timeframe"] == "{timeframe}")
            {field_filter}
            |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> sort(columns: ["_time"])
            """
        else:
            query = f"""
            from(bucket: "{self.bucket}")
            |> range(start: {start or "-10y"}, stop: {end or "now()"})
            |> filter(fn: (r) => r["_measurement"] == "metrics")
            |> filter(fn: (r) => r["ticker"] == "{ticker}")
            |> filter(fn: (r) => r["timeframe"] == "{timeframe}")
            {field_filter}
            |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> sort(columns: ["_time"])
            """

        try:
            result = self.query_api.query(query, org=self.org)
            records = []

            for table in result:
                for record in table.records:
                    record_data = {"datetime": record.get_time()}

                    # Add all metric values, but filter out metadata columns
                    for key, value in record.values.items():
                        if key not in [
                            "_time",
                            "_measurement",
                            "ticker",
                            "timeframe",
                            "result",
                            "table",
                            "_start",
                            "_stop",
                        ]:
                            record_data[key] = value

                    records.append(record_data)

            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
            df = df.sort_index()

            return df

        except Exception as e:
            print(f"Error querying metrics: {e}")
            return pd.DataFrame()

    def has_metrics(
        self,
        ticker: str,
        timeframe: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> bool:
        """
        Check if metrics exist for a ticker and timeframe.

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe for the metrics
            start: Start date filter
            end: End date filter

        Returns:
            True if metrics exist
        """
        df = self.get_metrics(ticker, timeframe, start=start, end=end)
        return not df.empty

    def get_available_metrics(self, ticker: str, timeframe: str = "1d") -> list:
        """
        Get list of available metrics for a ticker and timeframe.

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe for the metrics

        Returns:
            List of available metric names
        """
        # Get metrics data and extract column names
        metrics_df = self.get_metrics(ticker, timeframe)

        if metrics_df.empty:
            return []

        # Filter out any metadata columns that might remain
        metadata_columns = ["datetime", "result", "table", "_start", "_stop"]
        available_metrics = [col for col in metrics_df.columns if col not in metadata_columns]

        return sorted(available_metrics)

    def populate_metrics_table(
        self,
        tickers: list,
        timeframe: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        indicators: Optional[list] = None,
    ):
        """
        Populate metrics table for multiple tickers from their price data.

        Args:
            tickers: List of ticker symbols
            timeframe: Timeframe for calculations
            start: Start date for calculation period
            end: End date for calculation period
            indicators: List of indicators to calculate
        """
        print(f"📊 Populating metrics table for {len(tickers)} tickers...")

        successful_tickers = 0

        for ticker in tickers:
            try:
                print(f"   Processing {ticker}...")
                self.calculate_and_store_metrics(
                    ticker=ticker, timeframe=timeframe, start=start, end=end, indicators=indicators
                )
                successful_tickers += 1

            except Exception as e:
                print(f"   ❌ Error processing {ticker}: {e}")

        print(f"✅ Successfully processed {successful_tickers}/{len(tickers)} tickers")

    def get_metrics_summary(
        self,
        ticker: str,
        timeframe: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get summary statistics for metrics data.

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe for the metrics
            start: Start date filter
            end: End date filter

        Returns:
            Dictionary with metrics summary
        """
        metrics_df = self.get_metrics(ticker, timeframe, start=start, end=end)

        if metrics_df.empty:
            return {"error": "No metrics data available"}

        summary = {
            "ticker": ticker,
            "timeframe": timeframe,
            "data_points": len(metrics_df),
            "date_range": {
                "start": (metrics_df.index.min().strftime("%Y-%m-%d") if len(metrics_df) > 0 else None),
                "end": metrics_df.index.max().strftime("%Y-%m-%d") if len(metrics_df) > 0 else None,
            },
            "available_metrics": list(metrics_df.columns),
            "metrics_stats": {},
        }

        # Calculate basic stats for each metric
        for column in metrics_df.columns:
            if pd.api.types.is_numeric_dtype(metrics_df[column]):
                summary["metrics_stats"][column] = {
                    "mean": metrics_df[column].mean(),
                    "std": metrics_df[column].std(),
                    "min": metrics_df[column].min(),
                    "max": metrics_df[column].max(),
                    "last_value": metrics_df[column].iloc[-1] if len(metrics_df) > 0 else None,
                }

        return summary

    def clear_metrics_data(self, ticker: str, timeframe: str = "1d"):
        """
        Clear all metrics data for a specific ticker and timeframe.

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe for the metrics
        """
        delete_api = self.client.delete_api()

        # Delete all metrics for this ticker and timeframe
        try:
            delete_api.delete(
                start="1970-01-01T00:00:00Z",
                stop="2030-01-01T00:00:00Z",
                predicate=f'_measurement="metrics" AND ticker="{ticker}" AND timeframe="{timeframe}"',
                bucket=self.bucket,
                org=self.org,
            )
            print(f"✅ Cleared metrics data for {ticker} ({timeframe})")
        except Exception as e:
            print(f"Error clearing metrics data: {e}")
