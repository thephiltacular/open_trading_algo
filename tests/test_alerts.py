import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from open_trading_algo.alerts.alerts import (
    extract_ticker,
    is_positive_alert,
    is_negative_alert,
    TickerAlertCounts,
)


@pytest.fixture
def sample_alert_data():
    """Sample alert data for testing."""
    return [
        "ALRT:AAPL, Positive momentum signal detected",
        "ALRT:GOOGL, Negative earnings revision",
        "ALRT:MSFT, Strong buy signal",
        "ALRT:TSLA, Oversold condition",
        "ALRT:NFLX, Bearish divergence",
    ]


@pytest.fixture
def sample_alert_dataframe():
    """Sample alert DataFrame for testing."""
    data = {
        "description": [
            "Positive momentum signal detected",
            "Negative earnings revision",
            "Strong buy signal",
            "Oversold condition",
            "Bearish divergence",
        ],
        "timestamp": pd.date_range("2023-01-01", periods=5, freq="H"),
        "ticker": ["AAPL", "GOOGL", "MSFT", "TSLA", "NFLX"],
    }
    return pd.DataFrame(data)


class TestAlertParsing:
    """Test alert parsing and extraction functions."""

    def test_extract_ticker_valid(self):
        """Test ticker extraction from valid alert strings."""
        test_cases = [
            ("ALRT:AAPL, some description", "AAPL"),
            ("ALRT:GOOGL, another alert", "GOOGL"),
            ("ALRT:TSLA, test alert", "TSLA"),
        ]

        for alert_string, expected_ticker in test_cases:
            result = extract_ticker(alert_string)
            assert result == expected_ticker

    def test_extract_ticker_edge_cases(self):
        """Test ticker extraction edge cases."""
        edge_cases = [
            ("ALRT:AAPL", "AAPL"),  # No comma
            ("ALRT:", ""),  # Empty ticker
            ("ALRT:AAPL,", "AAPL"),  # Trailing comma
            ("ALRT:SPY500, market alert", "SPY500"),  # Multi-character ticker
            ("", ""),  # Empty string
            ("ALRT:AAPL, description with, comma", "AAPL"),  # Multiple commas
        ]

        for alert_string, expected_ticker in edge_cases:
            result = extract_ticker(alert_string)
            assert result == expected_ticker

    def test_is_positive_alert(self):
        """Test positive alert detection."""
        positive_cases = [
            "Positive momentum signal detected",
            "Strong buy signal",
            "Bullish trend confirmed",
            "positive earnings surprise",
            "POSITIVE technical signal",
        ]

        negative_cases = [
            "Negative earnings revision",
            "Bearish divergence",
            "Sell signal triggered",
            "negative momentum",
        ]

        for description in positive_cases:
            assert is_positive_alert(description) is True

        for description in negative_cases:
            assert is_positive_alert(description) is False

    def test_is_negative_alert(self):
        """Test negative alert detection."""
        negative_cases = [
            "Negative earnings revision",
            "Bearish divergence",
            "Sell signal triggered",
            "negative momentum",
            "NEGATIVE technical signal",
        ]

        positive_cases = [
            "Positive momentum signal detected",
            "Strong buy signal",
            "Bullish trend confirmed",
        ]

        for description in negative_cases:
            assert is_negative_alert(description) is True

        for description in positive_cases:
            assert is_negative_alert(description) is False

    def test_alert_detection_none_input(self):
        """Test alert detection with None input."""
        assert is_positive_alert(None) is False
        assert is_negative_alert(None) is False


class TestTickerAlertCounts:
    """Test TickerAlertCounts class functionality."""

    def test_ticker_alert_counts_initialization(self):
        """Test TickerAlertCounts initialization."""
        counts = TickerAlertCounts()

        assert counts.positive == 0
        assert counts.negative == 0
        assert counts.latest_time is None
        assert counts.latest_dt is None

    def test_ticker_alert_counts_custom_initialization(self):
        """Test TickerAlertCounts with custom values."""
        latest_time = "2023-01-01T10:00:00Z"
        counts = TickerAlertCounts(positive=5, negative=3, latest_time=latest_time)

        assert counts.positive == 5
        assert counts.negative == 3
        assert counts.latest_time == latest_time

    def test_ticker_alert_counts_updates(self):
        """Test updating alert counts."""
        counts = TickerAlertCounts()

        # Simulate positive alert
        counts.positive += 1
        counts.latest_time = "2023-01-01T10:00:00Z"

        # Simulate negative alert
        counts.negative += 1
        counts.latest_time = "2023-01-01T11:00:00Z"

        assert counts.positive == 1
        assert counts.negative == 1
        assert counts.latest_time == "2023-01-01T11:00:00Z"


class TestAlertProcessing:
    """Test alert processing and aggregation."""

    def test_alert_aggregation_by_ticker(self, sample_alert_data):
        """Test aggregating alerts by ticker."""
        ticker_counts = {}

        for alert in sample_alert_data:
            ticker = extract_ticker(alert)
            if ticker not in ticker_counts:
                ticker_counts[ticker] = TickerAlertCounts()

            if is_positive_alert(alert):
                ticker_counts[ticker].positive += 1
            elif is_negative_alert(alert):
                ticker_counts[ticker].negative += 1

        assert len(ticker_counts) == 5  # All unique tickers
        assert all(isinstance(counts, TickerAlertCounts) for counts in ticker_counts.values())

    def test_alert_summary_statistics(self, sample_alert_dataframe):
        """Test alert summary statistics."""
        df = sample_alert_dataframe

        # Count positive vs negative alerts
        positive_count = df["description"].apply(is_positive_alert).sum()
        negative_count = df["description"].apply(is_negative_alert).sum()

        assert positive_count == 2  # AAPL, MSFT
        assert negative_count == 2  # GOOGL, NFLX

    def test_alert_temporal_analysis(self, sample_alert_dataframe):
        """Test temporal analysis of alerts."""
        df = sample_alert_dataframe

        # Test time-based grouping
        hourly_counts = df.groupby(df["timestamp"].dt.hour).size()

        assert len(hourly_counts) > 0
        assert all(count >= 0 for count in hourly_counts)

    def test_alert_data_validation(self, sample_alert_dataframe):
        """Test alert data validation."""
        df = sample_alert_dataframe

        # Check required columns exist
        required_columns = ["description", "timestamp", "ticker"]
        for col in required_columns:
            assert col in df.columns

        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        assert pd.api.types.is_object_dtype(df["description"])
        assert pd.api.types.is_object_dtype(df["ticker"])

    def test_alert_filtering(self, sample_alert_dataframe):
        """Test alert filtering by various criteria."""
        df = sample_alert_dataframe

        # Filter positive alerts
        positive_alerts = df[df["description"].apply(is_positive_alert)]
        assert len(positive_alerts) == 2

        # Filter by ticker
        aapl_alerts = df[df["ticker"] == "AAPL"]
        assert len(aapl_alerts) == 1

        # Filter by time range
        recent_alerts = df[df["timestamp"] > df["timestamp"].min()]
        assert len(recent_alerts) == 4  # All alerts except the first (minimum timestamp)


class TestAlertEdgeCases:
    """Test alert processing edge cases."""

    def test_empty_alert_list(self):
        """Test processing empty alert list."""
        alerts = []
        ticker_counts = {}

        for alert in alerts:
            ticker = extract_ticker(alert)
            if ticker and ticker not in ticker_counts:
                ticker_counts[ticker] = TickerAlertCounts()

        assert len(ticker_counts) == 0

    def test_malformed_alert_strings(self):
        """Test handling malformed alert strings."""
        malformed_alerts = [
            "ALRT:",  # Missing ticker
            "ALRT:AAPL",  # Missing description
            "INVALID:AAPL, test",  # Wrong prefix
            "ALRT:AAPL, ",  # Empty description
        ]

        for alert in malformed_alerts:
            ticker = extract_ticker(alert)
            assert isinstance(ticker, str)  # Should not crash

    def test_unicode_alert_descriptions(self):
        """Test alerts with unicode characters."""
        unicode_alerts = [
            "ALRT:AAPL, Café bullish signal 📈",
            "ALRT:GOOGL, naïve bearish signal 📉",
            "ALRT:TSLA, naïve momentum signal 🚀",
        ]

        for alert in unicode_alerts:
            ticker = extract_ticker(alert)
            assert len(ticker) > 0
            assert is_positive_alert(alert) or is_negative_alert(alert)

    def test_case_insensitive_alert_detection(self):
        """Test case insensitive alert detection."""
        mixed_case_alerts = [
            "POSITIVE signal detected",
            "negative SIGNAL detected",
            "Positive Momentum",
            "NEGATIVE divergence",
        ]

        for alert in mixed_case_alerts:
            # Should detect regardless of case
            has_sentiment = is_positive_alert(alert) or is_negative_alert(alert)
            assert has_sentiment
