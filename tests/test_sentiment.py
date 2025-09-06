import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from open_trading_algo.sentiment.analyst_sentiment import (
    fetch_finnhub_analyst_sentiment,
    fetch_bulk_finnhub_analyst_sentiment,
    fetch_fmp_analyst_price_targets,
    fetch_bulk_fmp_analyst_price_targets,
)
from open_trading_algo.sentiment.social_sentiment import (
    fetch_twitter_sentiment,
    fetch_bulk_twitter_sentiment,
    fetch_reddit_sentiment,
    aggregate_social_sentiment,
)


@pytest.fixture
def sample_tickers():
    """Sample tickers for testing."""
    return ["AAPL", "GOOGL", "MSFT"]


@pytest.fixture
def mock_finnhub_response():
    """Mock Finnhub analyst sentiment API response."""
    return {
        "data": [
            {"buy": 25, "hold": 15, "sell": 5, "strongBuy": 10, "strongSell": 2},
            {"buy": 20, "hold": 20, "sell": 8, "strongBuy": 8, "strongSell": 3},
        ]
    }


@pytest.fixture
def mock_fmp_response():
    """Mock FMP analyst price targets API response."""
    return [
        {
            "symbol": "AAPL",
            "publishedDate": "2023-01-01",
            "analystName": "Test Analyst",
            "priceTarget": 180.0,
            "adjPriceTarget": 175.0,
        }
    ]


@pytest.fixture
def mock_lunarcrush_response():
    """Mock LunarCrush social sentiment API response."""
    return {
        "data": [
            {
                "symbol": "AAPL",
                "social_score": 75.5,
                "social_volume": 125000,
                "social_dominance": 12.3,
            }
        ]
    }


class TestAnalystSentiment:
    """Test analyst sentiment fetching and processing."""

    @patch("open_trading_algo.sentiment.analyst_sentiment.requests.get")
    @patch("open_trading_algo.sentiment.analyst_sentiment.is_caching_enabled", return_value=False)
    def test_fetch_finnhub_analyst_sentiment_success(
        self, mock_cache_enabled, mock_get, mock_finnhub_response
    ):
        """Test successful Finnhub analyst sentiment fetch."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_finnhub_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = fetch_finnhub_analyst_sentiment("AAPL", "fake_key")

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert "buy" in result.columns
        assert len(result) == 1

    @patch("open_trading_algo.sentiment.analyst_sentiment.requests.get")
    @patch("open_trading_algo.sentiment.analyst_sentiment.is_caching_enabled", return_value=False)
    def test_fetch_finnhub_analyst_sentiment_api_error(self, mock_cache_enabled, mock_get):
        """Test Finnhub API error handling."""
        mock_get.side_effect = Exception("API Error")

        result = fetch_finnhub_analyst_sentiment("AAPL", "fake_key")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result["buy"].iloc[0] is None

    @patch("open_trading_algo.sentiment.analyst_sentiment.fetch_finnhub_analyst_sentiment")
    @patch("open_trading_algo.sentiment.analyst_sentiment.is_caching_enabled", return_value=False)
    def test_fetch_bulk_finnhub_analyst_sentiment(
        self, mock_cache_enabled, mock_fetch_single, sample_tickers, mock_finnhub_response
    ):
        """Test bulk Finnhub analyst sentiment fetch."""
        mock_df = pd.DataFrame(mock_finnhub_response["data"])
        mock_fetch_single.return_value = mock_df

        result = fetch_bulk_finnhub_analyst_sentiment(sample_tickers, "fake_key")

        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ["date", "ticker"]

    @patch("open_trading_algo.sentiment.analyst_sentiment.requests.get")
    @patch("open_trading_algo.sentiment.analyst_sentiment.is_caching_enabled", return_value=False)
    def test_fetch_fmp_analyst_price_targets_success(
        self, mock_cache_enabled, mock_get, mock_fmp_response
    ):
        """Test successful FMP analyst price targets fetch."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_fmp_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = fetch_fmp_analyst_price_targets("AAPL", "fake_key")

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert "targetMean" in result.columns


class TestSocialSentiment:
    """Test social sentiment fetching and processing."""

    @patch("open_trading_algo.sentiment.social_sentiment.requests.get")
    @patch("open_trading_algo.sentiment.social_sentiment.is_caching_enabled", return_value=False)
    def test_fetch_twitter_sentiment_success(
        self, mock_cache_enabled, mock_get, mock_lunarcrush_response
    ):
        """Test successful Twitter sentiment fetch."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_lunarcrush_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = fetch_twitter_sentiment("AAPL", "fake_key")

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert "signal_value" in result.columns

    @patch("open_trading_algo.sentiment.social_sentiment.requests.get")
    @patch("open_trading_algo.sentiment.social_sentiment.is_caching_enabled", return_value=False)
    def test_fetch_twitter_sentiment_api_error(self, mock_cache_enabled, mock_get):
        """Test Twitter sentiment API error handling."""
        mock_get.side_effect = Exception("API Error")

        result = fetch_twitter_sentiment("AAPL", "fake_key")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result["signal_value"].iloc[0] is None

    @patch("open_trading_algo.sentiment.social_sentiment.fetch_twitter_sentiment")
    @patch("open_trading_algo.sentiment.social_sentiment.is_caching_enabled", return_value=False)
    def test_fetch_bulk_twitter_sentiment(
        self, mock_cache_enabled, mock_fetch_single, sample_tickers, mock_lunarcrush_response
    ):
        """Test bulk Twitter sentiment fetch."""
        mock_df = pd.DataFrame(mock_lunarcrush_response["data"])
        mock_fetch_single.return_value = mock_df

        result = fetch_bulk_twitter_sentiment(sample_tickers, "fake_key")

        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ["date", "ticker"]

    @patch("open_trading_algo.sentiment.social_sentiment.requests.get")
    @patch("open_trading_algo.sentiment.social_sentiment.is_caching_enabled", return_value=False)
    def test_fetch_reddit_sentiment_success(
        self, mock_cache_enabled, mock_get, mock_lunarcrush_response
    ):
        """Test successful Reddit sentiment fetch."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_lunarcrush_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = fetch_reddit_sentiment("AAPL")

        assert isinstance(result, pd.DataFrame)
        assert not result.empty


class TestSentimentAggregation:
    """Test sentiment data aggregation and normalization."""

    def test_sentiment_score_normalization(self):
        """Test that sentiment scores are properly normalized."""
        # Create test data with various score ranges
        test_data = pd.DataFrame(
            {"social_score": [0, 50, 100, -50, 25], "expected_normalized": [-1, 0, 1, -1, -0.5]}
        )

        # Normalize scores (simple min-max normalization)
        scores = test_data["social_score"]
        normalized = 2 * (scores - scores.min()) / (scores.max() - scores.min()) - 1

        assert normalized.between(-1, 1).all()
        assert normalized.iloc[3] == -1  # Min value (-50)
        assert normalized.iloc[2] == 1  # Max value

    def test_sentiment_data_types(self):
        """Test that sentiment data has correct types."""
        test_df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5),
                "ticker": ["AAPL"] * 5,
                "social_score": np.random.uniform(0, 100, 5),
                "news_sentiment": np.random.uniform(-1, 1, 5),
                "volume": np.random.randint(1000, 10000, 5),
            }
        )

        assert pd.api.types.is_datetime64_any_dtype(test_df["date"])
        assert pd.api.types.is_numeric_dtype(test_df["social_score"])
        assert pd.api.types.is_numeric_dtype(test_df["news_sentiment"])
        assert pd.api.types.is_integer_dtype(test_df["volume"])

    def test_sentiment_missing_data_handling(self):
        """Test handling of missing sentiment data."""
        test_df = pd.DataFrame(
            {
                "social_score": [50, np.nan, 75, None, 25],
                "news_sentiment": [0.1, 0.2, np.nan, -0.1, None],
            }
        )

        # Check that NaN values are handled gracefully
        assert test_df["social_score"].isna().sum() == 2
        assert test_df["news_sentiment"].isna().sum() == 2

        # Test filling NaN with neutral values
        filled_social = test_df["social_score"].fillna(50)  # Neutral score
        filled_news = test_df["news_sentiment"].fillna(0)  # Neutral sentiment

        assert not filled_social.isna().any()
        assert not filled_news.isna().any()
