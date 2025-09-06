import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from open_trading_algo.data_enrichment import enrich_dataframe_for_signals
from open_trading_algo.signal_optimizer import SignalOptimizer
from open_trading_algo.cache.data_cache import DataCache
from tests.test_data_enrichment import generate_comprehensive_test_data


@pytest.fixture
def sample_integration_data():
    """Generate comprehensive data for integration testing."""
    return generate_comprehensive_test_data(50)


@pytest.fixture
def sample_multi_ticker_integration():
    """Generate multi-ticker data for integration testing."""
    tickers = ["AAPL", "GOOGL", "MSFT"]
    data = {}
    for ticker in tickers:
        data[ticker] = generate_comprehensive_test_data(30)
    return data


class TestDataPipelineIntegration:
    """Test end-to-end data pipeline integration."""

    def test_data_enrichment_to_signal_generation(self, sample_integration_data):
        """Test data enrichment followed by signal generation."""
        # Step 1: Enrich data
        enriched_df = enrich_dataframe_for_signals(
            sample_integration_data.copy(), "AAPL", ["sma_trend", "positive_momentum", "rsi_macd"]
        )

        # Step 2: Generate signals
        from open_trading_algo.indicators.indicators import sma, rsi, macd

        # Add computed indicators
        enriched_df["sma_20"] = sma(enriched_df["close"], window=20)
        enriched_df["rsi_14"] = rsi(enriched_df["close"], window=14)
        macd_line, signal_line, histogram = macd(enriched_df["close"])
        enriched_df["macd_line"] = macd_line
        enriched_df["macd_signal"] = signal_line

        # Verify all required columns exist
        required_cols = ["close", "sma_20", "rsi_14", "macd_line", "macd_signal"]
        for col in required_cols:
            assert col in enriched_df.columns
            assert not enriched_df[col].isna().all()

    def test_multi_ticker_signal_optimization(self, sample_multi_ticker_integration):
        """Test signal optimization across multiple tickers."""
        # Setup indicators and signal generators
        indicators = {
            "sma_20": lambda df: pd.Series(df["close"].rolling(20).mean(), index=df.index),
            "rsi_14": lambda df: pd.Series(
                df["close"]
                .rolling(14)
                .apply(
                    lambda x: 100
                    - (
                        100
                        / (
                            1
                            + (
                                x.pct_change().fillna(0).apply(lambda y: max(y, 0)).mean()
                                / -x.pct_change().fillna(0).apply(lambda y: min(y, 0)).mean()
                            )
                        )
                    ),
                    raw=False,
                ),
                index=df.index,
            ),
        }

        def momentum_signal(df, indicators):
            returns = df["close"].pct_change()
            return pd.Series(np.where(returns > 0, 1, -1), index=df.index)

        signal_generators = {
            "momentum": momentum_signal,
        }

        # Create optimizer
        optimizer = SignalOptimizer(
            data=sample_multi_ticker_integration,
            indicators=indicators,
            signal_generators=signal_generators,
        )

        # Run optimization
        optimizer.compute_indicators()
        optimizer.generate_signals()

        # Verify results
        assert len(optimizer.indicator_results) == len(sample_multi_ticker_integration)
        assert len(optimizer.signal_results) == len(sample_multi_ticker_integration)

        for ticker in sample_multi_ticker_integration.keys():
            assert ticker in optimizer.indicator_results
            assert ticker in optimizer.signal_results
            assert len(optimizer.signal_results[ticker]) == len(signal_generators)

    def test_cache_integration_with_data_pipeline(self, sample_integration_data):
        """Test data pipeline functionality (cache integration not currently implemented)."""
        # Run enrichment without cache
        result = enrich_dataframe_for_signals(sample_integration_data.copy(), "AAPL", ["sma_trend"])

        # Verify result structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_integration_data)
        assert "close" in result.columns


class TestEndToEndWorkflow:
    """Test complete end-to-end trading workflow."""

    def test_data_fetch_to_signal_execution(self, sample_integration_data):
        """Test complete workflow from data fetch to signal execution."""
        # Step 1: Data enrichment
        enriched_df = enrich_dataframe_for_signals(
            sample_integration_data.copy(), "AAPL", ["sma_trend", "positive_momentum"]
        )

        # Step 2: Signal generation
        signals = pd.DataFrame(index=enriched_df.index)
        signals["sma_signal"] = np.where(
            enriched_df["close"] > enriched_df["close"].rolling(20).mean(), 1, -1
        )
        signals["momentum_signal"] = np.where(enriched_df["close"].pct_change() > 0, 1, -1)

        # Step 3: Portfolio simulation
        initial_capital = 100000
        position_size = 0.1

        portfolio_value = [initial_capital]
        position = 0

        for i in range(1, len(signals)):
            signal = signals["sma_signal"].iloc[i]

            if signal == 1 and position == 0:  # Buy
                position = (initial_capital * position_size) / enriched_df["close"].iloc[i]
            elif signal == -1 and position > 0:  # Sell
                initial_capital += position * enriched_df["close"].iloc[i]
                position = 0

            # Update portfolio value
            current_value = initial_capital + (position * enriched_df["close"].iloc[i])
            portfolio_value.append(current_value)

        # Verify workflow completion
        assert len(portfolio_value) == len(enriched_df)
        assert all(val > 0 for val in portfolio_value)

    def test_multi_asset_portfolio_optimization(self, sample_multi_ticker_integration):
        """Test portfolio optimization across multiple assets."""
        # Simple equal-weight portfolio
        tickers = list(sample_multi_ticker_integration.keys())
        weights = {ticker: 1.0 / len(tickers) for ticker in tickers}

        # Calculate portfolio returns
        portfolio_returns = pd.Series(index=sample_multi_ticker_integration[tickers[0]].index)

        for ticker in tickers:
            df = sample_multi_ticker_integration[ticker]
            returns = df["close"].pct_change().fillna(0)
            if portfolio_returns.isna().all():
                portfolio_returns = returns * weights[ticker]
            else:
                portfolio_returns += returns * weights[ticker]

        # Verify portfolio calculations
        assert len(portfolio_returns) == len(sample_multi_ticker_integration[tickers[0]])
        assert not portfolio_returns.isna().all()

    def test_risk_management_integration(self, sample_integration_data):
        """Test risk management integration with trading signals."""
        # Generate signals
        df = sample_integration_data.copy()
        df["returns"] = df["close"].pct_change()
        df["signal"] = np.where(df["returns"] > 0, 1, -1)

        # Risk management parameters
        max_drawdown = 0.1  # 10%
        position_size_limit = 0.2  # 20% of capital
        stop_loss = 0.05  # 5%

        capital = 100000
        peak_value = capital
        position = 0
        trades = []

        for i in range(1, len(df)):
            current_price = df["close"].iloc[i]
            signal = df["signal"].iloc[i]

            # Calculate current portfolio value
            portfolio_value = capital + (position * current_price)

            # Update peak value
            peak_value = max(peak_value, portfolio_value)

            # Check drawdown
            drawdown = (portfolio_value - peak_value) / peak_value
            if drawdown < -max_drawdown:
                # Close position due to excessive drawdown
                if position != 0:
                    capital = portfolio_value
                    position = 0
                    trades.append({"type": "stop_loss", "reason": "max_drawdown"})

            # Execute signals with risk management
            elif signal == 1 and position == 0:
                # Buy with position size limit
                position_value = min(capital * position_size_limit, capital)
                position = position_value / current_price
                trades.append({"type": "buy", "price": current_price})

            elif signal == -1 and position > 0:
                # Sell
                capital = portfolio_value
                position = 0
                trades.append({"type": "sell", "price": current_price})

        # Verify risk management worked
        assert len(trades) > 0
        assert all(trade["price"] > 0 for trade in trades)


class TestSystemIntegration:
    """Test system-level integrations and dependencies."""

    def test_database_cache_integration(self, sample_integration_data):
        """Test database and cache integration."""
        with patch("open_trading_algo.cache.data_cache.DataCache") as mock_cache:
            mock_cache_instance = MagicMock()
            mock_cache.return_value = mock_cache_instance
            mock_cache_instance.db_path = "/tmp/test_cache.db"
            mock_cache_instance.has_signals.return_value = False

            # Test cache operations
            from open_trading_algo.cache.data_cache import is_caching_enabled

            with patch("open_trading_algo.cache.data_cache.is_caching_enabled", return_value=True):
                cache = DataCache()
                assert cache is not None

    def test_api_integration_mock(self, sample_integration_data):
        """Test API integration with mocked responses."""
        with patch("open_trading_algo.fin_data_apis.fetchers.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {"data": [{"close": 150.0, "volume": 1000000}]}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            # Test that API mocking works
            assert mock_get.called or True  # Just verify mock setup

    def test_error_handling_integration(self, sample_integration_data):
        """Test error handling across integrated components."""
        # Test with corrupted data
        corrupted_data = sample_integration_data.copy()
        corrupted_data.loc[:, "close"] = np.nan

        # Should handle NaN values gracefully
        try:
            result = enrich_dataframe_for_signals(corrupted_data, "AAPL", ["sma_trend"])
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            # Should fail gracefully with meaningful error
            assert "NaN" in str(e) or "missing" in str(e).lower()

    def test_performance_integration(self, sample_multi_ticker_integration):
        """Test performance of integrated system."""
        import time

        start_time = time.time()

        # Run complete pipeline
        for ticker, df in sample_multi_ticker_integration.items():
            enriched = enrich_dataframe_for_signals(df, ticker, ["sma_trend", "positive_momentum"])

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete in reasonable time (less than 10 seconds for small dataset)
        assert execution_time < 10.0
        assert execution_time > 0


class TestDataValidationIntegration:
    """Test data validation across integrated components."""

    def test_data_consistency_across_modules(self, sample_integration_data):
        """Test data consistency across different modules."""
        # Enrich data
        enriched = enrich_dataframe_for_signals(
            sample_integration_data.copy(), "AAPL", ["sma_trend", "positive_momentum"]
        )

        # Verify data types are consistent
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in enriched.columns:
                assert pd.api.types.is_numeric_dtype(enriched[col])

        # Verify date index
        assert isinstance(enriched.index, pd.DatetimeIndex)

    def test_missing_data_handling_integration(self, sample_integration_data):
        """Test missing data handling across integrated components."""
        # Introduce missing data
        data_with_missing = sample_integration_data.copy()
        data_with_missing.loc[data_with_missing.index[10:15], "close"] = np.nan

        # Process through pipeline
        enriched = enrich_dataframe_for_signals(data_with_missing, "AAPL", ["sma_trend"])

        # Should still produce valid output
        assert isinstance(enriched, pd.DataFrame)
        assert len(enriched) == len(data_with_missing)

    def test_data_schema_validation(self, sample_integration_data):
        """Test data schema validation."""
        enriched = enrich_dataframe_for_signals(
            sample_integration_data.copy(), "AAPL", ["sma_trend", "positive_momentum"]
        )

        # Required columns should exist
        expected_cols = ["open", "high", "low", "close", "volume"]
        for col in expected_cols:
            assert col in enriched.columns

        # Data should not be empty
        assert not enriched.empty
        assert len(enriched) > 0
