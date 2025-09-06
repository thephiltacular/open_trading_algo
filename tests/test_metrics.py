"""Tests for trading metrics functions."""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from open_trading_algo.indicators.metrics import (
    _compute_column_metrics,
    compute_volume_10d_avg,
    compute_volume_30d_avg,
    compute_volume_60d_avg,
    compute_volume_90d_avg,
    compute_volatility_ratio,
    compute_trend_strength,
    compute_volume_price_trend,
    compute_max_drawdown,
    compute_vwap,
    compute_beta,
    compute_alpha,
    compute_price_acceleration,
    compute_seasonal_strength,
    compute_monthly_returns,
    compute_correlation_matrix,
    compute_autocorrelation,
    compute_pivot_points,
    compute_fibonacci_levels,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_calmar_ratio,
    compute_win_rate,
    compute_profit_factor,
)


class TestMetrics:
    """Test suite for trading metrics functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        # Generate realistic price data
        price_changes = np.random.normal(0.001, 0.02, 100)
        prices = 100 * np.exp(np.cumsum(price_changes))

        # Generate OHLC data
        highs = prices * (1 + np.random.uniform(0, 0.02, 100))
        lows = prices * (1 - np.random.uniform(0, 0.02, 100))
        opens = prices * (1 + np.random.normal(0, 0.01, 100))
        volumes = np.random.uniform(100000, 1000000, 100)

        return pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": prices, "volume": volumes},
            index=dates,
        )

    def test_compute_column_metrics(self, sample_data):
        """Test generic column metrics computation."""
        result = _compute_column_metrics(sample_data, "volume", 10)

        assert len(result) == len(sample_data)
        assert result.dtype == float
        assert not result.isna().all()

        # Test missing column
        result_missing = _compute_column_metrics(sample_data, "nonexistent", 10)
        assert result_missing.isna().all()

    def test_volume_averages(self, sample_data):
        """Test volume average computations."""
        vol_10d = compute_volume_10d_avg(sample_data)
        vol_30d = compute_volume_30d_avg(sample_data)
        vol_60d = compute_volume_60d_avg(sample_data)
        vol_90d = compute_volume_90d_avg(sample_data)

        assert len(vol_10d) == len(sample_data)
        assert len(vol_30d) == len(sample_data)
        assert len(vol_60d) == len(sample_data)
        assert len(vol_90d) == len(sample_data)

        # 10-day average should be more responsive than 90-day
        assert vol_10d.std() > vol_90d.std()

    def test_volatility_ratio(self, sample_data):
        """Test volatility ratio computation."""
        ratio = compute_volatility_ratio(sample_data)

        assert len(ratio) == len(sample_data)
        assert ratio.dtype == float

        # Most values should be positive (ratios)
        assert (ratio > 0).sum() > len(ratio) * 0.8

    def test_trend_strength(self, sample_data):
        """Test trend strength (ADX) computation."""
        adx = compute_trend_strength(sample_data)

        assert len(adx) == len(sample_data)
        assert adx.dtype == float

        # ADX should be between 0 and 100
        assert adx.min() >= 0
        assert adx.max() <= 100

    def test_volume_price_trend(self, sample_data):
        """Test volume price trend computation."""
        vpt = compute_volume_price_trend(sample_data)

        assert len(vpt) == len(sample_data)
        assert vpt.dtype == float
        assert not vpt.isna().all()

    def test_max_drawdown(self, sample_data):
        """Test maximum drawdown computation."""
        mdd = compute_max_drawdown(sample_data)
        mdd_rolling = compute_max_drawdown(sample_data, window=30)

        assert len(mdd) == len(sample_data)
        assert len(mdd_rolling) == len(sample_data)

        # Drawdowns should be non-negative
        assert (mdd >= 0).all()
        assert (mdd_rolling >= 0).all()

    def test_vwap(self, sample_data):
        """Test VWAP computation."""
        vwap = compute_vwap(sample_data)

        assert len(vwap) == len(sample_data)
        assert vwap.dtype == float
        assert not vwap.isna().all()

        # VWAP should be close to typical price
        typical_price = (sample_data["high"] + sample_data["low"] + sample_data["close"]) / 3
        assert abs(vwap.iloc[-1] - typical_price.iloc[-1]) < typical_price.iloc[-1] * 0.1

    def test_beta(self, sample_data):
        """Test beta computation."""
        beta = compute_beta(sample_data)

        assert len(beta) == len(sample_data)
        assert beta.dtype == float

        # Beta should be reasonable (not extreme)
        assert beta.std() < 10

    def test_alpha(self, sample_data):
        """Test alpha computation."""
        alpha = compute_alpha(sample_data)

        assert len(alpha) == len(sample_data)
        assert alpha.dtype == float

    def test_price_acceleration(self, sample_data):
        """Test price acceleration computation."""
        accel = compute_price_acceleration(sample_data)

        assert len(accel) == len(sample_data)
        assert accel.dtype == float

    def test_seasonal_strength(self, sample_data):
        """Test seasonal strength computation."""
        seasonal = compute_seasonal_strength(sample_data)

        assert len(seasonal) == len(sample_data)
        assert seasonal.dtype == float

        # Check that non-NaN values are non-negative
        valid_seasonal = seasonal.dropna()
        assert len(valid_seasonal) > 0  # Should have some valid values
        assert (valid_seasonal >= 0).all()

    def test_monthly_returns(self, sample_data):
        """Test monthly returns computation."""
        monthly = compute_monthly_returns(sample_data)

        assert len(monthly) <= 12  # Max 12 months
        assert monthly.dtype == float

    def test_correlation_matrix(self, sample_data):
        """Test correlation matrix computation."""
        # Create multi-asset data
        multi_asset = pd.DataFrame(
            {
                "asset1": sample_data["close"],
                "asset2": sample_data["close"] * (1 + np.random.normal(0, 0.1, len(sample_data))),
                "asset3": sample_data["close"] * (1 + np.random.normal(0, 0.1, len(sample_data))),
            }
        )

        corr_matrix = compute_correlation_matrix(multi_asset)

        assert not corr_matrix.empty
        # Pandas rolling correlation returns MultiIndex DataFrame
        assert corr_matrix.shape[1] == 3  # Number of assets
        assert isinstance(corr_matrix.index, pd.MultiIndex)

    def test_autocorrelation(self, sample_data):
        """Test autocorrelation computation."""
        autocorr = compute_autocorrelation(sample_data)

        assert len(autocorr) == len(sample_data)
        assert autocorr.dtype == float

        # Autocorrelation should be between -1 and 1
        assert autocorr.min() >= -1
        assert autocorr.max() <= 1

    def test_pivot_points(self, sample_data):
        """Test pivot points computation."""
        pivots = compute_pivot_points(sample_data)

        assert not pivots.empty
        assert "pivot" in pivots.columns
        assert "s1" in pivots.columns
        assert "r1" in pivots.columns

        # Pivot should be between low and high
        assert (pivots["pivot"] >= sample_data["low"]).all()
        assert (pivots["pivot"] <= sample_data["high"]).all()

    def test_fibonacci_levels(self, sample_data):
        """Test Fibonacci levels computation."""
        fib_levels = compute_fibonacci_levels(sample_data)

        assert not fib_levels.empty
        assert "fib_0" in fib_levels.columns
        assert "fib_1" in fib_levels.columns

        # Fibonacci levels should be properly ordered
        assert (fib_levels["fib_0"] <= fib_levels["fib_0.236"]).all()
        assert (fib_levels["fib_0.618"] <= fib_levels["fib_1"]).all()

    def test_sharpe_ratio(self, sample_data):
        """Test Sharpe ratio computation."""
        sharpe = compute_sharpe_ratio(sample_data)

        assert len(sharpe) == len(sample_data)
        assert sharpe.dtype == float

    def test_sortino_ratio(self, sample_data):
        """Test Sortino ratio computation."""
        sortino = compute_sortino_ratio(sample_data)

        assert len(sortino) == len(sample_data)
        assert sortino.dtype == float

    def test_calmar_ratio(self, sample_data):
        """Test Calmar ratio computation."""
        calmar = compute_calmar_ratio(sample_data)

        assert len(calmar) == len(sample_data)
        assert calmar.dtype == float

    def test_win_rate(self, sample_data):
        """Test win rate computation."""
        # Create binary signal data
        signals = pd.DataFrame({"signal": np.random.choice([0, 1], len(sample_data))})

        win_rate = compute_win_rate(signals)

        assert len(win_rate) == len(sample_data)
        assert win_rate.dtype == float

        # Win rate should be between 0 and 1
        assert win_rate.min() >= 0
        assert win_rate.max() <= 1

    def test_profit_factor(self, sample_data):
        """Test profit factor computation."""
        # Create returns data
        returns = pd.DataFrame({"returns": np.random.normal(0.001, 0.02, len(sample_data))})

        pf = compute_profit_factor(returns)

        assert len(pf) == len(sample_data)
        assert pf.dtype == float

        # Profit factor should be non-negative
        assert (pf >= 0).all()

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        empty_df = pd.DataFrame()

        # Test with empty dataframe
        result = compute_volume_10d_avg(empty_df)
        assert len(result) == 0

        # Test with missing required columns
        df_no_close = pd.DataFrame({"volume": [1, 2, 3]})
        result = compute_volatility_ratio(df_no_close)
        assert result.isna().all()

        # Test with single data point
        single_point = pd.DataFrame({"close": [100], "volume": [1000]})
        result = compute_volume_10d_avg(single_point)
        assert len(result) == 1
        assert result.iloc[0] == 1000  # Should return the single value


if __name__ == "__main__":
    pytest.main([__file__])
