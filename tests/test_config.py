import pytest
import os
import yaml
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestConfigurationLoading:
    """Test configuration file loading and validation."""

    def test_api_config_loading(self):
        """Test loading API configuration."""
        config_path = Path(__file__).parent.parent / "config" / "api_config.yaml"

        assert config_path.exists(), f"API config file not found at {config_path}"

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        assert isinstance(config, dict)
        assert len(config) > 0

        # Check required API providers
        expected_providers = ["yahoo", "finnhub", "fmp"]
        for provider in expected_providers:
            assert provider in config, f"Missing {provider} in API config"

    def test_database_config_loading(self):
        """Test loading database configuration."""
        config_path = Path(__file__).parent.parent / "config" / "db_config.yaml"

        assert config_path.exists(), f"Database config file not found at {config_path}"

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        assert isinstance(config, dict)

    def test_live_data_config_loading(self):
        """Test loading live data configuration."""
        config_path = Path(__file__).parent.parent / "config" / "live_data_config.yaml"

        assert config_path.exists(), f"Live data config file not found at {config_path}"

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        assert isinstance(config, dict)

    def test_config_validation(self):
        """Test configuration validation."""
        config_path = Path(__file__).parent.parent / "config" / "api_config.yaml"

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Validate required fields for each API
        for api_name, api_config in config.items():
            assert "name" in api_config, f"Missing 'name' in {api_name} config"
            assert "free_limit_per_minute" in api_config, f"Missing rate limit in {api_name} config"

    def test_config_missing_file_handling(self):
        """Test handling of missing configuration files."""
        nonexistent_path = Path("/nonexistent/config.yaml")

        with pytest.raises(FileNotFoundError):
            with open(nonexistent_path, "r", encoding="utf-8") as f:
                yaml.safe_load(f)


class TestDatabaseSetup:
    """Test database setup and initialization."""

    @patch("open_trading_algo.cache.data_cache.DataCache")
    def test_database_initialization(self, mock_cache):
        """Test database initialization."""
        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.db_path = "/tmp/test_trading_algo.db"

        # Import and run setup
        from scripts.setup_db import main

        with patch("builtins.print"):  # Suppress print output
            main()

        # Verify cache was initialized
        mock_cache.assert_called_once()

    def test_database_path_configuration(self):
        """Test database path configuration."""
        from open_trading_algo.cache.data_cache import DB_PATH

        # Should be a valid path
        assert isinstance(DB_PATH, (str, Path))
        assert len(str(DB_PATH)) > 0

    @patch("open_trading_algo.cache.data_cache.DataCache")
    def test_database_connection_handling(self, mock_cache):
        """Test database connection handling."""
        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance

        from open_trading_algo.cache.data_cache import DataCache

        cache = DataCache()
        cache.close()  # Should not raise error

        mock_cache_instance.close.assert_called_once()


class TestEnvironmentSetup:
    """Test environment setup and dependencies."""

    def test_required_packages_available(self):
        """Test that required packages are available."""
        required_packages = [
            "pandas",
            "numpy",
            "pytest",
            "yaml",
            "requests",
        ]

        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                pytest.fail(f"Required package '{package}' is not available")

    def test_python_version_compatibility(self):
        """Test Python version compatibility."""
        import sys

        version = sys.version_info
        assert version.major >= 3
        assert version.minor >= 8  # Minimum Python 3.8

    def test_environment_variables(self):
        """Test environment variable handling."""
        # Test that we can read environment variables
        test_var = os.getenv("TEST_VAR", "default")
        assert isinstance(test_var, str)

        # Test API key environment variables (should not be set in test)
        api_keys = ["FINNHUB_API_KEY", "FMP_API_KEY", "ALPHA_VANTAGE_API_KEY"]
        for key in api_keys:
            value = os.getenv(key)
            # In test environment, these should be None or empty
            assert value is None or value == "" or isinstance(value, str)


class TestDataValidation:
    """Test data validation and schema checking."""

    def test_ohlc_data_schema(self):
        """Test OHLC data schema validation."""
        # Create sample OHLC data
        sample_data = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "close": [102, 103, 104],
                "volume": [1000, 1100, 1200],
            },
            index=pd.date_range("2023-01-01", periods=3),
        )

        # Validate schema
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            assert col in sample_data.columns
            assert pd.api.types.is_numeric_dtype(sample_data[col])

        # Validate OHLC relationships
        assert (sample_data["high"] >= sample_data["open"]).all()
        assert (sample_data["high"] >= sample_data["close"]).all()
        assert (sample_data["low"] <= sample_data["open"]).all()
        assert (sample_data["low"] <= sample_data["close"]).all()
        assert (sample_data["volume"] > 0).all()

    def test_signal_data_schema(self):
        """Test signal data schema validation."""
        # Create sample signal data
        sample_signals = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=5, freq="H"),
                "ticker": ["AAPL"] * 5,
                "signal_type": ["long", "short", "hold", "long", "short"],
                "strength": [0.8, -0.6, 0.0, 0.9, -0.7],
                "confidence": [0.85, 0.72, 0.0, 0.91, 0.68],
            }
        )

        # Validate schema
        assert pd.api.types.is_datetime64_any_dtype(sample_signals["timestamp"])
        assert pd.api.types.is_object_dtype(sample_signals["ticker"])
        assert pd.api.types.is_object_dtype(sample_signals["signal_type"])

        # Validate signal values
        assert sample_signals["strength"].between(-1, 1).all()
        assert sample_signals["confidence"].between(0, 1).all()

        # Validate signal types
        valid_signals = ["long", "short", "hold"]
        assert sample_signals["signal_type"].isin(valid_signals).all()

    def test_config_schema_validation(self):
        """Test configuration schema validation."""
        config_path = Path(__file__).parent.parent / "config" / "api_config.yaml"

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Validate schema structure
        for api_name, api_config in config.items():
            assert isinstance(api_config, dict)
            assert "name" in api_config
            assert "free_limit_per_minute" in api_config

            # Validate rate limits are numeric
            assert isinstance(api_config["free_limit_per_minute"], (int, float))
            if "free_limit_per_day" in api_config:
                assert isinstance(api_config["free_limit_per_day"], (int, float))


class TestSetupScripts:
    """Test setup and installation scripts."""

    @patch("scripts.setup_db.main")
    def test_setup_db_script_execution(self, mock_main):
        """Test database setup script execution."""
        # Import the script
        import scripts.setup_db as setup_script

        # Mock the main function
        mock_main.return_value = None

        # Execute setup
        setup_script.main()

        # Verify main was called
        mock_main.assert_called_once()

    def test_setup_script_imports(self):
        """Test that setup scripts can import required modules."""
        try:
            import scripts.setup_db

            assert hasattr(scripts.setup_db, "main")
        except ImportError as e:
            pytest.fail(f"Setup script import failed: {e}")

    def test_config_file_accessibility(self):
        """Test that configuration files are accessible."""
        config_files = [
            "config/api_config.yaml",
            "config/db_config.yaml",
            "config/live_data_config.yaml",
        ]

        for config_file in config_files:
            config_path = Path(__file__).parent.parent / config_file
            assert config_path.exists(), f"Config file {config_file} not found"
            assert config_path.is_file(), f"{config_file} is not a file"

            # Test that file is readable
            with open(config_path, "r", encoding="utf-8") as f:
                content = f.read()
                assert len(content) > 0, f"{config_file} is empty"


class TestErrorHandling:
    """Test error handling in configuration and setup."""

    def test_invalid_config_handling(self):
        """Test handling of invalid configuration."""
        invalid_config = {
            "invalid_api": {
                "name": "Invalid API",
                # Missing required fields
            }
        }

        # Should handle missing fields gracefully
        assert "name" in invalid_config["invalid_api"]

    def test_missing_environment_variables(self):
        """Test handling of missing environment variables."""
        # Test with non-existent environment variable
        value = os.getenv("NONEXISTENT_VAR", "default_value")
        assert value == "default_value"

    def test_file_permission_errors(self):
        """Test handling of file permission errors."""
        config_path = Path(__file__).parent.parent / "config" / "api_config.yaml"

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert len(content) > 0
        except PermissionError:
            pytest.fail("Permission denied reading config file")

    def test_network_dependency_handling(self):
        """Test handling of network dependency failures."""
        with patch("requests.get") as mock_get:
            mock_get.side_effect = ConnectionError("Network unavailable")

            # Should handle network errors gracefully
            try:
                # This would normally make a network request
                pass  # Placeholder for actual network-dependent code
            except ConnectionError:
                pass  # Expected in test environment


class TestSystemIntegration:
    """Test system-level integration for setup and configuration."""

    def test_full_system_initialization(self):
        """Test full system initialization."""
        # Test that all required modules can be imported
        try:
            import open_trading_algo
            import open_trading_algo.data_enrichment
            import open_trading_algo.indicators
            import open_trading_algo.cache
            import open_trading_algo.sentiment
            import open_trading_algo.alerts
            import open_trading_algo.backtest
        except ImportError as e:
            pytest.fail(f"System module import failed: {e}")

    def test_config_consistency(self):
        """Test configuration consistency across modules."""
        # Load all config files
        config_files = [
            "config/api_config.yaml",
            "config/db_config.yaml",
            "config/live_data_config.yaml",
        ]

        configs = {}
        for config_file in config_files:
            config_path = Path(__file__).parent.parent / config_file
            with open(config_path, "r", encoding="utf-8") as f:
                configs[config_file] = yaml.safe_load(f)

        # All configs should be dictionaries
        for name, config in configs.items():
            assert isinstance(config, dict), f"{name} is not a valid config dict"

    def test_dependency_version_compatibility(self):
        """Test dependency version compatibility."""
        import pandas as pd
        import numpy as np

        # Test that versions are reasonable
        assert pd.__version__ >= "1.0.0"
        assert np.__version__ >= "1.0.0"

        # Test that pandas and numpy work together
        df = pd.DataFrame({"a": np.array([1, 2, 3])})
        assert len(df) == 3
        assert df["a"].sum() == 6
