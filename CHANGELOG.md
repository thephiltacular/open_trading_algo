# Changelog

All notable changes to this project are documented in this file.

## [0.1.1] - 2025-09-06
### Added
- **Technical Indicators Expansion**: Implemented 13 additional technical indicators to complete Alpha Vantage API compatibility:
  - **Volatility Indicators**: NATR (Normalized ATR), TRANGE (True Range)
  - **Volume Indicators**: MFI (Money Flow Index)
  - **Momentum Indicators**: PLUS_DM, MINUS_DM, PLUS_DI, MINUS_DI, DX (Directional Movement System)
  - **Trend Indicators**: AROON, AROONOSC, TRIX, ULTOSC, SAR (Parabolic SAR)
  - **Cycle Indicators**: HT_DCPHASE, HT_PHASOR (additional Hilbert Transform indicators)
  - **Comprehensive Test Coverage**: Added 13 new test functions covering all newly implemented indicators with proper edge case handling and bounds validation
  - **Enhanced Indicator Framework**: Improved error handling for edge cases (NaN values, division by zero) and maintained consistent API patterns
- **Trading Models Architecture**: Created comprehensive models directory with extensible strategy framework:
  - **BaseTradingModel**: Abstract base class providing common functionality for data validation, indicator caching, and signal generation
  - **MomentumModel**: RSI and MACD-based momentum strategy with Stochastic confirmation
  - **MeanReversionModel**: Bollinger Bands and RSI-based mean reversion strategy
  - **TrendFollowingModel**: Moving average crossover with ADX trend confirmation
  - **Complete Test Suite**: 17 comprehensive tests covering all model functionality, edge cases, and integration with indicators
  - **Modular Design**: Easy extension for new strategy types with consistent API patterns### Changed
- Updated `__all__` exports in indicators module to include all new indicator functions
- Enhanced test suite with 61 total tests (48 existing + 13 new) ensuring no regressions

### Fixed
- Resolved edge cases in Stochastic RSI calculations with proper NaN handling
- Fixed test assertions for A/D Oscillator convergence testing

## [0.1.0] - 2025-09-06
### Added
- Reorganized package into focused subpackages: indicators/, fin_data_apis/, cache/, backtest/, sentiment/, alerts/.
- Implemented persistent local data cache (SQLite) with DataCache API for OHLCV and signal storage.
- Added fin_data_apis collection with fetchers and bulk endpoints for: Yahoo (yfinance), Finnhub, FMP, Alpha Vantage, Twelve Data, Tiingo, Polygon, TradingView.
- Implemented secure API key management (secure_api) and integrated with all API clients.
- Implemented robust rate limiting framework:
  - RateLimiter base class and API-specific subclasses.
  - rate_limit decorator and rate_limit_check utility.
  - api_config.yaml integration with rate limits and docs links.
- Implemented Live Data Feed (feed.py) supporting multiple providers, batching, caching and callbacks.
- Implemented DatabasePopulator class to populate DB with OHLCV across multiple APIs, date ranges, and intervals using bulk fetches and concurrent dispatch.
- Implemented sentiment subsystem:
  - social_sentiment and analyst_sentiment with bulk fetching, caching, and DataFrame output (indexed by date,ticker).
  - Integrated with secure_api and rate limiting.
- Implemented comprehensive signal suites:
  - long_signals, short_signals, options_signals, sentiment_signals with compute_and_cache_*_signals hooks.
  - SignalOptimizer with extensive backtesting strategies (walk-forward, Monte Carlo, ML ensemble, regime switching).
- Implemented risk_management utilities for position sizing, stop-loss, and portfolio hedges and hooked into backtesting.
- Implemented indicators module with many technical indicators (SMA, EMA, WMA, DEMA, TEMA, MACD, RSI, ATR, OBV, Bollinger Bands, etc.).
- Added populate_database flow to calculate indicators per-ticker after fetching, and store unified DataFrame by date,ticker.
- Added tests covering data enrichment, data cache, live data (mocked), and signal modules. Tests updated to reuse cached yfinance data to avoid rate limits.
- Created docs/ with detailed guides (quickstart, installation, data-apis, configuration, cache, contribution).
- Added CONTRIBUTING.md, MIT LICENSE, CHANGES.md, and CHANGES/CHANGELOG structure; updated pyproject.toml and dev dependencies.
- Added scripts: setup_db.py, cache_aapl_10y.py, run_model.py, and other utility scripts.

### Changed
- Refactored and cleaned up duplicate/stray function definitions in all signal modules; moved compute_and_cache_* functions into proper function bodies.
- Updated all imports and __init__.py files to reflect new package layout and ensure pip-installable behavior.
- Updated tests to align with new package paths and to minimize external API usage (single cached fetch / mocking).
- Normalized data schema for all API fetchers: consistent OHLCV columns and index naming to avoid indicator calculation errors.
- Updated README to link docs and contributing guide; added CHANGES/Release guidance.

### Fixed
- Fixed numerous syntax and reference errors introduced during refactors; removed duplicate definitions causing import errors.
- Fixed test failures by adding caching + mocking and ensuring single yfinance request per test session.s
- Ensured DB creation is automatic (reads db_config.yaml) and persistent on disk.

### Docs
- Documented all modules, classes and functions with module-level and function-level docstrings across the codebase.
- Added docs/ pages and updated README to link to full documentation and contribution guide.
- Documented API usage, credential handling (.env / secrets.env.example), and rate limits (api_config.yaml).
