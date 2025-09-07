# Changelog

All notable changes to this project are documented in this file.

## [0.1.3] - 2025-09-07
### Added
- **Parquet Cache Implementation**: New columnar storage cache optimized for analytical workloads:
  - **Apache Parquet Integration**: Efficient columnar storage with high compression ratios
  - **Automatic Partitioning**: Data partitioned by ticker for optimal query performance
  - **Pandas Integration**: Seamless integration with pandas DataFrames
  - **Configuration Support**: Configurable compression algorithms (snappy, gzip, brotli, lz4, zstd)
  - **Batch Operations**: Efficient batch storage and retrieval for multiple tickers

- **InfluxDB Time Series Cache**: High-performance time series database implementation:
  - **Automated Technical Indicators**: Automatically calculate and store 15+ technical indicators from price data
  - **Advanced Time Series Queries**: Optimized Flux queries for complex time-based analytics
  - **Retention Policies**: Configurable data retention (default: 10 years for price data)
  - **Real-time Analytics**: Built-in aggregation functions and time-based queries
  - **Docker Integration**: Easy setup with local InfluxDB container
  - **Production Ready**: Thread-safe operations for concurrent access

- **Enhanced Cache System Architecture**: Comprehensive multi-cache implementation:
  - **Three Cache Types**: SQLite (default), Parquet (analytics), InfluxDB (production)
  - **Unified API**: Same interface across all cache implementations for easy migration
  - **Performance Optimization**: Each cache type optimized for specific use cases
  - **Configuration Management**: YAML-based configuration for all cache types
  - **Migration Tools**: Easy switching between cache implementations

### Enhanced
- **Time Series Cache Interface**: Major improvements to the InfluxDB cache implementation:
  - **Advanced Query Capabilities**: Complex date range filtering and aggregation
  - **Technical Indicator Automation**: Built-in calculation and storage of technical indicators
  - **Error Handling**: Robust error handling for database operations
  - **Performance Monitoring**: Built-in performance metrics and query optimization
  - **Data Integrity**: Enhanced data validation and consistency checks

- **Cache Documentation**: Comprehensive documentation updates:
  - **Multi-Cache Overview**: Detailed comparison of all three cache implementations
  - **Migration Guide**: Step-by-step instructions for switching between cache types
  - **Performance Benchmarks**: Comparative performance analysis for different use cases
  - **Configuration Examples**: Complete configuration examples for all cache types
  - **Best Practices**: Recommendations for choosing the right cache for specific scenarios

### Fixed
- **Time Series Cache Tests**: Resolved 9 failing test cases:
  - **Data Accumulation Issues**: Fixed test isolation by clearing metrics data between tests
  - **Timezone Comparison**: Resolved timezone handling in date filtering tests
  - **Metadata Filtering**: Fixed metadata column pollution in query results
  - **Query Range Optimization**: Updated Flux queries to use dynamic ranges
  - **Test Data Extension**: Extended test data to support 50-period moving averages

- **Cache System Integration**: Improved compatibility across all cache implementations:
  - **API Consistency**: Ensured identical APIs across SQLite, Parquet, and InfluxDB caches
  - **Error Handling**: Standardized error handling and logging across all cache types
  - **Performance Optimization**: Optimized query patterns for each cache implementation

### Docs
- **Cache System Documentation**: Complete rewrite of cache documentation:
  - **Architecture Overview**: Detailed explanation of multi-cache architecture
  - **Implementation Guides**: Step-by-step setup for each cache type
  - **API Reference**: Comprehensive API documentation for all cache methods
  - **Troubleshooting Guide**: Common issues and solutions for cache implementations
  - **Performance Tuning**: Optimization tips for different cache types

## [0.1.2] - 2025-09-06
### Added
- **Comprehensive Trading Metrics Module**: Implemented 24 advanced trading metrics with full test coverage:
  - **Volume & Price Metrics**: VWAP, Volume Price Trend, Volume Averages (10d, 30d, 60d, 90d)
  - **Volatility & Risk Metrics**: Volatility Ratio, ADX (Trend Strength), Maximum Drawdown
  - **Momentum & Trend Metrics**: Price Acceleration, Seasonal Strength, Monthly Returns
  - **Performance & Statistical Metrics**: Sharpe/Sortino/Calmar Ratios, Beta/Alpha, Win Rate, Profit Factor
  - **Support/Resistance Levels**: Pivot Points, Fibonacci Retracement Levels
  - **Statistical Analysis**: Correlation Matrix, Autocorrelation, Rolling Statistics
  - **Complete Test Suite**: 22 comprehensive tests covering all metrics with edge cases and validation
  - **Production-Ready Code**: Google-style docstrings, type hints, robust error handling

- **Comprehensive Metrics Documentation**: Created extensive documentation suite for developers and analysts:
  - **Complete API Reference**: Detailed documentation for all 24 metrics with mathematical foundations
  - **Performance Benchmarks**: Industry-standard benchmarks and accuracy assessments
  - **Usage Examples**: Practical code examples for risk analysis, performance evaluation, and strategy development
  - **Academic References**: Links to authoritative sources (Investopedia, CFA Institute, academic research)
  - **Visual Examples**: ASCII charts and mathematical notation for complex concepts
  - **Integration Guide**: Updated README, docs index, and quickstart with metrics examples

### Enhanced
- **Documentation Structure**: Added metrics to main documentation index and quickstart guide
- **Code Quality**: Maintained consistent API patterns with existing indicators and signals modules
- **Test Coverage**: Achieved 100% test coverage for all new metrics functions

### Fixed
- **Documentation Links**: Updated all cross-references to include new metrics documentation
- **Quickstart Examples**: Added working code examples for metrics usage

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
- Added scripts: cache_aapl_10y.py, run_model.py, and other utility scripts.
- Moved setup_db.py to open_trading_algo/cache/ for better organization.

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
