# Contributing to open_trading_algo

Thank you for your interest in contributing to open_trading_algo! This document provides guidelines and information for contributors.

## 🎯 Ways to Contribute

### 🐛 Bug Reports
- Search existing issues before creating new ones
- Include a clear description of the problem
- Provide steps to reproduce the issue
- Include system information (Python version, OS, dependencies)
- Add relevant code snippets or error messages

### 💡 Feature Requests
- Check if the feature already exists in the documentation
- Explain the use case and benefits
- Provide examples of how the feature would be used
- Consider backward compatibility implications

### 📝 Code Contributions
- Bug fixes and improvements
- New technical indicators
- Additional data source integrations
- Performance optimizations
- Documentation improvements

### 📚 Documentation
- Fix typos or improve clarity
- Add code examples
- Translate documentation
- Create tutorials or guides

## 🚀 Getting Started

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/open_trading_algo.git
   cd open_trading_algo
   ```

2. **Create Development Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .
   pip install -r requirements-dev.txt
   ```

3. **Set Up Pre-commit Hooks**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

4. **Run Tests**
   ```bash
   pytest tests/
   ```

### API Keys for Testing
- Copy `.env.example` to `.env`
- Add your API keys for data providers you want to test
- Never commit API keys to the repository

## 📋 Development Guidelines

### Code Style

We follow Python best practices and maintain consistency:

- **PEP 8** compliance with 100-character line limit
- **Type hints** for all function parameters and returns
- **Docstrings** for all public functions (Google style)
- **Meaningful variable names** and clear function signatures

Example:
```python
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index.

    Args:
        prices: Series of closing prices
        period: RSI calculation period (default: 14)

    Returns:
        Series of RSI values (0-100)

    Raises:
        ValueError: If period is less than 2
    """
    if period < 2:
        raise ValueError("RSI period must be at least 2")
    # Implementation...
```

### Testing Requirements

- **Unit tests** for all new functions
- **Integration tests** for API interactions
- **Mock external dependencies** in tests
- **Test edge cases** and error conditions
- Maintain **>90% code coverage**

### Commit Guidelines

Use conventional commit format:
```
type(scope): description

feat(indicators): add Ichimoku Cloud indicator
fix(cache): resolve SQLite connection timeout
docs(api): update Alpha Vantage rate limits
test(signals): add unit tests for momentum signals
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

## 🔄 Pull Request Process

### Before Submitting

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write tests** for your changes

3. **Run the test suite**:
   ```bash
   pytest tests/ --cov=open_trading_algo
   ```

4. **Check code style**:
   ```bash
   black .
   flake8 .
   mypy open_trading_algo/
   ```

5. **Update documentation** if needed

### Pull Request Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated (if applicable)
- [ ] CHANGELOG.md updated (for significant changes)
- [ ] No breaking changes (or clearly documented)
- [ ] Commit messages follow convention

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

## 🏗️ Project Structure

Understanding the codebase structure helps with contributions:

```
open_trading_algo/
├── open_trading_algo/           # Main package
│   ├── fin_data_apis/         # Data source integrations
│   │   ├── fetchers.py        # Unified API interface
│   │   ├── yahoo_api.py       # Yahoo Finance
│   │   ├── finnhub_api.py     # Finnhub integration
│   │   └── ...                # Other providers
│   ├── indicators/            # Technical analysis
│   │   ├── indicators.py      # Core indicators
│   │   ├── long_signals.py    # Long position signals
│   │   ├── short_signals.py   # Short position signals
│   │   └── options_signals.py # Options signals
│   ├── cache/                 # Data caching system
│   ├── sentiment/             # Sentiment analysis
│   └── utils/                 # Utility functions
├── tests/                     # Test suite
├── docs/                      # Documentation
├── examples/                  # Usage examples
└── scripts/                   # Development scripts
```

## 🧪 Adding New Features

### New Technical Indicators

1. Add the indicator function to `indicators/indicators.py`
2. Include comprehensive docstring with formula reference
3. Add corresponding signal functions to appropriate signal modules
4. Write unit tests with known good values
5. Update documentation

Example structure:
```python
def calculate_your_indicator(
    prices: pd.Series,
    param1: int = 14,
    param2: float = 2.0
) -> pd.Series:
    """Calculate Your Custom Indicator.

    Formula: Detailed mathematical description
    Reference: Academic paper or source

    Args:
        prices: Series of prices (usually close)
        param1: Parameter description
        param2: Parameter description

    Returns:
        Series of indicator values
    """
```

### New Data Sources

1. Create new module in `fin_data_apis/`
2. Implement standard interface:
   - `fetch_quote(symbol: str) -> Dict`
   - `fetch_historical(symbol: str, period: str) -> pd.DataFrame`
   - Rate limiting and error handling
3. Add to `fetchers.py` unified interface
4. Update configuration files
5. Add integration tests (with mocking)

### New Signal Strategies

1. Add function to appropriate signal module
2. Follow naming convention: `{strategy_name}_signal`
3. Return standardized signal format
4. Include risk parameters and confidence scores
5. Add backtesting examples

## 🐛 Debugging Guidelines

### Common Issues

1. **API Rate Limits**: Use caching and batch requests
2. **Data Quality**: Validate inputs and handle missing data
3. **Performance**: Profile code and optimize hot paths
4. **Memory Usage**: Use generators for large datasets

### Debugging Tools

- Use `pytest --pdb` for interactive debugging
- Enable logging: `logging.basicConfig(level=logging.DEBUG)`
- Profile with `cProfile` for performance issues
- Monitor memory with `memory_profiler`

## 📊 Performance Considerations

- **Vectorize operations** using pandas/numpy
- **Cache expensive computations**
- **Use generators** for memory efficiency
- **Batch API requests** to minimize latency
- **Profile before optimizing**

## 🎯 Review Process

Maintainers will review PRs for:

- **Code quality** and style compliance
- **Test coverage** and quality
- **Documentation** completeness
- **Performance** implications
- **Backward compatibility**
- **Security** considerations (especially for API integrations)

## 📞 Getting Help

- **Discussions**: Use GitHub Discussions for questions
- **Issues**: Create issues for bugs or feature requests
- **Documentation**: Check docs/ folder first
- **Examples**: See examples/ folder for usage patterns

## 📝 License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

## 🙏 Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes for significant contributions
- GitHub contributors graph

Thank you for helping make open_trading_algo better! 🚀
