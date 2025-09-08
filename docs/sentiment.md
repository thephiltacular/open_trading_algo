# Sentiment Analysis

This document covers the sentiment analysis capabilities in open_trading_algo, including social media sentiment, news analysis, and analyst sentiment processing.

## Overview

The sentiment analysis system provides comprehensive tools for:

- **Social media sentiment** analysis from Twitter, Reddit, and other platforms
- **News sentiment** processing from financial news sources
- **Analyst sentiment** aggregation from analyst reports and ratings
- **Sentiment scoring** with confidence intervals
- **Multi-source sentiment** aggregation and weighting

## Social Media Sentiment

### Twitter Sentiment Analysis

```python
from open_trading_algo.sentiment.twitter_sentiment import TwitterSentimentAnalyzer
from open_trading_algo.cache.sentiment_cache import SentimentCache

# Initialize components
twitter_analyzer = TwitterSentimentAnalyzer(
    api_key='your_twitter_api_key',
    api_secret='your_twitter_api_secret'
)
cache = SentimentCache()

# Analyze sentiment for a ticker
ticker = 'AAPL'
sentiment_data = twitter_analyzer.analyze_ticker_sentiment(
    ticker=ticker,
    lookback_hours=24,
    min_tweets=100
)

# Cache the results
cache.store_sentiment_data(ticker, sentiment_data, 'twitter')

print("Twitter Sentiment Analysis:")
print(f"Overall Sentiment: {sentiment_data['overall_sentiment']:.2f}")
print(f"Confidence: {sentiment_data['confidence']:.2f}")
print(f"Total Tweets: {sentiment_data['total_tweets']}")
print(f"Bullish %: {sentiment_data['bullish_percentage']:.1%}")
print(f"Bearish %: {sentiment_data['bearish_percentage']:.1%}")
```

### Reddit Sentiment Analysis

```python
from open_trading_algo.sentiment.reddit_sentiment import RedditSentimentAnalyzer

# Initialize Reddit analyzer
reddit_analyzer = RedditSentimentAnalyzer(
    client_id='your_reddit_client_id',
    client_secret='your_reddit_client_secret',
    user_agent='open_trading_algo/1.0'
)

# Analyze sentiment across subreddits
subreddits = ['wallstreetbets', 'investing', 'stocks']
sentiment_results = reddit_analyzer.analyze_subreddit_sentiment(
    ticker='TSLA',
    subreddits=subreddits,
    time_filter='week',
    limit=500
)

print("Reddit Sentiment Results:")
for subreddit, data in sentiment_results.items():
    print(f"{subreddit}: {data['sentiment_score']:.2f} "
          f"(confidence: {data['confidence']:.2f})")
```

### Multi-Platform Sentiment Aggregation

```python
from open_trading_algo.sentiment.multi_platform_analyzer import MultiPlatformSentimentAnalyzer

# Initialize multi-platform analyzer
multi_analyzer = MultiPlatformSentimentAnalyzer(
    twitter_config={'api_key': 'key', 'api_secret': 'secret'},
    reddit_config={'client_id': 'id', 'client_secret': 'secret'},
    weights={'twitter': 0.4, 'reddit': 0.6}
)

# Get aggregated sentiment
aggregated_sentiment = multi_analyzer.analyze_sentiment(
    ticker='NVDA',
    platforms=['twitter', 'reddit'],
    time_window='24h'
)

print("Aggregated Sentiment:")
print(f"Weighted Score: {aggregated_sentiment['weighted_score']:.2f}")
print(f"Confidence Interval: [{aggregated_sentiment['confidence_lower']:.2f}, "
      f"{aggregated_sentiment['confidence_upper']:.2f}]")
print(f"Platform Contributions: {aggregated_sentiment['platform_contributions']}")
```

## News Sentiment Analysis

### Financial News Processing

```python
from open_trading_algo.sentiment.news_sentiment import NewsSentimentAnalyzer

# Initialize news analyzer
news_analyzer = NewsSentimentAnalyzer(
    api_key='your_news_api_key',  # Alpha Vantage, NewsAPI, etc.
    sources=['bloomberg', 'reuters', 'cnbc', 'wsj']
)

# Analyze news sentiment
news_sentiment = news_analyzer.analyze_news_sentiment(
    ticker='MSFT',
    days_back=7,
    min_articles=10
)

print("News Sentiment Analysis:")
print(f"Average Sentiment: {news_sentiment['avg_sentiment']:.2f}")
print(f"Sentiment Volatility: {news_sentiment['sentiment_volatility']:.2f}")
print(f"Total Articles: {news_sentiment['total_articles']}")
print(f"Positive Articles: {news_sentiment['positive_count']}")
print(f"Negative Articles: {news_sentiment['negative_count']}")
```

### Real-time News Monitoring

```python
# Set up real-time news monitoring
news_monitor = news_analyzer.create_news_monitor(
    tickers=['AAPL', 'GOOGL', 'MSFT'],
    update_interval_minutes=15,
    alert_threshold=0.3  # Alert on sentiment changes > 0.3
)

# Start monitoring
news_monitor.start_monitoring()

# Get current sentiment snapshot
current_sentiment = news_monitor.get_current_sentiment()
for ticker, sentiment in current_sentiment.items():
    print(f"{ticker}: {sentiment['score']:.2f} "
          f"(change: {sentiment['change_24h']:.2f})")
```

## Analyst Sentiment Analysis

### Analyst Ratings Processing

```python
from open_trading_algo.sentiment.analyst_sentiment import AnalystSentimentAnalyzer

# Initialize analyst analyzer
analyst_analyzer = AnalystSentimentAnalyzer(
    data_source='alpha_vantage'  # or 'yahoo_finance', 'investing_com'
)

# Get analyst ratings
analyst_data = analyst_analyzer.get_analyst_ratings('TSLA')

print("Analyst Sentiment:")
print(f"Average Rating: {analyst_data['avg_rating']:.1f}/5")
print(f"Rating Distribution: {analyst_data['rating_distribution']}")
print(f"Consensus: {analyst_data['consensus']}")
print(f"Number of Analysts: {analyst_data['analyst_count']}")
print(f"Price Target: ${analyst_data['avg_price_target']:.2f}")
```

### Analyst Recommendations Tracking

```python
# Track analyst recommendation changes
recommendation_tracker = analyst_analyzer.create_recommendation_tracker(
    tickers=['AAPL', 'NVDA', 'AMD'],
    track_changes=True
)

# Get recent changes
recent_changes = recommendation_tracker.get_recent_changes(days_back=30)

print("Recent Analyst Changes:")
for change in recent_changes:
    print(f"{change['ticker']}: {change['analyst']} changed from "
          f"{change['old_rating']} to {change['new_rating']} "
          f"({change['date']})")
```

## Advanced Sentiment Features

### Sentiment Scoring Models

```python
from open_trading_algo.sentiment.sentiment_model import AdvancedSentimentModel

# Initialize advanced sentiment model
sentiment_model = AdvancedSentimentModel(
    model_type='transformer',  # or 'lstm', 'bert'
    fine_tuned=True,
    confidence_threshold=0.7
)

# Score individual text
text = "Apple reported better than expected earnings, stock up 5%"
score = sentiment_model.score_text(text)

print("Advanced Sentiment Scoring:")
print(f"Text: {text}")
print(f"Sentiment Score: {score['sentiment']:.2f}")
print(f"Confidence: {score['confidence']:.2f}")
print(f"Emotion: {score['emotion']}")
print(f"Intensity: {score['intensity']:.2f}")
```

### Sentiment Time Series Analysis

```python
from open_trading_algo.sentiment.sentiment_timeseries import SentimentTimeSeriesAnalyzer

# Initialize time series analyzer
ts_analyzer = SentimentTimeSeriesAnalyzer()

# Analyze sentiment trends
sentiment_series = ts_analyzer.analyze_sentiment_trends(
    ticker='AAPL',
    sentiment_data=sentiment_history,
    timeframe='daily'
)

print("Sentiment Time Series Analysis:")
print(f"Trend Direction: {sentiment_series['trend']}")
print(f"Momentum: {sentiment_series['momentum']:.2f}")
print(f"Volatility: {sentiment_series['volatility']:.2f}")
print(f"Reversal Signals: {sentiment_series['reversal_signals']}")
```

### Sentiment-Price Correlation

```python
# Analyze correlation between sentiment and price movements
correlation_analysis = ts_analyzer.analyze_price_correlation(
    sentiment_data=sentiment_history,
    price_data=price_history,
    lag_periods=[0, 1, 2, 3, 5]  # Same day and lagged correlations
)

print("Sentiment-Price Correlation:")
for lag, corr in correlation_analysis['correlations'].items():
    print(f"Lag {lag} days: {corr:.3f}")

print(f"Optimal Lag: {correlation_analysis['optimal_lag']} days")
print(f"Max Correlation: {correlation_analysis['max_correlation']:.3f}")
```

## Sentiment Integration with Trading Signals

### Sentiment-Based Signals

```python
from open_trading_algo.sentiment.sentiment_signals import SentimentSignalGenerator

# Initialize sentiment signal generator
signal_generator = SentimentSignalGenerator(
    sentiment_threshold=0.2,
    confidence_threshold=0.75,
    lookback_periods=5
)

# Generate sentiment-based signals
sentiment_signals = signal_generator.generate_signals(
    ticker='TSLA',
    sentiment_data=sentiment_history,
    price_data=price_history
)

print("Sentiment Signals:")
for signal in sentiment_signals[-5:]:  # Last 5 signals
    print(f"{signal['date']}: {signal['signal']} "
          f"(sentiment: {signal['sentiment']:.2f}, "
          f"confidence: {signal['confidence']:.2f})")
```

### Multi-Factor Sentiment Signals

```python
# Combine multiple sentiment sources
multi_factor_signals = signal_generator.generate_multi_factor_signals(
    ticker='AAPL',
    sentiment_sources={
        'twitter': twitter_sentiment,
        'reddit': reddit_sentiment,
        'news': news_sentiment,
        'analyst': analyst_sentiment
    },
    weights={
        'twitter': 0.3,
        'reddit': 0.3,
        'news': 0.25,
        'analyst': 0.15
    }
)

print("Multi-Factor Sentiment Signals:")
print(f"Combined Signal: {multi_factor_signals['combined_signal']}")
print(f"Signal Strength: {multi_factor_signals['signal_strength']:.2f}")
print(f"Source Contributions: {multi_factor_signals['contributions']}")
```

## Configuration

### Sentiment Configuration

```yaml
# config/sentiment.yaml
sentiment:
  # Social Media
  twitter:
    api_key: ${TWITTER_API_KEY}
    api_secret: ${TWITTER_API_SECRET}
    rate_limit: 300  # requests per 15 minutes
    languages: ['en']

  reddit:
    client_id: ${REDDIT_CLIENT_ID}
    client_secret: ${REDDIT_CLIENT_SECRET}
    user_agent: 'open_trading_algo/1.0'
    subreddits: ['wallstreetbets', 'investing', 'stocks']

  # News Sources
  news:
    api_key: ${NEWS_API_KEY}
    sources:
      - bloomberg
      - reuters
      - cnbc
      - wall_street_journal
    update_interval_minutes: 15

  # Analyst Data
  analyst:
    data_source: alpha_vantage
    api_key: ${ALPHA_VANTAGE_API_KEY}
    cache_expiry_hours: 24

  # Advanced Settings
  scoring:
    model_type: transformer
    confidence_threshold: 0.7
    fine_tuned: true

  signals:
    sentiment_threshold: 0.2
    confidence_threshold: 0.75
    lookback_periods: 5
```

## Best Practices

### Data Quality

1. **Source Diversification**: Use multiple sentiment sources to reduce bias
2. **Quality Filtering**: Filter out spam, bots, and low-quality content
3. **Language Detection**: Focus on English content for better accuracy
4. **Time Zone Handling**: Account for global time zones in sentiment timing

### Signal Generation

1. **Confidence Thresholds**: Only act on high-confidence sentiment signals
2. **Volume Requirements**: Require minimum volume of sentiment data
3. **Freshness**: Use recent sentiment data for better relevance
4. **Context Awareness**: Consider market context when interpreting sentiment

### Risk Management

1. **Sentiment Volatility**: Account for sentiment noise and volatility
2. **Confirmation**: Use sentiment as confirmation, not primary signal
3. **Position Sizing**: Reduce position sizes for sentiment-based trades
4. **Stop Losses**: Always use stop losses with sentiment trades

## Troubleshooting

### Common Issues

**Low Confidence Scores**: Check data quality and increase sample size
**Sentiment Drift**: Regularly recalibrate sentiment models
**API Rate Limits**: Implement proper rate limiting and caching
**False Signals**: Adjust thresholds and add confirmation signals

### Debug Tools

```python
# Debug sentiment analysis
from open_trading_algo.sentiment.sentiment_debugger import SentimentDebugger

debugger = SentimentDebugger()
debugger.analyze_sentiment_distribution(sentiment_data)
debugger.check_source_quality(sentiment_sources)
debugger.plot_sentiment_vs_price(sentiment_data, price_data)
debugger.validate_sentiment_signals(signals)
```</content>
<parameter name="filePath">/home/philipmai/repos/TradingViewAlgoDev/docs/sentiment.md
