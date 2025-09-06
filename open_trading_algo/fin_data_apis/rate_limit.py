"""API rate limiting utilities for open_trading_algo.

This module provides a thread-safe `RateLimiter` base class and API-specific subclasses
for enforcing per-minute and per-day rate limits on external API calls. It also includes
function-based decorators for backward compatibility and a global configuration loader
for rate limit settings.

Classes:
    RateLimiter: Base class for rate limiting.
    FinnhubRateLimiter, FmpRateLimiter, AlphaVantageRateLimiter, TwelveDataRateLimiter,
    YahooRateLimiter, TradingViewRateLimiter, PolygonRateLimiter: API-specific subclasses.

Functions:
    rate_limit_check(endpoint): Enforces rate limits for a given endpoint.
    rate_limit(endpoint): Decorator to enforce rate limiting on API call functions.
"""

import threading
import time
import yaml
from collections import defaultdict, deque
from pathlib import Path

# Load API rate limit config
API_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "api_config.yaml"
with open(API_CONFIG_PATH, "r", encoding="utf-8") as f:
    API_CONFIG = yaml.safe_load(f)


class RateLimiter:
    """
    Thread-safe base class for enforcing API rate limits.

    Attributes:
        endpoint (str): The API endpoint name.
        per_minute (int or None): Allowed requests per minute.
        per_day (int or None): Allowed requests per day.
    """

    def __init__(self, endpoint: str, per_minute: int = None, per_day: int = None):
        """
        Initialize a RateLimiter for a specific endpoint.

        Args:
            endpoint (str): The API endpoint name.
            per_minute (int, optional): Override for per-minute limit.
            per_day (int, optional): Override for per-day limit.
        """
        self.endpoint = endpoint
        conf = API_CONFIG.get(endpoint, {})
        self.per_minute = per_minute if per_minute is not None else conf.get("free_limit_per_minute")
        self.per_day = per_day if per_day is not None else conf.get("free_limit_per_day")
        self._lock = threading.Lock()
        self._dq_min = deque()
        self._dq_day = deque()

    def check(self):
        """
        Enforce the rate limit for the endpoint.

        Sleeps if the per-minute limit is reached. Raises RuntimeError if the per-day limit is reached.
        """
        now = time.time()
        with self._lock:
            while self._dq_min and now - self._dq_min[0] > 60:
                self._dq_min.popleft()
            while self._dq_day and now - self._dq_day[0] > 86400:
                self._dq_day.popleft()
            if self.per_minute is not None and len(self._dq_min) >= int(self.per_minute):
                wait = 60 - (now - self._dq_min[0])
                if wait > 0:
                    time.sleep(wait)
            if self.per_day is not None and len(self._dq_day) >= int(self.per_day):
                raise RuntimeError(f"API daily rate limit reached for {self.endpoint}")
            self._dq_min.append(now)
            self._dq_day.append(now)


class FinnhubRateLimiter(RateLimiter):
    """Rate limiter for the Finnhub API."""

    def __init__(self):
        super().__init__("finnhub")


class FmpRateLimiter(RateLimiter):
    """Rate limiter for the FMP API."""

    def __init__(self):
        super().__init__("fmp")


class AlphaVantageRateLimiter(RateLimiter):
    """Rate limiter for the Alpha Vantage API."""

    def __init__(self):
        super().__init__("alpha_vantage")


class TwelveDataRateLimiter(RateLimiter):
    """Rate limiter for the Twelve Data API."""

    def __init__(self):
        super().__init__("twelve_data")


class YahooRateLimiter(RateLimiter):
    """Rate limiter for the Yahoo Finance API."""

    def __init__(self):
        super().__init__("yahoo")


class TradingViewRateLimiter(RateLimiter):
    """Rate limiter for the TradingView API."""

    def __init__(self):
        super().__init__("tradingview")


class PolygonRateLimiter(RateLimiter):
    """Rate limiter for the Polygon API."""

    def __init__(self):
        super().__init__("polygon")


# Backward compatible function-based API
_api_call_times = defaultdict(lambda: {"minute": deque(), "day": deque()})
_api_lock = threading.Lock()


def rate_limit_check(endpoint: str):
    """
    Checks and enforces API rate limits for a given endpoint.

    This function manages both per-minute and per-day rate limits for API calls.
    It uses internal tracking to determine if the current request exceeds the allowed
    number of calls within the specified timeframes. If the per-minute limit is reached,
    the function will pause execution until the limit resets. If the per-day limit is reached,
    a RuntimeError is raised.

    Args:
        endpoint (str): The API endpoint to check rate limits for.

    Raises:
        RuntimeError: If the daily rate limit for the endpoint is exceeded.
    """
    now = time.time()
    conf = API_CONFIG.get(endpoint)
    if not conf:
        return
    per_min = conf.get("free_limit_per_minute")
    per_day = conf.get("free_limit_per_day")
    with _api_lock:
        dq_min = _api_call_times[endpoint]["minute"]
        dq_day = _api_call_times[endpoint]["day"]
        while dq_min and now - dq_min[0] > 60:
            dq_min.popleft()
        while dq_day and now - dq_day[0] > 86400:
            dq_day.popleft()
        if per_min is not None and len(dq_min) >= int(per_min):
            wait = 60 - (now - dq_min[0])
            if wait > 0:
                time.sleep(wait)
        if per_day is not None and len(dq_day) >= int(per_day):
            raise RuntimeError(f"API daily rate limit reached for {endpoint}")
        dq_min.append(now)
        dq_day.append(now)


def rate_limit(endpoint: str):
    """
    Decorator to enforce rate limiting on API call functions for a given endpoint.

    Usage:
        @rate_limit("finnhub")
        def fetch_finnhub(...):
            ...
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            rate_limit_check(endpoint)
            return func(*args, **kwargs)

        return wrapper

    return decorator
