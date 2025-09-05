import threading
import time
import yaml
from collections import defaultdict, deque
from pathlib import Path

# Load API rate limit config
API_CONFIG_PATH = Path(__file__).parent.parent.parent / "api_config.yaml"
with open(API_CONFIG_PATH, "r", encoding="utf-8") as f:
    API_CONFIG = yaml.safe_load(f)

_api_call_times = defaultdict(lambda: {"minute": deque(), "day": deque()})
_api_lock = threading.Lock()


def rate_limit_check(endpoint: str):
    """
    Checks and enforces the rate limit for the given endpoint.
    Sleeps if the per-minute limit is reached, raises if the per-day limit is reached.
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
    Decorator to enforce rate limiting on API call functions.
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
