"""
Tiingo API interface for OHLCV data with rate limit tracking and pandas integration.
"""
import requests
import pandas as pd
import time
from tradingview_algo.fin_data_apis.secure_api import get_api_key


class TiingoRateLimiter:
    def __init__(self, max_per_minute=60, max_per_day=500):
        self.max_per_minute = max_per_minute
        self.max_per_day = max_per_day
        self.calls_this_minute = 0
        self.calls_today = 0
        self.minute_start = time.time()
        self.day_start = time.time()

    def check(self):
        now = time.time()
        # Reset per minute
        if now - self.minute_start > 60:
            self.calls_this_minute = 0
            self.minute_start = now
        # Reset per day
        if now - self.day_start > 86400:
            self.calls_today = 0
            self.day_start = now
        if self.calls_this_minute >= self.max_per_minute:
            sleep_time = 60 - (now - self.minute_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.calls_this_minute = 0
            self.minute_start = time.time()
        if self.calls_today >= self.max_per_day:
            raise RuntimeError("Tiingo daily API limit reached.")
        self.calls_this_minute += 1
        self.calls_today += 1


class TiingoAPI:
    BASE_URL = "https://api.tiingo.com/tiingo/daily"

    def __init__(self, api_key=None, rate_limiter=None):
        if not api_key:
            api_key = get_api_key("tiingo")
        if not api_key:
            raise ValueError("Tiingo API key not found in environment.")
        self.api_key = api_key
        self.rate_limiter = rate_limiter or TiingoRateLimiter()

    def get_ohlcv(self, ticker, start_date=None, end_date=None, resample_freq="daily"):
        """
        Fetch OHLCV data for a ticker as a pandas DataFrame.
        """
        self.rate_limiter.check()
        url = f"{self.BASE_URL}/{ticker}/prices"
        headers = {"Content-Type": "application/json", "Authorization": f"Token {self.api_key}"}
        params = {"resampleFreq": resample_freq}
        if start_date:
            params["startDate"] = start_date
        if end_date:
            params["endDate"] = end_date
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        # Standardize columns
        col_map = {
            "date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
        df = df.rename(columns=col_map)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.set_index("date")
        return df[["open", "high", "low", "close", "volume"]]


# Example usage:
# tiingo = TiingoAPI()
# df = tiingo.get_ohlcv("AAPL", start_date="2023-01-01", end_date="2023-06-01")
