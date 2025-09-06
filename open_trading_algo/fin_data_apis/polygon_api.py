"""
Polygon.io API interface for stocks, options, and indices using secure_api for API key management.
"""
import requests
from open_trading_algo.fin_data_apis.secure_api import get_api_key
from open_trading_algo.fin_data_apis.rate_limit import rate_limit


class PolygonAPI:
    def __init__(self, api_key: str = None):
        if not api_key:
            api_key = get_api_key("polygon")
        if not api_key:
            raise ValueError("Polygon API key not found in environment.")
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"

    @rate_limit("polygon")
    def get_stock_aggregates(
        self,
        symbol: str,
        timespan: str = "day",
        from_date: str = None,
        to_date: str = None,
        limit: int = 100,
    ):
        """Fetch aggregate (OHLCV) bars for a stock symbol."""
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/{timespan}/{from_date}/{to_date}"
        params = {"apiKey": self.api_key, "limit": limit}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    @rate_limit("polygon")
    def get_option_chain(
        self, underlying: str, expiration: str = None, option_type: str = None, limit: int = 100
    ):
        """Fetch option chain for an underlying symbol."""
        url = f"{self.base_url}/v3/reference/options/contracts"
        params = {"underlying_ticker": underlying, "apiKey": self.api_key, "limit": limit}
        if expiration:
            params["expiration_date"] = expiration
        if option_type:
            params["contract_type"] = option_type  # 'call' or 'put'
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    @rate_limit("polygon")
    def get_index_aggregates(
        self,
        symbol: str,
        timespan: str = "day",
        from_date: str = None,
        to_date: str = None,
        limit: int = 100,
    ):
        """Fetch aggregate (OHLCV) bars for an index symbol."""
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/{timespan}/{from_date}/{to_date}"
        params = {"apiKey": self.api_key, "limit": limit}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()


# Example usage:
# polygon = PolygonAPI()
# stock_bars = polygon.get_stock_aggregates("AAPL", from_date="2023-01-01", to_date="2023-01-31")
# options = polygon.get_option_chain("AAPL")
# index_bars = polygon.get_index_aggregates("SPX", from_date="2023-01-01", to_date="2023-01-31")
