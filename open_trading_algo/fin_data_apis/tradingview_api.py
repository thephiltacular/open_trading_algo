"""
TradingView API interface using secure_api for API key management.
"""
import requests
from open_trading_algo.fin_data_apis.secure_api import get_api_key
from open_trading_algo.fin_data_apis.rate_limit import rate_limit


class TradingViewAPI:
    def __init__(self, api_key: str = None):
        if not api_key:
            api_key = get_api_key("tradingview")
        if not api_key:
            raise ValueError("TradingView API key not found in environment.")
        self.api_key = api_key
        self.base_url = (
            "https://api.tradingview.com"  # Placeholder, update to real endpoint if available
        )

    @rate_limit("tradingview")
    def get_symbol_info(self, symbol: str):
        """Fetch symbol info from TradingView API."""
        url = f"{self.base_url}/symbol_info/{symbol}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    @rate_limit("tradingview")
    def get_chart_data(self, symbol: str, interval: str = "1d", limit: int = 100):
        """Fetch chart data (candles) for a symbol."""
        url = f"{self.base_url}/chart/{symbol}?interval={interval}&limit={limit}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()


# Example usage:
# tv_api = TradingViewAPI()
# info = tv_api.get_symbol_info("AAPL")
# candles = tv_api.get_chart_data("AAPL", interval="1h", limit=200)
