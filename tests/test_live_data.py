import os
import pytest
from pathlib import Path
from tradingview_algo.fin_data_apis import feed as live_data
from tradingview_algo.fin_data_apis.secure_api import get_api_key

# Use a common ticker for all tests to avoid excessive API usage
TICKERS = ["AAPL"]
FIELDS = ["price", "open", "high", "low", "close", "volume", "timestamp"]

# Helper to skip tests if API key is missing
def require_api_key(service):
    key = get_api_key(service)
    if not key:
        pytest.skip(f"API key for {service} not set in .env or environment.")
    return key


def test_yahoo_fetch(monkeypatch):
    # Patch yfinance to avoid real API call
    import yfinance as yf

    called = {}

    def fake_download(*args, **kwargs):
        called["called"] = True
        import pandas as pd

        idx = pd.date_range("2023-01-01", periods=2, freq="1min")
        df = pd.DataFrame(
            {
                ("AAPL", "Open"): [100, 101],
                ("AAPL", "Close"): [102, 103],
                ("AAPL", "High"): [104, 105],
                ("AAPL", "Low"): [99, 100],
                ("AAPL", "Volume"): [1000, 1100],
            },
            index=idx,
        )
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df

    monkeypatch.setattr(yf, "download", fake_download)
    result = live_data.fetch_yahoo(TICKERS, FIELDS)
    assert "AAPL" in result
    assert "price" in result["AAPL"]
    assert called["called"]


def test_finnhub_bulk(monkeypatch):
    key = require_api_key("finnhub")
    # Patch requests.get to avoid real API call
    import requests

    def fake_get(url, params=None, timeout=10):
        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"c": 150, "o": 149, "h": 151, "l": 148, "pc": 148, "t": 1234567890}

        return Resp()

    monkeypatch.setattr(requests, "get", fake_get)
    result = live_data.fetch_finnhub_bulk(TICKERS, FIELDS, key)
    assert "AAPL" in result
    assert result["AAPL"]["price"] == 150


def test_finnhub_single(monkeypatch):
    key = require_api_key("finnhub")
    import requests

    def fake_get(url, params=None, timeout=10):
        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"c": 150, "o": 149, "h": 151, "l": 148, "pc": 148, "t": 1234567890}

        return Resp()

    monkeypatch.setattr(requests, "get", fake_get)
    result = live_data.fetch_finnhub(TICKERS, FIELDS, key)
    assert "AAPL" in result
    assert result["AAPL"]["price"] == 150


def test_fmp_bulk(monkeypatch):
    key = require_api_key("fmp")
    import requests

    def fake_get(url, params=None, timeout=10):
        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return [
                    {
                        "symbol": "AAPL",
                        "price": 200,
                        "open": 199,
                        "dayHigh": 201,
                        "dayLow": 198,
                        "previousClose": 197,
                        "volume": 10000,
                        "timestamp": 1234567890,
                    }
                ]

        return Resp()

    monkeypatch.setattr(requests, "get", fake_get)
    result = live_data.fetch_fmp_bulk(TICKERS, FIELDS, key)
    assert "AAPL" in result
    assert result["AAPL"]["price"] == 200


def test_fmp_single(monkeypatch):
    key = require_api_key("fmp")
    import requests

    def fake_get(url, params=None, timeout=10):
        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return [
                    {
                        "symbol": "AAPL",
                        "price": 200,
                        "open": 199,
                        "dayHigh": 201,
                        "dayLow": 198,
                        "previousClose": 197,
                        "volume": 10000,
                        "timestamp": 1234567890,
                    }
                ]

        return Resp()

    monkeypatch.setattr(requests, "get", fake_get)
    result = live_data.fetch_fmp(TICKERS, FIELDS, key)
    assert "AAPL" in result
    assert result["AAPL"]["price"] == 200


def test_alpha_vantage_bulk(monkeypatch):
    key = require_api_key("alpha_vantage")
    import requests

    def fake_get(url, params=None, timeout=10):
        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "Global Quote": {
                        "05. price": "300",
                        "02. open": "299",
                        "03. high": "301",
                        "04. low": "298",
                        "08. previous close": "297",
                        "06. volume": "5000",
                        "07. latest trading day": "2023-01-01",
                    }
                }

        return Resp()

    monkeypatch.setattr(requests, "get", fake_get)
    result = live_data.fetch_alpha_vantage_bulk(TICKERS, FIELDS, key)
    assert "AAPL" in result
    assert float(result["AAPL"]["price"]) == 300


def test_alpha_vantage_single(monkeypatch):
    key = require_api_key("alpha_vantage")
    import requests

    def fake_get(url, params=None, timeout=10):
        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "Global Quote": {
                        "05. price": "300",
                        "02. open": "299",
                        "03. high": "301",
                        "04. low": "298",
                        "08. previous close": "297",
                        "06. volume": "5000",
                        "07. latest trading day": "2023-01-01",
                    }
                }

        return Resp()

    monkeypatch.setattr(requests, "get", fake_get)
    result = live_data.fetch_alpha_vantage(TICKERS, FIELDS, key)
    assert "AAPL" in result
    assert float(result["AAPL"]["price"]) == 300


def test_twelve_data_bulk(monkeypatch):
    key = require_api_key("twelve_data")
    import requests

    def fake_get(url, params=None, timeout=10):
        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "close": 400,
                    "open": 399,
                    "high": 401,
                    "low": 398,
                    "previous_close": 397,
                    "volume": 2000,
                    "datetime": "2023-01-01 09:30:00",
                }

        return Resp()

    monkeypatch.setattr(requests, "get", fake_get)
    result = live_data.fetch_twelve_data_bulk(TICKERS, FIELDS, key)
    assert "AAPL" in result
    assert float(result["AAPL"]["price"]) == 400


def test_twelve_data_single(monkeypatch):
    key = require_api_key("twelve_data")
    import requests

    def fake_get(url, params=None, timeout=10):
        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "close": 400,
                    "open": 399,
                    "high": 401,
                    "low": 398,
                    "previous_close": 397,
                    "volume": 2000,
                    "datetime": "2023-01-01 09:30:00",
                }

        return Resp()

    monkeypatch.setattr(requests, "get", fake_get)
    result = live_data.fetch_twelve_data(TICKERS, FIELDS, key)
    assert "AAPL" in result
    assert float(result["AAPL"]["price"]) == 400


def test_credentials():
    # Just check that credentials are present (or skipped)
    for service in ["finnhub", "fmp", "alpha_vantage", "twelve_data"]:
        key = get_api_key(service)
        if not key:
            pytest.skip(f"API key for {service} not set.")
        assert isinstance(key, str)
        assert len(key) > 0
