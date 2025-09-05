import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tradingview_algo.fin_data_apis.fetchers import (
    fetch_yahoo,
    fetch_finnhub_bulk,
    fetch_fmp_bulk,
    fetch_alpha_vantage_bulk,
    fetch_twelve_data_bulk,
)
from tradingview_algo.fin_data_apis.tradingview_api import TradingViewAPI
from tradingview_algo.fin_data_apis.polygon_api import PolygonAPI
from tradingview_algo.fin_data_apis.secure_api import get_api_key
from tradingview_algo.data_cache import DataCache


class DatabasePopulator:
    def __init__(self, config_path=None, tickers_path=None, db_path=None):
        self.config_path = Path(
            config_path or Path(__file__).parent.parent / "live_data_config.yaml"
        )
        self.tickers_path = Path(tickers_path or Path(__file__).parent.parent / "all_tickers.yaml")
        self.db = DataCache(db_path)
        self._load_config()

    def _load_config(self):
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        self.fields = config.get("fields", ["price", "volume", "open", "high", "low", "close"])
        self.interval = config.get("interval", "1d")
        self.date_range = config.get(
            "date_range", None
        )  # e.g. {"start": "2023-01-01", "end": "2023-12-31"}
        # Load tickers
        with open(self.tickers_path, "r") as f:
            self.tickers = yaml.safe_load(f)["tickers"]

    def _tickers_to_fetch(self, start=None, end=None):
        # Check which tickers are missing or incomplete in the DB for the date range
        to_fetch = []
        for ticker in self.tickers:
            if not self.db.has_data(ticker, start, end):
                to_fetch.append(ticker)
        return to_fetch

    def _split_apis(self, tickers):
        # Assign tickers to APIs for concurrent fetching, avoiding repeats
        apis = {}
        # Yahoo (no key, batch)
        apis["yahoo"] = tickers.copy()
        # Finnhub, FMP, Alpha Vantage, Twelve Data (if keys present)
        for api in ["finnhub", "fmp", "alpha_vantage", "twelve_data"]:
            key = get_api_key(api)
            if key:
                apis[api] = tickers.copy()
        # TradingView, Polygon (one by one)
        if get_api_key("tradingview"):
            apis["tradingview"] = tickers.copy()
        if get_api_key("polygon"):
            apis["polygon"] = tickers.copy()
        return apis

    def _fetch_and_store(self, api, tickers, start, end):
        results = []
        if api == "yahoo":
            data = fetch_yahoo(tickers, self.fields, cache=self.db)
            for ticker, d in data.items():
                df = pd.DataFrame([d])
                df["ticker"] = ticker
                self.db.store_price_data(ticker, df)
                results.append(df)
        elif api == "finnhub":
            key = get_api_key("finnhub")
            data = fetch_finnhub_bulk(tickers, self.fields, key)
            for ticker, d in data.items():
                df = pd.DataFrame([d])
                df["ticker"] = ticker
                self.db.store_price_data(ticker, df)
                results.append(df)
        elif api == "fmp":
            key = get_api_key("fmp")
            data = fetch_fmp_bulk(tickers, self.fields, key)
            for ticker, d in data.items():
                df = pd.DataFrame([d])
                df["ticker"] = ticker
                self.db.store_price_data(ticker, df)
                results.append(df)
        elif api == "alpha_vantage":
            key = get_api_key("alpha_vantage")
            data = fetch_alpha_vantage_bulk(tickers, self.fields, key)
            for ticker, d in data.items():
                df = pd.DataFrame([d])
                df["ticker"] = ticker
                self.db.store_price_data(ticker, df)
                results.append(df)
        elif api == "twelve_data":
            key = get_api_key("twelve_data")
            data = fetch_twelve_data_bulk(tickers, self.fields, key)
            for ticker, d in data.items():
                df = pd.DataFrame([d])
                df["ticker"] = ticker
                self.db.store_price_data(ticker, df)
                results.append(df)
        elif api == "tradingview":
            tv_api = TradingViewAPI(get_api_key("tradingview"))
            for ticker in tickers:
                try:
                    data = tv_api.get_chart_data(ticker)
                    df = pd.DataFrame(data)
                    df["ticker"] = ticker
                    self.db.store_price_data(ticker, df)
                    results.append(df)
                except Exception:
                    continue
        elif api == "polygon":
            polygon_api = PolygonAPI(get_api_key("polygon"))
            for ticker in tickers:
                try:
                    data = polygon_api.get_stock_aggregates(ticker)
                    df = pd.DataFrame(data.get("results", []))
                    df["ticker"] = ticker
                    self.db.store_price_data(ticker, df)
                    results.append(df)
                except Exception:
                    continue
        return results

    def run(self, start=None, end=None):
        # Use config date_range if not provided
        if not start or not end:
            if self.date_range:
                start = self.date_range.get("start")
                end = self.date_range.get("end")
        to_fetch = self._tickers_to_fetch(start, end)
        if not to_fetch:
            print("All data already present in database for the given range.")
            return
        apis = self._split_apis(to_fetch)
        results = []
        with ThreadPoolExecutor(max_workers=len(apis)) as executor:
            futures = {
                executor.submit(self._fetch_and_store, api, tickers, start, end): api
                for api, tickers in apis.items()
                if tickers
            }
            for future in as_completed(futures):
                api = futures[future]
                try:
                    res = future.result()
                    results.extend(res)
                except Exception as e:
                    print(f"Error fetching from {api}: {e}")
        if results:
            all_data = pd.concat(results, ignore_index=True)
            if "timestamp" in all_data.columns:
                all_data["date"] = pd.to_datetime(all_data["timestamp"], unit="s").dt.normalize()
            elif "date" in all_data.columns:
                all_data["date"] = pd.to_datetime(all_data["date"]).dt.normalize()
            else:
                all_data["date"] = pd.Timestamp.utcnow().normalize()
            all_data = all_data.set_index(["date", "ticker"]).sort_index()
            all_data.to_parquet("all_data.parquet")
            print("Unified data saved to all_data.parquet")
        else:
            print("No new data fetched.")


# Example usage:
# populator = DatabasePopulator()
# populator.run(start="2023-01-01", end="2023-12-31")
