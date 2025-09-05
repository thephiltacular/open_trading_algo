"""Database population script for open_trading_algo.

This module provides a class to populate the database with financial data from various APIs,
handling caching, rate limits, and concurrent fetching.
"""

import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from open_trading_algo.fin_data_apis.fetchers import (
    fetch_yahoo,
    fetch_finnhub_bulk,
    fetch_fmp_bulk,
    fetch_alpha_vantage_bulk,
    fetch_twelve_data_bulk,
)
from open_trading_algo.fin_data_apis.tradingview_api import TradingViewAPI
from open_trading_algo.fin_data_apis.polygon_api import PolygonAPI
from open_trading_algo.fin_data_apis.secure_api import get_api_key
from open_trading_algo.cache.data_cache import DataCache
from open_trading_algo.fin_data_apis.alpha_vantage_api import (
    AlphaVantageAPI,
    ALPHA_VANTAGE_TECHNICAL_INDICATORS,
)


class DatabasePopulator:
    """Class for populating the database with financial data.

    Attributes:
        config_path (Path): Path to config file.
        tickers_path (Path): Path to tickers file.
        db (DataCache): Database cache instance.
    """

    def __init__(self, config_path=None, tickers_path=None, db_path=None):
        """Initialize the DatabasePopulator.

        Args:
            config_path (str, optional): Path to config file.
            tickers_path (str, optional): Path to tickers file.
            db_path (str, optional): Path to database.
        """
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
        # Fetch OHLCV for all tickers in as few API calls as possible, return a single DataFrame
        def normalize_df(df, ticker):
            col_map = {
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }
            df = df.rename(columns=col_map)
            if "date" not in df.columns:
                if "timestamp" in df.columns:
                    df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.normalize()
                elif df.index.name in ["date", "datetime"]:
                    df = df.reset_index().rename(columns={df.index.name: "date"})
                else:
                    df["date"] = pd.Timestamp.utcnow().normalize()
            df["ticker"] = ticker
            keep = ["date", "ticker", "open", "high", "low", "close", "volume"]
            for col in keep:
                if col not in df.columns:
                    df[col] = None
            return df[keep]

        # Prefer Yahoo for bulk, fallback to others if needed
        if api == "yahoo":
            data = fetch_yahoo(tickers, self.fields, cache=self.db)
        elif api == "finnhub":
            key = get_api_key("finnhub")
            data = fetch_finnhub_bulk(tickers, self.fields, key)
        elif api == "fmp":
            key = get_api_key("fmp")
            data = fetch_fmp_bulk(tickers, self.fields, key)
        elif api == "alpha_vantage":
            key = get_api_key("alpha_vantage")
            data = fetch_alpha_vantage_bulk(tickers, self.fields, key)
        elif api == "twelve_data":
            key = get_api_key("twelve_data")
            data = fetch_twelve_data_bulk(tickers, self.fields, key)
        elif api == "tradingview":
            tv_api = TradingViewAPI(get_api_key("tradingview"))
            data = {ticker: tv_api.get_chart_data(ticker) for ticker in tickers}
        elif api == "polygon":
            polygon_api = PolygonAPI(get_api_key("polygon"))
            data = {ticker: polygon_api.get_stock_aggregates(ticker) for ticker in tickers}
        else:
            data = {}

        frames = []
        for ticker, d in data.items():
            if isinstance(d, dict) and "results" in d:
                df = pd.DataFrame(d["results"])
            elif isinstance(d, (list, pd.DataFrame)):
                df = pd.DataFrame(d)
            else:
                df = pd.DataFrame([d])
            if df.empty:
                continue
            df = normalize_df(df, ticker)
            self.db.store_price_data(ticker, df)
            frames.append(df)
        if frames:
            return [pd.concat(frames, ignore_index=True)]
        return []

    def run(self, start=None, end=None):
        """Run the database population process.

        Args:
            start (str, optional): Start date.
            end (str, optional): End date.
        """
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
            all_data["date"] = pd.to_datetime(all_data["date"]).dt.normalize()
            all_data = all_data.set_index(["date", "ticker"]).sort_index()
            # Calculate indicators for each ticker
            from open_trading_algo.indicators import indicators as ind

            indicator_frames = []
            for ticker in all_data.index.get_level_values("ticker").unique():
                tdf = all_data.xs(ticker, level="ticker")
                # Calculate indicators using close, high, low, open, volume
                out = pd.DataFrame(index=tdf.index)
                out["SMA_14"] = ind.sma(tdf["close"], 14)
                out["EMA_14"] = ind.ema(tdf["close"], 14)
                out["WMA_14"] = ind.wma(tdf["close"], 14)
                out["DEMA_14"] = ind.dema(tdf["close"], 14)
                out["TEMA_14"] = ind.tema(tdf["close"], 14)
                out["TRIMA_14"] = ind.trima(tdf["close"], 14)
                macd_line, macd_signal, macd_hist = ind.macd(tdf["close"])
                out["MACD"] = macd_line
                out["MACD_SIGNAL"] = macd_signal
                out["MACD_HIST"] = macd_hist
                out["RSI_14"] = ind.rsi(tdf["close"], 14)
                out["WILLR_14"] = ind.willr(tdf["high"], tdf["low"], tdf["close"], 14)
                out["CCI_20"] = ind.cci(tdf["high"], tdf["low"], tdf["close"], 20)
                out["ATR_14"] = ind.atr(tdf["high"], tdf["low"], tdf["close"], 14)
                out["OBV"] = ind.obv(tdf["close"], tdf["volume"])
                out["ROC_12"] = ind.roc(tdf["close"], 12)
                out["MOM_10"] = ind.mom(tdf["close"], 10)
                ma, upper, lower = ind.bbands(tdf["close"], 20)
                out["BB_MA"] = ma
                out["BB_UPPER"] = upper
                out["BB_LOWER"] = lower
                out["MIDPOINT_14"] = ind.midpoint(tdf["close"], 14)
                out["MIDPRICE_14"] = ind.midprice(tdf["high"], tdf["low"], 14)
                out["ticker"] = ticker
                indicator_frames.append(out.reset_index())
            if indicator_frames:
                indicators_df = pd.concat(indicator_frames, ignore_index=True)
                indicators_df = indicators_df.set_index(["date", "ticker"])
                all_data = all_data.merge(
                    indicators_df, left_index=True, right_index=True, how="left"
                )
            all_data.to_parquet("all_data.parquet")
            print("Unified data saved to all_data.parquet (with indicators)")
        else:
            print("No new data fetched.")

    def calculate_indicators(self, all_data):
        """
        Given a DataFrame indexed by [date, ticker] with columns open, high, low, close, volume,
        calculate technical indicators for each ticker and merge into the DataFrame.
        """
        av_key = get_api_key("alpha_vantage")
        if not av_key:
            return all_data
        av_api = AlphaVantageAPI(av_key)
        indicators = list(ALPHA_VANTAGE_TECHNICAL_INDICATORS.keys())
        indicator_frames = []
        for ticker in all_data.index.get_level_values("ticker").unique():
            for ind in indicators:
                try:
                    ind_df = av_api.technical_indicator(
                        symbol=ticker, indicator=ind, interval=self.interval, return_format="df"
                    )
                    if not ind_df.empty:
                        ind_df["ticker"] = ticker
                        ind_df["indicator"] = ind
                        indicator_frames.append(ind_df)
                except Exception:
                    continue
        if indicator_frames:
            indicators_df = pd.concat(indicator_frames)
            indicators_df = indicators_df.reset_index().pivot_table(
                index=["index", "ticker"],
                columns="indicator",
                values=indicators_df.columns.difference(["ticker", "indicator"])[0],
                aggfunc="first",
            )
            indicators_df.index.names = ["date", "ticker"]
            all_data = all_data.merge(indicators_df, left_index=True, right_index=True, how="left")
        return all_data


# Example usage:
# populator = DatabasePopulator()
# populator.run(start="2023-01-01", end="2023-12-31")
