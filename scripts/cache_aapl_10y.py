"""
Fetch 10 years of AAPL daily data from yfinance and cache it in the local database.
Only one request is made; all data is stored for future use.
"""
import yfinance as yf
import pandas as pd
from tradingview_algo.data_cache import DataCache


def main():
    ticker = "AAPL"
    period = "10y"
    interval = "1d"
    print(f"Fetching {ticker} {period} {interval} data from yfinance...")
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        print("No data fetched!")
        return
    df = df.rename(
        columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"}
    )
    print(f"Fetched {len(df)} rows. Caching in database...")
    cache = DataCache()
    cache.store_price_data(ticker, df)
    print("Done. Data is now cached for all future use.")


if __name__ == "__main__":
    main()
