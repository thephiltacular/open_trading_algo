"""
Alpha Vantage API interface for stocks, alpha intelligence, fundamental data, and free technical indicators.
"""
import requests
from tradingview_algo.fin_data_apis.secure_api import get_api_key
from tradingview_algo.fin_data_apis.rate_limit import rate_limit

ALPHA_VANTAGE_TECHNICAL_INDICATORS = {
    # Trend indicators
    "SMA": "Simple Moving Average",
    "EMA": "Exponential Moving Average",
    "WMA": "Weighted Moving Average",
    "DEMA": "Double Exponential Moving Average",
    "TEMA": "Triple Exponential Moving Average",
    "TRIMA": "Triangular Moving Average",
    "KAMA": "Kaufman Adaptive Moving Average",
    "MAMA": "MESA Adaptive Moving Average",
    "T3": "Triple Exponential Moving Average (T3)",
    # Momentum indicators
    "MACD": "Moving Average Convergence/Divergence",
    "MACDEXT": "MACD with controllable MA type",
    "STOCH": "Stochastic Oscillator",
    "STOCHF": "Stochastic Fast",
    "RSI": "Relative Strength Index",
    "STOCHRSI": "Stochastic RSI",
    "WILLR": "Williams %R",
    "ADX": "Average Directional Movement Index",
    "ADXR": "Average Directional Movement Rating",
    "APO": "Absolute Price Oscillator",
    "PPO": "Percentage Price Oscillator",
    "MOM": "Momentum",
    "BOP": "Balance Of Power",
    "CCI": "Commodity Channel Index",
    "CMO": "Chande Momentum Oscillator",
    "ROC": "Rate of Change",
    "ROCR": "Rate of Change Ratio",
    # Volatility indicators
    "ATR": "Average True Range",
    "NATR": "Normalized Average True Range",
    "TRANGE": "True Range",
    # Volume indicators
    "AD": "Chaikin A/D Line",
    "ADOSC": "Chaikin A/D Oscillator",
    "OBV": "On Balance Volume",
    "HT_TRENDLINE": "Hilbert Transform - Instantaneous Trendline",
    # Cycle indicators
    "HT_SINE": "Hilbert Transform - SineWave",
    "HT_TRENDMODE": "Hilbert Transform - Trend vs Cycle Mode",
    "HT_DCPERIOD": "Hilbert Transform - Dominant Cycle Period",
    "HT_DCPHASE": "Hilbert Transform - Dominant Cycle Phase",
    "HT_PHASOR": "Hilbert Transform - Phasor Components",
    # Others
    "BBANDS": "Bollinger Bands",
    "MIDPOINT": "MidPoint over period",
    "MIDPRICE": "Midpoint Price over period",
    "SAR": "Parabolic SAR",
    "TRIMA": "Triangular Moving Average",
    "ULTOSC": "Ultimate Oscillator",
    "MFI": "Money Flow Index",
    "MINUS_DI": "Minus Directional Indicator",
    "PLUS_DI": "Plus Directional Indicator",
    "MINUS_DM": "Minus Directional Movement",
    "PLUS_DM": "Plus Directional Movement",
    "AROON": "Aroon",
    "AROONOSC": "Aroon Oscillator",
    "DX": "Directional Movement Index",
    "TRIX": "1-day Rate-Of-Change (ROC) of a Triple Smooth EMA",
    "BBANDS": "Bollinger Bands",
    "STOCH": "Stochastic Oscillator",
    "STOCHF": "Stochastic Fast",
    "STOCHRSI": "Stochastic RSI",
    "AD": "Chaikin A/D Line",
    "ADOSC": "Chaikin A/D Oscillator",
    "OBV": "On Balance Volume",
    "HT_TRENDLINE": "Hilbert Transform - Instantaneous Trendline",
    # ... (see Alpha Vantage docs for full list)
}


class AlphaVantageAPI:
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str = None):
        if not api_key:
            api_key = get_api_key("alpha_vantage")
        if not api_key:
            raise ValueError("Alpha Vantage API key not found in environment.")
        self.api_key = api_key

    @rate_limit("alpha_vantage")
    def stock_time_series(
        self,
        symbol: str,
        interval: str = "daily",
        outputsize: str = "compact",
        return_format: str = "json",
    ):
        """Fetch stock time series (INTRADAY, DAILY, WEEKLY, MONTHLY). return_format: 'json' or 'df'"""
        import pandas as pd

        function_map = {
            "intraday": "TIME_SERIES_INTRADAY",
            "daily": "TIME_SERIES_DAILY",
            "weekly": "TIME_SERIES_WEEKLY",
            "monthly": "TIME_SERIES_MONTHLY",
        }
        function = function_map.get(interval.lower(), "TIME_SERIES_DAILY")
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": outputsize,
        }
        if interval.lower() == "intraday":
            params["interval"] = "5min"  # default, can be parameterized
        resp = requests.get(self.BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if return_format == "df":
            # Find the time series key
            ts_key = next((k for k in data if "Time Series" in k), None)
            if ts_key and ts_key in data:
                df = pd.DataFrame.from_dict(data[ts_key], orient="index")
                df.index = pd.to_datetime(df.index)
                return df
            return pd.DataFrame()
        return data

    @rate_limit("alpha_vantage")
    def insider_transactions(self, symbol: str, return_format: str = "json"):
        """
        Fetch insider transactions for a given symbol using the Alpha Vantage API.
        return_format: 'json' or 'df'.
        """
        import pandas as pd

        params = {
            "function": "INSIDER_TRADING",
            "symbol": symbol,
            "apikey": self.api_key,
        }
        resp = requests.get(self.BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if return_format == "df":
            # Try to find the key with the transactions
            tx_key = next((k for k in data if "transactions" in k.lower()), None)
            if tx_key and tx_key in data:
                return pd.DataFrame(data[tx_key])
            return pd.DataFrame()
        return data

    @rate_limit("alpha_vantage")
    def alpha_intelligence(
        self, symbol: str, function: str = "NEWS_SENTIMENT", return_format: str = "json"
    ):
        """Fetch Alpha Intelligence endpoints (e.g., news sentiment, analyst upgrades/downgrades). return_format: 'json' or 'df'"""
        import pandas as pd

        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.api_key,
        }
        resp = requests.get(self.BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if return_format == "df":
            # Try to find the key with the main data
            main_key = next((k for k in data if isinstance(data[k], list)), None)
            if main_key and main_key in data:
                return pd.DataFrame(data[main_key])
            return pd.DataFrame()
        return data

    @rate_limit("alpha_vantage")
    def fundamental_data(self, symbol: str, function: str = "OVERVIEW", return_format: str = "json"):
        """Fetch fundamental data (OVERVIEW, INCOME_STATEMENT, BALANCE_SHEET, CASH_FLOW, EARNINGS, LISTING_STATUS, etc.). return_format: 'json' or 'df'"""
        import pandas as pd

        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.api_key,
        }
        resp = requests.get(self.BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if return_format == "df":
            # If the main data is a list, convert to DataFrame; else, single row
            main_key = next((k for k in data if isinstance(data[k], list)), None)
            if main_key and main_key in data:
                return pd.DataFrame(data[main_key])
            return pd.DataFrame([data])
        return data

    @rate_limit("alpha_vantage")
    def technical_indicator(
        self,
        symbol: str,
        indicator: str,
        interval: str = "daily",
        time_period: int = 14,
        series_type: str = "close",
        return_format: str = "json",
    ):
        """Fetch any free technical indicator (see Alpha Vantage docs for indicator names). return_format: 'json' or 'df'"""
        import pandas as pd

        params = {
            "function": indicator,
            "symbol": symbol,
            "interval": interval,
            "time_period": time_period,
            "series_type": series_type,
            "apikey": self.api_key,
        }
        resp = requests.get(self.BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if return_format == "df":
            # Find the key with indicator data
            ind_key = next((k for k in data if "Technical Analysis" in k), None)
            if ind_key and ind_key in data:
                df = pd.DataFrame.from_dict(data[ind_key], orient="index")
                df.index = pd.to_datetime(df.index)
                return df
            return pd.DataFrame()
        return data


# Example usage:
# av = AlphaVantageAPI()
# ts = av.stock_time_series("AAPL", interval="daily")
# news = av.alpha_intelligence("AAPL", function="NEWS_SENTIMENT")
# overview = av.fundamental_data("AAPL", function="OVERVIEW")
# rsi = av.technical_indicator("AAPL", indicator="RSI", interval="daily", time_period=14)
