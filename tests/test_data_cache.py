import os
import pandas as pd
from tradingview_algo.data_cache import DataCache, DB_PATH


def test_database_setup_and_store():
    # Remove DB if exists for a clean test
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    cache = DataCache()
    # Should create the DB and table
    assert os.path.exists(DB_PATH)
    # Store some fake data
    df = pd.DataFrame(
        {
            "Open": [100, 101],
            "High": [102, 103],
            "Low": [99, 100],
            "Close": [101, 102],
            "Volume": [1000, 1100],
        },
        index=pd.to_datetime(["2023-01-01 09:30", "2023-01-01 09:31"]),
    )
    cache.store_price_data("AAPL", df)
    # Retrieve and check
    out = cache.get_price_data("AAPL")
    assert not out.empty
    assert "open" in out.columns
    assert out.shape[0] == 2
    cache.close()


def test_has_data():
    cache = DataCache()
    assert cache.has_data("AAPL")
    assert not cache.has_data("MSFT")
    cache.close()
