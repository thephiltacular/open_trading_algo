"""
Script to set up the SQLite database for TradingViewAlgoDev.
Creates the database and tables if not present, unless a custom path is specified in config.
"""
import os
from open_trading_algo.cache.data_cache import DataCache, DB_PATH
import yaml


def main():
    print("Setting up TradingViewAlgoDev SQLite database...")
    cache = DataCache()  # Uses config if present, else default
    print(f"Database setup complete at: {cache.db_path}")
    cache.close()


if __name__ == "__main__":
    main()
