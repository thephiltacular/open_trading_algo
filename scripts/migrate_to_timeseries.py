#!/usr/bin/env python3
"""
Migration script to move data from SQLite cache to InfluxDB time series cache.

This script helps migrate existing cached data from the SQLite-based DataCache
to the new InfluxDB-based TimeSeriesCache.

Usage:
    python scripts/migrate_to_timeseries.py
"""

import pandas as pd
from pathlib import Path
from open_trading_algo.cache.data_cache import DataCache
from open_trading_algo.cache.timeseries_cache import TimeSeriesCache


def get_all_tickers_from_sqlite(sqlite_cache: DataCache) -> list:
    """Get all tickers that have data in SQLite cache."""
    # This is a simplified approach - in practice, you'd query the database
    # For now, we'll use a predefined list or scan the database
    try:
        # Try to get tickers from a sample query
        # This would need to be adapted based on your actual data
        conn = sqlite_cache.conn
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT ticker FROM price_data")
        tickers = [row[0] for row in cursor.fetchall()]
        return tickers
    except Exception as e:
        print(f"Could not query tickers from SQLite: {e}")
        return []


def migrate_price_data(
    sqlite_cache: DataCache, ts_cache: TimeSeriesCache, tickers: list, batch_size: int = 50
):
    """Migrate price data from SQLite to InfluxDB."""
    print(f"📊 Migrating price data for {len(tickers)} tickers...")

    migrated_count = 0
    error_count = 0

    for i, ticker in enumerate(tickers):
        print(f"   Processing {ticker} ({i+1}/{len(tickers)})...")
        try:
            # Get all price data for this ticker from SQLite
            price_data = sqlite_cache.get_price_data(ticker)

            if not price_data.empty:
                # Store in InfluxDB
                ts_cache.store_price_data(ticker, price_data)
                migrated_count += 1
            else:
                print(f"   ⚠️  No price data found for {ticker}")

        except Exception as e:
            print(f"   ❌ Error migrating price data for {ticker}: {e}")
            error_count += 1

    print(f"✅ Migrated price data for {migrated_count} tickers")
    if error_count > 0:
        print(f"⚠️  {error_count} tickers had migration errors")


def migrate_signals(sqlite_cache: DataCache, ts_cache: TimeSeriesCache, tickers: list):
    """Migrate signals data from SQLite to InfluxDB."""
    print(f"🎯 Migrating signals data...")

    # Get all unique signal combinations from SQLite
    try:
        conn = sqlite_cache.conn
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT ticker, timeframe, signal_type
            FROM signals
        """
        )
        signal_combinations = cursor.fetchall()
    except Exception as e:
        print(f"Could not query signals from SQLite: {e}")
        return

    migrated_count = 0
    error_count = 0

    for i, (ticker, timeframe, signal_type) in enumerate(signal_combinations):
        print(
            f"   Processing {ticker}/{timeframe}/{signal_type} ({i+1}/{len(signal_combinations)})..."
        )
        try:
            # Get signals for this combination from SQLite
            signals_data = sqlite_cache.get_signals(ticker, timeframe, signal_type)

            if not signals_data.empty:
                # Store in InfluxDB
                ts_cache.store_signals(ticker, timeframe, signal_type, signals_data)
                migrated_count += 1

        except Exception as e:
            print(f"   ❌ Error migrating signals for {ticker}/{timeframe}/{signal_type}: {e}")
            error_count += 1

    print(f"✅ Migrated signals for {migrated_count} combinations")
    if error_count > 0:
        print(f"⚠️  {error_count} signal combinations had migration errors")


def verify_migration(sqlite_cache: DataCache, ts_cache: TimeSeriesCache, sample_tickers: list):
    """Verify that migration was successful by comparing sample data."""
    print("🔍 Verifying migration with sample data...")

    verification_results = []

    for ticker in sample_tickers[:3]:  # Check first 3 tickers
        try:
            # Get data from both caches
            sqlite_data = sqlite_cache.get_price_data(ticker)
            influx_data = ts_cache.get_price_data(ticker)

            result = {
                "ticker": ticker,
                "sqlite_points": len(sqlite_data),
                "influx_points": len(influx_data),
                "match": len(sqlite_data) == len(influx_data),
            }

            verification_results.append(result)

        except Exception as e:
            print(f"   ❌ Error verifying {ticker}: {e}")

    # Print verification results
    print("\n📋 Verification Results:")
    print("-" * 50)
    for result in verification_results:
        status = "✅" if result["match"] else "❌"
        print(
            f"{status} {result['ticker']}: SQLite={result['sqlite_points']}, "
            f"InfluxDB={result['influx_points']}"
        )

    return verification_results


def main():
    """Main migration function."""
    print("🚀 SQLite to InfluxDB Migration Tool")
    print("=" * 50)
    print("This tool migrates cached data from SQLite to InfluxDB time series database.")
    print()

    # Initialize caches
    print("🔌 Initializing caches...")
    try:
        sqlite_cache = DataCache()
        ts_cache = TimeSeriesCache()
        print("✅ Caches initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize caches: {e}")
        return

    try:
        # Get tickers to migrate
        print("🔍 Discovering tickers in SQLite cache...")
        tickers = get_all_tickers_from_sqlite(sqlite_cache)

        if not tickers:
            print("⚠️  No tickers found in SQLite cache")
            print("   You may need to manually specify tickers or check your SQLite database")
            return

        print(
            f"📋 Found {len(tickers)} tickers: {tickers[:5]}..."
            f"{' (showing first 5)' if len(tickers) > 5 else ''}"
        )

        # Confirm migration
        response = input(f"\n🚨 This will migrate data for {len(tickers)} tickers. Continue? (y/N): ")
        if response.lower() not in ["y", "yes"]:
            print("Migration cancelled.")
            return

        # Migrate price data
        migrate_price_data(sqlite_cache, ts_cache, tickers)

        # Migrate signals
        migrate_signals(sqlite_cache, ts_cache, tickers)

        # Verify migration
        if tickers:
            verify_migration(sqlite_cache, ts_cache, tickers)

        print("\n🎉 Migration completed!")
        print("\n💡 Next steps:")
        print("   1. Test your application with the new TimeSeriesCache")
        print("   2. Update your code to use TimeSeriesCache instead of DataCache")
        print("   3. Consider backing up and removing the old SQLite database")

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        print("Please check your database connections and try again.")

    finally:
        # Clean up
        sqlite_cache.close()
        ts_cache.close()


if __name__ == "__main__":
    main()
