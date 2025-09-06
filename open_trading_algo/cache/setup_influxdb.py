#!/usr/bin/env python3
"""
Setup script for InfluxDB time series database.

This script helps set up InfluxDB for local development and configures
the time series cache for the trading algorithm project.

Requirements:
- Docker (for running InfluxDB locally)
- Python 3.8+

Usage:
    python open_trading_algo/cache/setup_influxdb.py
"""

import os
import subprocess
import time
import requests
from pathlib import Path


def check_docker():
    """Check if Docker is installed and running."""
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Docker is not installed or not accessible")
            return False

        result = subprocess.run(["docker", "info"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Docker daemon is not running")
            return False

        print("✅ Docker is installed and running")
        return True
    except FileNotFoundError:
        print("❌ Docker command not found")
        return False


def start_influxdb():
    """Start InfluxDB using Docker."""
    print("🚀 Starting InfluxDB container...")

    # Stop any existing container
    subprocess.run(["docker", "stop", "trading-influxdb"], capture_output=True)
    subprocess.run(["docker", "rm", "trading-influxdb"], capture_output=True)

    # Start new container
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        "trading-influxdb",
        "-p",
        "8086:8086",
        "-v",
        "trading-influxdb-data:/var/lib/influxdb2",
        "-v",
        "trading-influxdb-config:/etc/influxdb2",
        "-e",
        "DOCKER_INFLUXDB_INIT_MODE=setup",
        "-e",
        "DOCKER_INFLUXDB_INIT_USERNAME=admin",
        "-e",
        "DOCKER_INFLUXDB_INIT_PASSWORD=admin123",
        "-e",
        "DOCKER_INFLUXDB_INIT_ORG=trading-org",
        "-e",
        "DOCKER_INFLUXDB_INIT_BUCKET=trading-data",
        "-e",
        "DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=my-token",
        "influxdb:2.7",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Failed to start InfluxDB: {result.stderr}")
        return False

    print("✅ InfluxDB container started")
    return True


def wait_for_influxdb(max_attempts=30):
    """Wait for InfluxDB to be ready."""
    print("⏳ Waiting for InfluxDB to be ready...")

    for attempt in range(max_attempts):
        try:
            response = requests.get("http://localhost:8086/health", timeout=5)
            if response.status_code == 200:
                print("✅ InfluxDB is ready!")
                return True
        except requests.RequestException:
            pass

        print(f"   Attempt {attempt + 1}/{max_attempts}...")
        time.sleep(2)

    print("❌ InfluxDB failed to start within timeout")
    return False


def create_config_file():
    """Create configuration file for the time series cache."""
    config_dir = Path(__file__).parent / "config"
    config_dir.mkdir(exist_ok=True)

    config_content = """# InfluxDB Configuration for Time Series Cache
influxdb:
  url: "http://localhost:8086"
  token: "my-token"
  org: "trading-org"
  bucket: "trading-data"

# Data retention policies (in days)
retention:
  price_data: 3650  # 10 years
  signals: 365      # 1 year

# Query optimization settings
query:
  default_range: "-365d"
  max_points_per_query: 100000
"""

    config_path = config_dir / "timeseries_config.yaml"
    with open(config_path, "w") as f:
        f.write(config_content)

    print(f"✅ Configuration file created: {config_path}")


def test_connection():
    """Test the InfluxDB connection."""
    try:
        from open_trading_algo.cache.timeseries_cache import TimeSeriesCache

        print("🔍 Testing connection to InfluxDB...")
        cache = TimeSeriesCache()

        # Test basic connection
        info = cache.get_database_info()
        if info:
            print("✅ Successfully connected to InfluxDB!")
            print(f"   Organization: {info.get('organization')}")
            print(f"   Bucket: {info.get('bucket')}")
            print(f"   Price data points: {info.get('price_data_points', 0)}")
            print(f"   Signals points: {info.get('signals_points', 0)}")
            cache.close()
            return True
        else:
            print("❌ Failed to get database info")
            cache.close()
            return False

    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up InfluxDB for Trading Algorithm Time Series Cache")
    print("=" * 60)

    # Check Docker
    if not check_docker():
        print("\n📋 Please install Docker and try again:")
        print("   https://docs.docker.com/get-docker/")
        return

    # Start InfluxDB
    if not start_influxdb():
        return

    # Wait for InfluxDB to be ready
    if not wait_for_influxdb():
        return

    # Create configuration file
    create_config_file()

    # Test connection
    if test_connection():
        print("\n🎉 Setup completed successfully!")
        print("\n📖 Usage:")
        print("   from open_trading_algo.cache.timeseries_cache import TimeSeriesCache")
        print("   cache = TimeSeriesCache()")
        print("   # Use cache.store_price_data(), cache.get_price_data(), etc.")
    else:
        print("\n⚠️  Setup completed but connection test failed.")
        print("   Please check your InfluxDB configuration.")


if __name__ == "__main__":
    main()
