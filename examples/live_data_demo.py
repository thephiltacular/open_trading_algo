# Example usage for live data feed
import time
from pathlib import Path

from tradingview_algo.live_data import LiveDataFeed


def print_update(data):
    print("Live update:")
    for ticker, fields in data.items():
        print(f"{ticker}: {fields}")


if __name__ == "__main__":
    config_path = Path("live_data_config.yaml")
    feed = LiveDataFeed(config_path, on_update=print_update)
    print("Starting live data feed...")
    feed.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping feed...")
        feed.stop()
