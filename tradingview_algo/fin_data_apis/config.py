# moved from tradingview_algo/live/config.py
from pathlib import Path
import yaml


class LiveDataConfig:
    def __init__(self, config_path: Path):
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.update_rate = int(cfg.get("update_rate", 300))
        self.tickers = list(cfg.get("tickers", []))
        self.source = str(cfg.get("source", "yahoo"))
        self.api_key = str(cfg.get("api_key", ""))
        self.fields = list(cfg.get("fields", ["price", "volume"]))
