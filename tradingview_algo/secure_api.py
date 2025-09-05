import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env from project root if present
load_dotenv(dotenv_path=Path(__file__).parent.parent / "secrets.env")


def get_api_key(service: str) -> Optional[str]:
    """Get API key for a given service from environment variables (.env or system env)."""
    env_map = {
        "finnhub": "FINNHUB_API_KEY",
        "fmp": "FMP_API_KEY",
        "alpha_vantage": "ALPHA_VANTAGE_API_KEY",
        "twelve_data": "TWELVE_DATA_API_KEY",
    }
    key = env_map.get(service.lower())
    if key:
        return os.getenv(key)
    return None
