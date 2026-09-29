from __future__ import annotations
import requests
import pandas as pd

BASE_URL = "https://api.coingecko.com/api/v3"

class MarketDataError(RuntimeError):
    pass

def fetch_market_history(coin_id: str, days: int = 90, vs_currency: str = "usd") -> pd.DataFrame:
    """Fetch historical prices from CoinGecko and return timestamp/price."""
    try:
        response = requests.get(
            f"{BASE_URL}/coins/{coin_id}/market_chart",
            params={"vs_currency": vs_currency, "days": days},
            timeout=15,
        )
        response.raise_for_status()
        prices = response.json().get("prices", [])
    except (requests.RequestException, ValueError) as exc:
        raise MarketDataError(f"Unable to load market data for {coin_id}.") from exc
    if not prices:
        raise MarketDataError(f"No price history returned for {coin_id}.")
    frame = pd.DataFrame(prices, columns=["timestamp", "price"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = frame.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    return frame
