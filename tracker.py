import time
import requests
import pandas as pd
from datetime import datetime
import logging

COINS = ["bitcoin", "ethereum", "solana"]
VS_CURRENCY = "usd"

POLL_SECONDS = 30
ALERT_PCT = 2.0
WINDOW = 5
MAX_HISTORY = 500  # Limit DataFrame size

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()]
)

def fetch_prices(coins, vs_currency="usd"):
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": ",".join(coins),
        "vs_currencies": vs_currency
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        return {c: float(data[c][vs_currency]) for c in coins if c in data}
    except Exception as e:
        logging.error(f"Error fetching prices: {e}")
        return {}

def update_history(history, row):
    history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    # Limit DataFrame size
    if len(history) > MAX_HISTORY:
        history = history.iloc[-MAX_HISTORY:].reset_index(drop=True)
    return history

def update_metrics(history, coin):
    if coin in history.columns and history[coin].notna().sum() >= 2:
        history[f"{coin}_pct_change"] = history[coin].pct_change() * 100
        history[f"{coin}_vol"] = history[f"{coin}_pct_change"].rolling(WINDOW).std()
    return history

def main():
    history = pd.DataFrame()

    while True:
        ts = datetime.now()
        prices = fetch_prices(COINS, VS_CURRENCY)
        if not prices:
            logging.warning("No prices fetched. Skipping this cycle.")
            time.sleep(POLL_SECONDS)
            continue

        row = {"time": ts}
        for coin, price in prices.items():
            row[coin] = price

        history = update_history(history, row)

        for coin in COINS:
            history = update_metrics(history, coin)
            if f"{coin}_pct_change" in history.columns and f"{coin}_vol" in history.columns:
                latest_pct = history[f"{coin}_pct_change"].iloc[-1]
                latest_vol = history[f"{coin}_vol"].iloc[-1]
                if pd.notna(latest_pct) and abs(latest_pct) >= ALERT_PCT:
                    logging.info(f"[ALERT] {coin.upper()} moved {latest_pct:.2f}% (vol {latest_vol:.2f})")

        latest_prices = {coin: float(history[coin].iloc[-1]) for coin in COINS if coin in history.columns}
        logging.info(f"Prices: {latest_prices}")

        try:
            history.to_csv("price_history.csv", index=False)
        except Exception as e:
            logging.error(f"Error saving CSV: {e}")

        logging.info(f"Sleeping {POLL_SECONDS}s...\n")
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
