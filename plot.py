import pandas as pd
import matplotlib.pyplot as plt

COINS = ["bitcoin", "ethereum", "solana"]

df = pd.read_csv("price_history.csv")
df["time"] = pd.to_datetime(df["time"])

for coin in COINS:
    if coin in df.columns:
        plt.figure()
        plt.plot(df["time"], df[coin])
        plt.title(f"{coin.upper()} Price")
        plt.xlabel("Time")
        plt.ylabel("Price (USD)")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()

for coin in COINS:
    vol_col = f"{coin}_vol"
    if vol_col in df.columns:
        plt.figure()
        plt.plot(df["time"], df[vol_col])
        plt.title(f"{coin.upper()} Rolling Volatility (std of % change)")
        plt.xlabel("Time")
        plt.ylabel("Volatility")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()
