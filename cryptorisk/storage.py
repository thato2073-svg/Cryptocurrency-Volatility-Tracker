from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path("cryptorisk.db")

def save_prices(asset: str, frame: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    data = frame.reset_index()[["timestamp", "price"]].copy()
    data["asset"] = asset
    data["timestamp"] = data["timestamp"].astype(str)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS prices (
                asset TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                price REAL NOT NULL,
                PRIMARY KEY (asset, timestamp)
            )"""
        )
        connection.executemany(
            "INSERT OR REPLACE INTO prices(asset, timestamp, price) VALUES (?, ?, ?)",
            data[["asset", "timestamp", "price"]].itertuples(index=False, name=None),
        )

def load_prices(asset: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame(columns=["price"])
    with sqlite3.connect(db_path) as connection:
        frame = pd.read_sql_query(
            "SELECT timestamp, price FROM prices WHERE asset = ? ORDER BY timestamp",
            connection,
            params=(asset,),
        )
    if frame.empty:
        return pd.DataFrame(columns=["price"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.set_index("timestamp")
