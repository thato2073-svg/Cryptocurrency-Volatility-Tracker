from pathlib import Path
import pandas as pd
from cryptorisk.storage import load_prices, save_prices

def test_sqlite_round_trip(tmp_path: Path):
    path = tmp_path / "test.db"
    frame = pd.DataFrame(
        {"price": [100.0, 101.0]},
        index=pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
    )
    frame.index.name = "timestamp"
    save_prices("bitcoin", frame, path)
    loaded = load_prices("bitcoin", path)
    assert loaded["price"].tolist() == [100.0, 101.0]
