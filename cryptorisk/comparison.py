from __future__ import annotations
import pandas as pd
from cryptorisk.analytics import add_returns

def align_returns(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    series = {}
    for name, frame in frames.items():
        data = add_returns(frame)
        daily = data["log_return"].resample("1D").sum(min_count=1)
        series[name] = daily
    return pd.DataFrame(series).dropna(how="all")

def correlation_matrix(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return align_returns(frames).corr()
