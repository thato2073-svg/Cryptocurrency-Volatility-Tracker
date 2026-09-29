import numpy as np
import pandas as pd
from cryptorisk.backtest import error_metrics, ewma_backtest

def test_backtest_produces_metrics():
    index = pd.date_range("2026-01-01", periods=60, freq="D")
    returns = pd.Series(np.sin(np.arange(60)) / 100, index=index)
    result = ewma_backtest(returns)
    metrics = error_metrics(result)
    assert not result.empty
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
