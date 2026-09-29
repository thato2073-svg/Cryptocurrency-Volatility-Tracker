from __future__ import annotations
import numpy as np
import pandas as pd
from cryptorisk.forecast import ewma_volatility

def realized_volatility(log_returns: pd.Series, window: int = 7, periods_per_year: int = 365) -> pd.Series:
    return log_returns.rolling(window).std() * np.sqrt(periods_per_year)

def ewma_backtest(log_returns: pd.Series, window: int = 7) -> pd.DataFrame:
    forecast = ewma_volatility(log_returns).shift(1)
    realized = realized_volatility(log_returns, window)
    result = pd.DataFrame({"forecast": forecast, "realized": realized}).dropna()
    result["absolute_error"] = (result["forecast"] - result["realized"]).abs()
    result["squared_error"] = (result["forecast"] - result["realized"]) ** 2
    return result

def error_metrics(backtest: pd.DataFrame) -> dict[str, float]:
    if backtest.empty:
        return {"mae": float("nan"), "rmse": float("nan")}
    return {
        "mae": float(backtest["absolute_error"].mean()),
        "rmse": float(np.sqrt(backtest["squared_error"].mean())),
    }
