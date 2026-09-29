from __future__ import annotations
import numpy as np
import pandas as pd

def add_returns(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["return"] = data["price"].pct_change()
    data["log_return"] = np.log(data["price"] / data["price"].shift(1))
    return data

def rolling_volatility(log_returns: pd.Series, window: int = 30, periods_per_year: int = 365) -> pd.Series:
    return log_returns.rolling(window).std() * np.sqrt(periods_per_year)

def annualized_volatility(log_returns: pd.Series, periods_per_year: int = 365) -> float:
    clean = log_returns.dropna()
    return float(clean.std() * np.sqrt(periods_per_year)) if len(clean) > 1 else float("nan")

def max_drawdown(prices: pd.Series) -> float:
    running_peak = prices.cummax()
    drawdown = prices / running_peak - 1
    return float(drawdown.min())

def drawdown_series(prices: pd.Series) -> pd.Series:
    return prices / prices.cummax() - 1

def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    clean = returns.dropna()
    return float(np.quantile(clean, 1 - confidence)) if len(clean) else float("nan")

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 365) -> float:
    clean = returns.dropna()
    if len(clean) < 2 or clean.std() == 0:
        return float("nan")
    daily_rf = risk_free_rate / periods_per_year
    return float((clean.mean() - daily_rf) / clean.std() * np.sqrt(periods_per_year))

def risk_summary(frame: pd.DataFrame) -> dict[str, float]:
    data = add_returns(frame)
    return {
        "price": float(data["price"].iloc[-1]),
        "change": float(data["return"].iloc[-1]),
        "volatility": annualized_volatility(data["log_return"]),
        "max_drawdown": max_drawdown(data["price"]),
        "var_95": historical_var(data["return"], 0.95),
        "sharpe": sharpe_ratio(data["return"]),
    }
