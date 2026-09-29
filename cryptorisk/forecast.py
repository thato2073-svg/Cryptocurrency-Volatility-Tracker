from __future__ import annotations
import numpy as np
import pandas as pd

def ewma_volatility(log_returns: pd.Series, lam: float = 0.94, periods_per_year: int = 365) -> pd.Series:
    clean = log_returns.fillna(0.0)
    variance = pd.Series(index=clean.index, dtype=float)
    if clean.empty:
        return variance
    variance.iloc[0] = clean.var() if len(clean) > 1 else 0.0
    for i in range(1, len(clean)):
        variance.iloc[i] = lam * variance.iloc[i - 1] + (1 - lam) * clean.iloc[i - 1] ** 2
    return np.sqrt(variance) * np.sqrt(periods_per_year)

def garch_forecast(log_returns: pd.Series, horizon: int = 7) -> pd.Series:
    """Forecast annualized volatility with a GARCH(1,1) model."""
    from arch import arch_model
    clean = log_returns.dropna() * 100
    if len(clean) < 30:
        raise ValueError("At least 30 return observations are required for GARCH.")
    model = arch_model(clean, mean="Zero", vol="GARCH", p=1, q=1, rescale=False)
    fit = model.fit(disp="off")
    variance = fit.forecast(horizon=horizon).variance.iloc[-1]
    annualized = np.sqrt(variance) / 100 * np.sqrt(365)
    annualized.index = range(1, horizon + 1)
    return annualized
