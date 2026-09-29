import numpy as np
import pandas as pd

from cryptorisk.analytics import add_returns, annualized_volatility, historical_var, max_drawdown, sharpe_ratio
from cryptorisk.forecast import ewma_volatility

def sample_prices():
    return pd.DataFrame({"price": [100.0, 110.0, 99.0, 120.0]})

def test_add_returns():
    result = add_returns(sample_prices())
    assert np.isnan(result["return"].iloc[0])
    assert np.isclose(result["return"].iloc[1], 0.10)

def test_max_drawdown():
    assert np.isclose(max_drawdown(sample_prices()["price"]), -0.10)

def test_historical_var_is_lower_tail():
    returns = pd.Series([-0.10, -0.05, 0.0, 0.05, 0.10])
    assert historical_var(returns, 0.80) <= 0

def test_annualized_volatility_nonnegative():
    returns = pd.Series([0.01, -0.02, 0.015, -0.01])
    assert annualized_volatility(returns) >= 0

def test_sharpe_zero_variance_returns_nan():
    assert np.isnan(sharpe_ratio(pd.Series([0.01, 0.01, 0.01])))

def test_ewma_shape_matches_input():
    returns = pd.Series([np.nan, 0.01, -0.02, 0.015])
    result = ewma_volatility(returns)
    assert len(result) == len(returns)
    assert (result >= 0).all()
