# CryptoRisk

A cryptocurrency **risk analytics and volatility forecasting platform** built with Python, Streamlit, Plotly and statistical volatility models.

CryptoRisk turns raw market prices into interpretable risk signals: historical and rolling volatility, drawdowns, Value at Risk, risk-adjusted returns, exponentially weighted volatility and GARCH forecasts.

## Why this project exists

Crypto prices are noisy. Looking at price alone does not tell you how much risk an asset has carried or how quickly market conditions are changing. CryptoRisk provides a compact research dashboard for exploring those questions.

## Features

- Near real-time historical market data from CoinGecko
- Bitcoin, Ethereum, Solana, Cardano and XRP support
- Simple and log returns
- 7, 14 and 30-period rolling volatility
- Annualized historical volatility
- Maximum drawdown and drawdown history
- Historical 95% Value at Risk (VaR)
- Sharpe-style risk-adjusted return metric
- EWMA volatility estimation
- GARCH(1,1) seven-period volatility forecast
- Cross-asset daily-return correlation heatmap
- SQLite market-data persistence
- EWMA forecast backtesting with MAE/RMSE
- FastAPI risk endpoints
- Interactive Plotly charts
- Streamlit caching and API error handling
- Unit tests with pytest
- GitHub Actions continuous integration

## Architecture

```text
CoinGecko API
     |
     v
cryptorisk/data.py
     |
     +----------------------+
     |                      |
     v                      v
analytics.py            forecast.py
returns / VaR           EWMA / GARCH
drawdown / Sharpe
     |                      |
     +----------+-----------+
                |
                v
          dashboard.py
       Streamlit + Plotly
```

## Project structure

```text
.
├── cryptorisk/
│   ├── analytics.py
│   ├── config.py
│   ├── data.py
│   └── forecast.py
├── tests/
│   └── test_analytics.py
├── .github/workflows/ci.yml
├── dashboard.py
├── pyproject.toml
└── requirements.txt
```

## Run locally

```bash
git clone https://github.com/thato2073-svg/Cryptocurrency-Volatility-Tracker.git
cd Cryptocurrency-Volatility-Tracker

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run dashboard.py

# Optional REST API
uvicorn api:app --reload
```

## Methodology

For price \(P_t\), log return is

```text
r_t = ln(P_t / P_(t-1))
```

Historical volatility is the standard deviation of log returns. CryptoRisk annualizes volatility using **365 periods per year** because cryptocurrency markets operate continuously.

Maximum drawdown is the largest observed decline from a running price peak.

Historical 95% VaR is estimated from the empirical lower 5% tail of observed simple returns. It describes historical downside behavior and is **not** a guarantee of future maximum loss.

EWMA variance follows

```text
sigma_t^2 = lambda * sigma_(t-1)^2 + (1-lambda) * r_(t-1)^2
```

with lambda = 0.94 by default.

The forecasting module also fits a GARCH(1,1) conditional variance model:

```text
sigma_t^2 = omega + alpha * epsilon_(t-1)^2 + beta * sigma_(t-1)^2
```

## Testing

```bash
pytest -q
```

Every push and pull request runs the test suite through GitHub Actions.

## Limitations

- CoinGecko availability and rate limits can affect data retrieval.
- Risk metrics depend on the selected historical sample.
- VaR can underestimate extreme tail events.
- GARCH is a statistical model, not a trading signal or price predictor.
- This project is for research and educational use, not financial advice.

## Roadmap

- Hosted public demo
- Additional forecast-model comparisons
- Optional authenticated market-data provider

## Author

**Thato Olayinka**  
Computing Science + Economics, University of Alberta
