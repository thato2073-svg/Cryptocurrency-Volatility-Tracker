import pandas as pd
import plotly.express as px
import streamlit as st

from cryptorisk.analytics import add_returns, drawdown_series, risk_summary, rolling_volatility
from cryptorisk.config import ASSETS
from cryptorisk.data import MarketDataError, fetch_market_history
from cryptorisk.forecast import ewma_volatility, garch_forecast

st.set_page_config(page_title="CryptoRisk", page_icon="📈", layout="wide")

@st.cache_data(ttl=300)
def load_data(coin_id: str, days: int) -> pd.DataFrame:
    return fetch_market_history(coin_id, days)

st.title("CryptoRisk")
st.caption("Cryptocurrency volatility, downside risk and statistical forecasting.")

with st.sidebar:
    st.header("Market controls")
    asset_name = st.selectbox("Asset", list(ASSETS))
    history_days = st.select_slider("History", options=[30, 90, 180, 365], value=90)
    vol_window = st.select_slider("Volatility window", options=[7, 14, 30], value=30)

try:
    raw = load_data(ASSETS[asset_name], history_days)
except MarketDataError as exc:
    st.error(str(exc))
    st.stop()

data = add_returns(raw)
data["rolling_volatility"] = rolling_volatility(data["log_return"], vol_window)
data["drawdown"] = drawdown_series(data["price"])
data["ewma_volatility"] = ewma_volatility(data["log_return"])
summary = risk_summary(raw)

st.subheader(f"{asset_name} risk snapshot")
cols = st.columns(6)
cols[0].metric("Price", f"${summary['price']:,.2f}")
cols[1].metric("Latest return", f"{summary['change']:.2%}")
cols[2].metric("Annualized vol.", f"{summary['volatility']:.1%}")
cols[3].metric("Max drawdown", f"{summary['max_drawdown']:.1%}")
cols[4].metric("95% Hist. VaR", f"{summary['var_95']:.2%}")
cols[5].metric("Sharpe", f"{summary['sharpe']:.2f}")

left, right = st.columns(2)
with left:
    st.plotly_chart(px.line(data, y="price", title="Price history"), use_container_width=True)
with right:
    vol = data[["rolling_volatility", "ewma_volatility"]].rename(
        columns={"rolling_volatility": f"{vol_window}-period rolling", "ewma_volatility": "EWMA"}
    )
    st.plotly_chart(px.line(vol, title="Annualized volatility"), use_container_width=True)

left, right = st.columns(2)
with left:
    st.plotly_chart(px.area(data, y="drawdown", title="Drawdown from running peak"), use_container_width=True)
with right:
    returns = data["return"].dropna()
    st.plotly_chart(px.histogram(returns, nbins=50, title="Return distribution"), use_container_width=True)

st.subheader("GARCH volatility forecast")
try:
    forecast = garch_forecast(data["log_return"], horizon=7)
    forecast_frame = pd.DataFrame({"day": forecast.index, "forecast_volatility": forecast.values})
    st.plotly_chart(px.line(forecast_frame, x="day", y="forecast_volatility", markers=True), use_container_width=True)
    st.caption("GARCH(1,1) forecast. Values are annualized volatility estimates.")
except (ValueError, ImportError) as exc:
    st.info(f"GARCH forecast unavailable: {exc}")

with st.expander("Methodology"):
    st.markdown(
        """
        - Returns include simple percentage returns and log returns.
        - Historical volatility is the standard deviation of log returns, annualized with √365.
        - Maximum drawdown measures the largest peak-to-trough decline in the selected history.
        - Historical 95% VaR is the empirical 5th percentile of observed returns.
        - EWMA gives more weight to recent squared returns (λ = 0.94).
        - GARCH(1,1) models time-varying conditional variance and forecasts the next seven periods.
        """
    )
