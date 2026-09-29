import pandas as pd
import plotly.express as px
import streamlit as st

from cryptorisk.analytics import add_returns, drawdown_series, risk_summary, rolling_volatility
from cryptorisk.backtest import error_metrics, ewma_backtest
from cryptorisk.comparison import correlation_matrix
from cryptorisk.config import ASSETS
from cryptorisk.data import MarketDataError, fetch_market_history
from cryptorisk.forecast import ewma_volatility, garch_forecast
from cryptorisk.storage import save_prices

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
    save_prices(ASSETS[asset_name], raw)
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
    st.plotly_chart(px.histogram(data["return"].dropna(), nbins=50, title="Return distribution"), use_container_width=True)

st.subheader("GARCH volatility forecast")
try:
    forecast = garch_forecast(data["log_return"], horizon=7)
    forecast_frame = pd.DataFrame({"day": forecast.index, "forecast_volatility": forecast.values})
    st.plotly_chart(px.line(forecast_frame, x="day", y="forecast_volatility", markers=True), use_container_width=True)
except (ValueError, ImportError) as exc:
    st.info(f"GARCH forecast unavailable: {exc}")

st.subheader("Forecast validation")
backtest = ewma_backtest(data["log_return"])
metrics = error_metrics(backtest)
b1, b2 = st.columns(2)
b1.metric("EWMA backtest MAE", f"{metrics['mae']:.2%}")
b2.metric("EWMA backtest RMSE", f"{metrics['rmse']:.2%}")
if not backtest.empty:
    st.plotly_chart(px.line(backtest[["forecast", "realized"]], title="EWMA forecast vs realized volatility"), use_container_width=True)

st.subheader("Cross-asset correlation")
try:
    comparison_frames = {name: load_data(coin_id, history_days) for name, coin_id in ASSETS.items()}
    corr = correlation_matrix(comparison_frames)
    st.plotly_chart(
        px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, title="Daily log-return correlation"),
        use_container_width=True,
    )
except MarketDataError as exc:
    st.info(f"Cross-asset comparison unavailable: {exc}")

with st.expander("Methodology & limitations"):
    st.markdown("""
- Returns include simple percentage returns and log returns.
- Historical volatility is log-return standard deviation annualized with √365 because crypto trades continuously.
- Maximum drawdown measures the largest peak-to-trough decline in the selected history.
- Historical 95% VaR is the empirical 5th percentile of observed returns and does not bound future losses.
- EWMA weights recent squared returns more heavily (λ = 0.94).
- GARCH(1,1) models conditional variance; it is not a price forecast or trading signal.
- Backtest MAE/RMSE compare lagged EWMA estimates with a rolling realized-volatility proxy.
""")
