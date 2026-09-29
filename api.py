from fastapi import FastAPI, HTTPException
from cryptorisk.analytics import risk_summary
from cryptorisk.config import ASSETS
from cryptorisk.data import MarketDataError, fetch_market_history

app = FastAPI(title="CryptoRisk API", version="2.0.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/assets")
def assets():
    return ASSETS

@app.get("/risk/{coin_id}")
def risk(coin_id: str, days: int = 90):
    try:
        frame = fetch_market_history(coin_id, days)
        return {"asset": coin_id, "days": days, **risk_summary(frame)}
    except MarketDataError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
