from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import yfinance as yf
import ta
import warnings

warnings.filterwarnings("ignore")

app = FastAPI(title="Stock Analysis API", version="1.0.0")


class StockRequest(BaseModel):
    tickers: List[str]


def fetch_stock_data(ticker, period="3mo", interval="1d"):
    return yf.download(ticker, period=period, interval=interval, progress=False)


def apply_indicators(df, ema_short=12, ema_long=26, macd_fast=12, macd_slow=26, macd_signal=9):
    close = df["Close"].astype(float).squeeze()
    df["EMA_Short"] = ta.trend.EMAIndicator(close, window=ema_short).ema_indicator()
    df["EMA_Long"] = ta.trend.EMAIndicator(close, window=ema_long).ema_indicator()

    macd = ta.trend.MACD(
        close, window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_signal
    )
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    return df


def find_last_crossover(df):
    ema_cross = None
    macd_cross = None

    ema_diff = df["EMA_Short"] - df["EMA_Long"]
    for i in range(len(ema_diff) - 1, 0, -1):
        if ema_diff.iloc[i] > 0 and ema_diff.iloc[i - 1] <= 0:
            ema_cross = {"signal": "BUY", "date": str(df.index[i].date())}
            break
        elif ema_diff.iloc[i] < 0 and ema_diff.iloc[i - 1] >= 0:
            ema_cross = {"signal": "SELL", "date": str(df.index[i].date())}
            break

    macd_diff = df["MACD"] - df["MACD_Signal"]
    for i in range(len(macd_diff) - 1, 0, -1):
        if macd_diff.iloc[i] > 0 and macd_diff.iloc[i - 1] <= 0:
            macd_cross = {"signal": "BUY", "date": str(df.index[i].date())}
            break
        elif macd_diff.iloc[i] < 0 and macd_diff.iloc[i - 1] >= 0:
            macd_cross = {"signal": "SELL", "date": str(df.index[i].date())}
            break

    return ema_cross, macd_cross


@app.post("/analyze")
async def analyze_stocks(request: StockRequest) -> Dict[str, Any]:
    results = {}
    for ticker in request.tickers:
        try:
            df = fetch_stock_data(ticker)
            df = apply_indicators(df)
            ema_cross, macd_cross = find_last_crossover(df)
            results[ticker] = {
                "EMA_Crossover": ema_cross or "No EMA crossover found",
                "MACD_Crossover": macd_cross or "No MACD crossover found",
            }
        except Exception as e:
            results[ticker] = {"error": str(e)}
    return {"status": "success", "data": results}
