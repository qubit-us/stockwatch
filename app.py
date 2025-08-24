from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import yfinance as yf
import pandas as pd
import ta
import warnings

warnings.filterwarnings("ignore")

app = FastAPI(title="Stock Analysis API", version="2.0.0")

# ✅ CORS for Lovable frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this down to your lovable domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StockRequest(BaseModel):
    tickers: List[str]
    # optional knobs if you want to tweak from frontend later
    period: str = "6mo"
    interval: str = "1d"
    ema_short: int = 12
    ema_long: int = 26
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    max_points: int = 200  # limit history to keep payload small


def fetch_stock_data(ticker: str, period="6mo", interval="1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise ValueError(f"No data for {ticker}")
    return df


def apply_indicators(
    df: pd.DataFrame,
    ema_short=12,
    ema_long=26,
    macd_fast=12,
    macd_slow=26,
    macd_signal=9,
) -> pd.DataFrame:
    close = df["Close"].astype(float)
    df["EMA_Short"] = ta.trend.EMAIndicator(close, window=ema_short).ema_indicator()
    df["EMA_Long"] = ta.trend.EMAIndicator(close, window=ema_long).ema_indicator()

    macd = ta.trend.MACD(
        close,
        window_slow=macd_slow,
        window_fast=macd_fast,
        window_sign=macd_signal,
    )
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()
    return df


def last_crossover(series_a: pd.Series, series_b: pd.Series):
    cross = None
    diff = series_a - series_b
    for i in range(len(diff) - 1, 0, -1):
        if pd.isna(diff.iloc[i]) or pd.isna(diff.iloc[i - 1]):
            continue
        if diff.iloc[i] > 0 and diff.iloc[i - 1] <= 0:
            cross = {"signal": "BUY", "date": str(series_a.index[i].date())}
            break
        elif diff.iloc[i] < 0 and diff.iloc[i - 1] >= 0:
            cross = {"signal": "SELL", "date": str(series_a.index[i].date())}
            break
    return cross


def build_history_payload(df: pd.DataFrame, limit: int = 200):
    # keep latest N rows
    if limit and len(df) > limit:
        df = df.iloc[-limit:]

    records = []
    for idx, row in df.iterrows():
        try:
            records.append({
                "date": str(idx.date()),
                "open": float(row["Open"]) if pd.notna(row["Open"]) else None,
                "high": float(row["High"]) if pd.notna(row["High"]) else None,
                "low": float(row["Low"]) if pd.notna(row["Low"]) else None,
                "close": float(row["Close"]) if pd.notna(row["Close"]) else None,
                "ema_short": float(row["EMA_Short"]) if pd.notna(row["EMA_Short"]) else None,
                "ema_long": float(row["EMA_Long"]) if pd.notna(row["EMA_Long"]) else None,
                "macd": float(row["MACD"]) if pd.notna(row["MACD"]) else None,
                "macd_signal": float(row["MACD_Signal"]) if pd.notna(row["MACD_Signal"]) else None,
                "macd_hist": float(row["MACD_Hist"]) if pd.notna(row["MACD_Hist"]) else None,
                "volume": float(row["Volume"]) if "Volume" in df.columns and pd.notna(row["Volume"]) else None,
            })
        except Exception:
            continue
    return records


@app.post("/analyze")
async def analyze(request: StockRequest) -> Dict[str, Any]:
    out = {}
    for t in request.tickers:
        try:
            df = fetch_stock_data(t, request.period, request.interval)
            df = apply_indicators(
                df,
                ema_short=request.ema_short,
                ema_long=request.ema_long,
                macd_fast=request.macd_fast,
                macd_slow=request.macd_slow,
                macd_signal=request.macd_signal,
            )
            ema_cross = last_crossover(df["EMA_Short"], df["EMA_Long"])
            macd_cross = last_crossover(df["MACD"], df["MACD_Signal"])
            history = build_history_payload(df, request.max_points)
            out[t] = {
                "EMA_Crossover": ema_cross or None,
                "MACD_Crossover": macd_cross or None,
                "history": history,  # ✅ charts feed on this
            }
        except Exception as e:
            out[t] = {"error": str(e)}
    return {"status": "success", "data": out}



// Old version for reference

# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import List, Dict, Any
# import yfinance as yf
# import ta
# import warnings

# warnings.filterwarnings("ignore")

# app = FastAPI(title="Stock Analysis API", version="1.0.0")


# class StockRequest(BaseModel):
#     tickers: List[str]


# def fetch_stock_data(ticker, period="3mo", interval="1d"):
#     return yf.download(ticker, period=period, interval=interval, progress=False)


# def apply_indicators(df, ema_short=12, ema_long=26, macd_fast=12, macd_slow=26, macd_signal=9):
#     close = df["Close"].astype(float).squeeze()
#     df["EMA_Short"] = ta.trend.EMAIndicator(close, window=ema_short).ema_indicator()
#     df["EMA_Long"] = ta.trend.EMAIndicator(close, window=ema_long).ema_indicator()

#     macd = ta.trend.MACD(
#         close, window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_signal
#     )
#     df["MACD"] = macd.macd()
#     df["MACD_Signal"] = macd.macd_signal()
#     return df


# def find_last_crossover(df):
#     ema_cross = None
#     macd_cross = None

#     ema_diff = df["EMA_Short"] - df["EMA_Long"]
#     for i in range(len(ema_diff) - 1, 0, -1):
#         if ema_diff.iloc[i] > 0 and ema_diff.iloc[i - 1] <= 0:
#             ema_cross = {"signal": "BUY", "date": str(df.index[i].date())}
#             break
#         elif ema_diff.iloc[i] < 0 and ema_diff.iloc[i - 1] >= 0:
#             ema_cross = {"signal": "SELL", "date": str(df.index[i].date())}
#             break

#     macd_diff = df["MACD"] - df["MACD_Signal"]
#     for i in range(len(macd_diff) - 1, 0, -1):
#         if macd_diff.iloc[i] > 0 and macd_diff.iloc[i - 1] <= 0:
#             macd_cross = {"signal": "BUY", "date": str(df.index[i].date())}
#             break
#         elif macd_diff.iloc[i] < 0 and macd_diff.iloc[i - 1] >= 0:
#             macd_cross = {"signal": "SELL", "date": str(df.index[i].date())}
#             break

#     return ema_cross, macd_cross


# @app.post("/analyze")
# async def analyze_stocks(request: StockRequest) -> Dict[str, Any]:
#     results = {}
#     for ticker in request.tickers:
#         try:
#             df = fetch_stock_data(ticker)
#             df = apply_indicators(df)
#             ema_cross, macd_cross = find_last_crossover(df)
#             results[ticker] = {
#                 "EMA_Crossover": ema_cross or "No EMA crossover found",
#                 "MACD_Crossover": macd_cross or "No MACD crossover found",
#             }
#         except Exception as e:
#             results[ticker] = {"error": str(e)}
#     return {"status": "success", "data": results}
