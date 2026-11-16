# app.py

from typing import List

import numpy as np
import yfinance as yf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Indian Stock Forecaster – Educational Purpose",
    description=(
        "A simple, educational model that uses recent closing prices and a "
        "straight-line trend to predict the next 5 days of Indian stock prices. "
        "For educational use only. Not for trading."
    ),
    version="1.0.0",
)


class PredictionResponse(BaseModel):
    ticker: str
    last_close: float
    predictions: List[float]
    disclaimer: str


def get_closing_prices(ticker: str) -> np.ndarray:
    """
    Download the last 2 years of daily data and return an array
    of closing prices as floats.
    """
    df = yf.download(ticker, period="2y", interval="1d")
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}")

    # Take the Close column, drop missing values, make a 1-D float array
    closes = df["Close"].dropna().to_numpy(dtype=float).flatten()

    if closes.size < 20:
        raise ValueError("Not enough data to build a simple trend")

    return closes


def forecast_next_5_days(closes: np.ndarray) -> List[float]:
    """
    Fit a very simple straight-line trend on the last N days and
    project 5 days into the future.

    This is NOT a real trading model — it's purely for learning.
    """
    # Use at most the last 60 days to fit the line
    n = min(60, closes.size)
    y = closes[-n:]
    x = np.arange(n, dtype=float)

    # Fit y ≈ a*x + b
    a, b = np.polyfit(x, y, 1)

    predictions: List[float] = []
    for i in range(1, 6):
        xi = n - 1 + i  # days ahead
        yi = a * xi + b
        predictions.append(round(float(yi), 2))

    return predictions


@app.get("/predict", response_model=PredictionResponse)
def predict(ticker: str):
    """
    Example: /predict?ticker=RELIANCE.NS

    For NSE stocks, use the .NS suffix (e.g., TCS.NS, HDFCBANK.NS, INFY.NS).
    """
    try:
        closes = get_closing_prices(ticker)
        last_close = float(closes[-1])
        preds = forecast_next_5_days(closes)
    except ValueError as e:
        # User / data problems (no data, too little data, etc.)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Unexpected internal problem
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    return PredictionResponse(
        ticker=ticker.upper(),
        last_close=round(last_close, 2),
        predictions=preds,
        disclaimer=(
            "This is a toy educational model using a simple trend line on "
            "historical prices. Predictions are NOT investment advice. "
            "Do NOT use for trading."
        ),
    )


@app.get("/")
def root():
    return {
        "message": "Welcome to the Indian Stock Forecaster – Educational Purpose.",
        "example": "/predict?ticker=RELIANCE.NS",
    }
