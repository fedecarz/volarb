import pandas as pd
import numpy as np
import yfinance as yf


def fetch_market_data(start: str, end: str) -> pd.DataFrame:
    prices = yf.download(["SPY", "^VIX", "^IRX"], start=start, end=end, auto_adjust=True)["Close"]
    prices = prices.rename(columns = {"SPY": "spot", "^VIX": "sigma", "^IRX": "r",})
    prices.columns.name = None
    prices["sigma"] = prices["sigma"] / 100
    prices["r"] = prices["r"] / 100
    return prices.dropna()


def compute_realized_vol(prices: pd.Series, window: int = 21) -> pd.Series:
    log_returns = np.log(prices / prices.shift(1))
    realized_vol = log_returns.rolling(window).std() * np.sqrt(252)
    return realized_vol

