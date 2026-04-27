import yfinance as yf
import numpy as np
import pandas as pd

"""
Note: historical volatility should be replaced with implied volatility (IV) extracted by inverting the Black-Scholes formula given real observed option prices. Historical volatility is used here for simplicity and data availability reasons.
"""

hedge_instrument = "SPY"
risk_free_ticker = "^IRX"   # 13 WEEK TREASURY BILL (^IRX)
vix_ticker = "^VIX"
trading_days = 252


def fetch_prices(tickers: list[str], start: str = None, end: str = None, period: str = "1y") -> pd.DataFrame:
    """
    Fetch daily adjusted closing prices for a list of tickers.

    Params:
    tickers:    list of ticker symbols e.g. ["AAPL", "MSFT", "NVDA"]
    start:      start date in "YYYY-MM-DD" format. Ignored if period is provided.
    end:        end date in "YYYY-MM-DD" format. Ignored if period is provided.
    period:     yfinance period string e.g. "3mo", "6mo", "1y", "2y". If provided, start and end are ignored. Default is "1y"
    """

    if period:
        raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    else:
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    prices = raw["Close"]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    prices.dropna(how="all")

    return prices

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from a DataFrame of prices.
    Params
    prices:     DataFrame of prices as returned by fetch_prices.
    """

    return np.log(prices/prices.shift(1)).dropna()

def compute_historical_volatility(prices: pd.DataFrame, window: int = 30):
    """
    Compute annualised rolling historical volatility from prices.
    Note: Implied volatility would be more appropriate in a real hedging context as it reflects the market's forward-looking expectation of risk. This function serves as a practical approximation.
    Params
    
    prices:     DataFrame of prices as returned by fetch_prices.
    window:     Rolling window in trading days. Default is 30.
    """

    returns = compute_returns(prices=prices)
    vol = returns.rolling(window=window).std() * np.sqrt(trading_days)
    return vol.dropna()


def compute_betas(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Compute rolling betas for each stock relative to SPY.
    Beta is estimated daily using OLS over a rolling window: beta = cov(stock returns, SPY returns) / var(SPY returns)
    Params
    returns: daily log returns for all tickers including SPY
    window: rolling window in trading days. Default is 60.
    """
    spy_returns = returns["SPY"]
    betas = {}

    for ticker in returns.columns:
        if ticker == "SPY":
            betas[ticker] = pd.Series(1.0, index=returns.index)
            continue
        cov = returns[ticker].rolling(window).cov(spy_returns)
        var = spy_returns.rolling(window).var()
        betas[ticker] = cov / var

    return pd.DataFrame(betas).dropna()


def fetch_risk_free() -> float:
    """
    Fetch the current annualised risk-free rate using the 3-month US T-bill yield (^IRX).
    Returns the most recent available value as a decimal.
    Default is 0.05 if data is unavailable.
    """

    try:
        irx = yf.Ticker(risk_free_ticker)
        rate = irx.history(period="5d")["Close"].dropna().iloc[-1]
        return float(rate) / 100
    except Exception:
        print("Could not fetch risk-free rate. Defaulting to 0.05.")
        return 0.05
    

def fetch_vix() -> float:
    """
    Fetch historical VIX index levels as a proxy for SPY implied volatility.
    Returns daily VIX closing levels as decimals.
    """

    try:
        vix = yf.Ticker(vix_ticker)
        hist = vix.history(period="1y")["Close"].dropna()
        return hist / 100
    except Exception:
        print("Could not fetch VIX data.")
        return pd.Series(dtype=float)
    

def fetch_all (tickers: list[str], start: str = None, end: str = None, period: str = "1y", vol_window: int = 30) -> dict:
    """
    Convenience function that fetches all data needed. SPY is automatically included.
    Fetches prices, computes returns and historical volatility, and retrieves the risk-free rate and VIX.
    Params
    tickers:    Portfolio tickers e.g. ["AAPL", "MSFT", "NVDA"]. SPY is added automatically.
    start:      Start date in "YYYY-MM-DD" format.
    end:        End date in "YYYY-MM-DD" format.
    period:     yfinance period string e.g. "3mo", "6mo", "1y", "2y". Overrides start/end if provided. Default is "1y"
    vol_window: Rolling window for historical volatility computation. Default is 30.
    Returns a dict with keys:
        "prices"    : pd.DataFrame of adjusted closing prices
        "returns"   : pd.DataFrame of daily log returns
        "vol"       : pd.DataFrame of annualised rolling historical volatility
        "vix"       : pd.Series of daily VIX levels as decimals
        "risk_free" : float, annualised risk-free rate
    """
    
    all_tickers = list(set(tickers + [hedge_instrument]))
    prices = fetch_prices(all_tickers, start=start, end=end, period=period)
    rets = compute_returns(prices)
    vol = compute_historical_volatility(prices, window=vol_window)
    vix = fetch_vix()
    rf = fetch_risk_free()

    return {
        "prices":   prices,
        "returns":  rets,
        "vol":      vol,
        "vix":      vix,
        "risk_free": rf,
        "betas":     compute_betas(rets)
    }



    

