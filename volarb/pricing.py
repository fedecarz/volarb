import numpy as np
from scipy.stats import norm


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Compute d1 component of the Black-Scholes formula.
    Params
    S:      Current spot price of the underlying.
    K:      Strike price of the option.
    T:      Time to maturity in years.
    r:      Annualised risk-free rate as a decimal.
    sigma:  Annualised volatility of the underlying as a decimal.
    """
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Compute d2 component of the Black-Scholes formula.
    Params
    S:      Current spot price of the underlying.
    K:      Strike price of the option.
    T:      Time to maturity in years.
    r:      Annualised risk-free rate as a decimal.
    sigma:  Annualised volatility of the underlying as a decimal.
    """
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    """
    Compute the Black-Scholes theoretical price of a European option.
    Params
    S:              Current spot price of the underlying.
    K:              Strike price of the option.
    T:              Time to maturity in years.
    r:              Annualised risk-free rate as a decimal.
    sigma:          Annualised volatility of the underlying as a decimal.
    option_type:    "call" or "put".
    """

    # Input validation
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")
    
    # Edge case: option has expired -> return the intrinsic payoff
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    
    # Edge case: vol is zero -> equivalent to a forward contract on max(S_T - K, 0)
    if sigma <= 0:
        if option_type == "call":
            return max(S - K * np.exp(-r * T), 0)
        else:
            return max(K * np.exp(-r * T) - S, 0)

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    

def bs_delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    """
    Compute the Black-Scholes delta of a European option.
    Delta measures the sensitivity of the option price to a unit change in the spot price of the underlying. For a call, delta is in [0, 1]. For a put, delta is in [-1, 0].
    Params
    S:              Current spot price of the underlying.
    K:              Strike price of the option.
    T:              Time to maturity in years.
    r:              Annualised risk-free rate as a decimal.
    sigma:          Annualised volatility of the underlying as a decimal.
    option_type:    "call" or "put".
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put'.")

    d1 = _d1(S, K, T, r, sigma)

    if option_type == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Compute the Black-Scholes gamma of a European option.
    Gamma measures the rate of change of delta with respect to the spot price.
    Params
    S:              Current spot price of the underlying.
    K:              Strike price of the option.
    T:              Time to maturity in years.
    r:              Annualised risk-free rate as a decimal.
    sigma:          Annualised volatility of the underlying as a decimal.
    """
    d1 = _d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Compute the Black-Scholes vega of a European option.
    Vega measures the sensitivity of the option price to a unit change in volatility.
    Returned as sensitivity to a 1-point (100%) move in sigma — divide by 100 to get sensitivity per 1 volatility point.
    Params
    S:              Current spot price of the underlying.
    K:              Strike price of the option.
    T:              Time to maturity in years.
    r:              Annualised risk-free rate as a decimal.
    sigma:          Annualised volatility of the underlying as a decimal.
    """
    d1 = _d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)


def bs_theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    """
    Compute the Black-Scholes theta of a European option.
    Theta measures the sensitivity of the option price to the passage of time.
    Returned as the change in option value per calendar day (divided by 365).
    Params
    S:              Current spot price of the underlying.
    K:              Strike price of the option.
    T:              Time to maturity in years.
    r:              Annualised risk-free rate as a decimal.
    sigma:          Annualised volatility of the underlying as a decimal.
    option_type:    "call" or "put".
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put'.")

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)

    common = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

    if option_type == "call":
        return (common - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        return (common + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365


def bs_rho(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    """
    Compute the Black-Scholes rho of a European option.
    Rho measures the sensitivity of the option price to a unit change in the risk-free rate.
    Params
    S:              Current spot price of the underlying.
    K:              Strike price of the option.
    T:              Time to maturity in years.
    r:              Annualised risk-free rate as a decimal.
    sigma:          Annualised volatility of the underlying as a decimal.
    option_type:    "call" or "put".
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put'.")

    d2 = _d2(S, K, T, r, sigma)

    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> dict:
    """
    Compute all Black-Scholes Greeks for a European option.
    Optimised: d1, d2 and shared terms computed once and reused across all Greeks.
    Params
    S:              Current spot price of the underlying.
    K:              Strike price of the option.
    T:              Time to maturity in years.
    r:              Annualised risk-free rate as a decimal.
    sigma:          Annualised volatility of the underlying as a decimal.
    option_type:    "call" or "put".
    Returns dict with keys:
        "price":    float, theoretical option price
        "delta":    float
        "gamma":    float
        "vega":     float
        "theta":    float, per calendar day
        "rho":      float
    """

    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    is_call = option_type == "call"

    # Edge case: expiry → intrinsic payoff, greeks collapse
    if T <= 0:
        price = max(S - K, 0) if is_call else max(K - S, 0)
        if S > K:
            delta = 1.0 if is_call else 0.0
        elif S < K:
            delta = 0.0 if is_call else -1.0
        else:
            delta = 0.5 if is_call else -0.5
        return {"price": price,
                "delta": delta,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": 0.0,
                "rho": 0.0
                }

    # Edge case: zero vol → deterministic pricing, greeks degenerate
    if sigma <= 0:
        disc   = np.exp(-r * T)
        price  = max(S - K * disc, 0) if is_call else max(K * disc - S, 0)
        return {"price": price,
                "delta": 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": 0.0,
                "rho": 0.0
                }

    # Shared terms — computed once
    d1     = _d1(S, K, T, r, sigma)
    d2     = _d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    disc   = np.exp(-r * T)
    sqrtT  = np.sqrt(T)

    # Price
    if is_call:
        price = S * cdf_d1 - K * disc * cdf_d2
    else:
        price = K * disc * (1 - cdf_d2) - S * (1 - cdf_d1)

    # Delta
    delta = cdf_d1 if is_call else cdf_d1 - 1.0

    # Gamma (identical for call and put)
    gamma = pdf_d1 / (S * sigma * sqrtT)

    # Vega (identical for call and put) — per 1.0 change in sigma
    vega = S * pdf_d1 * sqrtT

    # Theta — per calendar day (divided by 365)
    common = -(S * pdf_d1 * sigma) / (2 * sqrtT)
    if is_call:
        theta = (common - r * K * disc * cdf_d2) / 365
    else:
        theta = (common + r * K * disc * (1 - cdf_d2)) / 365

    # Rho
    if is_call:
        rho = K * T * disc * cdf_d2
    else:
        rho = -K * T * disc * (1 - cdf_d2)

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho
        }