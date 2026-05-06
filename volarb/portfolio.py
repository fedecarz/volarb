from dataclasses import dataclass
from datetime import date
from volarb.pricing import bs_price, bs_greeks

@dataclass
class Straddle:
    strike: float
    expiry: date
    open_date: date
    n_contracts: float = 1.0
    contract_size: int = 100
    opening_premium: float = 0.0

    def time_to_expiry(self, current_date: date) -> float:
        years = max((self.expiry - current_date).days / 365, 0)
        return years
    

    def days_to_expiry(self, current_date: date) -> int:
        days = max((self.expiry - current_date).days, 0)
        return days

    def is_expired(self, current_date: date) -> bool:
        return current_date >= self.expiry


    def value(self, S: float, r: float, sigma: float, current_date: date) -> float:
        T = self.time_to_expiry(current_date)
        call_value = bs_price(S, self.strike, T, r, sigma, "call")
        put_value = bs_price(S, self.strike, T, r, sigma, "put")
        straddle_price_per_share = call_value + put_value
        return straddle_price_per_share * self.n_contracts * self.contract_size


    def greeks(self, S: float, r: float, sigma: float, current_date: date) -> dict:
        T = self.time_to_expiry(current_date)
        call_greeks = bs_greeks(S, self.strike, T, r, sigma, "call")
        put_greeks = bs_greeks(S, self.strike, T, r, sigma, "put")
        agg_greeks = {
            "delta": call_greeks["delta"] + put_greeks["delta"],
            "gamma": call_greeks["gamma"] + put_greeks["gamma"],
            "vega": (call_greeks["vega"] + put_greeks["vega"]) / 100,
            "theta": call_greeks["theta"] + put_greeks["theta"],
            "price": call_greeks["price"] + put_greeks["price"],
        }
        scale = self.n_contracts * self.contract_size
        return {k: v * scale for k, v in agg_greeks.items()}