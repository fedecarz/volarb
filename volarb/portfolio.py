from dataclasses import dataclass
from datetime import date
from volarb.pricing import bs_price, bs_greeks


@dataclass
class OptionPosition:
    """
    Base class for any option position.
    """
    strike: float
    expiry: date
    open_date: date
    n_contracts: float = 1.0
    contract_size: int = 100
    opening_premium: float = 0.0

    def time_to_expiry(self, current_date: date) -> float:
        return max((self.expiry - current_date).days / 365, 0.0)

    def days_to_expiry(self, current_date: date) -> int:
        return max((self.expiry - current_date).days, 0)

    def is_expired(self, current_date: date) -> bool:
        return current_date >= self.expiry

    def value(self, S: float, r: float, sigma: float, current_date: date) -> float:
        """Total dollar value of the position (signed by long/short)."""
        T = self.time_to_expiry(current_date)
        scale = self.n_contracts * self.contract_size
        return sum(leg["qty"] * bs_price(S, leg["strike"], T, r, sigma, leg["type"]) for leg in self._legs) * scale

    def greeks(self, S: float, r: float, sigma: float, current_date: date) -> dict:
        """
        Aggregated dollar greeks.
        vega: per vol point
        theta: per calendar day
        """
        T = self.time_to_expiry(current_date)
        scale = self.n_contracts * self.contract_size

        agg = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "price": 0.0}
        for leg in self._legs:
            g = bs_greeks(S, leg["strike"], T, r, sigma, leg["type"])
            agg["delta"] += leg["qty"] * g["delta"]
            agg["gamma"] += leg["qty"] * g["gamma"]
            agg["vega"]  += leg["qty"] * g["vega"] / 100
            agg["theta"] += leg["qty"] * g["theta"]
            agg["price"] += leg["qty"] * g["price"]

        return {k: v * scale for k, v in agg.items()}


@dataclass
class LongStraddle(OptionPosition):
    """Long ATM straddle: +1 call +1 put at the same strike."""
    N_LEGS: int = 2

    def __post_init__(self):
        self._legs = [
            {"strike": self.strike, "type": "call", "qty": +1},
            {"strike": self.strike, "type": "put",  "qty": +1},
        ]


@dataclass
class ShortStraddle(OptionPosition):
    """Short ATM straddle: -1 call -1 put at the same strike."""
    N_LEGS: int = 2

    def __post_init__(self):
        self._legs = [
            {"strike": self.strike, "type": "call", "qty": -1},
            {"strike": self.strike, "type": "put",  "qty": -1},
        ]
