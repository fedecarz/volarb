from dataclasses import dataclass
from volarb.costs import TransactionCosts


@dataclass
class DeltaHedger:
    n_shares: float = 0.0    # stock postion (can be short)
    cash: float = 0.0        # cash account of the hedge leg

    def required_hedge(self, straddle, S, r, sigma, current_date) -> float:
        return - straddle.greeks(S, r, sigma, current_date)["delta"]


    def rebalance(self, straddle, S, r, sigma, current_date, costs: TransactionCosts) -> dict:
        target = self.required_hedge(straddle, S, r, sigma, current_date)
        trade = target - self.n_shares
        cost = costs.share_cost(trade_size=trade, spot=S)
        self.cash -= trade * S + cost
        self.n_shares = target
        return {
            "trade":  trade,   # shares bought (positive) or bought (negative)
            "cost":   cost,   # dollar transaction cost
            "n_shares": self.n_shares, # new position
            "cash":   self.cash,   # cash account after trade
        }
    

    def hedge_value(self, S) -> float:
        return (self.n_shares * S) + self.cash
