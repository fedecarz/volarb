from dataclasses import dataclass

@dataclass
class TransactionCosts:
    option_spread_per_contract: float = 0.10
    option_commission_per_contract: float = 0.65
    share_spread_bps: float = 0.5


    def option_costs(self, n_contracts: float, n_legs: float = 2) -> float:
        return (self.option_spread_per_contract + self.option_commission_per_contract) * n_contracts * n_legs

    def share_cost(self, trade_size: float, spot: float) -> float:
        return abs(trade_size) * spot * (self.share_spread_bps / 10000)