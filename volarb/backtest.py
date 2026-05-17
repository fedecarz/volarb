from dataclasses import dataclass
from datetime import timedelta
import pandas as pd
from volarb.portfolio import LongStraddle, OptionPosition, ShortStraddle
from volarb.hedger import DeltaHedger
from volarb.costs import TransactionCosts


@dataclass
class BacktestConfig:
    position_class: type = LongStraddle
    strike_fn: object = None          # callable: spot -> strike
    maturity_days: int = 30
    roll_when_dte: int = 7
    rebalance_freq: int = 1
    n_contracts: float = 1.0
    initial_cash: float = 100_000
    adaptive: bool = False

    def __post_init__(self):
        if self.strike_fn is None:
            self.strike_fn = lambda spot: round(spot)


@dataclass
class BacktestResults:
    daily: pd.DataFrame
    rolls: pd.DataFrame
    config: BacktestConfig


class Backtest:
    def __init__(self, config: BacktestConfig, market_data: pd.DataFrame, costs: TransactionCosts):
        self.config      = config
        self.market_data = market_data
        self.costs       = costs

    def run(self) -> BacktestResults:

        position           = None
        hedger             = None
        iv_open            = None
        cycle_initial_cash = None
        daily_records      = []
        roll_records       = []

        for i, (date, row) in enumerate(self.market_data.iterrows()):
            spot  = row["spot"]
            sigma = row["sigma"]
            r     = row["r"]

            # Roll
            if position is None or position.days_to_expiry(date) <= self.config.roll_when_dte:

                if position is not None:
                    close_value  = position.value(spot, r, sigma, date)
                    close_cost   = self.costs.option_costs(self.config.n_contracts, n_legs=position.N_LEGS)
                    position_pnl = close_value - position.opening_premium * position.n_contracts * 100
                    hedge_pnl    = (hedger.cash + hedger.n_shares * spot) - cycle_initial_cash
                    pnl_gross    = position_pnl + hedge_pnl
                    pnl_net      = pnl_gross - close_cost

                    roll_records.append({
                        "open_date":    position.open_date,
                        "close_date":   date,
                        "strike":       position.strike,
                        "iv_open":      iv_open,
                        "pnl_position": position_pnl,
                        "pnl_hedge":    hedge_pnl,
                        "pnl_gross":    pnl_gross,
                        "costs":        close_cost,
                        "pnl_net":      pnl_net,
                    })

                    rollover_cash = hedger.cash + hedger.n_shares * spot + close_value - close_cost
                else:
                    rollover_cash = self.config.initial_cash

                # Adaptive strategy
                if self.config.adaptive:
                    rv = row["rv"]
                    position_class = LongStraddle if rv > sigma else ShortStraddle
                else:
                    position_class = self.config.position_class

                # Open new position
                strike      = self.config.strike_fn(spot)
                T           = self.config.maturity_days / 365
                iv_open     = sigma
                option_cost = self.costs.option_costs(
                    self.config.n_contracts,
                    n_legs=position_class.N_LEGS
                )

                # Create position to get signed premium
                position = position_class(
                    strike          = strike,
                    expiry          = date + timedelta(days=self.config.maturity_days),
                    open_date       = date,
                    n_contracts     = self.config.n_contracts,
                    opening_premium = 0.0,   # temporary
                )

                # Compute signed premium per share via value() at T
                premium_per_share  = position.value(spot, r, sigma, date) / (self.config.n_contracts * 100)
                position.opening_premium = premium_per_share

                # Cash flow: subtract signed premium (positive = paying, negative = receiving)
                new_cash           = rollover_cash - premium_per_share * self.config.n_contracts * 100 - option_cost
                hedger             = DeltaHedger(cash=new_cash)
                cycle_initial_cash = new_cash

            # Rebalance
            hedge_cost = 0.0
            if i % self.config.rebalance_freq == 0:
                rebalance_result = hedger.rebalance(position, spot, r, sigma, date, self.costs)
                hedge_cost = rebalance_result["cost"]

            # Daily record
            position_value = position.value(spot, r, sigma, date)
            hedge_val      = hedger.hedge_value(spot)
            nav            = position_value + hedge_val
            greeks         = position.greeks(spot, r, sigma, date)

            daily_records.append({
                "date":           date,
                "spot":           spot,
                "sigma":          sigma,
                "r":              r,
                "position_value": position_value,
                "hedge_value":    hedge_val,
                "nav":            nav,
                "delta":          greeks["delta"],
                "gamma":          greeks["gamma"],
                "vega":           greeks["vega"],
                "theta":          greeks["theta"],
                "hedge_cost":     hedge_cost,
            })

        return BacktestResults(
            daily=pd.DataFrame(daily_records),
            rolls=pd.DataFrame(roll_records),
            config=self.config,
        )