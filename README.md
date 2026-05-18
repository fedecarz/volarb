# volarb

**A Python backtesting library for volatility arbitrage strategies on SPY options using long and short straddles — demonstrating the Variance Risk Premium empirically.**

*Personal project — built to implement and evaluate delta-hedged straddle strategies on real market data.*

---

## Overview

volarb provides a complete backtesting pipeline for volatility arbitrage on SPY:

- price ATM straddles using Black-Scholes and compute Greeks
- delta-hedge dynamically using SPY shares with configurable rebalancing frequency
- roll positions automatically at a target DTE threshold
- decompose daily P&L into gamma, theta, vega, hedge cost, and residual components
- compare strategies across multiple years: long straddle, short straddle, and adaptive

The main demonstration lives in `demo.ipynb` — end-to-end walkthrough with data, backtests, attribution, and charts.

---

## The Core Idea — Variance Risk Premium

Equity options are systematically overpriced relative to the volatility that actually realizes. The market consistently pays more for protection than volatility warrants. This gap is the **Variance Risk Premium (VRP)**:

```
VRP = IV − RV
```

When VRP > 0, implied vol exceeds realized vol — options are expensive, and selling volatility collects premium that more than compensates for realized moves.

volarb tests this empirically on SPY using three strategies:

| Strategy | Bet | Expected outcome |
|----------|-----|-----------------|
| **Long straddle** (delta-hedged) | RV > IV | Loses systematically — VRP exists |
| **Short straddle** (delta-hedged) | IV > RV | Gains systematically — captures VRP |
| **Adaptive** | Switch based on RV vs IV signal | Mixed — signal lag limits performance |

---

## Results

**Key takeaways:**
- The long straddle loses in every year tested — IV is structurally above RV. The VRP exists and is persistent.
- The short straddle gains in every year tested — it systematically captures the VRP.
- The adaptive strategy works in stable regimes (2023, 2024) but breaks down in volatile ones (2022) due to signal lag and switching costs.

### P&L Attribution Example — Short Straddle 2023

The waterfall below decomposes where the $3,923 P&L comes from:

- **Theta +$6,478** — the primary driver. Time decay collected every day.
- **Gamma −$5,200** — the main cost. Being short gamma means realized moves hurt.
- **Vega +$261** — small positive contribution from IV declining during 2023.
- **Hedge cost −$53** — minimal friction from daily delta rebalancing.

The short straddle profits because **theta beats gamma**. This is the empirical signature of the VRP.

---

## Setup

```
Underlying:     SPY (S&P 500 ETF)
Instrument:     ATM straddle (call + put, same strike)
Hedge:          SPY shares, delta-hedged daily
IV proxy:       VIX / 100
Risk-free rate: ^IRX / 100
Roll:           Open 30-day straddle, close at 7 DTE (~23 days per cycle)
Data:           yfinance, daily close prices
Capital:        $100,000 initial cash
Contract size:  100 shares per contract
```

---

## Quickstart

```python
from volarb.data import fetch_market_data
from volarb.costs import TransactionCosts
from volarb.backtest import Backtest, BacktestConfig
from volarb.portfolio import LongStraddle, ShortStraddle
from volarb.analytics import summary_stats, plot_equity_curve
from volarb.pnl_attribution import attribute_pnl, aggregate_attribution

costs = TransactionCosts()
data  = fetch_market_data("2023-01-01", "2024-01-01")

# run all three strategies
results = {
    "Long Straddle":  Backtest(BacktestConfig(position_class=LongStraddle), data, costs).run(),
    "Short Straddle": Backtest(BacktestConfig(position_class=ShortStraddle), data, costs).run(),
    "Adaptive":       Backtest(BacktestConfig(adaptive=True), data, costs).run(),
}

# summary stats
stats = summary_stats(results["Short Straddle"])

# equity curve comparison
plot_equity_curve(results, title="SPY Straddle Strategies — 2023")

# P&L attribution
attr = attribute_pnl(results["Short Straddle"].daily)
agg  = aggregate_attribution(attr)
# {'gamma_pnl': -5200, 'theta_pnl': 6478, 'vega_pnl': 261, ...}
```

---

## Modules

| Module | Purpose |
|--------|---------|
| **pricing.py** | Black-Scholes pricer and all five Greeks implemented from scratch |
| **portfolio.py** | `LongStraddle` and `ShortStraddle` position classes with Greeks aggregation |
| **data.py** | Fetch SPY, VIX, and IRX from yfinance; compute rolling realized vol |
| **costs.py** | Transaction costs — option bid-ask spread, commissions, share spread |
| **hedger.py** | `DeltaHedger` — computes required hedge and rebalances cash and shares |
| **backtest.py** | Main engine — rolls positions, rebalances hedge, records daily and per-roll P&L |
| **pnl_attribution.py** | Decomposes daily P&L into gamma, theta, vega, hedge cost, and residual |
| **analytics.py** | Summary stats, equity curve, RV vs IV chart, P&L waterfall, per-roll bar chart |

---

## P&L Attribution

The `pnl_attribution` module applies a Taylor expansion to decompose daily P&L:

```
total_pnl  = gamma_pnl + theta_pnl + vega_pnl + hedge_cost + residual

gamma_pnl  = 0.5 × Γ_scaled × (ΔS)²
theta_pnl  = (θ_t + θ_{t-1}) / 2          # averaged to smooth intraday
vega_pnl   = ν_scaled × Δσ (in vol points)
hedge_cost = − transaction costs from delta rebalancing
residual   = total_pnl − sum(above)
```

**Units:** all Greeks in `daily` are scaled by `n_contracts × contract_size` — they are already in dollar terms. Vega is per vol point (per 0.01 of sigma), so `Δσ` is multiplied by 100 before applying.

The residual captures everything not explained by the first and second-order Greeks. In practice it can represent a significant fraction of P&L: the Taylor expansion is an approximation, and higher-order terms (vanna, volga), discrete rebalancing, and model error all contribute.

---

## Installation

Clone the repo and install in editable mode:

```bash
git clone https://github.com/fedecarz/volarb.git
cd volarb
pip install -e .
```

---

## Repository Structure

```
volarb/
│
├── volarb/
│   ├── __init__.py
│   ├── pricing.py          # Black-Scholes pricer and Greeks
│   ├── portfolio.py        # LongStraddle and ShortStraddle position classes
│   ├── data.py             # Market data fetch and realized vol computation
│   ├── costs.py            # Transaction cost model
│   ├── hedger.py           # Delta hedger
│   ├── backtest.py         # Backtesting engine
│   ├── pnl_attribution.py  # P&L decomposition
│   └── analytics.py        # Summary stats and plots
│
├── demo.ipynb          # End-to-end demo
├── setup.py
├── requirements.txt
└── README.md
```

---

## Requirements

Python 3.8+ · numpy · scipy · pandas · yfinance · matplotlib · jupyter

---

## Limitations

| Area | Current implementation |
|------|----------------------|
| IV proxy | VIX / 100 — index-level, not SPY chain |
| Realized vol | 21-day rolling close-to-close |
| Strike selection | Round spot price |
| Adaptive signal | RV vs IV with 21-day lag |
| P&L attribution | Taylor expansion to second order |
| Capital model | Fixed $100k, no margin model |
| Transaction costs | Fixed spread + commission |

### On the scope of this project

volarb is not a production backtesting engine. It is a study project — built to implement the mechanics of delta-hedged options strategies from scratch and to observe the VRP empirically on real data. The backtest makes several simplifying assumptions (daily close prices, VIX as IV proxy, no margin model, fixed transaction costs) that would not be acceptable in a real trading context.

### On the adaptive strategy

The adaptive strategy is academically interesting but practically limited. The RV signal is computed on a 21-day rolling window — it is backward-looking by construction, and lags the true volatility regime by approximately three weeks. In a rapidly changing environment, the signal consistently arrives after the regime has already shifted, generating switching costs without capturing the intended edge.

### On the VIX as IV proxy

The consequence of using VIX as IV proxy is that option premiums in the backtest are priced using a vol that is not the exact market IV for the specific contract being traded. This is a documented limitation.

---

## License

MIT
