import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from volarb.backtest import BacktestResults

def summary_stats(result: BacktestResults) -> dict:
    daily_pnl = result.daily["nav"].diff().dropna()
    total_pnl = result.rolls["pnl_net"].sum()
    sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)
    cum_nav = result.daily["nav"]
    rolling_max = cum_nav.cummax()
    drawdown = (cum_nav - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    win_rate = (result.rolls["pnl_net"] > 0).mean()
    n_rolls = len(result.rolls)
    avg_pnl_roll = result.rolls["pnl_net"].mean()
    return_on_capital = total_pnl / result.config.initial_cash

    return {

    "total_pnl":          total_pnl,
    "return_on_capital":  return_on_capital,
    "sharpe":             sharpe,
    "max_drawdown":       max_drawdown,
    "win_rate":           win_rate,
    "n_rolls":            n_rolls,
    "avg_pnl_per_roll":   avg_pnl_roll,
    }


def plot_equity_curve(results: dict, title: str = "Equity Curve") -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    
    colors = {"Long Straddle": "red", "Short Straddle": "green", "Adaptive": "steelblue"}
    
    for name, result in results.items():
        cum_pnl = result.daily["nav"] - result.daily["nav"].iloc[0]
        ax.plot(result.daily["date"], cum_pnl, label=name, color=colors.get(name), linewidth=1.5)
    
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_rv_vs_iv(daily: pd.DataFrame, market_data: pd.DataFrame, title: str = "RV vs IV (VRP)") -> None:
    fig, ax = plt.subplots(figsize=(12, 5))

    dates = daily["date"]
    iv    = daily["sigma"]
    rv    = market_data.loc[market_data.index.isin(dates), "rv"].values

    ax.plot(dates, iv * 100, label="IV (VIX)", color="steelblue", linewidth=1.5)
    ax.plot(dates, rv * 100, label="RV (21d)", color="darkorange", linewidth=1.5)

    ax.fill_between(dates, iv * 100, rv * 100,
                    where=(iv > rv), alpha=0.2, color="green", label="IV > RV (VRP positive)")
    ax.fill_between(dates, iv * 100, rv * 100,
                    where=(iv <= rv), alpha=0.2, color="red", label="RV > IV (VRP negative)")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pnl_waterfall(agg: dict, title: str = "P&L Attribution") -> None:
    components = ["gamma_pnl", "theta_pnl", "vega_pnl", "hedge_cost", "residual"]
    labels     = ["Gamma", "Theta", "Vega", "Hedge Cost", "Residual", "Total"]
    values     = [agg[c] for c in components] + [agg["total_pnl"]]
    colors     = ["red" if v < 0 else "green" for v in values]
    colors[-1] = "steelblue"  # total sempre blu

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (50 if val >= 0 else -150),
                f"${val:,.0f}",
                ha="center", va="bottom", fontsize=9)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("P&L ($)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()


def plot_per_roll(rolls: pd.DataFrame, title: str = "P&L per Roll") -> None:
    fig, ax = plt.subplots(figsize=(12, 5))

    labels = [f"{r['open_date'].strftime('%b %d')}" for _, r in rolls.iterrows()]
    values = rolls["pnl_net"].values
    colors = ["green" if v > 0 else "red" for v in values]

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (10 if val >= 0 else -30),
                f"${val:,.0f}",
                ha="center", va="bottom", fontsize=8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("P&L ($)")
    ax.set_xlabel("Cycle Open Date")
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    from volarb.data import fetch_market_data
    from volarb.costs import TransactionCosts
    from volarb.backtest import Backtest, BacktestConfig
    from volarb.portfolio import LongStraddle, ShortStraddle
    from volarb.pnl_attribution import attribute_pnl, aggregate_attribution

    data  = fetch_market_data("2023-01-01", "2024-01-01")
    costs = TransactionCosts()

    result_long  = Backtest(BacktestConfig(position_class=LongStraddle), data, costs).run()
    result_short = Backtest(BacktestConfig(position_class=ShortStraddle), data, costs).run()
    result_adapt = Backtest(BacktestConfig(adaptive=True), data, costs).run()

    plot_equity_curve({
        "Long Straddle":  result_long,
        "Short Straddle": result_short,
        "Adaptive":       result_adapt,
    }, title="SPY Straddle Strategies — 2023")

    plot_rv_vs_iv(result_short.daily, data, title="RV vs IV — SPY 2023")

    attr = attribute_pnl(result_short.daily)
    agg  = aggregate_attribution(attr)
    plot_pnl_waterfall(agg, title="P&L Attribution — Short Straddle 2023")

    plot_per_roll(result_short.rolls, title="P&L per Roll — Short Straddle 2023")


    stats = summary_stats(result_short)
    print("\n--- Summary Stats (Short Straddle 2023) ---")
    for k, v in stats.items():
        print(f"  {k:25s}: {v:.4f}")