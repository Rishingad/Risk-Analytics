"""
stress_testing.py — Historical Stress Testing Module.

Replays historical market crash scenarios through the current portfolio
to estimate potential P&L impacts and drawdowns.

Scenarios included:
  • 2008 Global Financial Crisis (Sep – Dec 2008)
  • 2020 COVID-19 Crash (Feb – Apr 2020)

The module downloads scenario-period returns from yfinance, applies them
to a user-defined portfolio (default: equal-weight), and outputs:
  • P&L impact tables (total loss, max single-day loss, CVaR).
  • Cumulative drawdown charts for each scenario.

Usage
-----
    from src.stress_testing import run_stress_tests, plot_drawdowns
    results = run_stress_tests(portfolio_value=1_000_000)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import os

# ================================================================
# SCENARIO DEFINITIONS
# ================================================================

SCENARIOS = {
    "2008 GFC (Sep–Dec 2008)": {
        "start": "2008-09-01",
        "end":   "2008-12-31",
        "description": "Lehman Brothers collapse and global credit freeze.",
    },
    "2020 COVID Crash (Feb–Apr 2020)": {
        "start": "2020-02-01",
        "end":   "2020-04-30",
        "description": "Pandemic-driven global market selloff.",
    },
}

# Default portfolio assets (same as data_loader.py)
ASSETS = ["^GSPC", "^IXIC", "AAPL", "MSFT", "JPM"]


# ================================================================
# DATA RETRIEVAL
# ================================================================

def fetch_scenario_data(tickers, start, end):
    """
    Download price data for a specific stress scenario period.

    Parameters
    ----------
    tickers : list of str
        Asset tickers.
    start, end : str
        Date range for the scenario.

    Returns
    -------
    pd.DataFrame
        Adjusted closing prices.
    """
    prices = yf.download(tickers, start=start, end=end,
                         auto_adjust=True, progress=False)["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    return prices.dropna()


def compute_scenario_returns(prices):
    """
    Compute daily log returns from scenario prices.

    Parameters
    ----------
    prices : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    return np.log(prices / prices.shift(1)).dropna()


# ================================================================
# P&L IMPACT CALCULATION
# ================================================================

def portfolio_pnl(returns, weights=None, portfolio_value=1_000_000):
    """
    Calculate P&L metrics when scenario returns are applied to a portfolio.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily asset returns during the stress period.
    weights : array-like or None
        Portfolio weights. Defaults to equal-weight.
    portfolio_value : float
        Notional portfolio value in dollars.

    Returns
    -------
    dict
        Keys: 'daily_pnl', 'cumulative_pnl', 'total_loss', 'total_loss_pct',
              'max_daily_loss', 'max_daily_loss_pct', 'max_drawdown',
              'max_drawdown_pct', 'cvar_95', 'worst_5_days'.
    """
    n_assets = returns.shape[1]
    if weights is None:
        weights = np.ones(n_assets) / n_assets
    weights = np.array(weights)

    # Portfolio-level daily return
    port_ret = returns.dot(weights)

    # Daily P&L in dollars
    daily_pnl = port_ret * portfolio_value

    # Cumulative P&L
    cumulative_pnl = daily_pnl.cumsum()

    # Cumulative return and drawdown
    cum_return = (1 + port_ret).cumprod()
    running_max = cum_return.cummax()
    drawdown = (cum_return - running_max) / running_max

    # CVaR at 95%
    alpha = 0.05
    cutoff = port_ret.quantile(alpha)
    cvar_95 = -port_ret[port_ret <= cutoff].mean()

    # Worst 5 days
    worst_5 = port_ret.nsmallest(5)

    return {
        "daily_pnl":          daily_pnl,
        "cumulative_pnl":     cumulative_pnl,
        "port_returns":       port_ret,
        "cum_return":         cum_return,
        "drawdown":           drawdown,
        "total_loss":         round(cumulative_pnl.iloc[-1], 2),
        "total_loss_pct":     round(port_ret.sum() * 100, 2),
        "max_daily_loss":     round(daily_pnl.min(), 2),
        "max_daily_loss_pct": round(port_ret.min() * 100, 2),
        "max_drawdown_pct":   round(drawdown.min() * 100, 2),
        "cvar_95":            round(cvar_95 * 100, 4),
        "worst_5_days":       worst_5,
    }


# ================================================================
# MASTER STRESS TEST RUNNER
# ================================================================

def run_stress_tests(tickers=None, weights=None, portfolio_value=1_000_000,
                     scenarios=None):
    """
    Run all stress scenarios and compile results.

    Parameters
    ----------
    tickers : list of str or None
        Defaults to ASSETS.
    weights : array-like or None
        Defaults to equal-weight.
    portfolio_value : float
        Notional portfolio value.
    scenarios : dict or None
        Defaults to SCENARIOS.

    Returns
    -------
    dict
        {scenario_name: {'metrics': dict, 'pnl_data': dict, 'info': dict}}
    """
    if tickers is None:
        tickers = ASSETS
    if scenarios is None:
        scenarios = SCENARIOS

    all_results = {}

    for name, info in scenarios.items():
        print(f"  ⏳ Running scenario: {name} ...", end="", flush=True)

        prices = fetch_scenario_data(tickers, info["start"], info["end"])
        returns = compute_scenario_returns(prices)
        pnl = portfolio_pnl(returns, weights, portfolio_value)

        all_results[name] = {
            "metrics": {
                "Scenario":             name,
                "Period":               f"{info['start']} → {info['end']}",
                "Trading Days":         len(returns),
                "Total P&L ($)":        pnl["total_loss"],
                "Total P&L (%)":        pnl["total_loss_pct"],
                "Max Daily Loss ($)":   pnl["max_daily_loss"],
                "Max Daily Loss (%)":   pnl["max_daily_loss_pct"],
                "Max Drawdown (%)":     pnl["max_drawdown_pct"],
                "CVaR 95% (daily %)":   pnl["cvar_95"],
            },
            "pnl_data": pnl,
            "info":     info,
        }
        print(" done")

    return all_results


def impact_table(results):
    """
    Format stress test results as a comparison DataFrame.

    Parameters
    ----------
    results : dict
        Output of ``run_stress_tests``.

    Returns
    -------
    pd.DataFrame
    """
    rows = [v["metrics"] for v in results.values()]
    df = pd.DataFrame(rows).set_index("Scenario")
    return df


# ================================================================
# DRAWDOWN VISUALISATION
# ================================================================

def plot_drawdowns(results, save_dir=None):
    """
    Plot cumulative P&L and drawdown charts for each stress scenario.

    Parameters
    ----------
    results : dict
        Output of ``run_stress_tests``.
    save_dir : str or None
        Directory to save figures.

    Returns
    -------
    list of matplotlib.figure.Figure
    """
    figs = []

    for name, res in results.items():
        pnl = res["pnl_data"]
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # --- Cumulative P&L ---
        ax1 = axes[0]
        cum_pnl = pnl["cumulative_pnl"]
        colors = ["#F44336" if v < 0 else "#4CAF50" for v in cum_pnl]
        ax1.fill_between(cum_pnl.index, cum_pnl.values, 0,
                         where=cum_pnl.values < 0,
                         color="#F44336", alpha=0.3, label="Loss")
        ax1.fill_between(cum_pnl.index, cum_pnl.values, 0,
                         where=cum_pnl.values >= 0,
                         color="#4CAF50", alpha=0.3, label="Gain")
        ax1.plot(cum_pnl.index, cum_pnl.values, color="#333", linewidth=1.2)
        ax1.set_ylabel("Cumulative P&L ($)", fontsize=11)
        ax1.set_title(f"Stress Test: {name}", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.2)
        ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")

        # --- Drawdown ---
        ax2 = axes[1]
        dd = pnl["drawdown"] * 100
        ax2.fill_between(dd.index, dd.values, 0,
                         color="#FF7043", alpha=0.5)
        ax2.plot(dd.index, dd.values, color="#D32F2F", linewidth=1.2)
        ax2.set_ylabel("Drawdown (%)", fontsize=11)
        ax2.set_xlabel("Date", fontsize=11)
        ax2.grid(alpha=0.2)

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            safe_name = name.replace(" ", "_").replace("–", "-")
            path = os.path.join(save_dir, f"stress_{safe_name}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  📊 Saved: {path}")

        figs.append(fig)

    return figs


# ================================================================
# CLI ENTRY POINT
# ================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  Historical Stress Testing")
    print("=" * 65)
    print(f"  Portfolio : Equal-weight across {ASSETS}")
    print(f"  Notional  : $1,000,000\n")

    results = run_stress_tests(portfolio_value=1_000_000)

    print("\n" + "=" * 65)
    print("  P&L Impact Table")
    print("=" * 65)
    table = impact_table(results)
    print(table.to_string())

    # Worst days detail
    for name, res in results.items():
        print(f"\n  📅 Worst 5 days — {name}:")
        w5 = res["pnl_data"]["worst_5_days"]
        for dt, ret in w5.items():
            print(f"     {dt.date()}  {ret*100:+.2f}%")

    # Drawdown charts
    plot_drawdowns(results, save_dir=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "report"))

    print("\n✅ Stress testing complete!")
