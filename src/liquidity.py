"""
liquidity.py — Liquidity-adjusted Value-at-Risk (L-VaR).

Academic VaR assumes you can exit instantly at the mid price. Funds cannot: a
large position takes several days to unwind under a participation-rate cap, and
each clip pays the bid-ask spread. Liquidity was repeatedly flagged in the
hedge-fund risk literature as the missing piece, so this module adds it.

Two standard adjustments on top of the engine's existing VaR:

  1. Liquidation-horizon scaling (exogenous-spread / BDSS style):
         L-VaR = VaR · √(days_to_liquidate)
     where days_to_liquidate = position_shares / (ADV · participation_rate),
     capped at a configurable minimum of 1 day.

  2. Bid-ask cost: half-spread × position value, added to the loss.

ADV (average daily volume) and the latest price are pulled from yfinance and
cached to ``data/liquidity.csv`` so the dashboard does not hit the network on
every run.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from returns import load_prices, log_returns, portfolio_returns  # noqa: E402
from var_models import historical_var  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LIQ_FILE = os.path.join(DATA_DIR, "liquidity.csv")

# Default assumptions (overridable per call).
DEFAULT_PARTICIPATION = 0.20   # can trade 20% of ADV per day without moving the market
DEFAULT_SPREAD_BPS = {         # typical round-trip half-spread in basis points
    "^GSPC": 0.5, "^IXIC": 0.5, "AAPL": 1.0, "MSFT": 1.0, "JPM": 1.5,
}


# =====================================================================
# Market microstructure data (ADV + price), cached
# =====================================================================

def fetch_liquidity_data(tickers, refresh=False):
    """Return a DataFrame [ADV (shares), Price] per ticker, cached to CSV.

    Indices (^GSPC, ^IXIC) report index-level 'volume' which is not tradable
    share volume; their ADV is still useful as a relative liquidity proxy.
    """
    if not refresh and os.path.exists(LIQ_FILE):
        cached = pd.read_csv(LIQ_FILE, index_col=0)
        if set(tickers).issubset(cached.index):
            return cached.loc[tickers]

    import yfinance as yf

    rows = {}
    for t in tickers:
        hist = yf.Ticker(t).history(period="3mo")
        if hist.empty:
            rows[t] = {"ADV": np.nan, "Price": np.nan}
        else:
            rows[t] = {"ADV": float(hist["Volume"].mean()),
                       "Price": float(hist["Close"].iloc[-1])}
    df = pd.DataFrame(rows).T
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(LIQ_FILE)
    return df


# =====================================================================
# Liquidity-adjusted VaR
# =====================================================================

def days_to_liquidate(position_value, price, adv, participation=DEFAULT_PARTICIPATION):
    """Trading days to unwind a position under a participation-rate cap (min 1)."""
    if not np.isfinite(adv) or adv <= 0 or not np.isfinite(price) or price <= 0:
        return 1.0
    shares = position_value / price
    capacity = adv * participation            # shares tradable per day
    return max(1.0, shares / capacity)


def liquidity_metrics(weights=None, portfolio_value=1_000_000, confidence=0.95,
                      participation=DEFAULT_PARTICIPATION, spread_bps=None,
                      refresh=False):
    """Per-asset and portfolio liquidity-adjusted VaR.

    Returns (per_asset_df, summary_dict). Each asset's standalone daily VaR is
    scaled by √(days_to_liquidate) and a bid-ask cost is added; the portfolio
    L-VaR is the position-weighted sum (a conservative no-diversification
    aggregation appropriate for a forced-liquidation view).
    """
    prices = load_prices()
    rets = log_returns(prices)
    tickers = list(rets.columns)
    n = len(tickers)
    if weights is None:
        weights = np.ones(n) / n
    weights = np.asarray(weights, dtype=float)
    if spread_bps is None:
        spread_bps = DEFAULT_SPREAD_BPS

    liq = fetch_liquidity_data(tickers, refresh=refresh)

    rows = []
    for i, t in enumerate(tickers):
        pos_value = portfolio_value * weights[i]
        base_var = historical_var(rets[t], confidence)        # 1-day, fraction
        price = liq.loc[t, "Price"]
        adv = liq.loc[t, "ADV"]
        ttl = days_to_liquidate(pos_value, price, adv, participation)
        liq_var_frac = base_var * np.sqrt(ttl)                 # horizon-scaled
        half_spread = spread_bps.get(t, 2.0) / 1e4 / 2
        spread_cost = half_spread * pos_value
        lvar_dollar = liq_var_frac * pos_value + spread_cost
        rows.append({
            "Ticker": t,
            "Position ($)": pos_value,
            "Base VaR (%)": base_var * 100,
            "Days to Liquidate": ttl,
            "L-VaR (%)": liq_var_frac * 100,
            "Spread Cost ($)": spread_cost,
            "L-VaR ($)": lvar_dollar,
        })

    per_asset = pd.DataFrame(rows).set_index("Ticker")

    # Portfolio-level: undiversified base VaR vs liquidity-adjusted.
    port_ret = portfolio_returns(rets, weights)
    base_port_var = historical_var(port_ret, confidence)
    summary = {
        "Base Portfolio VaR ($)": base_port_var * portfolio_value,
        "Liquidity-Adjusted VaR ($)": per_asset["L-VaR ($)"].sum(),
        "Total Spread Cost ($)": per_asset["Spread Cost ($)"].sum(),
        "Worst Liquidation (days)": per_asset["Days to Liquidate"].max(),
    }
    return per_asset, summary


if __name__ == "__main__":
    pd.set_option("display.float_format", lambda v: f"{v:12.2f}")
    per_asset, summary = liquidity_metrics(portfolio_value=10_000_000)
    print("=== Liquidity-Adjusted VaR (95%, $10M equal-weight) ===")
    print(per_asset)
    print("\n--- Portfolio summary ---")
    for k, v in summary.items():
        print(f"  {k:32s} {v:>15,.2f}")
