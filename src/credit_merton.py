"""
credit_merton.py — Merton structural credit risk layer.

This module plugs into the existing risk engine and provides:
  * Equity‑volatility estimates via the GARCH utilities already present.
  * Balance‑sheet extraction (debt, market‑cap, shares) from yfinance.
  * Asset‑value, asset volatility, distance‑to‑default (DD) and probability of default (PD).
  * Scenario‑adjusted PDs for the 2008 GFC and 2020 COVID stress periods already defined
    in ``src/stress_testing.py``.

All functions are pure Python and have **no external dependencies** beyond the
project's own modules, ``yfinance``, ``numpy``, ``pandas`` and ``scipy``.
"""

import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# Local utilities ---------------------------------------------------
# Ensure sibling modules resolve whether this file is run as a script
# (``python credit_merton.py``) or imported as a package (``src.credit_merton``).
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from volatility import fit_garch  # GARCH fitting function defined in src/volatility.py
from returns import load_prices, log_returns, portfolio_returns

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------
# 1. Balance‑sheet helpers
# -----------------------------------------------------------------

def _latest_balance_sheet(ticker: str) -> Tuple[float, float, float, float]:
    """Return (debt, equity, market_cap, shares_outstanding) for *ticker*.

    Returns (nan, nan, nan, nan) for index tickers (^GSPC, ^IXIC) or any ticker
    whose yfinance balance sheet is empty.
    """
    _nan4 = (np.nan, np.nan, np.nan, np.nan)
    try:
        yf_tkr = yf.Ticker(ticker)
        bs = yf_tkr.balance_sheet

        # Index tickers and some ETFs return an empty DataFrame – skip.
        if bs is None or bs.empty or len(bs.columns) == 0:
            return _nan4

        latest_col = bs.columns[0]

        # ----- Debt -----------------------------------------------
        debt = np.nan
        for debt_key in ["Total Debt", "Net Debt"]:
            if debt_key in bs.index:
                val = bs.loc[debt_key, latest_col]
                if pd.notna(val):
                    debt = float(val)
                    break

        # ----- Equity ---------------------------------------------
        equity = np.nan
        for eq_key in ["Stockholders Equity", "Total Stockholder Equity",
                        "Common Stock Equity"]:
            if eq_key in bs.index:
                val = bs.loc[eq_key, latest_col]
                if pd.notna(val):
                    equity = float(val)
                    break

        # ----- Market cap & shares --------------------------------
        info = yf_tkr.info
        market_cap = float(info.get("marketCap", np.nan))
        shares = float(info.get("sharesOutstanding", np.nan))

        return debt, equity, market_cap, shares

    except Exception:
        return _nan4


# -----------------------------------------------------------------
# 2. Volatility extraction (GARCH) ---------------------------------
# -----------------------------------------------------------------

def _equity_volatility_series(ticker: str, returns: pd.Series) -> pd.Series:
    """Fit a GARCH(1,1) model to *returns* and return the daily conditional vol.

    ``fit_garch`` requires returns in **percentage points** for the optimizer to
    converge (raw decimal returns ~0.01 are poorly scaled and yield garbage
    volatility). We therefore scale by 100 before fitting and divide the
    resulting ``conditional_volatility`` back to decimal units so callers can
    annualise with √252 directly.
    """
    # fit_garch expects a pd.Series in percentage points, not a DataFrame.
    result = fit_garch(returns * 100)
    vol = result.conditional_volatility.squeeze() / 100  # back to decimal units
    vol.name = ticker
    return vol


# -----------------------------------------------------------------
# 3. Merton model core --------------------------------------------
# -----------------------------------------------------------------

def _merton_parameters(
    market_cap: float,
    debt: float,
    sigma_e: float,
    T: float = 1.0,
    r: float = 0.01,
) -> Tuple[float, float, float, float]:
    """Calculate asset value V, asset volatility sigma_V, DD and PD.

    Parameters
    ----------
    market_cap : float
        Current equity market value (E) in dollars.
    debt : float
        Total debt (D) – assumed a zero‑coupon liability due at *T*.
    sigma_e : float
        Annualised equity volatility (σ_E).
    T : float, optional
        Time horizon (years). Default is 1 year.
    r : float, optional
        Risk‑free rate (used as the drift μ). Default 1 %.

    Returns
    -------
    V : float
        Implied asset value.
    sigma_v : float
        Implied asset volatility.
    dd : float
        Distance‑to‑default.
    pd : float
        Probability of default (N(-DD)).
    """
    # Simple approximation: sigma_V = sigma_E * (E / V)  ⇒  V solves:
    #   V = E + D  (first guess) -> iterate for consistency.
    V = market_cap + debt
    for _ in range(5):
        sigma_v = sigma_e * market_cap / V
        # DD formula (Merton, zero‑coupon debt)
        dd = (np.log(V / debt) + (r - 0.5 * sigma_v ** 2) * T) / (sigma_v * np.sqrt(T))
        # Update V using Black‑Scholes call‑option equivalence (not required for PD).
        # We keep V fixed for simplicity; convergence is sufficient.
    pd = norm.cdf(-dd)
    return V, sigma_v, dd, pd


# -----------------------------------------------------------------
# 4. Public API ----------------------------------------------------
# -----------------------------------------------------------------

def credit_metrics(tickers: List[str] = None) -> pd.DataFrame:
    """Compute current PD and DD for each ticker.

    The function:
      1. Loads historic price data via ``load_prices``.
      2. Calculates daily log returns.
      3. Fits a GARCH(1,1) model **once per ticker** to obtain a *daily* volatility.
      4. Annualises the volatility (σ_E_daily × √252).
      5. Pulls the latest debt, equity, market cap and shares using ``yfinance``.
      6. Applies the Merton formulas to produce a DataFrame:
         ``Ticker | Debt | Equity | MarketCap | σ_E (annual %) | DD | PD (%)``.
    """
    if tickers is None:
        from data_loader import ASSETS as default_assets
        tickers = default_assets

    # Load price data and compute log returns for the full sample.
    prices = load_prices()
    log_ret = log_returns(prices)

    rows = []
    for ticker in tickers:
        if ticker not in log_ret.columns:
            continue
        # 1️⃣ Equity volatility (daily) → annualised %
        vol_series = _equity_volatility_series(ticker, log_ret[ticker])
        sigma_e_annual = vol_series.mean() * np.sqrt(252)  # simple average

        # 2️⃣ Balance‑sheet info
        debt, equity, market_cap, shares = _latest_balance_sheet(ticker)

        # Guard against missing data – skip if critical fields absent.
        if any(np.isnan(v) for v in [debt, market_cap, sigma_e_annual]):
            continue

        # 3️⃣ Merton calculations (T=1 year, r=1%)
        V, sigma_v, dd, pd_val = _merton_parameters(
            market_cap=market_cap,
            debt=debt,
            sigma_e=sigma_e_annual,
            T=1.0,
            r=0.01,
        )

        rows.append(
            {
                "Ticker": ticker,
                "Debt": round(debt / 1e9, 2),  # billions USD
                "Equity_BS": round(equity / 1e9, 2) if not np.isnan(equity) else np.nan,
                "MarketCap": round(market_cap / 1e9, 2),
                "σ_E (% annual)": round(sigma_e_annual * 100, 2),
                "DD": round(dd, 4),
                "PD (%)": round(pd_val * 100, 4),
            }
        )

    return pd.DataFrame(rows).set_index("Ticker")


# -----------------------------------------------------------------
# 5. Scenario‑adjusted PDs ----------------------------------------
# -----------------------------------------------------------------

def scenario_pd(
    scenario_name: str,
    tickers: List[str] = None,
    risk_free: float = 0.01,
) -> pd.DataFrame:
    """Return the **average PD** for each ticker under a given stress scenario.

    The function re‑uses ``stress_testing`` helpers to fetch scenario price
    data, computes log returns, fits a GARCH model on the scenario window and
    plugs the resulting *annualised* equity volatility into the Merton model.
    Debt and market‑cap are taken at the *start* of the scenario (the most
    recent balance‑sheet snapshot) – this mirrors a typical credit‑risk
    stress‑testing practice.
    """
    from stress_testing import SCENARIOS

    if tickers is None:
        from data_loader import ASSETS as default_assets
        tickers = default_assets

    if scenario_name not in SCENARIOS:
        raise ValueError(f"Scenario '{scenario_name}' not defined.")

    scen = SCENARIOS[scenario_name]
    start, end = scen["start"], scen["end"]

    # Pull scenario price data (adjusted closes).
    prices = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    prices = prices.dropna()
    log_ret = np.log(prices / prices.shift(1)).dropna()

    rows = []
    for ticker in tickers:
        if ticker not in log_ret.columns:
            continue
        # 1️⃣ GARCH volatility on scenario returns.
        vol_series = _equity_volatility_series(ticker, log_ret[ticker])
        sigma_e_ann = vol_series.mean() * np.sqrt(252)

        # 2️⃣ Balance‑sheet snapshot (most recent).
        debt, equity, market_cap, shares = _latest_balance_sheet(ticker)
        if any(np.isnan(v) for v in [debt, market_cap, sigma_e_ann]):
            continue

        # 3️⃣ Merton PD (T=1y, same as baseline).
        _, _, dd, pd_val = _merton_parameters(
            market_cap=market_cap,
            debt=debt,
            sigma_e=sigma_e_ann,
            T=1.0,
            r=risk_free,
        )
        rows.append({
            "Ticker": ticker,
            "Scenario": scenario_name,
            "DD": round(dd, 4),
            "PD (%)": round(pd_val * 100, 4),
        })

    return pd.DataFrame(rows).set_index(["Ticker", "Scenario"]) 


# -----------------------------------------------------------------
# 6. Simple CLI for quick inspection
# -----------------------------------------------------------------
if __name__ == "__main__":
    print("=== Current Credit‑Risk Metrics (Merton) ===")
    print(credit_metrics())
    print("\n=== Scenario PDs ===")
    for name in ["2008 GFC (Sep–Dec 2008)", "2020 COVID Crash (Feb–Apr 2020)"]:
        print(f"\n--- {name} ---")
        print(scenario_pd(name))
