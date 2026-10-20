"""
factor_model.py — Multi-factor risk model (Fama-French 5 + Momentum).

This is the centrepiece a desk-grade risk stack adds on top of raw VaR:
instead of treating each asset as an opaque return series, it decomposes
returns into *systematic factor* exposures (market, size, value, profitability,
investment, momentum) plus an idiosyncratic residual.  That decomposition is
what powers risk attribution ("how much of my variance is momentum?"), hedge
construction and scenario generation — exactly what Barra / Axioma provide
commercially.

Factors are sourced (free) from the Kenneth R. French data library via
``pandas_datareader`` and cached to ``data/ff_factors.csv`` the same way
``data_loader.py`` caches prices.

Functions
---------
download_factors(start, end)      Fetch FF5 + Momentum, cache to CSV (decimal units).
load_factors(refresh=False)       Load cached factors (download on first use).
factor_exposures(asset_returns)   Per-asset OLS betas, annualised alpha, R².
portfolio_exposures(returns, w)   Portfolio-level factor betas + alpha + R².
risk_decomposition(returns, w)    % of portfolio variance from each factor vs idiosyncratic.
summary_table(...)                Convenience formatted view used by the dashboard.
"""

import os
import warnings

import numpy as np
import pandas as pd

# Ensure sibling modules resolve whether run as a script or imported as src.factor_model.
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from returns import load_prices, log_returns  # noqa: E402

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FACTOR_FILE = os.path.join(DATA_DIR, "ff_factors.csv")

# The six priced factors (RF held separately as the risk-free rate).
FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
TRADING_DAYS = 252


# =====================================================================
# 1. Factor data loading / caching
# =====================================================================

def download_factors(start="2015-01-01", end="2025-01-01"):
    """Fetch FF 5-factor + Momentum daily series and cache to ``ff_factors.csv``.

    Ken French publishes factors in *percent*; we convert to decimal so they
    are on the same scale as the log returns produced by ``returns.py``.
    """
    import pandas_datareader.data as web

    ff5 = web.DataReader(
        "F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start=start, end=end
    )[0]
    mom = web.DataReader(
        "F-F_Momentum_Factor_daily", "famafrench", start=start, end=end
    )[0]
    mom.columns = ["Mom"]  # library ships it as 'Mom   ' with stray whitespace

    factors = ff5.join(mom, how="inner") / 100.0  # percent -> decimal
    factors.index = pd.to_datetime(factors.index)

    os.makedirs(DATA_DIR, exist_ok=True)
    factors.to_csv(FACTOR_FILE)
    return factors


def load_factors(refresh=False):
    """Load cached factors, downloading on first use (or when ``refresh=True``)."""
    if refresh or not os.path.exists(FACTOR_FILE):
        return download_factors()
    factors = pd.read_csv(FACTOR_FILE, index_col=0, parse_dates=True)
    return factors


# =====================================================================
# 2. Date alignment helper
# =====================================================================

def _align(asset_returns, factors):
    """Inner-join asset returns and factors on common dates."""
    if isinstance(asset_returns, pd.Series):
        asset_returns = asset_returns.to_frame()
    common = asset_returns.index.intersection(factors.index)
    return asset_returns.loc[common], factors.loc[common]


def _ols(y, X):
    """OLS with intercept. Returns (alpha, betas[Series], r2, resid_var)."""
    import statsmodels.api as sm

    Xc = sm.add_constant(X)
    model = sm.OLS(y.values, Xc.values).fit()
    alpha = model.params[0]
    betas = pd.Series(model.params[1:], index=X.columns)
    resid_var = float(np.var(model.resid, ddof=Xc.shape[1]))
    return alpha, betas, float(model.rsquared), resid_var


# =====================================================================
# 3. Per-asset factor exposures
# =====================================================================

def factor_exposures(asset_returns, factors=None):
    """Regress each asset's *excess* return on the six factors.

    Returns a DataFrame indexed by ticker with one column per factor beta,
    plus annualised ``Alpha (%)`` and ``R²``.
    """
    if factors is None:
        factors = load_factors()
    rets, fac = _align(asset_returns, factors)
    rf = fac["RF"]
    F = fac[FACTORS]

    rows = {}
    for ticker in rets.columns:
        excess = rets[ticker] - rf
        alpha, betas, r2, _ = _ols(excess, F)
        row = betas.to_dict()
        row["Alpha (%)"] = alpha * TRADING_DAYS * 100  # annualised
        row["R²"] = r2
        rows[ticker] = row

    return pd.DataFrame(rows).T[FACTORS + ["Alpha (%)", "R²"]]


# =====================================================================
# 4. Portfolio-level exposures & risk decomposition
# =====================================================================

def _portfolio_excess(returns, weights, factors):
    """Build the portfolio *excess* return series aligned to the factors."""
    rets, fac = _align(returns, factors)
    if weights is None:
        weights = np.ones(rets.shape[1]) / rets.shape[1]
    weights = np.asarray(weights, dtype=float)
    port = rets.values @ weights
    port = pd.Series(port, index=rets.index)
    return port - fac["RF"], fac


def portfolio_exposures(returns, weights=None, factors=None):
    """Portfolio factor betas, annualised alpha and R² (single OLS fit)."""
    if factors is None:
        factors = load_factors()
    excess, fac = _portfolio_excess(returns, weights, factors)
    alpha, betas, r2, _ = _ols(excess, fac[FACTORS])
    out = betas.copy()
    out["Alpha (%)"] = alpha * TRADING_DAYS * 100
    out["R²"] = r2
    return out


def risk_decomposition(returns, weights=None, factors=None):
    """Decompose annualised portfolio variance into per-factor and idiosyncratic parts.

    Variance identity:  Var(r_p) = bᵀ Σ_f b + σ_e²
      * b      = portfolio factor betas
      * Σ_f    = factor covariance matrix
      * σ_e²   = residual (idiosyncratic) variance

    Each factor's contribution uses the standard marginal decomposition
    (b ∘ Σ_f b), which sums to the systematic variance.

    Returns a DataFrame: index = factors + 'Idiosyncratic', columns =
    ['Variance (ann.)', 'Risk Contribution (%)'].
    """
    if factors is None:
        factors = load_factors()
    excess, fac = _portfolio_excess(returns, weights, factors)
    F = fac[FACTORS]
    _, b, _, resid_var = _ols(excess, F)

    Sigma_f = F.cov().values                      # daily factor covariance
    b_vec = b.values
    factor_contrib = b_vec * (Sigma_f @ b_vec)    # marginal contribution, daily
    systematic_var = float(b_vec @ Sigma_f @ b_vec)
    total_var = systematic_var + resid_var

    contrib = np.append(factor_contrib, resid_var) * TRADING_DAYS  # annualise
    pct = np.append(factor_contrib, resid_var) / total_var * 100

    return pd.DataFrame(
        {"Variance (ann.)": contrib, "Risk Contribution (%)": pct},
        index=FACTORS + ["Idiosyncratic"],
    )


def summary_table(returns, weights=None, factors=None):
    """Combined view for the dashboard: portfolio betas + risk contributions."""
    if factors is None:
        factors = load_factors()
    exp = portfolio_exposures(returns, weights, factors)
    decomp = risk_decomposition(returns, weights, factors)
    tbl = pd.DataFrame(index=FACTORS)
    tbl["Beta"] = exp[FACTORS]
    tbl["Risk Contribution (%)"] = decomp.loc[FACTORS, "Risk Contribution (%)"]
    return tbl, exp["Alpha (%)"], exp["R²"], decomp.loc["Idiosyncratic", "Risk Contribution (%)"]


# =====================================================================
# 5. CLI
# =====================================================================
if __name__ == "__main__":
    prices = load_prices()
    rets = log_returns(prices)
    # Equity sleeve only (factors are equity factors; indices are fine too but noisy).
    print("=== Per-Asset Factor Exposures (FF5 + Momentum) ===")
    print(factor_exposures(rets).round(3))

    print("\n=== Equal-Weight Portfolio Factor Exposures ===")
    print(portfolio_exposures(rets).round(3))

    print("\n=== Portfolio Risk Decomposition ===")
    print(risk_decomposition(rets).round(4))
