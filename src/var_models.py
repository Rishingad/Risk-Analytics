"""
var_models.py — Value-at-Risk and Conditional VaR (Expected Shortfall) Engine.

Implements three VaR estimation methodologies:
  1. Parametric (Variance–Covariance) — assumes normally distributed returns.
  2. Historical Simulation — uses the empirical distribution of past returns.
  3. Monte Carlo Simulation — generates synthetic return paths via a fitted
     multivariate normal distribution.

Each method computes both VaR and CVaR (Expected Shortfall) at user-specified
confidence levels (default: 95 % and 99 %).

Usage
-----
    from src.returns import load_prices, log_returns, portfolio_returns
    from src.var_models import compute_all_var, summary_table

    prices  = load_prices()
    rets    = log_returns(prices)
    port_r  = portfolio_returns(rets)   # equal-weight

    results = compute_all_var(port_r, rets)
    print(summary_table(results))
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


# ================================================================
# 1. PARAMETRIC (Variance–Covariance) VaR & CVaR
# ================================================================

def parametric_var(returns, confidence=0.95):
    """
    Parametric (Variance–Covariance) Value-at-Risk.

    Assumes returns are normally distributed.  VaR is computed as:
        VaR = -(μ + z_α · σ)
    where z_α is the standard-normal quantile.

    Parameters
    ----------
    returns : pd.Series
        Portfolio return series.
    confidence : float
        Confidence level (e.g. 0.95 or 0.99).

    Returns
    -------
    float
        VaR expressed as a positive loss.
    """
    mu = returns.mean()
    sigma = returns.std()
    z = norm.ppf(1 - confidence)
    var = -(mu + z * sigma)
    return var


def parametric_cvar(returns, confidence=0.95):
    """
    Parametric Conditional VaR (Expected Shortfall).

    CVaR under normality:
        CVaR = -(μ - σ · φ(z_α) / (1 - α))

    Parameters
    ----------
    returns : pd.Series
        Portfolio return series.
    confidence : float
        Confidence level.

    Returns
    -------
    float
        CVaR expressed as a positive loss.
    """
    mu = returns.mean()
    sigma = returns.std()
    alpha = 1 - confidence
    z = norm.ppf(alpha)
    cvar = -(mu - sigma * norm.pdf(z) / alpha)
    return cvar


# ================================================================
# 2. HISTORICAL SIMULATION VaR & CVaR
# ================================================================

def historical_var(returns, confidence=0.95):
    """
    Historical Simulation Value-at-Risk.

    Uses the empirical quantile of the actual return distribution.

    Parameters
    ----------
    returns : pd.Series
        Portfolio return series.
    confidence : float
        Confidence level.

    Returns
    -------
    float
        VaR expressed as a positive loss.
    """
    alpha = 1 - confidence
    var = -returns.quantile(alpha)
    return var


def historical_cvar(returns, confidence=0.95):
    """
    Historical Conditional VaR (Expected Shortfall).

    Average of all losses that exceed the VaR threshold.

    Parameters
    ----------
    returns : pd.Series
        Portfolio return series.
    confidence : float
        Confidence level.

    Returns
    -------
    float
        CVaR expressed as a positive loss.
    """
    alpha = 1 - confidence
    cutoff = returns.quantile(alpha)
    cvar = -returns[returns <= cutoff].mean()
    return cvar


# ================================================================
# 3. MONTE CARLO SIMULATION VaR & CVaR
# ================================================================

def monte_carlo_var(returns, asset_returns=None, confidence=0.95,
                    n_simulations=100_000, seed=42):
    """
    Monte Carlo Value-at-Risk.

    If ``asset_returns`` is provided (multi-asset DataFrame), a multivariate
    normal distribution is fitted and correlated simulations are generated.
    Otherwise, a univariate normal is used on the portfolio return series.

    Parameters
    ----------
    returns : pd.Series
        Portfolio return series (used for univariate fallback).
    asset_returns : pd.DataFrame or None
        Individual asset log-returns for multivariate simulation.
    confidence : float
        Confidence level.
    n_simulations : int
        Number of simulated return paths.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    float
        VaR expressed as a positive loss.
    """
    rng = np.random.default_rng(seed)

    if asset_returns is not None and asset_returns.shape[1] > 1:
        mu = asset_returns.mean().values
        cov = asset_returns.cov().values
        n_assets = len(mu)
        weights = np.ones(n_assets) / n_assets

        simulated = rng.multivariate_normal(mu, cov, size=n_simulations)
        port_sim = simulated @ weights
    else:
        mu = returns.mean()
        sigma = returns.std()
        port_sim = rng.normal(mu, sigma, size=n_simulations)

    alpha = 1 - confidence
    var = -np.percentile(port_sim, alpha * 100)
    return var


def monte_carlo_cvar(returns, asset_returns=None, confidence=0.95,
                     n_simulations=100_000, seed=42):
    """
    Monte Carlo Conditional VaR (Expected Shortfall).

    Parameters
    ----------
    returns : pd.Series
        Portfolio return series.
    asset_returns : pd.DataFrame or None
        Individual asset log-returns for multivariate simulation.
    confidence : float
        Confidence level.
    n_simulations : int
        Number of simulated return paths.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    float
        CVaR expressed as a positive loss.
    """
    rng = np.random.default_rng(seed)

    if asset_returns is not None and asset_returns.shape[1] > 1:
        mu = asset_returns.mean().values
        cov = asset_returns.cov().values
        n_assets = len(mu)
        weights = np.ones(n_assets) / n_assets

        simulated = rng.multivariate_normal(mu, cov, size=n_simulations)
        port_sim = simulated @ weights
    else:
        mu = returns.mean()
        sigma = returns.std()
        port_sim = rng.normal(mu, sigma, size=n_simulations)

    alpha = 1 - confidence
    cutoff = np.percentile(port_sim, alpha * 100)
    cvar = -port_sim[port_sim <= cutoff].mean()
    return cvar


# ================================================================
# MASTER COMPUTE FUNCTION
# ================================================================

CONFIDENCE_LEVELS = [0.95, 0.99]


def compute_all_var(portfolio_returns, asset_returns=None,
                    confidence_levels=None, n_simulations=100_000, seed=42):
    """
    Compute VaR and CVaR across all three methods and confidence levels.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio return series.
    asset_returns : pd.DataFrame or None
        Individual asset returns for Monte Carlo multivariate simulation.
    confidence_levels : list of float or None
        Defaults to [0.95, 0.99].
    n_simulations : int
        Monte Carlo sample count.
    seed : int
        Random seed.

    Returns
    -------
    list of dict
        One dict per (method, confidence) combination with keys:
        'Method', 'Confidence', 'VaR', 'CVaR'.
    """
    if confidence_levels is None:
        confidence_levels = CONFIDENCE_LEVELS

    rows = []

    for cl in confidence_levels:
        pct = f"{cl:.0%}"

        # --- Parametric ---
        rows.append({
            "Method":     "Parametric",
            "Confidence": pct,
            "VaR":        parametric_var(portfolio_returns, cl),
            "CVaR":       parametric_cvar(portfolio_returns, cl),
        })

        # --- Historical ---
        rows.append({
            "Method":     "Historical",
            "Confidence": pct,
            "VaR":        historical_var(portfolio_returns, cl),
            "CVaR":       historical_cvar(portfolio_returns, cl),
        })

        # --- Monte Carlo ---
        rows.append({
            "Method":     "Monte Carlo",
            "Confidence": pct,
            "VaR":        monte_carlo_var(
                portfolio_returns, asset_returns, cl, n_simulations, seed),
            "CVaR":       monte_carlo_cvar(
                portfolio_returns, asset_returns, cl, n_simulations, seed),
        })

    return rows


def summary_table(results):
    """
    Format VaR/CVaR results as a styled summary DataFrame.

    Parameters
    ----------
    results : list of dict
        Output of ``compute_all_var``.

    Returns
    -------
    pd.DataFrame
        Columns: Method | Confidence | VaR (%) | CVaR (%)
    """
    df = pd.DataFrame(results)
    df["VaR (%)"]  = (df["VaR"]  * 100).round(4)
    df["CVaR (%)"] = (df["CVaR"] * 100).round(4)
    df = df[["Method", "Confidence", "VaR (%)", "CVaR (%)"]]
    return df


# ================================================================
# CLI ENTRY POINT
# ================================================================

if __name__ == "__main__":
    from returns import load_prices, log_returns, portfolio_returns as port_ret

    prices = load_prices()
    rets = log_returns(prices)
    port = port_ret(rets)

    print("=" * 60)
    print("  VaR & CVaR Summary  —  Equal-Weight Portfolio")
    print("=" * 60)
    print(f"  Assets : {list(rets.columns)}")
    print(f"  Period : {rets.index[0].date()} → {rets.index[-1].date()}")
    print(f"  Obs    : {len(rets)}")
    print("=" * 60)

    results = compute_all_var(port, rets)
    table = summary_table(results)
    print(table.to_string(index=False))
    print()
