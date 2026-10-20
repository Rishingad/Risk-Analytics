"""
backtesting.py — VaR Model Backtesting Framework.

Validates VaR model accuracy using two standard regulatory tests:

  1. **Kupiec Proportion-of-Failures (POF) Test**
     Tests whether the observed exceedance rate matches the expected rate
     under the null hypothesis.  A rejection (p < 0.05) means the model
     is systematically over- or under-estimating risk.

  2. **Christoffersen Independence Test**
     Tests whether VaR exceedances cluster in time.  Even if the overall
     exceedance rate is correct, clustered violations indicate the model
     fails to capture changing market conditions.

Both tests use a likelihood-ratio statistic distributed as χ²(1).

Usage
-----
    from src.backtesting import run_backtest, results_table
    results = run_backtest(portfolio_returns, asset_returns)
    print(results_table(results))
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2

# Ensure sibling modules resolve whether this file is run as a script
# (``python backtesting.py``) or imported as a package (``src.backtesting``).
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from var_models import (
    parametric_var,
    historical_var,
    monte_carlo_var,
)


# ================================================================
# VaR EXCEEDANCE GENERATION
# ================================================================

def _rolling_var(returns, method, confidence, window=252, **mc_kwargs):
    """
    Compute a rolling VaR series using a trailing window.

    For each day *t*, VaR is estimated on the window
    [t − window, t − 1] and compared to the actual return on day *t*.

    Parameters
    ----------
    returns : pd.Series
        Portfolio return series.
    method : str
        One of 'parametric', 'historical', 'montecarlo'.
    confidence : float
        Confidence level.
    window : int
        Look-back window in trading days (default: 1 year).

    Returns
    -------
    pd.Series
        Daily VaR estimates aligned to the return dates.
    """
    var_series = pd.Series(index=returns.index, dtype=float)

    for i in range(window, len(returns)):
        hist = returns.iloc[i - window:i]

        if method == "parametric":
            var_series.iloc[i] = parametric_var(hist, confidence)
        elif method == "historical":
            var_series.iloc[i] = historical_var(hist, confidence)
        elif method == "montecarlo":
            var_series.iloc[i] = monte_carlo_var(
                hist, confidence=confidence,
                n_simulations=mc_kwargs.get("n_simulations", 10_000),
                seed=mc_kwargs.get("seed", 42) + i,
            )

    return var_series.dropna()


def exceedances(returns, var_series):
    """
    Identify days where the actual loss exceeded VaR.

    An exceedance occurs when the actual return is more negative than
    -VaR (since VaR is stored as a positive number).

    Parameters
    ----------
    returns : pd.Series
        Actual portfolio returns.
    var_series : pd.Series
        Rolling VaR estimates (positive values).

    Returns
    -------
    pd.Series of int
        1 = exceedance, 0 = no exceedance.
    """
    common = returns.index.intersection(var_series.index)
    hits = (returns.loc[common] < -var_series.loc[common]).astype(int)
    return hits


# ================================================================
# KUPIEC POF TEST
# ================================================================

def kupiec_pof(hits, confidence):
    """
    Kupiec Proportion-of-Failures likelihood-ratio test.

    H₀: The true exceedance probability equals (1 − confidence).
    H₁: The true exceedance probability differs from (1 − confidence).

    The test statistic follows χ²(1).

    Parameters
    ----------
    hits : array-like of {0, 1}
        Exceedance indicator series.
    confidence : float
        Confidence level used to compute VaR.

    Returns
    -------
    dict
        Keys: 'test_statistic', 'p_value', 'expected_rate',
              'observed_rate', 'n_exceedances', 'n_observations',
              'reject_H0'.
    """
    hits = np.asarray(hits)
    T = len(hits)
    x = hits.sum()

    p_expected = 1 - confidence
    p_observed = x / T if T > 0 else 0.0

    # Avoid log(0) edge cases
    if x == 0 or x == T:
        lr_stat = 0.0
    else:
        lr_stat = -2 * (
            np.log((1 - p_expected) ** (T - x) * p_expected ** x)
            - np.log((1 - p_observed) ** (T - x) * p_observed ** x)
        )

    p_value = 1 - chi2.cdf(lr_stat, df=1)

    return {
        "test_statistic":  round(lr_stat, 4),
        "p_value":         round(p_value, 4),
        "expected_rate":   round(p_expected, 4),
        "observed_rate":   round(p_observed, 4),
        "n_exceedances":   int(x),
        "n_observations":  T,
        "reject_H0":      p_value < 0.05,
    }


# ================================================================
# CHRISTOFFERSEN INDEPENDENCE TEST
# ================================================================

def christoffersen_independence(hits):
    """
    Christoffersen test for independence of VaR exceedances.

    H₀: Exceedances are independent (no clustering).
    H₁: Exceedances are serially dependent (clustered).

    Constructs a 2×2 transition matrix of consecutive {0, 1} states
    and uses a likelihood-ratio statistic ~ χ²(1).

    Parameters
    ----------
    hits : array-like of {0, 1}
        Exceedance indicator series.

    Returns
    -------
    dict
        Keys: 'test_statistic', 'p_value', 'reject_H0',
              'transition_matrix'.
    """
    hits = np.asarray(hits)

    # Build transition counts  n[i][j] = count(state_t=i → state_{t+1}=j)
    n = np.zeros((2, 2))
    for t in range(len(hits) - 1):
        i = int(hits[t])
        j = int(hits[t + 1])
        n[i][j] += 1

    n00, n01, n10, n11 = n[0, 0], n[0, 1], n[1, 0], n[1, 1]

    # Row totals
    n0 = n00 + n01  # transitions from state 0
    n1 = n10 + n11  # transitions from state 1

    # Transition probabilities
    pi01 = n01 / n0 if n0 > 0 else 0.0
    pi11 = n11 / n1 if n1 > 0 else 0.0

    # Unconditional probability
    total = n0 + n1
    pi = (n01 + n11) / total if total > 0 else 0.0

    # Log-likelihood under independence (H₀)
    ll_indep = 0.0
    if 0 < pi < 1 and total > 0:
        ll_indep = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)

    # Log-likelihood under dependence (H₁)
    ll_dep = 0.0
    if n0 > 0 and 0 < pi01 < 1:
        ll_dep += n00 * np.log(1 - pi01) + n01 * np.log(pi01)
    if n1 > 0 and 0 < pi11 < 1:
        ll_dep += n10 * np.log(1 - pi11) + n11 * np.log(pi11)

    lr_stat = -2 * (ll_indep - ll_dep)
    lr_stat = max(lr_stat, 0.0)  # numerical guard

    p_value = 1 - chi2.cdf(lr_stat, df=1)

    return {
        "test_statistic":    round(lr_stat, 4),
        "p_value":           round(p_value, 4),
        "reject_H0":         p_value < 0.05,
        "transition_matrix": n.astype(int).tolist(),
    }


# ================================================================
# MASTER BACKTEST RUNNER
# ================================================================

METHODS = ["parametric", "historical", "montecarlo"]
CONFIDENCE_LEVELS = [0.95, 0.99]


def run_backtest(portfolio_returns, asset_returns=None, window=252,
                 confidence_levels=None, methods=None):
    """
    Run the full backtesting suite across methods and confidence levels.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio return series.
    asset_returns : pd.DataFrame or None
        Individual asset returns (currently unused; reserved for future
        multivariate MC backtesting).
    window : int
        Rolling look-back window in trading days.
    confidence_levels : list of float
        Defaults to [0.95, 0.99].
    methods : list of str
        Defaults to ['parametric', 'historical', 'montecarlo'].

    Returns
    -------
    list of dict
        One dict per (method, confidence) pair containing Kupiec and
        Christoffersen test results.
    """
    if confidence_levels is None:
        confidence_levels = CONFIDENCE_LEVELS
    if methods is None:
        methods = METHODS

    all_results = []

    for method in methods:
        for cl in confidence_levels:
            pct = f"{cl:.0%}"
            print(f"  ⏳ Backtesting {method:12s} @ {pct} ...", end="", flush=True)

            var_s = _rolling_var(portfolio_returns, method, cl, window)
            hits = exceedances(portfolio_returns, var_s)

            kupiec = kupiec_pof(hits, cl)
            christo = christoffersen_independence(hits)

            all_results.append({
                "Method":           method.title(),
                "Confidence":       pct,
                "Exceedances":      kupiec["n_exceedances"],
                "Observations":     kupiec["n_observations"],
                "Expected Rate":    kupiec["expected_rate"],
                "Observed Rate":    kupiec["observed_rate"],
                "Kupiec Stat":      kupiec["test_statistic"],
                "Kupiec p-value":   kupiec["p_value"],
                "Kupiec Pass":      "✅ Pass" if not kupiec["reject_H0"] else "❌ Fail",
                "Christo Stat":     christo["test_statistic"],
                "Christo p-value":  christo["p_value"],
                "Christo Pass":     "✅ Pass" if not christo["reject_H0"] else "❌ Fail",
            })
            print(" done")

    return all_results


def results_table(results):
    """
    Format backtesting results as a readable DataFrame.

    Parameters
    ----------
    results : list of dict
        Output of ``run_backtest``.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.DataFrame(results)
    return df


# ================================================================
# CLI ENTRY POINT
# ================================================================

if __name__ == "__main__":
    from returns import load_prices, log_returns, portfolio_returns as port_ret

    prices = load_prices()
    rets = log_returns(prices)
    port = port_ret(rets)

    print("=" * 70)
    print("  VaR Backtesting Framework")
    print("=" * 70)
    print(f"  Portfolio : Equal-weight across {list(rets.columns)}")
    print(f"  Period    : {rets.index[0].date()} → {rets.index[-1].date()}")
    print(f"  Window    : 252 trading days\n")

    results = run_backtest(port, window=252)

    print("\n" + "=" * 70)
    print("  Results")
    print("=" * 70)
    table = results_table(results)
    display_cols = [
        "Method", "Confidence", "Exceedances", "Expected Rate",
        "Observed Rate", "Kupiec p-value", "Kupiec Pass",
        "Christo p-value", "Christo Pass",
    ]
    print(table[display_cols].to_string(index=False))

    print("\n📘 Interpretation Guide:")
    print("  • Kupiec POF  : Tests if the exceedance *count* matches expectations.")
    print("                   Fail → model systematically over/under-estimates risk.")
    print("  • Christoffersen: Tests if exceedances *cluster* in time.")
    print("                   Fail → model misses regime changes / volatility shifts.")
    print("  • p-value < 0.05 → Reject H₀ → Test FAILS.")
    print()
