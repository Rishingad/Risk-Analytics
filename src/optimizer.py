"""
optimizer.py — Portfolio construction / weight optimisation.

The base engine hardcodes equal weights; a fund *solves* for them. This module
provides the standard construction methods on top of the existing return data,
using ``scipy.optimize`` (SLSQP) so no heavy convex-solver dependency is needed.

Methods
-------
min_variance   Global minimum-variance portfolio.
max_sharpe     Maximum Sharpe-ratio (tangency) portfolio.
risk_parity    Equal risk contribution (ERC) portfolio.
min_cvar       Minimum Conditional-VaR portfolio (Rockafellar–Uryasev LP form).
equal_weight   1/N baseline (for comparison).

All optimisers are long-only and fully invested (wᵢ ≥ 0, Σw = 1) by default,
with an optional per-name ``max_weight`` cap. They return a pd.Series of weights
indexed by ticker.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize, linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from returns import load_prices, log_returns  # noqa: E402

warnings.filterwarnings("ignore")

TRADING_DAYS = 252


# =====================================================================
# Moment estimation
# =====================================================================

def _moments(returns):
    """Annualised mean vector and covariance matrix from daily returns."""
    mu = returns.mean().values * TRADING_DAYS
    cov = returns.cov().values * TRADING_DAYS
    return mu, cov


def _bounds_constraints(n, max_weight):
    bounds = [(0.0, max_weight) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    return bounds, constraints


def _as_series(weights, returns):
    return pd.Series(weights, index=returns.columns).round(6)


# =====================================================================
# Optimisers
# =====================================================================

def equal_weight(returns):
    n = returns.shape[1]
    return _as_series(np.ones(n) / n, returns)


def min_variance(returns, max_weight=1.0):
    mu, cov = _moments(returns)
    n = len(mu)
    bounds, cons = _bounds_constraints(n, max_weight)
    obj = lambda w: w @ cov @ w
    res = minimize(obj, np.ones(n) / n, method="SLSQP", bounds=bounds, constraints=cons)
    return _as_series(res.x, returns)


def max_sharpe(returns, rf=0.02, max_weight=1.0):
    mu, cov = _moments(returns)
    n = len(mu)
    bounds, cons = _bounds_constraints(n, max_weight)
    # minimise the negative Sharpe ratio
    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        return -(ret - rf) / vol if vol > 0 else 0.0
    res = minimize(neg_sharpe, np.ones(n) / n, method="SLSQP", bounds=bounds, constraints=cons)
    return _as_series(res.x, returns)


def risk_parity(returns, max_weight=1.0):
    """Equal-risk-contribution (ERC) portfolio.

    Uses the convex log-barrier formulation (Spinu / Maillard et al.):
        minimise  ½ wᵀΣw − (1/n) Σ ln(wᵢ),   wᵢ > 0
    whose normalised solution equalises each asset's risk contribution. This is
    far more reliable than minimising the variance-of-contributions directly,
    which has a flat gradient at the 1/N starting point. ``max_weight`` is
    applied as a post-normalisation safeguard.
    """
    _, cov = _moments(returns)
    n = cov.shape[0]
    obj = lambda w: 0.5 * w @ cov @ w - (1.0 / n) * np.sum(np.log(w))
    res = minimize(obj, np.ones(n) / n, method="SLSQP",
                   bounds=[(1e-8, None)] * n, options={"maxiter": 500, "ftol": 1e-12})
    w = res.x / res.x.sum()
    if (w > max_weight).any():  # rare; renormalise capped weights
        w = np.minimum(w, max_weight)
        w = w / w.sum()
    return _as_series(w, returns)


def min_cvar(returns, alpha=0.95, max_weight=1.0):
    """Minimum-CVaR portfolio via the Rockafellar–Uryasev linear program.

    Decision vector x = [w (n), ζ (1), u (T)] with loss_j = −(R w)_j:
        minimise   ζ + 1/((1-α)T) Σ uⱼ
        s.t.       uⱼ ≥ loss_j − ζ,   uⱼ ≥ 0,   Σw = 1,   0 ≤ wᵢ ≤ max_weight

    Solved with HiGHS (``linprog``) — exact and fast even for thousands of
    scenarios, unlike a general nonlinear solver over T auxiliary variables.
    """
    R = returns.values
    T, n = R.shape

    # Objective coefficients over [w, ζ, u]
    c = np.concatenate([np.zeros(n), [1.0], np.full(T, 1.0 / ((1 - alpha) * T))])

    # Inequality: loss_j - ζ - uⱼ ≤ 0  →  -R_j·w - ζ - uⱼ ≤ 0
    A_ub = np.hstack([-R, -np.ones((T, 1)), -np.eye(T)])
    b_ub = np.zeros(T)

    # Equality: Σw = 1
    A_eq = np.concatenate([np.ones(n), [0.0], np.zeros(T)]).reshape(1, -1)
    b_eq = [1.0]

    bounds = [(0.0, max_weight)] * n + [(None, None)] + [(0.0, None)] * T
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    return _as_series(res.x[:n], returns)


# =====================================================================
# Reporting
# =====================================================================

METHODS = {
    "Equal Weight": equal_weight,
    "Min Variance": min_variance,
    "Max Sharpe": max_sharpe,
    "Risk Parity": risk_parity,
    "Min CVaR": min_cvar,
}


def portfolio_stats(weights, returns, rf=0.02):
    """Annualised return, volatility and Sharpe for a weight vector."""
    mu, cov = _moments(returns)
    w = weights.values
    ret = w @ mu
    vol = np.sqrt(w @ cov @ w)
    sharpe = (ret - rf) / vol if vol > 0 else np.nan
    return ret, vol, sharpe


def compare_methods(returns, rf=0.02):
    """Run every method and tabulate weights + annualised return/vol/Sharpe."""
    rows = []
    for name, fn in METHODS.items():
        w = fn(returns)
        ret, vol, sharpe = portfolio_stats(w, returns, rf)
        row = {t: w[t] for t in returns.columns}
        row.update({"Return (%)": ret * 100, "Vol (%)": vol * 100, "Sharpe": sharpe})
        rows.append(pd.Series(row, name=name))
    return pd.DataFrame(rows)


def efficient_frontier(returns, n_points=40, rf=0.02):
    """Trace the long-only efficient frontier for plotting."""
    mu, cov = _moments(returns)
    n = len(mu)
    targets = np.linspace(mu.min(), mu.max(), n_points)
    bounds = [(0.0, 1.0)] * n
    front = []
    for t in targets:
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=t: w @ mu - t},
        ]
        res = minimize(lambda w: w @ cov @ w, np.ones(n) / n, method="SLSQP",
                       bounds=bounds, constraints=cons)
        if res.success:
            vol = np.sqrt(res.x @ cov @ res.x)
            front.append((vol * 100, t * 100))
    return pd.DataFrame(front, columns=["Vol (%)", "Return (%)"])


if __name__ == "__main__":
    rets = log_returns(load_prices())
    pd.set_option("display.float_format", lambda v: f"{v:8.4f}")
    print("=== Optimised Portfolios (annualised) ===")
    print(compare_methods(rets))
