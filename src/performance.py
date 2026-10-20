"""
performance.py — Risk-adjusted performance metrics ("tear sheet").

The existing ``backtesting.py`` only *validates VaR*; it does not measure how a
portfolio actually performed. This module supplies the standard metrics a fund
reports: annualised return/vol, Sharpe, Sortino, Calmar, maximum drawdown,
hit rate, tail stats and the drawdown path.

All functions take a daily return ``pd.Series`` (decimal, e.g. 0.01 = 1%).
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from returns import load_prices, log_returns, portfolio_returns  # noqa: E402

TRADING_DAYS = 252


# =====================================================================
# Core building blocks
# =====================================================================

def cumulative_returns(returns):
    """Growth of $1 (compounded)."""
    return (1 + returns).cumprod()


def drawdown_series(returns):
    """Drawdown path: current value relative to the running peak (≤ 0)."""
    cum = cumulative_returns(returns)
    peak = cum.cummax()
    return cum / peak - 1.0


def max_drawdown(returns):
    return drawdown_series(returns).min()


def annualized_return(returns):
    """Geometric annualised return."""
    n = len(returns)
    if n == 0:
        return np.nan
    total_growth = (1 + returns).prod()
    return total_growth ** (TRADING_DAYS / n) - 1


def annualized_vol(returns):
    return returns.std(ddof=1) * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns, rf=0.02):
    excess = annualized_return(returns) - rf
    vol = annualized_vol(returns)
    return excess / vol if vol > 0 else np.nan


def sortino_ratio(returns, rf=0.02):
    """Like Sharpe but penalises only downside volatility."""
    downside = returns[returns < 0]
    dd_vol = downside.std(ddof=1) * np.sqrt(TRADING_DAYS)
    excess = annualized_return(returns) - rf
    return excess / dd_vol if dd_vol > 0 else np.nan


def calmar_ratio(returns):
    """Annualised return divided by the magnitude of max drawdown."""
    mdd = abs(max_drawdown(returns))
    return annualized_return(returns) / mdd if mdd > 0 else np.nan


def hit_rate(returns):
    """Fraction of days with positive return."""
    return (returns > 0).mean()


# =====================================================================
# Summary tear sheet
# =====================================================================

def performance_summary(returns, rf=0.02):
    """One-column DataFrame of headline performance & risk statistics."""
    metrics = {
        "Cumulative Return (%)": (cumulative_returns(returns).iloc[-1] - 1) * 100,
        "Annualised Return (%)": annualized_return(returns) * 100,
        "Annualised Vol (%)": annualized_vol(returns) * 100,
        "Sharpe Ratio": sharpe_ratio(returns, rf),
        "Sortino Ratio": sortino_ratio(returns, rf),
        "Calmar Ratio": calmar_ratio(returns),
        "Max Drawdown (%)": max_drawdown(returns) * 100,
        "Hit Rate (%)": hit_rate(returns) * 100,
        "Best Day (%)": returns.max() * 100,
        "Worst Day (%)": returns.min() * 100,
        "Skew": returns.skew(),
        "Excess Kurtosis": returns.kurtosis(),  # pandas returns excess kurtosis
    }
    return pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"]).round(4)


def rolling_sharpe(returns, window=126, rf=0.02):
    """Rolling annualised Sharpe over a trailing window (default ~6 months)."""
    rf_daily = rf / TRADING_DAYS
    excess = returns - rf_daily
    mean = excess.rolling(window).mean() * TRADING_DAYS
    vol = returns.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)
    return (mean / vol).dropna()


if __name__ == "__main__":
    rets = log_returns(load_prices())
    port = portfolio_returns(rets)
    print("=== Equal-Weight Portfolio — Performance Summary ===")
    print(performance_summary(port))
