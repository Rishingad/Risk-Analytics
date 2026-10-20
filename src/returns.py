"""
returns.py — Return computation utilities for portfolio risk analytics.

Provides functions for computing log returns, simple returns, and
weighted portfolio returns from a price DataFrame.
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_prices(path=None):
    """
    Load the prices CSV produced by data_loader.py.

    Parameters
    ----------
    path : str or None
        Path to CSV file.  Defaults to ``data/prices.csv``.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex, one column per ticker.
    """
    if path is None:
        path = os.path.join(DATA_DIR, "prices.csv")
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    return prices


def log_returns(prices):
    """
    Compute daily log returns: ln(P_t / P_{t-1}).

    Parameters
    ----------
    prices : pd.DataFrame or pd.Series

    Returns
    -------
    pd.DataFrame or pd.Series
        Log returns with the first row dropped.
    """
    return np.log(prices / prices.shift(1)).dropna()


def simple_returns(prices):
    """
    Compute daily simple (arithmetic) returns: (P_t - P_{t-1}) / P_{t-1}.

    Parameters
    ----------
    prices : pd.DataFrame or pd.Series

    Returns
    -------
    pd.DataFrame or pd.Series
    """
    return prices.pct_change().dropna()


def portfolio_returns(returns, weights=None):
    """
    Compute portfolio returns as a weighted sum of individual asset returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset return matrix (rows = dates, columns = tickers).
    weights : array-like or None
        Portfolio weights summing to 1.  If None, equal-weight is used.

    Returns
    -------
    pd.Series
        Daily portfolio return series.
    """
    if weights is None:
        n = returns.shape[1]
        weights = np.ones(n) / n
    weights = np.array(weights)
    return returns.dot(weights)
