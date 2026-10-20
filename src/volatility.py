"""
volatility.py — GARCH model fitting and conditional volatility extraction.

Provides reusable utilities for fitting GARCH / GJR-GARCH / EGARCH models
with skewed-t innovations, comparing them, and generating forecasts.
"""

import pandas as pd
import numpy as np
from arch import arch_model


# ==============================
# MODEL FITTING
# ==============================

def fit_garch(returns, vol='Garch', p=1, o=0, q=1, dist='skewt', mean='Constant',
              last_obs=None):
    """
    Fit a GARCH-family model.

    Parameters
    ----------
    returns : pd.Series
        Return series (in percentage points for numerical stability).
    vol : str
        Volatility model: 'Garch' or 'EGARCH'.
    p, o, q : int
        Lag orders for GARCH, asymmetry (o), and ARCH terms.
    dist : str
        Innovation distribution: 'normal', 't', 'skewt'.
    mean : str
        Mean model: 'Constant', 'Zero', 'AR'.
    last_obs : str or None
        If set, fit only up to this date (for out-of-sample testing).

    Returns
    -------
    ARCHModelResult
    """
    model = arch_model(returns, vol=vol, p=p, o=o, q=q, dist=dist, mean=mean)
    result = model.fit(last_obs=last_obs, disp='off')
    return result


def fit_all_models(returns, last_obs=None):
    """
    Fit GARCH(1,1), GJR-GARCH(1,1), and EGARCH(1,1) with skewed-t innovations.

    Returns dict of {name: ARCHModelResult}.
    """
    specs = {
        'GARCH(1,1)':     dict(vol='Garch',  p=1, o=0, q=1),
        'GJR-GARCH(1,1)': dict(vol='Garch',  p=1, o=1, q=1),
        'EGARCH(1,1)':    dict(vol='EGARCH', p=1, o=1, q=1),
    }

    results = {}
    for name, spec in specs.items():
        results[name] = fit_garch(returns, last_obs=last_obs, **spec)

    return results


# ==============================
# MODEL COMPARISON
# ==============================

def compare_models(results):
    """
    Build a comparison DataFrame with AIC, BIC, log-likelihood.

    Parameters
    ----------
    results : dict
        {model_name: ARCHModelResult}

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for name, res in results.items():
        rows.append({
            'Model':   name,
            'AIC':     res.aic,
            'BIC':     res.bic,
            'Log-Lik': res.loglikelihood,
            'Params':  res.num_params,
        })
    df = pd.DataFrame(rows).set_index('Model').round(2)
    return df


def best_model(results, criterion='aic'):
    """Return (name, result) of the model with lowest AIC or BIC."""
    comp = compare_models(results)
    col = 'AIC' if criterion == 'aic' else 'BIC'
    winner = comp[col].idxmin()
    return winner, results[winner]


# ==============================
# CONDITIONAL VOLATILITY
# ==============================

def conditional_volatility(result, scale=100):
    """
    Extract conditional volatility from a fitted model.

    Parameters
    ----------
    result : ARCHModelResult
    scale : float
        If returns were scaled to %, pass 100 to convert back to decimal.

    Returns
    -------
    pd.Series
    """
    return result.conditional_volatility / scale


# ==============================
# FORECASTING
# ==============================

def one_step_forecast_variance(result, start, horizon=1):
    """
    Generate one-step-ahead variance forecasts from `start` onward.

    Returns
    -------
    pd.Series of forecasted variance values.
    """
    fcast = result.forecast(start=start, horizon=horizon, reindex=False)
    return fcast.variance.dropna().iloc[:, 0]


def forecast_rmse(forecast_var, realized_returns):
    """
    Compute RMSE and MAE of variance forecasts vs squared returns.

    Parameters
    ----------
    forecast_var : pd.Series
        Forecasted variance.
    realized_returns : pd.Series
        Actual returns (will be squared as a variance proxy).

    Returns
    -------
    dict with 'rmse' and 'mae'.
    """
    realized_var = realized_returns ** 2
    common = forecast_var.index.intersection(realized_var.index)

    errors = forecast_var.loc[common] - realized_var.loc[common]
    rmse = np.sqrt((errors ** 2).mean())
    mae = errors.abs().mean()

    return {'rmse': rmse, 'mae': mae}
