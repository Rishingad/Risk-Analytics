"""
Smoke + correctness tests for the risk engine.

Runs fully offline against the cached CSVs in ``data/`` (prices, Fama-French
factors, liquidity) plus small synthetic series with known answers. Modules
that always hit the network (credit_merton, stress_testing) are not covered.

    pytest -q
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.returns import load_prices, log_returns, simple_returns, portfolio_returns
from src import var_models, backtesting, optimizer, performance, copula, volatility


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture(scope="module")
def rets():
    return log_returns(load_prices())


@pytest.fixture(scope="module")
def port(rets):
    return portfolio_returns(rets)


@pytest.fixture
def normal_series():
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0005, 0.01, 20_000),
                     index=pd.bdate_range("2000-01-03", periods=20_000))


# =====================================================================
# Returns
# =====================================================================

def test_prices_loaded(rets):
    assert list(rets.columns) == ["^GSPC", "^IXIC", "AAPL", "MSFT", "JPM"]
    assert isinstance(rets.index, pd.DatetimeIndex)
    assert not rets.isna().any().any()


def test_simple_vs_log_returns():
    p = pd.DataFrame({"A": [100.0, 110.0, 99.0]})
    assert simple_returns(p)["A"].tolist() == pytest.approx([0.10, -0.10])
    assert log_returns(p)["A"].tolist() == pytest.approx([np.log(1.1), np.log(0.9)])


def test_equal_weight_portfolio(rets, port):
    assert port.iloc[0] == pytest.approx(rets.iloc[0].mean())


# =====================================================================
# VaR / CVaR
# =====================================================================

def test_parametric_var_matches_normal_quantile(normal_series):
    v = var_models.parametric_var(normal_series, 0.99)
    expected = -(normal_series.mean() + norm.ppf(0.01) * normal_series.std())
    assert v == pytest.approx(expected)


def test_historical_var_close_to_parametric_on_normal_data(normal_series):
    h = var_models.historical_var(normal_series, 0.95)
    p = var_models.parametric_var(normal_series, 0.95)
    assert h == pytest.approx(p, rel=0.05)


@pytest.mark.parametrize("cl", [0.95, 0.99])
def test_cvar_exceeds_var(port, cl):
    assert var_models.historical_cvar(port, cl) >= var_models.historical_var(port, cl)
    assert var_models.parametric_cvar(port, cl) >= var_models.parametric_var(port, cl)


def test_summary_table_matches_readme(port, rets):
    df = var_models.summary_table(var_models.compute_all_var(port, rets))
    assert len(df) == 6
    hist99 = df[(df["Method"] == "Historical") & (df["Confidence"] == "99%")]["VaR (%)"].iloc[0]
    assert hist99 == pytest.approx(3.846, abs=0.01)


# =====================================================================
# Backtesting
# =====================================================================

def test_kupiec_accepts_correct_rate():
    hits = np.zeros(1000, dtype=int)
    hits[::100] = 1                       # exactly 1% exceedances
    res = backtesting.kupiec_pof(hits, 0.99)
    assert res["n_exceedances"] == 10
    assert not res["reject_H0"]


def test_kupiec_rejects_too_many_exceedances():
    hits = np.zeros(1000, dtype=int)
    hits[::20] = 1                        # 5% vs expected 1%
    assert backtesting.kupiec_pof(hits, 0.99)["reject_H0"]


def test_christoffersen_detects_clustering():
    spread = np.zeros(1000, dtype=int)
    spread[::50] = 1
    clustered = np.zeros(1000, dtype=int)
    clustered[500:520] = 1
    assert not backtesting.christoffersen_independence(spread)["reject_H0"]
    assert backtesting.christoffersen_independence(clustered)["reject_H0"]


def test_rolling_var_and_exceedances(port):
    sample = port.iloc[:400]
    v = backtesting._rolling_var(sample, "historical", 0.95, window=252)
    assert len(v) == 400 - 252
    assert (v > 0).all()
    hits = backtesting.exceedances(sample, v)
    assert set(hits.unique()) <= {0, 1}


# =====================================================================
# Optimiser
# =====================================================================

@pytest.mark.parametrize("fn", [optimizer.min_variance, optimizer.max_sharpe,
                                optimizer.risk_parity, optimizer.min_cvar])
def test_weights_long_only_fully_invested(rets, fn):
    w = fn(rets)
    assert w.sum() == pytest.approx(1.0, abs=1e-4)
    assert (w >= -1e-6).all()


def test_max_weight_cap(rets):
    w = optimizer.max_sharpe(rets, max_weight=0.3)
    assert w.max() <= 0.3 + 1e-4


def test_min_variance_has_lowest_vol(rets):
    cmp = optimizer.compare_methods(rets)
    assert cmp["Vol (%)"].idxmin() == "Min Variance" or \
        cmp.loc["Min Variance", "Vol (%)"] == pytest.approx(cmp["Vol (%)"].min(), abs=1e-3)


def test_max_sharpe_has_highest_sharpe(rets):
    cmp = optimizer.compare_methods(rets)
    assert cmp.loc["Max Sharpe", "Sharpe"] == pytest.approx(cmp["Sharpe"].max(), abs=1e-3)


# =====================================================================
# Performance
# =====================================================================

def test_drawdown_known_path():
    r = pd.Series([0.10, 0.5 / 1.1 - 1, 0.2])   # $1 → 1.10 → 0.50 → 0.60
    assert performance.cumulative_returns(r).iloc[-1] == pytest.approx(0.6)
    assert performance.max_drawdown(r) == pytest.approx(0.5 / 1.1 - 1)


def test_performance_summary_fields(port):
    s = performance.performance_summary(port)["Value"]
    for key in ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Max Drawdown (%)"]:
        assert np.isfinite(s[key])
    assert s["Max Drawdown (%)"] < 0
    assert 0 < s["Hit Rate (%)"] < 100


# =====================================================================
# Copula
# =====================================================================

def test_tail_dependence_t_properties():
    lo = copula.tail_dependence_t(0.3, 5)["lambda_lower"]
    hi = copula.tail_dependence_t(0.9, 5)["lambda_lower"]
    thin = copula.tail_dependence_t(0.9, 200)["lambda_lower"]
    assert 0 < lo < hi < 1
    assert thin < hi                          # ν → ∞ approaches Gaussian (λ = 0)
    assert copula.tail_dependence_gaussian()["lambda_lower"] == 0.0


def test_gaussian_copula_correlation_shape(rets):
    res = copula.fit_gaussian_copula(rets.iloc[-500:])
    corr = res["correlation_matrix"].to_numpy()
    assert corr.shape == (5, 5)
    assert np.allclose(np.diag(corr), 1.0, atol=1e-3)


# =====================================================================
# Volatility (GARCH)
# =====================================================================

def test_garch_fits_and_best_model(port):
    fits = volatility.fit_all_models(port * 100)
    comp = volatility.compare_models(fits)
    assert set(comp.index) == {"GARCH(1,1)", "GJR-GARCH(1,1)", "EGARCH(1,1)"}
    name, res = volatility.best_model(fits)
    assert comp.loc[name, "AIC"] == comp["AIC"].min()
    vol = volatility.conditional_volatility(res)
    assert len(vol) == len(port) and (vol > 0).all()


# =====================================================================
# Factor model & liquidity (cached CSVs)
# =====================================================================

def test_factor_risk_decomposition_sums_to_100(rets):
    from src.factor_model import risk_decomposition
    rd = risk_decomposition(rets)
    assert rd["Risk Contribution (%)"].sum() == pytest.approx(100, abs=0.5)


def test_liquidity_lvar_exceeds_base_var():
    from src.liquidity import liquidity_metrics, days_to_liquidate
    assert days_to_liquidate(1_000_000, 100, 1_000, participation=0.2) == pytest.approx(50)
    per_asset, summary = liquidity_metrics(portfolio_value=10_000_000)
    assert summary["Liquidity-Adjusted VaR ($)"] >= summary["Base Portfolio VaR ($)"]
    assert len(per_asset) == 5
