"""
copula.py — Copula-Based Dependence Modelling for Multi-Asset Portfolios.

Fits Gaussian and Student-t copulas to equity return data and analyses how
the dependence structure shifts between calm and crisis regimes.

Key outputs:
  • Fitted copula parameters (correlation matrices, degrees of freedom).
  • Tail dependence coefficients (λ_upper, λ_lower) for each copula.
  • Crisis vs. calm period comparison tables and scatter visualisations.

Dependencies: scipy, numpy, pandas, matplotlib, copulas (SDV copulas library).

Usage
-----
    from src.copula import fit_copulas, tail_dependence, plot_tail_comparison
    results = fit_copulas(returns, crisis_start, crisis_end)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t as t_dist, kendalltau
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore")


# ================================================================
# PERIOD DEFINITIONS
# ================================================================

CRISIS_PERIODS = {
    "2020 COVID Crash":  ("2020-02-15", "2020-04-30"),
    "2018 Q4 Selloff":   ("2018-10-01", "2018-12-31"),
}

CALM_PERIODS = {
    "2017 Bull Run":     ("2017-01-01", "2017-12-31"),
    "2019 Steady":       ("2019-01-01", "2019-12-31"),
}


# ================================================================
# PSEUDO-OBSERVATION TRANSFORM (Probability Integral Transform)
# ================================================================

def _to_pseudo_obs(data):
    """
    Transform data to pseudo-observations (uniform margins) using
    the empirical CDF (rank-based) method.

    Parameters
    ----------
    data : pd.DataFrame
        Raw returns.

    Returns
    -------
    np.ndarray
        Values in (0, 1).
    """
    n = len(data)
    ranks = data.rank()
    u = ranks / (n + 1)  # avoids 0 and 1
    return u.values


# ================================================================
# GAUSSIAN COPULA
# ================================================================

def fit_gaussian_copula(data):
    """
    Fit a Gaussian copula to the data.

    Transforms margins to standard normal and computes the Pearson
    correlation matrix of the transformed data.

    Parameters
    ----------
    data : pd.DataFrame
        Return data (rows = dates, columns = assets).

    Returns
    -------
    dict
        'type', 'correlation_matrix', 'columns'.
    """
    u = _to_pseudo_obs(data)
    z = norm.ppf(u)
    z = np.clip(z, -6, 6)  # numerical guard

    corr = np.corrcoef(z, rowvar=False)

    return {
        "type":               "Gaussian",
        "correlation_matrix": pd.DataFrame(corr, index=data.columns,
                                           columns=data.columns).round(4),
        "columns":            list(data.columns),
    }


# ================================================================
# STUDENT-t COPULA
# ================================================================

def _t_copula_log_likelihood(nu, z, corr):
    """
    Log-likelihood for the t-copula given correlation matrix and DoF.

    Parameters
    ----------
    nu : float
        Degrees of freedom.
    z : np.ndarray
        Standard-normal transformed pseudo-observations.
    corr : np.ndarray
        Correlation matrix.

    Returns
    -------
    float
        Negative log-likelihood (for minimization).
    """
    d = z.shape[1]
    n = z.shape[0]

    # Transform from normal quantiles to t-quantiles
    t_quantiles = t_dist.ppf(norm.cdf(z), df=nu)
    t_quantiles = np.clip(t_quantiles, -50, 50)

    try:
        corr_inv = np.linalg.inv(corr)
        log_det = np.linalg.slogdet(corr)[1]
    except np.linalg.LinAlgError:
        return 1e12

    from scipy.special import gammaln

    # t-copula density (simplified)
    ll = 0.0
    ll += n * (gammaln((nu + d) / 2) - gammaln(nu / 2)
               - (d / 2) * np.log(nu * np.pi) - 0.5 * log_det)

    for i in range(n):
        xi = t_quantiles[i]
        quad = xi @ corr_inv @ xi
        ll += -(nu + d) / 2 * np.log(1 + quad / nu)
        # Subtract marginal t densities
        for j in range(d):
            ll -= np.log(t_dist.pdf(xi[j], df=nu) + 1e-300)

    return -ll  # negative for minimization


def fit_t_copula(data, nu_range=(2.5, 30)):
    """
    Fit a Student-t copula by MLE over degrees of freedom.

    Parameters
    ----------
    data : pd.DataFrame
        Return data.
    nu_range : tuple
        Bounds for the degrees-of-freedom search.

    Returns
    -------
    dict
        'type', 'correlation_matrix', 'nu' (degrees of freedom), 'columns'.
    """
    u = _to_pseudo_obs(data)
    z = norm.ppf(u)
    z = np.clip(z, -6, 6)

    corr = np.corrcoef(z, rowvar=False)

    # Ensure positive-definite
    eigvals = np.linalg.eigvalsh(corr)
    if eigvals.min() < 1e-6:
        corr += np.eye(corr.shape[0]) * 1e-4

    result = minimize_scalar(
        _t_copula_log_likelihood,
        bounds=nu_range,
        args=(z, corr),
        method='bounded',
    )
    nu = result.x

    return {
        "type":               "Student-t",
        "correlation_matrix": pd.DataFrame(corr, index=data.columns,
                                           columns=data.columns).round(4),
        "nu":                 round(nu, 2),
        "columns":            list(data.columns),
    }


# ================================================================
# TAIL DEPENDENCE COEFFICIENTS
# ================================================================

def tail_dependence_gaussian():
    """
    The Gaussian copula has **zero** tail dependence (λ_U = λ_L = 0).

    This is a well-known theoretical result: the Gaussian copula
    cannot capture joint extreme events.

    Returns
    -------
    dict
    """
    return {"lambda_upper": 0.0, "lambda_lower": 0.0}


def tail_dependence_t(rho, nu):
    """
    Compute the tail dependence coefficient of the bivariate t-copula.

    The upper and lower tail dependence coefficients are symmetric
    for the standard t-copula:

        λ = 2 · t_{ν+1}( -√((ν+1)(1−ρ)/(1+ρ)) )

    Parameters
    ----------
    rho : float
        Pairwise correlation.
    nu : float
        Degrees of freedom.

    Returns
    -------
    dict
        'lambda_upper', 'lambda_lower'.
    """
    arg = -np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
    lam = 2 * t_dist.cdf(arg, df=nu + 1)
    return {"lambda_upper": round(lam, 4), "lambda_lower": round(lam, 4)}


def compute_all_tail_dependence(t_copula_result):
    """
    Compute tail dependence for all asset pairs from a fitted t-copula.

    Parameters
    ----------
    t_copula_result : dict
        Output of ``fit_t_copula``.

    Returns
    -------
    pd.DataFrame
        Columns: Pair, ρ, ν, λ_upper, λ_lower.
    """
    corr = t_copula_result["correlation_matrix"]
    nu = t_copula_result["nu"]
    cols = t_copula_result["columns"]
    rows = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rho = corr.iloc[i, j]
            td = tail_dependence_t(rho, nu)
            rows.append({
                "Pair":      f"{cols[i]} / {cols[j]}",
                "ρ":         round(rho, 4),
                "ν":         nu,
                "λ_upper":   td["lambda_upper"],
                "λ_lower":   td["lambda_lower"],
            })

    return pd.DataFrame(rows)


# ================================================================
# CRISIS vs CALM COMPARISON
# ================================================================

def fit_copulas_for_period(data, start, end):
    """
    Fit both copulas on a specific sub-period.

    Parameters
    ----------
    data : pd.DataFrame
        Full return series.
    start, end : str
        Date strings for slicing.

    Returns
    -------
    dict
        'gaussian', 't', 'period', 'n_obs'.
    """
    subset = data.loc[start:end].dropna()
    if len(subset) < 20:
        raise ValueError(f"Period {start}→{end} has only {len(subset)} obs.")

    return {
        "gaussian": fit_gaussian_copula(subset),
        "t":        fit_t_copula(subset),
        "period":   f"{start} → {end}",
        "n_obs":    len(subset),
    }


def compare_periods(data, crisis_start, crisis_end,
                    calm_start, calm_end):
    """
    Compare copula fits between a crisis and a calm period.

    Returns
    -------
    dict with keys 'crisis' and 'calm', each containing copula results.
    """
    crisis = fit_copulas_for_period(data, crisis_start, crisis_end)
    calm = fit_copulas_for_period(data, calm_start, calm_end)
    return {"crisis": crisis, "calm": calm}


def correlation_shift_table(comparison):
    """
    Build a table showing how average pairwise correlation changes
    between calm and crisis periods for both copulas.

    Parameters
    ----------
    comparison : dict
        Output of ``compare_periods``.

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for regime, label in [("calm", "Calm"), ("crisis", "Crisis")]:
        for copula_type in ["gaussian", "t"]:
            result = comparison[regime][copula_type]
            corr = result["correlation_matrix"]

            # Average off-diagonal correlation
            mask = ~np.eye(corr.shape[0], dtype=bool)
            avg_corr = corr.values[mask].mean()

            row = {
                "Regime":   label,
                "Copula":   result["type"],
                "Avg ρ":    round(avg_corr, 4),
                "Period":   comparison[regime]["period"],
                "N":        comparison[regime]["n_obs"],
            }
            if copula_type == "t":
                row["ν (DoF)"] = result["nu"]
            else:
                row["ν (DoF)"] = "∞"

            rows.append(row)

    return pd.DataFrame(rows)


# ================================================================
# VISUALISATION
# ================================================================

def plot_tail_dependence_comparison(data, crisis_start, crisis_end,
                                   calm_start, calm_end,
                                   save_path=None):
    """
    Bar chart comparing tail dependence (λ) for each asset pair
    across crisis and calm periods (t-copula only, since Gaussian λ = 0).

    Parameters
    ----------
    data : pd.DataFrame
        Full return series.
    crisis_start, crisis_end : str
    calm_start, calm_end : str
    save_path : str or None
        If set, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    comp = compare_periods(data, crisis_start, crisis_end,
                           calm_start, calm_end)

    td_crisis = compute_all_tail_dependence(comp["crisis"]["t"])
    td_calm = compute_all_tail_dependence(comp["calm"]["t"])

    pairs = td_calm["Pair"]
    x = np.arange(len(pairs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width / 2, td_calm["λ_upper"], width,
                   label=f'Calm ({comp["calm"]["period"]})',
                   color="#4CAF50", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width / 2, td_crisis["λ_upper"], width,
                   label=f'Crisis ({comp["crisis"]["period"]})',
                   color="#F44336", alpha=0.85, edgecolor="white")

    ax.set_xlabel("Asset Pair", fontsize=12)
    ax.set_ylabel("Tail Dependence Coefficient (λ)", fontsize=12)
    ax.set_title("Tail Dependence: Calm vs Crisis Period (Student-t Copula)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  📊 Saved: {save_path}")

    return fig


def plot_copula_scatter(data, pair, crisis_start, crisis_end,
                        calm_start, calm_end, save_path=None):
    """
    Side-by-side scatter plots of pseudo-observations for one asset pair
    during calm vs. crisis periods.

    Parameters
    ----------
    data : pd.DataFrame
        Full return series.
    pair : tuple of str
        Two column names, e.g. ('AAPL', 'MSFT').
    """
    calm_data = data.loc[calm_start:calm_end, list(pair)].dropna()
    crisis_data = data.loc[crisis_start:crisis_end, list(pair)].dropna()

    u_calm = _to_pseudo_obs(calm_data)
    u_crisis = _to_pseudo_obs(crisis_data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(u_calm[:, 0], u_calm[:, 1], alpha=0.5, s=15,
                    color="#4CAF50", edgecolors="none")
    axes[0].set_title(f"Calm: {calm_start} → {calm_end}", fontsize=12)
    axes[0].set_xlabel(pair[0])
    axes[0].set_ylabel(pair[1])

    axes[1].scatter(u_crisis[:, 0], u_crisis[:, 1], alpha=0.5, s=15,
                    color="#F44336", edgecolors="none")
    axes[1].set_title(f"Crisis: {crisis_start} → {crisis_end}", fontsize=12)
    axes[1].set_xlabel(pair[0])
    axes[1].set_ylabel(pair[1])

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)

    fig.suptitle(f"Copula Pseudo-Observations: {pair[0]} vs {pair[1]}",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  📊 Saved: {save_path}")

    return fig


# ================================================================
# CLI ENTRY POINT
# ================================================================

if __name__ == "__main__":
    from returns import load_prices, log_returns
    import os

    prices = load_prices()
    rets = log_returns(prices)

    print("=" * 65)
    print("  Copula-Based Dependence Analysis")
    print("=" * 65)

    # --- Fit on full sample ---
    print("\n📈 Full-Sample Copula Fit")
    gauss = fit_gaussian_copula(rets)
    print("  Gaussian Copula — Correlation Matrix:")
    print(gauss["correlation_matrix"].to_string())

    t_cop = fit_t_copula(rets)
    print(f"\n  Student-t Copula — ν = {t_cop['nu']}")
    print("  Correlation Matrix:")
    print(t_cop["correlation_matrix"].to_string())

    # --- Tail dependence ---
    print("\n📉 Tail Dependence Coefficients (Student-t):")
    td = compute_all_tail_dependence(t_cop)
    print(td.to_string(index=False))
    print("  (Gaussian copula: λ = 0 for all pairs — no tail dependence)")

    # --- Crisis vs Calm ---
    crisis_s, crisis_e = "2020-02-15", "2020-04-30"
    calm_s, calm_e = "2019-01-01", "2019-12-31"

    print(f"\n🔄 Comparing Calm ({calm_s}→{calm_e}) vs Crisis ({crisis_s}→{crisis_e})")
    comp = compare_periods(rets, crisis_s, crisis_e, calm_s, calm_e)
    shift = correlation_shift_table(comp)
    print(shift.to_string(index=False))

    # --- Plots ---
    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "report")
    os.makedirs(report_dir, exist_ok=True)
    plot_tail_dependence_comparison(
        rets, crisis_s, crisis_e, calm_s, calm_e,
        save_path=os.path.join(report_dir, "tail_dependence_comparison.png"))

    plot_copula_scatter(
        rets, ("AAPL", "MSFT"), crisis_s, crisis_e, calm_s, calm_e,
        save_path=os.path.join(report_dir, "copula_scatter_aapl_msft.png"))

    print("\n✅ Copula analysis complete!")
