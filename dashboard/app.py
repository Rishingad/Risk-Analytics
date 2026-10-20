"""
Dashboard — institutional-grade Streamlit app unifying the risk engine's
market, backtesting, volatility, factor, credit, dependence, construction,
performance, liquidity and stress-testing analytics behind a single polished interface.

Plugs directly into the ``src`` package (imported lazily per tab):
    src.var_models · src.backtesting · src.volatility · src.factor_model · src.optimizer · src.performance
    src.liquidity · src.credit_merton · src.copula · src.stress_testing · src.returns
"""

import os
import sys

# ``streamlit run dashboard/app.py`` puts dashboard/ on sys.path, not the
# project root — so ``import src`` fails. Add the project root so ``src`` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# Design system — colour palette shared by CSS and charts
# =====================================================================
BG        = "#0b0f17"
PANEL     = "#141a24"
PANEL_2   = "#10151f"
BORDER    = "#232c3b"
TEXT      = "#e8ecf3"
MUTED     = "#8a93a6"
ACCENT    = "#4f8cff"   # primary blue
ACCENT_2  = "#22d3ee"   # cyan
POS       = "#34d399"   # green
NEG       = "#f87171"   # red
WARN      = "#fbbf24"   # amber
VIOLET    = "#a78bfa"
CHART_BG  = "#0e131d"
CYCLE     = [ACCENT, ACCENT_2, VIOLET, "#f472b6", WARN, POS, "#fb7185", "#38bdf8"]


# =====================================================================
# Global styling
# =====================================================================

def configure_charts():
    """Make every matplotlib figure match the dark theme."""
    plt.rcParams.update({
        "figure.facecolor":  CHART_BG,
        "axes.facecolor":    CHART_BG,
        "savefig.facecolor": CHART_BG,
        "axes.edgecolor":    BORDER,
        "axes.labelcolor":   "#c7cedb",
        "axes.titlecolor":   TEXT,
        "axes.titlesize":    12,
        "axes.titleweight":  "bold",
        "text.color":        TEXT,
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "grid.color":        "#1b2230",
        "grid.linewidth":    0.8,
        "axes.grid":         True,
        "axes.axisbelow":    True,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.size":         10,
        "legend.frameon":    False,
        "figure.autolayout": True,
        "axes.prop_cycle":   plt.cycler(color=CYCLE),
    })


def inject_css():
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"], .stApp {{
            font-family: 'Inter', -apple-system, sans-serif;
        }}
        .stApp {{ background: radial-gradient(1200px 600px at 20% -5%, #131a27 0%, {BG} 45%); color: {TEXT}; }}

        /* hide default streamlit chrome — but KEEP the toolbar container itself,
           because it holds the sidebar reopen chevron (stExpandSidebarButton)
           when the sidebar is collapsed */
        #MainMenu, footer {{ visibility: hidden; }}
        header[data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stToolbarActions"], [data-testid="stStatusWidget"],
        [data-testid="stMainMenu"], [data-testid="stMainMenuButton"],
        [data-testid="stAppDeployButton"], .stDeployButton {{ display: none !important; }}
        [data-testid="stExpandSidebarButton"] {{
            color: {TEXT} !important; background: {PANEL};
            border: 1px solid {BORDER}; border-radius: 10px;
        }}
        [data-testid="stExpandSidebarButton"]:hover {{ border-color: {ACCENT}; }}
        .block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }}

        /* ---------- explainer expander ---------- */
        [data-testid="stExpander"] {{
            border: 1px solid {BORDER}; border-radius: 12px;
            background: linear-gradient(160deg, {PANEL_2} 0%, {PANEL} 140%);
            margin-bottom: .6rem;
        }}
        [data-testid="stExpander"] summary {{ font-weight: 600; color: {ACCENT_2}; }}
        [data-testid="stExpander"] p, [data-testid="stExpander"] li {{
            color: #b9c2d4; font-size: .88rem; line-height: 1.55;
        }}

        /* ---------- hero header ---------- */
        .hero {{
            background: linear-gradient(120deg, #16213a 0%, #101725 55%, #0e1420 100%);
            border: 1px solid {BORDER}; border-radius: 16px;
            padding: 1.4rem 1.8rem; margin-bottom: 1.4rem;
            display: flex; align-items: center; justify-content: space-between;
            box-shadow: 0 8px 30px rgba(0,0,0,.35);
        }}
        .hero-title {{ font-size: 1.7rem; font-weight: 800; letter-spacing: -.02em;
            background: linear-gradient(90deg, {TEXT}, {ACCENT_2}); -webkit-background-clip: text;
            -webkit-text-fill-color: transparent; }}
        .hero-sub {{ color: {MUTED}; font-size: .92rem; margin-top: .15rem; font-weight: 500; }}
        .hero-badge {{ text-align: right; }}
        .hero-badge .lbl {{ color: {MUTED}; font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; }}
        .hero-badge .val {{ color: {TEXT}; font-size: 1.05rem; font-weight: 700; }}

        /* ---------- KPI cards ---------- */
        .kpi {{
            background: linear-gradient(160deg, {PANEL} 0%, {PANEL_2} 100%);
            border: 1px solid {BORDER}; border-radius: 14px;
            padding: 1.05rem 1.15rem; height: 100%;
            transition: transform .15s ease, border-color .15s ease;
        }}
        .kpi:hover {{ transform: translateY(-3px); border-color: {ACCENT}; }}
        .kpi-label {{ color: {MUTED}; font-size: .78rem; font-weight: 600;
            text-transform: uppercase; letter-spacing: .06em; }}
        .kpi-value {{ font-size: 1.75rem; font-weight: 700; margin-top: .35rem; line-height: 1.1; }}
        .kpi-sub {{ color: {MUTED}; font-size: .78rem; margin-top: .3rem; }}

        /* ---------- section headers ---------- */
        .section {{ margin: 1.6rem 0 .7rem 0; }}
        .section h3 {{ font-size: 1.12rem; font-weight: 700; margin: 0;
            display: flex; align-items: center; gap: .5rem; }}
        .section h3::before {{ content: ""; width: 4px; height: 18px;
            background: linear-gradient(180deg, {ACCENT}, {ACCENT_2}); border-radius: 3px; display: inline-block; }}
        .section p {{ color: {MUTED}; font-size: .85rem; margin: .25rem 0 0 .95rem; }}

        /* ---------- dataframes ---------- */
        [data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 12px; overflow: hidden; }}

        /* ---------- sidebar ---------- */
        section[data-testid="stSidebar"] {{ background: {PANEL_2}; border-right: 1px solid {BORDER}; }}
        .side-brand {{ font-size: 1.15rem; font-weight: 800; letter-spacing: -.01em; }}
        .side-brand span {{ color: {ACCENT}; }}
        .side-tag {{ color: {MUTED}; font-size: .78rem; margin-bottom: .4rem; }}
        .side-foot {{ color: {MUTED}; font-size: .72rem; line-height: 1.5; border-top: 1px solid {BORDER};
            padding-top: .8rem; margin-top: 1rem; }}

        div[role="radiogroup"] label {{
            background: transparent; border-radius: 9px; padding: .35rem .6rem; margin: .12rem 0;
            transition: background .12s ease; width: 100%;
        }}
        div[role="radiogroup"] label:hover {{ background: {PANEL}; }}

        .stSlider label {{ color: {MUTED}; font-weight: 600; }}
    </style>
    """, unsafe_allow_html=True)


# =====================================================================
# Reusable presentation helpers
# =====================================================================

def hero(as_of, universe):
    st.markdown(f"""
    <div class="hero">
      <div>
        <div class="hero-title">Portfolio Risk Analytics</div>
        <div class="hero-sub">Market · Factor · Credit · Liquidity · Stress — one desk view</div>
      </div>
      <div class="hero-badge">
        <div class="lbl">Data as of</div>
        <div class="val">{as_of}</div>
        <div class="lbl" style="margin-top:.4rem">Universe</div>
        <div class="val" style="font-size:.9rem">{universe}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def section(title, desc=None):
    html = f'<div class="section"><h3>{title}</h3>'
    if desc:
        html += f'<p>{desc}</p>'
    st.markdown(html + '</div>', unsafe_allow_html=True)


def explain(text):
    """Client-friendly 'how to read this' expander shown at the top of a tab."""
    with st.expander("💡  How to read this tab", expanded=False):
        st.markdown(text)


def kpi(col, label, value, sub=None, color=ACCENT):
    html = (f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value" style="color:{color}">{value}</div>')
    if sub:
        html += f'<div class="kpi-sub">{sub}</div>'
    col.markdown(html + '</div>', unsafe_allow_html=True)


def styled_fig(fig):
    """Final polish + render a matplotlib figure inside the app."""
    for ax in fig.get_axes():
        ax.title.set_color(TEXT)
    st.pyplot(fig)
    plt.close(fig)


# =====================================================================
# Data loaders
# =====================================================================

@st.cache_data(ttl=86400)
def load_prices():
    from src.returns import load_prices as lp
    return lp()

@st.cache_data(ttl=86400)
def load_returns():
    from src.returns import log_returns
    return log_returns(load_prices())

@st.cache_data(ttl=86400)
def get_portfolio_returns():
    from src.returns import portfolio_returns
    return portfolio_returns(load_returns())


# =====================================================================
# 1 · Overview
# =====================================================================
def render_overview():
    from src.performance import performance_summary, cumulative_returns, drawdown_series
    port = get_portfolio_returns()
    rets = load_returns()
    latest = port.iloc[-1] * 100
    pv = 1_000_000
    pnl = pv * latest / 100
    perf = performance_summary(port)["Value"]

    explain("""
**This is your portfolio's front page** — a $1M portfolio spread equally across 5 assets (S&P 500, Nasdaq, Apple, Microsoft, JPMorgan), tracked over 10 years.

- **Ann. Return / Volatility** — what the portfolio earned per year on average, and how bumpy the ride was. You want return high relative to volatility.
- **Sharpe Ratio** — return earned *per unit of risk taken*. Above 1.0 is good; 0.5–1.0 is typical for equity portfolios.
- **Max Drawdown** — the worst peak-to-bottom fall ever experienced. This is the pain you would have had to sit through.
- **The chart** — top panel shows $1 growing over time; the red panel below shows every drawdown episode. Deep red valleys = crisis periods (note early 2020).
""")
    section("Portfolio Snapshot", "Headline positioning and risk-adjusted performance for the equal-weight book.")
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Portfolio Value", f"${pv:,.0f}")
    kpi(c2, "Latest Daily P&L", f"${pnl:,.0f}", f"{latest:+.2f}%", POS if pnl >= 0 else NEG)
    kpi(c3, "Ann. Return", f"{perf['Annualised Return (%)']:.1f}%", "geometric", POS)
    kpi(c4, "Sharpe Ratio", f"{perf['Sharpe Ratio']:.2f}", "rf = 2%",
        POS if perf['Sharpe Ratio'] >= 1 else WARN)

    c5, c6, c7, c8 = st.columns(4)
    kpi(c5, "Ann. Volatility", f"{perf['Annualised Vol (%)']:.1f}%")
    kpi(c6, "Max Drawdown", f"{perf['Max Drawdown (%)']:.1f}%", "peak-to-trough", NEG)
    kpi(c7, "Sortino", f"{perf['Sortino Ratio']:.2f}", "downside-adj", ACCENT_2)
    kpi(c8, "Assets", f"{rets.shape[1]}", ", ".join(rets.columns), VIOLET)

    section("Cumulative Performance & Drawdown")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    cum = cumulative_returns(port)
    ax1.plot(cum.index, cum.values, color=ACCENT, lw=1.8)
    ax1.fill_between(cum.index, cum.values, cum.values.min(), color=ACCENT, alpha=0.08)
    ax1.set_ylabel("Growth of $1")
    dd = drawdown_series(port) * 100
    ax2.fill_between(dd.index, dd.values, 0, color=NEG, alpha=0.55)
    ax2.set_ylabel("Drawdown %")
    styled_fig(fig)


# =====================================================================
# 2 · VaR / CVaR
# =====================================================================
def render_var():
    from src.var_models import compute_all_var, summary_table
    explain("""
**"How much could I lose tomorrow?"** — that's the question this tab answers.

- **VaR 95%** — on 19 days out of 20, your daily loss should stay *below* this number. Roughly one day a month will be worse.
- **VaR 99%** — the loss threshold breached only ~2–3 days a year.
- **CVaR (Expected Shortfall)** — *when* one of those bad days hits, this is the **average** loss you should expect. Always larger than VaR; this is the number risk committees care most about.
- **Three methods, one question** — Parametric assumes markets are well-behaved (bell curve); Historical replays what actually happened; Monte Carlo simulates 100,000 possible tomorrows. **When Historical is much bigger than Parametric, real markets have fatter tails than the textbook assumes** — which is exactly what you see at 99%.
""")
    section("Value-at-Risk & Expected Shortfall",
            "Potential loss over one day at 95% and 99% confidence across three methodologies.")
    port_ret = get_portfolio_returns()
    rets = load_returns()
    df = summary_table(compute_all_var(port_ret, rets))

    worst = df.loc[df["VaR (%)"].idxmax()]
    c1, c2, c3 = st.columns(3)
    kpi(c1, "Max 1-Day VaR", f"{df['VaR (%)'].max():.2f}%",
        f"{worst['Method']} · {worst['Confidence']}", NEG)
    kpi(c2, "Max Expected Shortfall", f"{df['CVaR (%)'].max():.2f}%", "worst-case tail", NEG)
    kpi(c3, "On $1M", f"${df['VaR (%)'].max()/100*1_000_000:,.0f}", "at risk / day", WARN)

    section("All Methods")
    st.dataframe(df.style.format({"VaR (%)": "{:.2f}", "CVaR (%)": "{:.2f}"})
                 .background_gradient(cmap="Reds", subset=["VaR (%)", "CVaR (%)"]),
                 width="stretch", hide_index=True)
    st.caption("Parametric assumes normality · Historical uses the empirical distribution · "
               "Monte Carlo simulates correlated asset paths. Divergence signals fat tails.")


# =====================================================================
# 3 · VaR Backtesting
# =====================================================================
@st.cache_data(ttl=86400, show_spinner="Running rolling VaR backtests…")
def get_backtest():
    from src.backtesting import run_backtest, results_table
    return results_table(run_backtest(get_portfolio_returns(), window=252))

@st.cache_data(ttl=86400)
def get_rolling_var(method, confidence):
    from src.backtesting import _rolling_var
    return _rolling_var(get_portfolio_returns(), method, confidence, window=252)


def render_backtest():
    from src.backtesting import exceedances
    explain("""
**"Can I trust the VaR numbers?"** — a VaR model is only useful if reality agrees with it. Each day we estimate VaR from the previous year of data, then check whether the next day's loss broke through it.

- **Exceedance** — a day where the actual loss was bigger than VaR. At 99% you expect ~1 in 100 days; at 95%, ~1 in 20.
- **Kupiec test** — are there the *right number* of exceedances? Fail = the model systematically under- (or over-) states risk.
- **Christoffersen test** — do exceedances *cluster* together? Fail = the model is slow to react when volatility jumps, so bad days arrive in bunches.
- **Reading the results** — every method passes Kupiec at 95% but fails at 99%, and all fail Christoffersen: a 1-year rolling window misses volatility regimes. This is the motivation for the **Volatility (GARCH)** tab.
""")
    section("VaR Model Backtesting",
            "Rolling 252-day VaR tested against realised next-day returns (Kupiec POF + Christoffersen).")
    df = get_backtest()

    kupiec_pass = (df["Kupiec Pass"].str.contains("Pass")).sum()
    christo_pass = (df["Christo Pass"].str.contains("Pass")).sum()
    c1, c2, c3 = st.columns(3)
    kpi(c1, "Kupiec Passes", f"{kupiec_pass} / {len(df)}", "exceedance count OK",
        POS if kupiec_pass == len(df) else WARN)
    kpi(c2, "Christoffersen Passes", f"{christo_pass} / {len(df)}", "no clustering",
        POS if christo_pass == len(df) else NEG)
    kpi(c3, "Test Window", f"{int(df['Observations'].iloc[0]):,}", "out-of-sample days", ACCENT_2)

    section("Test Results")
    cols = ["Method", "Confidence", "Exceedances", "Expected Rate", "Observed Rate",
            "Kupiec p-value", "Kupiec Pass", "Christo p-value", "Christo Pass"]
    st.dataframe(df[cols].style.format({"Expected Rate": "{:.2%}", "Observed Rate": "{:.2%}",
                                        "Kupiec p-value": "{:.4f}", "Christo p-value": "{:.4f}"})
                 .background_gradient(cmap="RdYlGn", vmin=0, vmax=0.1,
                                      subset=["Kupiec p-value", "Christo p-value"]),
                 width="stretch", hide_index=True)
    st.caption("p-value < 0.05 → reject H₀ → test fails.")

    section("Exceedance Plot", "Realised daily returns vs the rolling VaR threshold; red dots are breaches.")
    cset = st.columns([1, 1, 2])
    method = cset[0].selectbox("Method", ["historical", "parametric"], format_func=str.title)
    conf = cset[1].selectbox("Confidence", [0.99, 0.95], format_func=lambda c: f"{c:.0%}")

    port = get_portfolio_returns()
    var_s = get_rolling_var(method, conf)
    hits = exceedances(port, var_s)
    r = port.loc[var_s.index] * 100
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(r.index, r.values, color=MUTED, lw=0.6, alpha=0.7, label="Daily return")
    ax.plot(var_s.index, -var_s.values * 100, color=ACCENT, lw=1.4, label=f"−VaR {conf:.0%}")
    breach = hits[hits == 1].index
    ax.scatter(breach, r.loc[breach], color=NEG, s=12, zorder=3, label=f"Exceedances ({len(breach)})")
    ax.set_ylabel("Return %")
    ax.legend(fontsize=8, loc="lower left")
    styled_fig(fig)


# =====================================================================
# 4 · Volatility (GARCH)
# =====================================================================
@st.cache_resource(ttl=86400, show_spinner="Fitting GARCH models…")
def get_garch_fits(asset):
    from src.volatility import fit_all_models
    series = get_portfolio_returns() if asset == "Portfolio" else load_returns()[asset]
    return fit_all_models(series * 100)


def render_volatility():
    from src.volatility import compare_models, best_model, conditional_volatility
    explain("""
**"How risky is the market *right now*?"** — volatility isn't constant. Calm stretches are followed by storms, and storms tend to persist. GARCH models capture this by letting today's volatility depend on yesterday's shocks.

- **GARCH(1,1)** — the classic: big moves raise tomorrow's volatility, which then decays back to normal.
- **GJR-GARCH / EGARCH** — add the *leverage effect*: sell-offs raise volatility more than rallies of the same size.
- **AIC / BIC** — model-fit scores penalised for complexity. **Lower is better**; the winner is used for the chart and forecast.
- **The chart** — the blue line is the model's estimate of daily volatility through time; grey is the size of each day's actual move. Note how the line spikes in 2020 and decays over the following months.
- **GARCH VaR** — tomorrow's VaR using *today's* volatility instead of a 1-year average. It reacts far faster than the rolling VaR that failed the clustering test in **VaR Backtesting**.
""")
    section("Conditional Volatility (GARCH)",
            "GARCH-family models with skewed-t innovations — time-varying risk and next-day forecasts.")
    options = ["Portfolio"] + list(load_returns().columns)
    asset = st.selectbox("Series", options)

    fits = get_garch_fits(asset)
    comp = compare_models(fits)
    name, res = best_model(fits, "aic")

    vol = conditional_volatility(res) * 100
    fc_vol = float(np.sqrt(res.forecast(horizon=1, reindex=False).variance.iloc[-1, 0]))
    long_run = float(vol.mean())
    # 1% quantile of the fitted standardised skewed-t, scaled by tomorrow's σ
    dist_params = res.params[res.model.distribution.parameter_names()]
    q01 = float(res.model.distribution.ppf(0.01, dist_params))
    garch_var99 = -(float(res.params.get("mu", 0.0)) + q01 * fc_vol)

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Best Model (AIC)", name, f"AIC {comp.loc[name, 'AIC']:,.0f}", ACCENT)
    kpi(c2, "Next-Day Vol", f"{fc_vol:.2f}%", f"{fc_vol * np.sqrt(252):.1f}% annualised",
        NEG if fc_vol > long_run else POS)
    kpi(c3, "Long-Run Avg Vol", f"{long_run:.2f}%", "daily, in-sample mean", ACCENT_2)
    kpi(c4, "GARCH VaR 99%", f"{garch_var99:.2f}%", "skewed-t quantile × σₜ₊₁", WARN)

    left, right = st.columns([1, 1.6])
    with left:
        section("Model Comparison")
        st.dataframe(comp.style.format({"AIC": "{:,.1f}", "BIC": "{:,.1f}", "Log-Lik": "{:,.1f}"})
                     .highlight_min(subset=["AIC", "BIC"], color="#1d3b2f"),
                     width="stretch")
        section("Fitted Parameters", name)
        st.dataframe(res.params.to_frame("Estimate").style.format("{:.4f}"), width="stretch")
    with right:
        section("Conditional Volatility vs |Return|")
        fig, ax = plt.subplots(figsize=(7, 4.2))
        absr = res.resid.abs()
        ax.plot(absr.index, absr.values, color=MUTED, lw=0.5, alpha=0.5, label="|Return| %")
        ax.plot(vol.index, vol.values, color=ACCENT, lw=1.4, label=f"σₜ {name}")
        ax.axhline(long_run, color=WARN, lw=0.9, ls="--", label="Long-run avg")
        ax.set_ylabel("Daily %")
        ax.legend(fontsize=8, loc="upper left")
        styled_fig(fig)


# =====================================================================
# 5 · Factor Model
# =====================================================================
def render_factors():
    from src.factor_model import factor_exposures, summary_table
    explain("""
**"What is my portfolio *really* a bet on?"** — five stocks can secretly be one single bet. This tab decomposes returns into six well-known market forces (the Fama-French factors used by BlackRock, AQR, and every major fund):

- **Mkt-RF** — overall stock market · **SMB** — small vs large companies · **HML** — cheap "value" vs expensive "growth" stocks · **RMW** — profitable vs unprofitable · **CMA** — conservative vs aggressive investment · **Mom** — recent winners vs losers.
- **Beta** — the portfolio's sensitivity to each force. Green bars = you gain when that factor rises; red = you lose.
- **Risk Contribution** — the key column: what % of your total risk each force explains. If Mkt-RF dominates (~90%+ here), diversification across these 5 names is largely an illusion — **you own one big market bet**.
- **Alpha** — return left over *after* all factors are accounted for: skill (or luck), not market exposure.
""")
    section("Multi-Factor Risk Model",
            "Fama-French 5 + Momentum decomposition — what systematically drives portfolio risk.")
    rets = load_returns()
    tbl, alpha, r2, idio = summary_table(rets)

    c1, c2, c3 = st.columns(3)
    kpi(c1, "Annualised Alpha", f"{alpha:+.2f}%", "factor-neutral excess", POS if alpha >= 0 else NEG)
    kpi(c2, "R²", f"{r2:.2f}", "variance explained by factors", ACCENT_2)
    kpi(c3, "Idiosyncratic Risk", f"{idio:.1f}%", "stock-specific share", VIOLET)

    left, right = st.columns([1.1, 1])
    with left:
        section("Portfolio Factor Exposures")
        fig, ax = plt.subplots(figsize=(6, 3.6))
        colors = [POS if v >= 0 else NEG for v in tbl["Beta"]]
        ax.barh(tbl.index, tbl["Beta"], color=colors, edgecolor="none")
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_xlabel("Beta")
        ax.invert_yaxis()
        styled_fig(fig)
    with right:
        section("Risk Contribution")
        st.dataframe(tbl.style.format({"Beta": "{:.3f}", "Risk Contribution (%)": "{:.1f}"})
                     .background_gradient(cmap="RdBu_r", subset=["Beta"]),
                     width="stretch")

    section("Per-Asset Factor Exposures")
    st.dataframe(factor_exposures(rets).style.format("{:.3f}")
                 .background_gradient(cmap="RdBu_r", vmin=-1, vmax=1),
                 width="stretch")


# =====================================================================
# 6 · Optimisation
# =====================================================================
def render_optimizer():
    from src.optimizer import compare_methods, efficient_frontier
    explain("""
**"Is equal-weight actually the best way to split my money?"** — this tab computes four alternative recipes and compares them:

- **Equal Weight** — the naive 20% each baseline.
- **Min Variance** — the smoothest possible ride (lowest volatility), usually at the cost of return.
- **Max Sharpe** — the mathematically best return-per-risk trade-off. This is what "optimal" means in textbooks.
- **Risk Parity** — each asset contributes the *same amount of risk* (popularised by Bridgewater). More robust than Max Sharpe when forecasts are uncertain.
- **Min CVaR** — minimises the average of the worst days, protecting the tail specifically.
- **The frontier chart** — every point on the blue curve is the best return achievable for that level of risk. Portfolios *below* the curve are leaving money on the table; dots show where each recipe lands.
""")
    section("Portfolio Construction",
            "Optimised weightings vs the equal-weight baseline, with the long-only efficient frontier.")
    rets = load_returns()
    cmp = compare_methods(rets)

    best = cmp["Sharpe"].idxmax()
    c1, c2, c3 = st.columns(3)
    kpi(c1, "Best Sharpe Method", best, f"Sharpe {cmp.loc[best,'Sharpe']:.2f}", POS)
    kpi(c2, "Its Return", f"{cmp.loc[best,'Return (%)']:.1f}%", "annualised", ACCENT_2)
    kpi(c3, "Its Volatility", f"{cmp.loc[best,'Vol (%)']:.1f}%", "annualised", WARN)

    left, right = st.columns([1, 1.05])
    with left:
        section("Optimised Weights & Stats")
        st.dataframe(cmp.style.format("{:.3f}")
                     .format({"Return (%)": "{:.1f}", "Vol (%)": "{:.1f}", "Sharpe": "{:.2f}"})
                     .background_gradient(cmap="YlGnBu", subset=["Sharpe"]),
                     width="stretch")
    with right:
        section("Efficient Frontier")
        front = efficient_frontier(rets)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(front["Vol (%)"], front["Return (%)"], color=ACCENT, lw=2, zorder=1)
        for i, (name, row) in enumerate(cmp.iterrows()):
            ax.scatter(row["Vol (%)"], row["Return (%)"], s=90,
                       color=CYCLE[i % len(CYCLE)], edgecolor=BG, lw=1.2, zorder=3, label=name)
        ax.set_xlabel("Volatility (%)"); ax.set_ylabel("Return (%)")
        ax.legend(fontsize=8, loc="lower right")
        styled_fig(fig)


# =====================================================================
# 7 · Performance
# =====================================================================
def render_performance():
    from src.performance import performance_summary, cumulative_returns, drawdown_series, rolling_sharpe
    explain("""
**The full report card** — did the returns justify the risk taken?

- **Sharpe** — excess return per unit of *total* volatility. **Sortino** — same idea but only punishes *downside* moves (upside volatility isn't a problem). **Calmar** — annual return divided by the worst drawdown: "how fast do I earn back my worst-case loss?"
- **Hit Rate** — % of days that were positive. Even great portfolios sit near 53–55%.
- **Skew / Kurtosis** — the shape of the return distribution. Negative skew = crashes are bigger than rallies; high kurtosis (here ~13 vs 0 for a bell curve) = extreme days happen **far** more often than a normal distribution predicts.
- **Rolling Sharpe** (bottom chart) — performance quality through time. Long stretches below zero are regimes where you weren't being paid to hold this risk.
""")
    section("Performance & Risk-Adjusted Returns",
            "Full tear sheet for the equal-weight portfolio.")
    port = get_portfolio_returns()
    s = performance_summary(port)["Value"]

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Cumulative Return", f"{s['Cumulative Return (%)']:.0f}%", color=POS)
    kpi(c2, "Sharpe", f"{s['Sharpe Ratio']:.2f}", color=ACCENT)
    kpi(c3, "Calmar", f"{s['Calmar Ratio']:.2f}", "return / max-DD", ACCENT_2)
    kpi(c4, "Hit Rate", f"{s['Hit Rate (%)']:.0f}%", "positive days", VIOLET)

    left, right = st.columns([1, 1.3])
    with left:
        section("Statistics")
        st.dataframe(performance_summary(port).style.format("{:.3f}")
                     .background_gradient(cmap="YlOrRd"), width="stretch")
    with right:
        section("Growth, Drawdown & Rolling Sharpe")
        fig, axes = plt.subplots(3, 1, figsize=(7, 5.4), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1, 1]})
        cum = cumulative_returns(port)
        axes[0].plot(cum.index, cum.values, color=POS, lw=1.6)
        axes[0].fill_between(cum.index, cum.values, cum.values.min(), color=POS, alpha=0.08)
        axes[0].set_ylabel("Growth $1")
        dd = drawdown_series(port) * 100
        axes[1].fill_between(dd.index, dd.values, 0, color=NEG, alpha=0.55)
        axes[1].set_ylabel("DD %")
        rs = rolling_sharpe(port)
        axes[2].plot(rs.index, rs.values, color=ACCENT, lw=1.3)
        axes[2].axhline(0, color=MUTED, lw=0.7)
        axes[2].set_ylabel("Roll Sharpe")
        styled_fig(fig)


# =====================================================================
# 8 · Liquidity
# =====================================================================
def render_liquidity():
    from src.liquidity import liquidity_metrics
    explain("""
**"If I had to get out, could I — and what would it cost?"** — standard VaR assumes you can sell everything instantly at the screen price. In reality, dumping a large position too fast moves the market against you.

- **Participation rate** — the fraction of a stock's normal daily volume you can trade without moving the price (20% is a common desk rule). Larger portfolio + lower participation = longer exit.
- **Days to Liquidate** — how many trading days a forced unwind would take per position.
- **L-VaR** — VaR scaled up by the exit horizon (√days) plus the bid-ask spread cost. **The gap between Base VaR and L-VaR is the price of illiquidity.**
- **Try it**: drag the portfolio value slider up to $500M and watch the liquidation days and L-VaR explode — this is why size is the enemy of returns.
""")
    section("Liquidity-Adjusted VaR",
            "How the cost and time to exit positions inflates headline risk under a participation cap.")
    cset = st.columns([1, 1, 2])
    pv = cset[0].slider("Portfolio value ($)", 1_000_000, 500_000_000, 10_000_000,
                        step=1_000_000, format="$%d")
    part = cset[1].slider("Daily participation rate", 0.01, 0.50, 0.20, step=0.01)

    per_asset, summary = liquidity_metrics(portfolio_value=pv, participation=part)
    items = list(summary.items())
    cols = st.columns(len(items))
    for col, (k, v) in zip(cols, items):
        if "days" in k:
            kpi(col, k, f"{v:.1f}", "worst name", WARN)
        else:
            tone = NEG if "Adjusted" in k else ACCENT
            kpi(col, k, f"${v:,.0f}", None, tone)

    section("Per-Asset Breakdown")
    st.dataframe(per_asset.style.format({
        "Position ($)": "${:,.0f}", "Base VaR (%)": "{:.2f}",
        "Days to Liquidate": "{:.1f}", "L-VaR (%)": "{:.2f}",
        "Spread Cost ($)": "${:,.0f}", "L-VaR ($)": "${:,.0f}",
    }).background_gradient(cmap="Blues", subset=["Days to Liquidate"]),
                 width="stretch")


# =====================================================================
# 9 · Credit (Merton)
# =====================================================================
def render_credit():
    from src.credit_merton import credit_metrics, scenario_pd
    explain("""
**"How likely is each company to go bust?"** — the Merton model (the foundation of Moody's KMV, used by banks worldwide) treats a company's equity as an option on its assets: default happens if asset value falls below the debt owed.

- **DD (Distance-to-Default)** — how many standard deviations the company's asset value sits *above* its debt. Bigger = safer. Healthy large-caps sit at 5–15; below ~2 is a warning zone.
- **PD** — that distance translated into a probability of default within one year.
- **Scenario PDs** — the same calculation re-run with crisis-level volatility. Note how JPMorgan's PD jumps in the 2008 scenario (a leveraged bank) while Apple/Microsoft barely move (low debt) — **leverage is what turns market stress into default risk**.
- Index tickers (^GSPC, ^IXIC) have no balance sheet, so only the three companies appear.
""")
    section("Merton Structural Credit Risk",
            "Probability of default and distance-to-default implied from equity markets.")
    cur = credit_metrics()

    riskiest = cur["PD (%)"].idxmax()
    c1, c2, c3 = st.columns(3)
    kpi(c1, "Highest PD", riskiest, f"{cur.loc[riskiest,'PD (%)']:.2f}%", NEG)
    kpi(c2, "Lowest DD", f"{cur['DD'].min():.2f}", cur['DD'].idxmin(), WARN)
    kpi(c3, "Names Covered", f"{len(cur)}", "equity issuers", ACCENT_2)

    section("Current PD & Distance-to-Default")
    st.dataframe(cur.style.format({"σ_E (% annual)": "{:.2f}", "DD": "{:.3f}", "PD (%)": "{:.4f}"})
                 .background_gradient(cmap="Reds", subset=["PD (%)"]),
                 width="stretch")

    section("Scenario-Adjusted PDs", "Default probabilities re-estimated with crisis-period volatility.")
    for scen in ["2008 GFC (Sep–Dec 2008)", "2020 COVID Crash (Feb–Apr 2020)"]:
        st.markdown(f"**{scen}**")
        df = scenario_pd(scen)
        st.dataframe(df.reset_index().style.format({"PD (%)": "{:.4f}", "DD": "{:.3f}"})
                     .background_gradient(cmap="Reds", subset=["PD (%)"]),
                     width="stretch", hide_index=True)


# =====================================================================
# 10 · Copula Dependence
# =====================================================================
def render_copula():
    from src.copula import compare_periods, correlation_shift_table, plot_tail_dependence_comparison
    explain("""
**"Will my diversification survive a crash?"** — assets that look independent in calm markets tend to crash *together*. This tab measures exactly that.

- **Avg ρ (correlation)** — compare the Calm 2019 row vs the Crisis 2020 row: average correlation jumped from ~0.68 to ~0.90. **When you needed diversification most, it vanished.**
- **Copulas** — a statistical lens on *how* assets co-move. The Gaussian copula (used before 2008, infamously) assumes extreme moves are unrelated; the Student-t copula allows joint crashes.
- **Tail dependence (λ)** — the probability that one asset has an extreme crash *given* another one just did. Gaussian says ~0; the t-copula measures it directly. The bar chart shows it rising in crisis for every pair.
- **Takeaway for the portfolio**: position limits and stress tests should assume crisis correlations, not calm ones.
""")
    section("Copula Dependence & Tail Risk",
            "Whether diversification holds in a crisis — correlation and tail dependence shift from calm to stress.")
    rets = load_returns()
    calm = ("2019-01-01", "2019-12-31")
    crisis = ("2020-02-15", "2020-04-30")
    comp = compare_periods(rets, crisis[0], crisis[1], calm[0], calm[1])
    shift = correlation_shift_table(comp)
    # 'ν (DoF)' mixes the '∞' symbol with floats (object dtype) — cast to string
    # so Arrow can serialise the table.
    if "ν (DoF)" in shift.columns:
        shift = shift.copy()
        shift["ν (DoF)"] = shift["ν (DoF)"].astype(str)

    st.dataframe(shift.style.background_gradient(cmap="PuBu"),
                 width="stretch", hide_index=True)
    section("Tail Dependence — Calm vs Crisis")
    fig = plot_tail_dependence_comparison(rets, crisis[0], crisis[1], calm[0], calm[1])
    styled_fig(fig)


# =====================================================================
# 11 · Stress Testing
# =====================================================================
def render_stress():
    from src.stress_testing import run_stress_tests, plot_drawdowns, impact_table
    explain("""
**"What would 2008 do to this portfolio, today?"** — instead of predicting the future, we replay the two worst market episodes of the modern era through the current $1M book, day by day.

- **Total P&L** — the dollar loss over the full episode. **Max Drawdown** — the deepest point along the way (often worse than the end-to-end number).
- **Worst single days** — 2008's worst day cost ~13% in one session; March 2020 hit ~14%. One day.
- **Why two scenarios?** 2008 was a slow 4-month grind lower (drip-feed losses, nowhere to hide); 2020 was a violent 5-week crash followed by a fast rebound. A portfolio can survive one shape and not the other.
- Use this next to the **VaR tab**: VaR describes a *normal* bad day — this tab shows what the abnormal ones look like.
""")
    section("Historical Stress Testing",
            "Replaying the 2008 GFC and 2020 COVID crash through the current book.")
    results = run_stress_tests(portfolio_value=1_000_000)
    tbl = impact_table(results)

    st.dataframe(tbl.style.format({"Total P&L ($)": "${:,.0f}", "Total P&L (%)": "{:.2f}%"}),
                 width="stretch")
    section("P&L & Drawdown Paths")
    for f in plot_drawdowns(results):
        styled_fig(f)


# =====================================================================
# Main
# =====================================================================

def main():
    st.set_page_config(page_title="Risk Analytics", page_icon="📈",
                       layout="wide", initial_sidebar_state="expanded")
    inject_css()
    configure_charts()

    rets = load_returns()
    as_of = rets.index[-1].strftime("%d %b %Y")
    universe = ", ".join(rets.columns)

    tabs = {
        "🏠  Overview": render_overview,
        "📉  VaR / CVaR": render_var,
        "🎯  VaR Backtesting": render_backtest,
        "🌊  Volatility (GARCH)": render_volatility,
        "🧬  Factor Model": render_factors,
        "⚖️  Optimisation": render_optimizer,
        "📊  Performance": render_performance,
        "💧  Liquidity": render_liquidity,
        "🏦  Credit (Merton)": render_credit,
        "🔗  Copula Dependence": render_copula,
        "⚡  Stress Testing": render_stress,
    }

    with st.sidebar:
        st.markdown('<div class="side-brand">Risk<span>Analytics</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="side-tag">Institutional risk desk</div>', unsafe_allow_html=True)
        st.markdown("")
        choice = st.radio("Navigate", list(tabs.keys()), label_visibility="collapsed")
        st.markdown(f"""<div class="side-foot">
            <b>Universe</b><br>{universe}<br><br>
            <b>As of</b> {as_of}<br>
            <b>Base value</b> $1,000,000<br><br>
            Educational &amp; research use only.
        </div>""", unsafe_allow_html=True)

    hero(as_of, universe)
    tabs[choice]()


if __name__ == "__main__":
    main()
