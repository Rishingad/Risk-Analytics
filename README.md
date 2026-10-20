# 🚀 Portfolio Risk Analytics Engine

A production-grade, modular Python risk engine covering **market risk**, **factor risk**, **credit risk**, **tail dependence**, **portfolio construction**, **performance**, **liquidity**, and **stress testing** — unified in an interactive Streamlit dashboard.

---

## 🎯 Use Cases

This is a **portfolio risk analytics platform** — the tool a risk manager, quant, or PM reaches for to answer *"how much can I lose, why, and what if?"* It serves three audiences:

**1. Quant / risk portfolio project.**
A broad, working demonstration of quantitative finance — VaR methodologies, GARCH, copulas, Merton credit, Fama-French factor modelling, portfolio optimisation, and liquidity risk — in clean modular code with a polished dashboard. Covers most of an FRM / quant-finance syllabus end-to-end.

**2. Practical risk monitoring for a real portfolio.**
Point it at any set of tickers (edit `ASSETS` in `src/data_loader.py`) and each tab answers a concrete question:

| Question | Tab |
|----------|-----|
| How much could I lose tomorrow? | **VaR / CVaR** |
| Can I trust those VaR numbers? | **VaR Backtesting** (Kupiec / Christoffersen) |
| How risky is the market *right now*? | **Volatility (GARCH)** |
| What's actually driving my risk? | **Factor Model** (e.g. "93 % a market bet") |
| Am I being paid for my risk? | **Performance** (Sharpe / Sortino / Calmar) |
| How should I weight the book? | **Optimisation** (max-Sharpe vs risk-parity) |
| Could I survive a crash? | **Stress-Testing** (2008 / 2020 replay) |
| Can I actually exit this size? | **Liquidity** (days-to-liquidate) |
| What's my default risk? | **Credit (Merton)** (PD / distance-to-default) |
| Does diversification hold in a crisis? | **Copula Dependence** (tail dependence) |

**3. An extensible foundation.**
Each risk domain is one importable module + one dashboard tab, making it a clean base to add new analytics (attribution, leverage modelling, etc.).

> ⚠️ **Scope:** built on daily Yahoo Finance data for a handful of large-caps using historical/parametric assumptions. Excellent for *understanding and monitoring* risk — **for educational and research use only**, not for trading real capital without institutional data and validation.

---

## 📁 Project Structure

```
Risk Analytics/
├── data/
│   ├── prices.csv               # Cached daily adjusted close prices
│   ├── ff_factors.csv           # Cached Fama-French 5 + Momentum factors
│   └── liquidity.csv            # Cached ADV / price for liquidity analysis
├── dashboard/
│   └── app.py                   # Streamlit interactive dashboard (11 sections)
├── notebooks/
│   ├── phase1_eda.ipynb          # Exploratory data analysis
│   └── phase2_garch.ipynb        # GARCH volatility modelling
├── report/                      # Auto-generated charts and plots
├── src/
│   ├── data_loader.py           # Price ingestion via yfinance
│   ├── returns.py               # Return computation utilities
│   ├── volatility.py            # GARCH family model fitting
│   ├── var_models.py            # VaR & CVaR (3 methods × 2 confidence levels)
│   ├── backtesting.py           # Kupiec POF + Christoffersen independence tests
│   ├── copula.py                # Gaussian & Student-t copula + tail dependence
│   ├── stress_testing.py        # 2008 GFC & 2020 COVID historical scenario replay
│   ├── credit_merton.py         # Merton structural model — PD & Distance-to-Default
│   ├── factor_model.py          # Fama-French 5 + Momentum factor risk decomposition
│   ├── optimizer.py             # Min-var / Max-Sharpe / Risk-parity / Min-CVaR construction
│   ├── performance.py           # Sharpe / Sortino / Calmar / drawdown tear sheet
│   └── liquidity.py             # Liquidity-adjusted VaR (days-to-liquidate + spread)
├── tests/
│   └── test_risk_engine.py      # Offline pytest suite (26 tests)
├── requirements.txt             # Pinned dependencies
└── README.md
```

---

## 🔧 Setup & Installation

### Prerequisites

- Python ≥ 3.11 (required by the pinned SciPy / NumPy versions)
- pip
- Internet access on first run (Yahoo Finance, Kenneth French data library)

### Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🏃 Quick Start

> Run commands from the **project root**. All modules resolve `data/` and `report/` relative to the project itself, so they write to the same place wherever you launch them from.

### 1. Download price data

```bash
python src/data_loader.py
# Saves data/prices.csv  (^GSPC, ^IXIC, AAPL, MSFT, JPM — 2015 → 2025)
```

Fama-French factors (`data/ff_factors.csv`) and liquidity data (`data/liquidity.csv`) are downloaded and cached automatically on first use.

### 2. Run individual modules

```bash
# VaR & CVaR summary table
python src/var_models.py

# Backtesting (Kupiec + Christoffersen)
python src/backtesting.py

# Copula analysis + tail dependence charts
python src/copula.py

# Stress test (2008 GFC, 2020 COVID)
python src/stress_testing.py

# Merton credit risk (PD + DD)
python src/credit_merton.py

# Fama-French factor exposures + risk decomposition
python src/factor_model.py

# Portfolio optimisation (min-var / max-Sharpe / risk-parity / min-CVaR)
python src/optimizer.py

# Performance tear sheet
python src/performance.py

# Liquidity-adjusted VaR
python src/liquidity.py
```

### 3. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

### 4. Run the tests

```bash
pytest -q
```

The suite runs offline against the cached CSVs in `data/` and synthetic series with known answers. It covers returns, VaR/CVaR, the Kupiec and Christoffersen tests, the optimisers, performance, copula tail dependence, GARCH, the factor decomposition and liquidity. `credit_merton` and `stress_testing` always fetch live data, so the suite leaves them out.

---

## 📦 Module Reference

### `src/data_loader.py`

Downloads daily adjusted prices for the portfolio universe using `yfinance`, with automatic retry logic and logging.

| Function | Description |
|----------|-------------|
| `download_ticker(ticker)` | Downloads a single ticker with retries |
| `run_pipeline()` | Downloads all tickers, aligns, drops NaNs, saves CSV |

**Portfolio universe:** `^GSPC  ^IXIC  AAPL  MSFT  JPM`

---

### `src/returns.py`

Utility functions for computing return series used by all downstream modules.

| Function | Description |
|----------|-------------|
| `load_prices(path)` | Loads `data/prices.csv` into a DatetimeIndex DataFrame |
| `log_returns(prices)` | ln(Pₜ / Pₜ₋₁) |
| `simple_returns(prices)` | (Pₜ − Pₜ₋₁) / Pₜ₋₁ |
| `portfolio_returns(returns, weights)` | Weighted sum of asset returns (default: equal-weight) |

---

### `src/volatility.py`

GARCH-family model fitting using the `arch` library.

| Function | Description |
|----------|-------------|
| `fit_garch(returns, vol, p, o, q, dist)` | Fits a single GARCH/GJR-GARCH/EGARCH model |
| `fit_all_models(returns)` | Fits all three variants simultaneously |
| `compare_models(results)` | AIC / BIC / log-likelihood comparison table |
| `best_model(results, criterion)` | Returns winner by AIC or BIC |
| `conditional_volatility(result)` | Extracts σₜ series |
| `one_step_forecast_variance(result, start)` | Out-of-sample variance forecasts |
| `forecast_rmse(forecast_var, realized_returns)` | RMSE & MAE vs squared returns |

**Models:** `GARCH(1,1)`  `GJR-GARCH(1,1)`  `EGARCH(1,1)` — all with skewed-t innovations.

---

### `src/var_models.py`

Three VaR methodologies at 95 % and 99 % confidence, plus CVaR (Expected Shortfall).

| Method | Assumption | Formula |
|--------|-----------|---------|
| **Parametric** | Normal returns | VaR = −(μ + z_α · σ) |
| **Historical** | Empirical distribution | VaR = −Q_α(returns) |
| **Monte Carlo** | Multivariate normal simulation | VaR = −percentile of simulated portfolio returns |

| Function | Description |
|----------|-------------|
| `parametric_var / cvar` | Closed-form normal VaR & ES |
| `historical_var / cvar` | Empirical quantile approach |
| `monte_carlo_var / cvar` | 100 k simulated paths (correlated assets) |
| `compute_all_var(port_ret, asset_ret)` | Runs all 6 combinations |
| `summary_table(results)` | Formatted comparison DataFrame |

**Sample output:**

```
     Method Confidence  VaR (%)  CVaR (%)
 Parametric        95%   2.1086    2.6619
 Historical        95%   2.0205    3.2126
Monte Carlo        95%   2.0897    2.6485
 Parametric        99%   3.0110    3.4598
 Historical        99%   3.8460    5.2529
Monte Carlo        99%   2.9905    3.4390
```

---

### `src/backtesting.py`

Validates VaR models using two standard regulatory tests.

| Test | Null Hypothesis | Rejection means |
|------|----------------|-----------------|
| **Kupiec POF** | Exceedance rate = (1 − confidence) | Model is mis-calibrated |
| **Christoffersen** | Exceedances are independent | Exceedances cluster (model misses regime changes) |

| Function | Description |
|----------|-------------|
| `_rolling_var(returns, method, confidence, window=252)` | Rolling in-sample VaR series |
| `exceedances(returns, var_series)` | Binary hit series |
| `kupiec_pof(hits, confidence)` | LR test statistic + p-value |
| `christoffersen_independence(hits)` | Transition-matrix LR test |
| `run_backtest(portfolio_returns)` | Full suite across all methods × confidence levels |
| `results_table(results)` | Formatted pass/fail DataFrame |

---

### `src/copula.py`

Models the joint dependence structure across assets beyond linear correlation.

| Copula | Tail Dependence | When to use |
|--------|----------------|-------------|
| **Gaussian** | λ = 0 (no tail dependence) | Calm markets |
| **Student-t** | λ > 0 (symmetric tail dependence) | Crisis periods, fat-tailed returns |

| Function | Description |
|----------|-------------|
| `fit_gaussian_copula(data)` | Rank-transform → normal → Pearson corr |
| `fit_t_copula(data)` | MLE over degrees of freedom ν |
| `tail_dependence_t(rho, nu)` | λ = 2·t_{ν+1}(−√((ν+1)(1−ρ)/(1+ρ))) |
| `compute_all_tail_dependence(t_cop)` | λ for all asset pairs |
| `compare_periods(data, crisis, calm)` | Fit both copulas on two sub-periods |
| `correlation_shift_table(comparison)` | Avg ρ change between regimes |
| `plot_tail_dependence_comparison(...)` | Bar chart saved to `report/` |

**Key insight:** During COVID-19 (2020), average portfolio correlation jumped from **0.68 → 0.90**, and the t-copula ν dropped from 4.12 → 9.68 — confirming concentration of tail risk during the crisis.

---

### `src/stress_testing.py`

Replays two historical crash scenarios through the portfolio.

| Scenario | Period | Total P&L | Max Drawdown |
|----------|--------|-----------|-------------|
| 2008 GFC | Sep – Dec 2008 | −$396,701 (−39.7 %) | −46.2 % |
| 2020 COVID | Feb – Apr 2020 | −$97,312 (−9.7 %) | −35.9 % |

| Function | Description |
|----------|-------------|
| `fetch_scenario_data(tickers, start, end)` | Downloads scenario-period prices |
| `portfolio_pnl(returns, weights, value)` | P&L, drawdown, CVaR, worst days |
| `run_stress_tests(...)` | Runs all scenarios, returns metrics dict |
| `impact_table(results)` | Cross-scenario comparison DataFrame |
| `plot_drawdowns(results, save_dir)` | Dual-panel P&L + drawdown charts |

---

### `src/credit_merton.py`

Merton structural model — estimates **Probability of Default (PD)** and **Distance-to-Default (DD)** from equity market data.

**Model:**

```
DD  = [ln(V/D) + (r − ½σ_V²)T] / (σ_V √T)
PD  = N(−DD)
```

Where V = asset value (implied from equity + debt), σ_V = asset volatility (derived from GARCH equity vol).

| Function | Description |
|----------|-------------|
| `_latest_balance_sheet(ticker)` | Pulls Total Debt, Stockholders Equity, Market Cap from yfinance |
| `_equity_volatility_series(ticker, returns)` | GARCH(1,1) conditional vol |
| `_merton_parameters(E, D, σ_E, T, r)` | Iterative solve for V, σ_V, DD, PD |
| `credit_metrics(tickers)` | Current PD & DD table for all equity tickers |
| `scenario_pd(scenario_name, tickers)` | PD re-estimated using crisis-period volatility |

> ⚠️ Note: Index tickers (`^GSPC`, `^IXIC`) have no balance-sheet data and are automatically skipped in credit calculations.

---

### `src/factor_model.py`

Multi-factor risk model — decomposes returns into **Fama-French 5 factors + Momentum** (`Mkt-RF`, `SMB`, `HML`, `RMW`, `CMA`, `Mom`) plus idiosyncratic risk. Factors are pulled free from the Kenneth R. French data library via `pandas_datareader` and cached to `data/ff_factors.csv`.

| Function | Description |
|----------|-------------|
| `load_factors(refresh)` | Load cached factors (downloads on first use) |
| `factor_exposures(asset_returns)` | Per-asset betas + annualised alpha + R² |
| `portfolio_exposures(returns, weights)` | Portfolio-level factor betas |
| `risk_decomposition(returns, weights)` | % of portfolio variance per factor vs idiosyncratic |

**Key insight:** the equal-weight book is ~93 % market-driven (Mkt-RF) with ~7 % idiosyncratic risk; JPM loads strongly on value (HML ≈ +1.16), AAPL/MSFT on growth (negative HML).

---

### `src/optimizer.py`

Portfolio construction — replaces the equal-weight assumption with optimised weights via `scipy.optimize` (no heavy solver dependency). Long-only and fully invested by default, with an optional per-name `max_weight` cap.

| Method | Objective |
|--------|-----------|
| **Min Variance** | Minimise `wᵀΣw` |
| **Max Sharpe** | Maximise `(wᵀμ − r_f) / √(wᵀΣw)` |
| **Risk Parity** | Equal risk contribution (convex log-barrier ERC) |
| **Min CVaR** | Rockafellar–Uryasev linear program (HiGHS) |

| Function | Description |
|----------|-------------|
| `compare_methods(returns)` | Weights + annualised return/vol/Sharpe for all methods |
| `efficient_frontier(returns)` | Long-only frontier for plotting |
| `portfolio_stats(weights, returns)` | Annualised return, vol, Sharpe for any weight vector |

---

### `src/performance.py`

Risk-adjusted performance tear sheet from a daily return series.

| Function | Description |
|----------|-------------|
| `performance_summary(returns)` | Headline table: return, vol, Sharpe, Sortino, Calmar, max-DD, hit rate, skew/kurtosis |
| `drawdown_series(returns)` | Drawdown path relative to running peak |
| `cumulative_returns(returns)` | Growth of $1 |
| `rolling_sharpe(returns, window)` | Trailing annualised Sharpe |

---

### `src/liquidity.py`

Liquidity-adjusted VaR — academic VaR assumes instant exit; this models the cost of actually unwinding.

```
days_to_liquidate = position_shares / (ADV × participation_rate)
L-VaR = VaR × √(days_to_liquidate) + half-spread × position_value
```

ADV and price are pulled from `yfinance` and cached to `data/liquidity.csv`.

| Function | Description |
|----------|-------------|
| `days_to_liquidate(pos_value, price, adv, participation)` | Trading days to unwind under a participation cap |
| `liquidity_metrics(weights, portfolio_value, ...)` | Per-asset & portfolio L-VaR, spread cost, liquidation horizon |

---

### `dashboard/app.py`

Unified interactive Streamlit dashboard with a **dark premium theme**.

| Tab | Content |
|-----|---------|
| **Overview** | Portfolio value, latest daily return, current P&L |
| **VaR / CVaR** | Parametric / Historical / Monte Carlo table (95 % & 99 %) |
| **VaR Backtesting** | Kupiec + Christoffersen pass/fail table, rolling-VaR exceedance plot |
| **Volatility (GARCH)** | GARCH / GJR / EGARCH comparison, conditional-vol chart, next-day vol + GARCH VaR |
| **Factor Model** | FF5 + Momentum betas, alpha/R², risk decomposition |
| **Optimisation** | Min-var / Max-Sharpe / Risk-parity / Min-CVaR weights + efficient frontier |
| **Performance** | Sharpe / Sortino / Calmar, growth curve + drawdown chart |
| **Liquidity** | Liquidity-adjusted VaR with interactive position-size / participation sliders |
| **Credit (Merton)** | Current PD & DD table + scenario-adjusted PDs |
| **Copula Dependence** | Correlation-shift table + tail-dependence bar chart |
| **Stress Testing** | P&L impact table + draw-down charts for GFC & COVID |

Launch:

```bash
streamlit run dashboard/app.py
```

---

## 🔄 End-to-End Data Flow

```
data_loader.py ──► data/prices.csv
                         │
                         ▼
                    returns.py  (log / simple returns, portfolio returns)
                         │
   ┌──────────┬──────────┼────────────┬────────────┬─────────────┬──────────────┐
   ▼          ▼          ▼            ▼            ▼             ▼              ▼
volatility  var_models  factor_model optimizer   performance   copula      credit_merton
(GARCH σₜ)  (VaR/CVaR)  (FF5 + Mom)  (weights)   (Sharpe, DD)  (tail dep.)  (PD, DD)
              │
              ├──► backtesting.py  (Kupiec / Christoffersen)
              └──► liquidity.py    (L-VaR, days-to-liquidate)

stress_testing.py  (fetches its own GFC / COVID window prices via yfinance)

                 all modules ──► dashboard/app.py  (Streamlit — 11 sections)
```

---

## 📊 Key Risk Numbers (2015–2025, Equal-Weight Portfolio)

| Metric | 95% Confidence | 99% Confidence |
|--------|---------------|---------------|
| Parametric VaR | 2.11 % | 3.01 % |
| Historical VaR | 2.02 % | 3.85 % |
| Monte Carlo VaR | 2.09 % | 2.99 % |
| Historical CVaR | 3.21 % | 5.25 % |

---

## 📝 License

For educational and research use only. Data sourced from Yahoo Finance via `yfinance`.
