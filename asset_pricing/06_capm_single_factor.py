"""
06. CAPM - Capital Asset Pricing Model (Single Factor)
========================================================
Goal: Understand the simplest asset pricing model.

Theory:
  E[R_i] - R_f = beta_i * (E[R_m] - R_f)

  Where:
    R_i     = return of asset i
    R_f     = risk-free rate
    R_m     = market return
    beta_i  = sensitivity of asset i to the market
    E[R_m] - R_f = market risk premium

  In regression form:
    (R_i - R_f) = alpha + beta * (R_m - R_f) + epsilon

  Key Predictions:
    1. alpha = 0       (no free lunch)
    2. beta explains all cross-sectional return variation
    3. Higher beta -> higher expected return (linear relationship)

What Students Should Learn:
  - How to estimate beta from data
  - What alpha means (abnormal return / mispricing)
  - Security Market Line (SML)
  - CAPM is a ONE-factor model: only market risk is priced
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# ============================================================
# 1. Download Data
# ============================================================
print("=" * 60)
print("CAPM: Capital Asset Pricing Model")
print("=" * 60)

# Download a mix of stocks: tech, defensive, high-beta, low-beta
tickers = ["AAPL", "MSFT", "NVDA", "JNJ", "PG", "KO", "TSLA", "XOM", "JPM", "META"]
market = "SPY"  # Market proxy
rf_ticker = "^IRX"  # 13-week Treasury Bill (annualized)

print(f"\nDownloading: {tickers} + {market}")
start, end = "2019-01-01", "2025-12-31"

# Download all at once
all_tickers = tickers + [market]
prices = yf.download(all_tickers, start=start, end=end, auto_adjust=True)["Close"]
prices.columns = prices.columns.droplevel(1) if isinstance(prices.columns, pd.MultiIndex) else prices.columns

# Monthly returns (more standard for CAPM)
monthly_prices = prices.resample("ME").last()
monthly_returns = monthly_prices.pct_change().dropna()

# Risk-free rate: use constant approximation (Treasury ~4% annual -> ~0.33% monthly)
# For simplicity; real research would use actual T-bill rates
RF_MONTHLY = 0.04 / 12  # ~0.33% per month

# Excess returns
excess_returns = monthly_returns.sub(RF_MONTHLY)
market_excess = excess_returns[market]
stock_excess = excess_returns[tickers]

print(f"Period: {monthly_returns.index[0].strftime('%Y-%m')} to {monthly_returns.index[-1].strftime('%Y-%m')}")
print(f"Months: {len(monthly_returns)}")
print(f"Risk-free rate: {RF_MONTHLY*100:.2f}% monthly ({RF_MONTHLY*12*100:.1f}% annual)")

# ============================================================
# 2. Estimate CAPM for Each Stock
# ============================================================
print("\n" + "=" * 60)
print("Step 2: CAPM Regression for Each Stock")
print("=" * 60)
print(f"\n{'Ticker':<8} {'Alpha':>8} {'Beta':>8} {'R²':>8} {'p(alpha)':>10} {'p(beta)':>10}")
print("-" * 55)

capm_results = {}
for ticker in tickers:
    y = stock_excess[ticker].values
    X = sm.add_constant(market_excess.values)  # Add intercept (alpha)
    model = sm.OLS(y, X).fit()

    alpha = model.params[0]
    beta = model.params[1]
    r2 = model.rsquared
    p_alpha = model.pvalues[0]
    p_beta = model.pvalues[1]

    capm_results[ticker] = {
        "alpha": alpha, "beta": beta, "r2": r2,
        "p_alpha": p_alpha, "p_beta": p_beta,
        "alpha_annual": alpha * 12,  # Annualize monthly alpha
        "avg_excess": y.mean() * 12,  # Annualized avg excess return
    }

    sig_alpha = "*" if p_alpha < 0.05 else " "
    print(f"{ticker:<8} {alpha:>8.4f}{sig_alpha} {beta:>7.2f}  {r2:>7.2f}  {p_alpha:>9.4f}  {p_beta:>9.4f}")

print("\n* = alpha significantly different from 0 at 5% level")
print("If CAPM holds perfectly, all alphas should be zero.")

# ============================================================
# 3. Security Market Line (SML)
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Security Market Line")
print("=" * 60)

betas = [capm_results[t]["beta"] for t in tickers]
avg_excess = [capm_results[t]["avg_excess"] for t in tickers]

# Theoretical SML: E[R_excess] = beta * market_premium
market_premium = market_excess.mean() * 12  # Annualized
beta_range = np.linspace(0, 2.5, 100)
sml_line = beta_range * market_premium

print(f"Market Risk Premium (annualized): {market_premium*100:.2f}%")

# ============================================================
# 4. Plots
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("CAPM - Capital Asset Pricing Model", fontsize=16, fontweight="bold")

# (a) Security Market Line
ax = axes[0, 0]
ax.plot(beta_range, sml_line * 100, "r--", linewidth=2, label="SML (theoretical)")
for i, t in enumerate(tickers):
    ax.scatter(betas[i], avg_excess[i] * 100, s=80, zorder=5)
    ax.annotate(t, (betas[i], avg_excess[i] * 100), fontsize=8,
                xytext=(5, 5), textcoords="offset points")
ax.set_xlabel("Beta (β)")
ax.set_ylabel("Average Excess Return (% annual)")
ax.set_title("Security Market Line (SML)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(0, color="gray", linewidth=0.5)

# (b) Beta bar chart
ax = axes[0, 1]
colors = ["#ef4444" if b > 1.2 else "#3b82f6" if b > 0.8 else "#10b981" for b in betas]
bars = ax.barh(tickers, betas, color=colors, edgecolor="black", alpha=0.8)
ax.axvline(1, color="red", linestyle="--", linewidth=1, label="β=1 (market)")
ax.set_xlabel("Beta")
ax.set_title("Stock Betas")
ax.legend()
ax.grid(True, alpha=0.3, axis="x")

# (c) Alpha bar chart (annualized)
ax = axes[0, 2]
alphas_annual = [capm_results[t]["alpha_annual"] * 100 for t in tickers]
colors = ["#10b981" if a > 0 else "#ef4444" for a in alphas_annual]
ax.barh(tickers, alphas_annual, color=colors, edgecolor="black", alpha=0.8)
ax.axvline(0, color="black", linewidth=1)
ax.set_xlabel("Alpha (% annual)")
ax.set_title("CAPM Alpha (abnormal return)")
ax.grid(True, alpha=0.3, axis="x")

# (d-f) Individual CAPM regressions for 3 stocks
for idx, ticker in enumerate(["NVDA", "KO", "SPY"]):
    ax = axes[1, idx]
    if ticker == "SPY":
        # Show R² comparison
        r2s = [capm_results[t]["r2"] for t in tickers]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(tickers)))
        ax.barh(tickers, r2s, color=colors, edgecolor="black", alpha=0.8)
        ax.set_xlabel("R²")
        ax.set_title("CAPM R² (% variance explained by market)")
        ax.grid(True, alpha=0.3, axis="x")
    else:
        y = stock_excess[ticker].values
        x = market_excess.values
        ax.scatter(x * 100, y * 100, alpha=0.5, s=20, color="#3b82f6")

        # Regression line
        b = capm_results[ticker]["beta"]
        a = capm_results[ticker]["alpha"]
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line * 100, (a + b * x_line) * 100, "r-", linewidth=2,
                label=f"β={b:.2f}, α={a*12*100:.1f}%/yr")
        ax.set_xlabel("Market Excess Return (%)")
        ax.set_ylabel(f"{ticker} Excess Return (%)")
        ax.set_title(f"CAPM: {ticker}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)

plt.tight_layout()
plt.savefig("/mnt/nas/Class/2026/deeplearning/06_capm_single_factor.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: 06_capm_single_factor.png")

# ============================================================
# 5. Summary
# ============================================================
print("\n" + "=" * 60)
print("CAPM SUMMARY")
print("=" * 60)
print(f"""
CAPM Equation: E[Ri] - Rf = βi × (E[Rm] - Rf)

Results for {start} to {end}:
  Market Risk Premium: {market_premium*100:.2f}% per year
  Risk-Free Rate:      {RF_MONTHLY*12*100:.1f}% per year

Interpretation:
  β > 1  → More volatile than market (NVDA, TSLA, META)
  β = 1  → Same as market
  β < 1  → Less volatile (JNJ, PG, KO = defensive stocks)

  α > 0  → Stock outperformed CAPM prediction (positive abnormal return)
  α < 0  → Stock underperformed CAPM prediction
  α = 0  → CAPM explains the stock's return perfectly

Limitations of CAPM:
  1. Only one factor (market) - ignores size, value, momentum
  2. Assumes beta is constant over time
  3. Empirical evidence shows alpha ≠ 0 for many stocks
  4. Low-beta stocks often earn more than CAPM predicts (low-beta anomaly)

  → This motivates multi-factor models (Fama-French 3-factor, next file!)
""")
