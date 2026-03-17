"""
09. Markowitz Efficient Frontier & Portfolio Optimization
==========================================================
Goal: Construct the mean-variance efficient frontier and find optimal portfolios.

Theory (Markowitz 1952):
  Investors choose portfolios that maximize expected return for a given risk,
  or minimize risk for a given expected return.

  Portfolio return:    E[R_p] = w' @ μ
  Portfolio variance:  σ²_p   = w' @ Σ @ w

  Where:
    w = (N×1) vector of portfolio weights (sum to 1)
    μ = (N×1) vector of expected returns
    Σ = (N×N) covariance matrix of returns

  Key Portfolios:
    1. Minimum Variance Portfolio (MVP): min σ²_p s.t. w'1 = 1
    2. Tangency Portfolio (Max Sharpe): max (E[R_p] - R_f) / σ_p
    3. Efficient Frontier: set of all optimal portfolios

  Optimization:
    min  w' @ Σ @ w
    s.t. w' @ μ = target_return
         w' @ 1 = 1
         w >= 0  (optional: long-only constraint)

What Students Should Learn:
  - How diversification reduces risk
  - The shape of the efficient frontier (hyperbola)
  - Difference between MVP and tangency portfolio
  - Effect of constraints (long-only vs unconstrained)
  - Why sample estimates lead to unstable portfolios
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ============================================================
# 1. Download Data
# ============================================================
print("=" * 60)
print("Markowitz Efficient Frontier")
print("=" * 60)

tickers = ["AAPL", "MSFT", "NVDA", "JNJ", "PG", "XOM", "JPM", "GLD", "TLT", "VNQ"]
labels = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "JNJ": "J&J",
    "PG": "P&G", "XOM": "Exxon", "JPM": "JPMorgan", "GLD": "Gold ETF",
    "TLT": "Long Treasury", "VNQ": "Real Estate",
}

print(f"Universe: {tickers}")
print("Mix of: Tech, Defensive, Energy, Finance, Gold, Bonds, Real Estate")

prices = yf.download(tickers, start="2019-01-01", end="2025-12-31", auto_adjust=True)["Close"]
if isinstance(prices.columns, pd.MultiIndex):
    prices.columns = prices.columns.droplevel(1)

# Monthly returns
monthly = prices.resample("ME").last().pct_change().dropna()
N = len(tickers)

# Expected returns and covariance (annualized)
mu = monthly.mean().values * 12        # Annualized mean
sigma = monthly.cov().values * 12       # Annualized covariance
rf = 0.04                               # Risk-free rate (4%)

print(f"\nAnnualized Returns:")
for i, t in enumerate(tickers):
    print(f"  {t:<6} {labels[t]:<14} {mu[i]*100:>6.1f}%  (vol: {np.sqrt(sigma[i,i])*100:.1f}%)")

# ============================================================
# 2. Portfolio Optimization Functions
# ============================================================
def portfolio_return(w, mu):
    return w @ mu

def portfolio_volatility(w, sigma):
    return np.sqrt(w @ sigma @ w)

def portfolio_sharpe(w, mu, sigma, rf):
    ret = portfolio_return(w, mu)
    vol = portfolio_volatility(w, sigma)
    return (ret - rf) / vol

def neg_sharpe(w, mu, sigma, rf):
    return -portfolio_sharpe(w, mu, sigma, rf)

# Constraints
cons_sum1 = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
bounds_long = tuple((0, 1) for _ in range(N))   # Long-only
bounds_short = tuple((-0.3, 1) for _ in range(N))  # Allow short

# ============================================================
# 3. Find Key Portfolios
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Key Portfolios")
print("=" * 60)

w0 = np.ones(N) / N  # Equal weight starting point

# (a) Minimum Variance Portfolio
res_mvp = minimize(lambda w: portfolio_volatility(w, sigma), w0,
                   method="SLSQP", bounds=bounds_long, constraints=cons_sum1)
w_mvp = res_mvp.x
ret_mvp = portfolio_return(w_mvp, mu)
vol_mvp = portfolio_volatility(w_mvp, sigma)

# (b) Maximum Sharpe (Tangency) Portfolio
res_tan = minimize(neg_sharpe, w0, args=(mu, sigma, rf),
                   method="SLSQP", bounds=bounds_long, constraints=cons_sum1)
w_tan = res_tan.x
ret_tan = portfolio_return(w_tan, mu)
vol_tan = portfolio_volatility(w_tan, sigma)
sharpe_tan = portfolio_sharpe(w_tan, mu, sigma, rf)

# (c) Equal Weight Portfolio
w_eq = np.ones(N) / N
ret_eq = portfolio_return(w_eq, mu)
vol_eq = portfolio_volatility(w_eq, sigma)
sharpe_eq = portfolio_sharpe(w_eq, mu, sigma, rf)

print(f"\n{'Portfolio':<25} {'Return':>8} {'Vol':>8} {'Sharpe':>8}")
print("-" * 50)
print(f"{'Min Variance (MVP)':<25} {ret_mvp*100:>7.1f}% {vol_mvp*100:>7.1f}% {(ret_mvp-rf)/vol_mvp:>7.2f}")
print(f"{'Max Sharpe (Tangency)':<25} {ret_tan*100:>7.1f}% {vol_tan*100:>7.1f}% {sharpe_tan:>7.2f}")
print(f"{'Equal Weight (1/N)':<25} {ret_eq*100:>7.1f}% {vol_eq*100:>7.1f}% {sharpe_eq:>7.2f}")

# Print weights
print(f"\n{'Ticker':<8} {'MVP':>8} {'Tangency':>10} {'Equal':>8}")
print("-" * 36)
for i, t in enumerate(tickers):
    print(f"{t:<8} {w_mvp[i]*100:>7.1f}% {w_tan[i]*100:>9.1f}% {w_eq[i]*100:>7.1f}%")

# ============================================================
# 4. Generate Efficient Frontier
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Efficient Frontier")
print("=" * 60)

target_returns = np.linspace(mu.min(), mu.max(), 100)
frontier_vols = []
frontier_rets = []

for target in target_returns:
    cons = [
        cons_sum1,
        {"type": "eq", "fun": lambda w, t=target: portfolio_return(w, mu) - t},
    ]
    res = minimize(lambda w: portfolio_volatility(w, sigma), w0,
                   method="SLSQP", bounds=bounds_long, constraints=cons)
    if res.success:
        frontier_vols.append(portfolio_volatility(res.x, sigma))
        frontier_rets.append(target)

frontier_vols = np.array(frontier_vols)
frontier_rets = np.array(frontier_rets)

# Random portfolios for comparison (Monte Carlo)
n_random = 5000
rand_rets = np.zeros(n_random)
rand_vols = np.zeros(n_random)
rand_sharpes = np.zeros(n_random)

rng = np.random.default_rng(42)
for i in range(n_random):
    w = rng.dirichlet(np.ones(N))  # Random weights summing to 1
    rand_rets[i] = portfolio_return(w, mu)
    rand_vols[i] = portfolio_volatility(w, sigma)
    rand_sharpes[i] = (rand_rets[i] - rf) / rand_vols[i]

print(f"Generated {n_random} random portfolios")
print(f"Frontier points: {len(frontier_vols)}")

# Capital Market Line (CML)
cml_vols = np.linspace(0, max(frontier_vols) * 1.1, 100)
cml_rets = rf + sharpe_tan * cml_vols

# ============================================================
# 5. Plots
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Markowitz Efficient Frontier & Portfolio Optimization", fontsize=14, fontweight="bold")

# (a) Main Efficient Frontier
ax = axes[0, 0]
scatter = ax.scatter(rand_vols * 100, rand_rets * 100, c=rand_sharpes, cmap="viridis",
                     s=3, alpha=0.3, label="Random Portfolios")
plt.colorbar(scatter, ax=ax, label="Sharpe Ratio")
ax.plot(frontier_vols * 100, frontier_rets * 100, "r-", linewidth=2.5, label="Efficient Frontier")
ax.plot(cml_vols * 100, cml_rets * 100, "b--", linewidth=1.5, alpha=0.7, label="Capital Market Line")
ax.scatter(vol_mvp * 100, ret_mvp * 100, marker="D", s=150, color="green", zorder=10, label="Min Variance")
ax.scatter(vol_tan * 100, ret_tan * 100, marker="*", s=250, color="red", zorder=10, label="Tangency")
ax.scatter(vol_eq * 100, ret_eq * 100, marker="s", s=100, color="orange", zorder=10, label="Equal Weight")
# Individual stocks
for i, t in enumerate(tickers):
    ax.scatter(np.sqrt(sigma[i, i]) * 100, mu[i] * 100, marker="^", s=60, color="white",
               edgecolor="black", zorder=8)
    ax.annotate(t, (np.sqrt(sigma[i, i]) * 100, mu[i] * 100), fontsize=7,
                xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("Annualized Volatility (%)")
ax.set_ylabel("Annualized Return (%)")
ax.set_title("Efficient Frontier")
ax.legend(fontsize=7, loc="upper left")
ax.grid(True, alpha=0.3)

# (b) Portfolio weights: MVP
ax = axes[0, 1]
sorted_idx = np.argsort(w_mvp)[::-1]
ax.barh([tickers[i] for i in sorted_idx], [w_mvp[i] * 100 for i in sorted_idx],
        color="#10b981", alpha=0.8, edgecolor="black")
ax.set_xlabel("Weight (%)")
ax.set_title(f"Minimum Variance Portfolio\n(Vol={vol_mvp*100:.1f}%, Ret={ret_mvp*100:.1f}%)")
ax.grid(True, alpha=0.3, axis="x")

# (c) Portfolio weights: Tangency
ax = axes[0, 2]
sorted_idx = np.argsort(w_tan)[::-1]
ax.barh([tickers[i] for i in sorted_idx], [w_tan[i] * 100 for i in sorted_idx],
        color="#ef4444", alpha=0.8, edgecolor="black")
ax.set_xlabel("Weight (%)")
ax.set_title(f"Tangency Portfolio (Max Sharpe)\n(Sharpe={sharpe_tan:.2f}, Ret={ret_tan*100:.1f}%)")
ax.grid(True, alpha=0.3, axis="x")

# (d) Correlation matrix
ax = axes[1, 0]
corr = monthly.corr()
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(N))
ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(N))
ax.set_yticklabels(tickers, fontsize=8)
for i in range(N):
    for j in range(N):
        ax.text(j, i, f"{corr.values[i,j]:.1f}", ha="center", va="center", fontsize=6)
plt.colorbar(im, ax=ax)
ax.set_title("Correlation Matrix")

# (e) Risk contribution
ax = axes[1, 1]
# Risk contribution of tangency portfolio
w = w_tan
mrc = sigma @ w  # Marginal risk contribution
rc = w * mrc     # Risk contribution
rc_pct = rc / rc.sum() * 100
sorted_idx = np.argsort(rc_pct)[::-1]
ax.barh([tickers[i] for i in sorted_idx], [rc_pct[i] for i in sorted_idx],
        color="#8b5cf6", alpha=0.8, edgecolor="black")
ax.set_xlabel("Risk Contribution (%)")
ax.set_title("Risk Budget (Tangency Portfolio)")
ax.grid(True, alpha=0.3, axis="x")

# (f) Diversification effect
ax = axes[1, 2]
# Show how volatility decreases with number of assets
vols_by_n = []
for n in range(1, N + 1):
    # Pick first n assets, equal weight
    w_n = np.zeros(N)
    w_n[:n] = 1 / n
    vol_n = portfolio_volatility(w_n, sigma)
    vols_by_n.append(vol_n)

# Also show average individual vol
avg_ind_vol = np.mean([np.sqrt(sigma[i, i]) for i in range(N)])
ax.plot(range(1, N + 1), np.array(vols_by_n) * 100, "bo-", linewidth=2, label="Portfolio Vol")
ax.axhline(avg_ind_vol * 100, color="red", linestyle="--", label=f"Avg Stock Vol ({avg_ind_vol*100:.1f}%)")
ax.axhline(vol_mvp * 100, color="green", linestyle="--", label=f"MVP Vol ({vol_mvp*100:.1f}%)")
ax.set_xlabel("Number of Assets (equal weight)")
ax.set_ylabel("Portfolio Volatility (%)")
ax.set_title("Diversification Effect")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/mnt/nas/Class/2026/asset_pricing/09_efficient_frontier.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: 09_efficient_frontier.png")

# ============================================================
# 6. Summary
# ============================================================
print(f"""
{'='*60}
EFFICIENT FRONTIER SUMMARY
{'='*60}

Markowitz Mean-Variance Optimization:
  min  w'Σw           (minimize portfolio variance)
  s.t. w'μ = target   (achieve target return)
       w'1 = 1        (weights sum to 1)
       w ≥ 0          (long-only constraint)

Key Portfolios:
  MVP (Min Variance):   Lowest risk achievable via diversification
  Tangency (Max Sharpe): Best risk-adjusted return (on the CML)
  Equal Weight (1/N):    Simple benchmark, surprisingly hard to beat!

Capital Market Line (CML):
  All optimal portfolios are combinations of risk-free + tangency portfolio
  E[R_p] = R_f + Sharpe_tan × σ_p

Practical Issues:
  1. Sample estimates of μ and Σ are noisy → unstable weights
  2. Concentrated portfolios (tangency often puts 50%+ in one stock)
  3. Sensitive to outliers in return estimates
  4. Need regularization: shrinkage, constraints, resampling

  → This motivates robust methods like HRP (next file!)
""")
