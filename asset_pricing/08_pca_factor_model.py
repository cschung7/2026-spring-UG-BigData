"""
08. PCA Factor Model (Statistical Factor Model)
==================================================
Goal: Extract latent factors from stock returns using PCA.

Theory:
  Unlike CAPM/FF3 where factors have economic meaning (market, size, value),
  PCA extracts STATISTICAL factors from the covariance matrix of returns.

  R = F @ Λ' + ε

  Where:
    R = (T × N) matrix of returns for N stocks over T periods
    F = (T × K) matrix of K latent factors
    Λ = (N × K) factor loading matrix
    ε = idiosyncratic returns (noise)

  PCA Process:
    1. Standardize returns
    2. Compute covariance matrix (N × N)
    3. Eigendecomposition: Σ = V @ D @ V'
    4. Top K eigenvectors = factor loadings
    5. Project returns onto eigenvectors = factor realizations

  Key Insight:
    - PC1 ≈ market factor (highly correlated with MKT)
    - PC2 often ≈ size or sector rotation
    - PC3 often ≈ value/growth spread
    - But PCA factors are ROTATIONS, not directly interpretable

What Students Should Learn:
  - How to run PCA on a return matrix
  - Scree plot and explained variance
  - Comparing PCA factors to economic factors (FF3)
  - How many factors are "enough"?
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# ============================================================
# 1. Download a Broader Universe
# ============================================================
print("=" * 60)
print("PCA Factor Model")
print("=" * 60)

# Use more stocks to make PCA meaningful
tickers = [
    # Tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN",
    # Finance
    "JPM", "BAC", "GS", "MS",
    # Healthcare
    "JNJ", "PFE", "UNH", "ABBV",
    # Consumer
    "PG", "KO", "WMT", "MCD",
    # Energy
    "XOM", "CVX", "COP",
    # Industrial
    "CAT", "BA", "GE",
    # Other
    "TSLA", "DIS", "NFLX", "V", "MA", "HD",
]

print(f"Downloading {len(tickers)} stocks...")
prices = yf.download(tickers, start="2019-01-01", end="2025-12-31", auto_adjust=True)["Close"]
if isinstance(prices.columns, pd.MultiIndex):
    prices.columns = prices.columns.droplevel(1)

# Drop any stocks with missing data
prices = prices.dropna(axis=1, how="any")
valid_tickers = list(prices.columns)
print(f"Valid stocks after dropping NaN: {len(valid_tickers)}")

# Monthly returns
monthly = prices.resample("ME").last().pct_change().dropna()
print(f"Return matrix shape: {monthly.shape} (months × stocks)")

# ============================================================
# 2. Run PCA
# ============================================================
print("\n" + "=" * 60)
print("Step 2: Principal Component Analysis")
print("=" * 60)

# Standardize returns (zero mean, unit variance)
scaler = StandardScaler()
returns_std = scaler.fit_transform(monthly.values)

# Full PCA
pca = PCA()
factors = pca.fit_transform(returns_std)  # (T × N) factor scores
loadings = pca.components_                 # (N × N) each row = one PC's loadings
explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

print(f"\nExplained Variance Ratio (first 10 PCs):")
for i in range(min(10, len(explained))):
    bar = "█" * int(explained[i] * 100)
    print(f"  PC{i+1:2d}: {explained[i]*100:5.1f}%  {bar}  (cumul: {cumulative[i]*100:.1f}%)")

# How many PCs to keep? (Kaiser criterion: eigenvalue > 1, or 80% variance)
n_components_80 = np.searchsorted(cumulative, 0.80) + 1
n_components_90 = np.searchsorted(cumulative, 0.90) + 1
print(f"\nPCs needed for 80% variance: {n_components_80}")
print(f"PCs needed for 90% variance: {n_components_90}")

# ============================================================
# 3. Interpret PC1, PC2, PC3
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Factor Interpretation")
print("=" * 60)

# PC1 loadings: which stocks contribute most?
for pc_idx in range(3):
    print(f"\nPC{pc_idx+1} loadings (top 5 positive and negative):")
    loading_series = pd.Series(loadings[pc_idx], index=valid_tickers)
    sorted_loadings = loading_series.sort_values()
    print(f"  Most negative: {dict(sorted_loadings.head(5).round(3))}")
    print(f"  Most positive: {dict(sorted_loadings.tail(5).round(3))}")

# ============================================================
# 4. Compare PC1 with Market Return
# ============================================================
print("\n" + "=" * 60)
print("Step 4: PC1 vs Market Return Correlation")
print("=" * 60)

# Download SPY for comparison
spy = yf.download("SPY", start="2019-01-01", end="2025-12-31", auto_adjust=True)["Close"]
spy_monthly = spy.resample("ME").last().pct_change().dropna()

# Align dates
common = monthly.index.intersection(spy_monthly.index)
pc1 = pd.Series(factors[:len(common), 0], index=common)
spy_ret = spy_monthly.loc[common]

corr_pc1_market = np.corrcoef(pc1.values, spy_ret.values.flatten())[0, 1]
print(f"Correlation(PC1, SPY return): {corr_pc1_market:.3f}")
print(f"→ PC1 is {'highly' if abs(corr_pc1_market) > 0.8 else 'moderately'} correlated with market factor")

# ============================================================
# 5. PCA as Factor Model: Regression
# ============================================================
print("\n" + "=" * 60)
print("Step 5: PCA Factor Model Regression (K=3)")
print("=" * 60)

K = 3  # Use first 3 PCs
factor_matrix = factors[:, :K]  # (T, K)

print(f"\n{'Ticker':<8} {'α':>7} {'β_PC1':>7} {'β_PC2':>7} {'β_PC3':>7} {'R²':>7}")
print("-" * 45)

pca_results = {}
for i, ticker in enumerate(valid_tickers):
    y = returns_std[:, i]
    X = sm.add_constant(factor_matrix)
    model = sm.OLS(y, X).fit()
    pca_results[ticker] = {
        "alpha": model.params[0],
        "betas": model.params[1:],
        "r2": model.rsquared,
    }
    print(f"{ticker:<8} {model.params[0]:>7.3f} {model.params[1]:>7.3f} "
          f"{model.params[2]:>7.3f} {model.params[3]:>7.3f} {model.rsquared:>7.3f}")

avg_r2 = np.mean([pca_results[t]["r2"] for t in valid_tickers])
print(f"\nAverage R² with 3 PCs: {avg_r2:.3f}")

# ============================================================
# 6. Plots
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("PCA Factor Model - Statistical Factors from Stock Returns", fontsize=14, fontweight="bold")

# (a) Scree plot
ax = axes[0, 0]
ax.bar(range(1, 11), explained[:10] * 100, color="#8b5cf6", alpha=0.7, edgecolor="black", label="Individual")
ax.plot(range(1, 11), cumulative[:10] * 100, "ro-", linewidth=2, label="Cumulative")
ax.axhline(80, color="gray", linestyle="--", alpha=0.5, label="80% threshold")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Explained Variance (%)")
ax.set_title("Scree Plot")
ax.legend()
ax.grid(True, alpha=0.3)

# (b) PC1 vs Market
ax = axes[0, 1]
# Flip PC1 sign if negatively correlated (PCA sign is arbitrary)
pc1_plot = -pc1 if corr_pc1_market < 0 else pc1
pc1_norm = (pc1_plot - pc1_plot.mean()) / pc1_plot.std()
spy_norm = (spy_ret - spy_ret.mean()) / spy_ret.std()
ax.plot(common, spy_norm.values.flatten(), label="SPY (standardized)", color="#ef4444", linewidth=1.5)
ax.plot(common, pc1_norm.values, label="PC1 (standardized)", color="#8b5cf6", linewidth=1.5, alpha=0.8)
ax.set_title(f"PC1 vs Market (ρ = {abs(corr_pc1_market):.2f})")
ax.legend()
ax.grid(True, alpha=0.3)
ax.tick_params(axis="x", rotation=45)

# (c) PC1 loadings bar chart
ax = axes[0, 2]
loading_series = pd.Series(loadings[0], index=valid_tickers).sort_values()
colors = ["#8b5cf6" if v > 0 else "#ef4444" for v in loading_series.values]
ax.barh(range(len(loading_series)), loading_series.values, color=colors, alpha=0.7)
ax.set_yticks(range(len(loading_series)))
ax.set_yticklabels(loading_series.index, fontsize=7)
ax.set_xlabel("Loading")
ax.set_title("PC1 Loadings (≈ Market Factor)")
ax.grid(True, alpha=0.3, axis="x")

# (d) PC2 vs PC1 scatter (stock map)
ax = axes[1, 0]
sectors = {
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "GOOGL": "Tech", "META": "Tech", "AMZN": "Tech",
    "JPM": "Finance", "BAC": "Finance", "GS": "Finance", "MS": "Finance",
    "JNJ": "Health", "PFE": "Health", "UNH": "Health", "ABBV": "Health",
    "PG": "Consumer", "KO": "Consumer", "WMT": "Consumer", "MCD": "Consumer",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "CAT": "Industrial", "BA": "Industrial", "GE": "Industrial",
    "TSLA": "Tech", "DIS": "Consumer", "NFLX": "Tech", "V": "Finance", "MA": "Finance", "HD": "Consumer",
}
sector_colors = {"Tech": "#3b82f6", "Finance": "#10b981", "Health": "#ef4444",
                 "Consumer": "#f59e0b", "Energy": "#8b5cf6", "Industrial": "#64748b"}
for ticker in valid_tickers:
    sector = sectors.get(ticker, "Other")
    color = sector_colors.get(sector, "gray")
    ax.scatter(loadings[0, valid_tickers.index(ticker)],
               loadings[1, valid_tickers.index(ticker)],
               color=color, s=60, zorder=5)
    ax.annotate(ticker, (loadings[0, valid_tickers.index(ticker)],
                         loadings[1, valid_tickers.index(ticker)]),
                fontsize=7, xytext=(3, 3), textcoords="offset points")
# Legend
for sector, color in sector_colors.items():
    ax.scatter([], [], color=color, label=sector, s=60)
ax.legend(fontsize=8, loc="best")
ax.set_xlabel("PC1 Loading")
ax.set_ylabel("PC2 Loading")
ax.set_title("Stock Map: PC1 vs PC2")
ax.grid(True, alpha=0.3)

# (e) Factor time series
ax = axes[1, 1]
for i, (name, color) in enumerate(zip(["PC1", "PC2", "PC3"], ["#3b82f6", "#10b981", "#f59e0b"])):
    ax.plot(common[:len(factors)], factors[:len(common), i], label=name, color=color, linewidth=1, alpha=0.8)
ax.set_title("PCA Factor Time Series")
ax.legend()
ax.grid(True, alpha=0.3)
ax.tick_params(axis="x", rotation=45)

# (f) R² distribution
ax = axes[1, 2]
r2_vals = [pca_results[t]["r2"] for t in valid_tickers]
ax.hist(r2_vals, bins=15, color="#8b5cf6", alpha=0.7, edgecolor="black")
ax.axvline(avg_r2, color="red", linestyle="--", linewidth=2, label=f"Mean R²={avg_r2:.2f}")
ax.set_xlabel("R²")
ax.set_ylabel("Count")
ax.set_title(f"R² Distribution (3 PCs, avg={avg_r2:.2f})")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/mnt/nas/Class/2026/asset_pricing/08_pca_factor_model.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: 08_pca_factor_model.png")

# ============================================================
# 7. Summary
# ============================================================
print(f"""
{'='*60}
PCA FACTOR MODEL SUMMARY
{'='*60}

What PCA does:
  Finds orthogonal directions of maximum variance in the return matrix.
  Each PC is a weighted portfolio of all stocks.

Results:
  PC1 explains {explained[0]*100:.1f}% of total return variance (≈ market factor)
  PC2 explains {explained[1]*100:.1f}% (often sector rotation or size)
  PC3 explains {explained[2]*100:.1f}% (often value/growth spread)
  First 3 PCs: {cumulative[2]*100:.1f}% total

PCA vs Economic Factors (FF3):
  ┌─────────┬──────────────────────┬──────────────────────┐
  │         │ Economic (FF3)       │ Statistical (PCA)    │
  ├─────────┼──────────────────────┼──────────────────────┤
  │ Factors │ MKT, SMB, HML       │ PC1, PC2, PC3, ...   │
  │ Source  │ Economic theory      │ Data-driven          │
  │ Meaning │ Directly interpretable│ Need rotation/mapping│
  │ # Fixed │ Yes (3 or 5)        │ Flexible (scree plot)│
  │ Pros    │ Intuitive, stable   │ Optimal variance fit │
  │ Cons    │ May miss patterns   │ Hard to interpret    │
  └─────────┴──────────────────────┴──────────────────────┘

When to use PCA:
  - Risk decomposition in large portfolios
  - Finding hidden common drivers in returns
  - Dimensionality reduction before ML models
  - When you don't know what the factors are
""")
