"""
10. Hierarchical Risk Parity (HRP) - de Prado (2016)
=======================================================
Goal: Build a robust portfolio using hierarchical clustering instead of
      mean-variance optimization.

Reference:
  López de Prado, M. (2016). "Building Diversified Portfolios that
  Outperform Out-of-Sample." Journal of Portfolio Management.

Why HRP?
  Markowitz optimization has serious practical problems:
    1. Inverts the covariance matrix → amplifies estimation errors
    2. Produces concentrated, unstable portfolios
    3. Small changes in inputs → large changes in weights

  HRP solves these by:
    1. NO matrix inversion needed
    2. Uses hierarchical clustering to group similar assets
    3. Allocates risk top-down through the hierarchy
    4. Produces stable, diversified portfolios

Algorithm (3 Steps):
  Step 1: TREE CLUSTERING
    - Compute distance matrix from correlation: d(i,j) = sqrt(0.5*(1-ρ_ij))
    - Apply hierarchical clustering (single/ward linkage)
    - Result: dendrogram showing asset groupings

  Step 2: QUASI-DIAGONALIZATION
    - Reorder the covariance matrix so that similar assets are adjacent
    - Follow the dendrogram leaf order
    - Result: block-diagonal-like covariance matrix

  Step 3: RECURSIVE BISECTION
    - Split assets into two clusters at each level
    - Allocate weight inversely proportional to cluster variance
    - Recursively bisect until individual assets reached
    - Result: portfolio weights

What Students Should Learn:
  - Why covariance matrix inversion is dangerous
  - How hierarchical clustering groups assets
  - The elegant recursive bisection algorithm
  - Comparison with Markowitz and equal-weight
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform
from scipy.optimize import minimize

# ============================================================
# 1. Download Data
# ============================================================
print("=" * 60)
print("HRP: Hierarchical Risk Parity (de Prado 2016)")
print("=" * 60)

tickers = ["AAPL", "MSFT", "NVDA", "JNJ", "PG", "XOM", "JPM", "GLD", "TLT", "VNQ"]

prices = yf.download(tickers, start="2019-01-01", end="2025-12-31", auto_adjust=True)["Close"]
if isinstance(prices.columns, pd.MultiIndex):
    prices.columns = prices.columns.droplevel(1)

monthly = prices.resample("ME").last().pct_change().dropna()
N = len(tickers)

# Annualized stats
mu = monthly.mean().values * 12
cov = monthly.cov().values * 12
corr = monthly.corr().values

print(f"Assets: {tickers}")
print(f"Months: {len(monthly)}")

# ============================================================
# 2. Step 1 - Tree Clustering
# ============================================================
print("\n" + "=" * 60)
print("HRP Step 1: Tree Clustering")
print("=" * 60)

# Distance matrix from correlations
# d(i,j) = sqrt(0.5 * (1 - rho_ij))
# This transforms correlation into a proper distance metric
dist = np.sqrt(0.5 * (1 - corr))
np.fill_diagonal(dist, 0)

# Convert to condensed form for scipy
dist_condensed = squareform(dist)

# Hierarchical clustering (Ward's method)
link = linkage(dist_condensed, method="ward")

print("Distance matrix d(i,j) = sqrt(0.5 * (1 - ρ_ij)):")
print("  ρ = 1.0  →  d = 0.00  (identical)")
print("  ρ = 0.0  →  d = 0.71  (uncorrelated)")
print("  ρ = -1.0 →  d = 1.00  (opposite)")

# ============================================================
# 3. Step 2 - Quasi-Diagonalization
# ============================================================
print("\n" + "=" * 60)
print("HRP Step 2: Quasi-Diagonalization")
print("=" * 60)

# Get the order from dendrogram leaves
sort_ix = list(leaves_list(link))
sorted_tickers = [tickers[i] for i in sort_ix]
print(f"Clustered order: {sorted_tickers}")

# Reorder covariance matrix
cov_sorted = cov[np.ix_(sort_ix, sort_ix)]
corr_sorted = corr[np.ix_(sort_ix, sort_ix)]

print("Covariance matrix reordered so similar assets are adjacent")
print("→ Creates a block-diagonal-like structure")

# ============================================================
# 4. Step 3 - Recursive Bisection (HRP Core Algorithm)
# ============================================================
print("\n" + "=" * 60)
print("HRP Step 3: Recursive Bisection")
print("=" * 60)


def get_cluster_var(cov, cluster_items):
    """Compute variance of the minimum-variance portfolio within a cluster."""
    cov_slice = cov[np.ix_(cluster_items, cluster_items)]
    # Inverse-variance weights (simple approximation)
    ivp = 1.0 / np.diag(cov_slice)
    ivp /= ivp.sum()
    return ivp @ cov_slice @ ivp


def hrp_allocation(cov, sort_ix):
    """
    Recursive bisection to allocate weights.

    Algorithm:
      1. Start with all assets in one cluster, weight = 1.0
      2. Split into two sub-clusters (left, right)
      3. Compute variance of each sub-cluster
      4. Allocate weight inversely proportional to variance:
         α = 1 - V_left / (V_left + V_right)
         w_left *= α, w_right *= (1 - α)
      5. Recurse on each sub-cluster until singleton
    """
    weights = np.ones(len(sort_ix))
    cluster_items = [sort_ix]  # Start with all items

    while len(cluster_items) > 0:
        # Bisect each cluster
        new_clusters = []
        for cluster in cluster_items:
            if len(cluster) <= 1:
                continue

            # Split in half
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]

            # Cluster variances
            var_left = get_cluster_var(cov, left)
            var_right = get_cluster_var(cov, right)

            # Allocation factor (more weight to lower variance cluster)
            alpha = 1 - var_left / (var_left + var_right)

            # Update weights
            for i in left:
                weights[i] *= alpha
            for i in right:
                weights[i] *= (1 - alpha)

            # Add sub-clusters for further bisection
            if len(left) > 1:
                new_clusters.append(left)
            if len(right) > 1:
                new_clusters.append(right)

        cluster_items = new_clusters

    return weights


w_hrp = hrp_allocation(cov, sort_ix)
# Normalize
w_hrp /= w_hrp.sum()

ret_hrp = w_hrp @ mu
vol_hrp = np.sqrt(w_hrp @ cov @ w_hrp)
sharpe_hrp = (ret_hrp - 0.04) / vol_hrp

print(f"\nHRP Weights:")
for i, t in enumerate(tickers):
    print(f"  {t:<6} {w_hrp[i]*100:>6.1f}%")

print(f"\nHRP Portfolio: Return={ret_hrp*100:.1f}%, Vol={vol_hrp*100:.1f}%, Sharpe={sharpe_hrp:.2f}")

# ============================================================
# 5. Comparison: HRP vs Markowitz vs Equal Weight
# ============================================================
print("\n" + "=" * 60)
print("Step 5: Model Comparison")
print("=" * 60)

rf = 0.04

# Equal Weight
w_eq = np.ones(N) / N
ret_eq = w_eq @ mu
vol_eq = np.sqrt(w_eq @ cov @ w_eq)
sharpe_eq = (ret_eq - rf) / vol_eq

# Minimum Variance (Markowitz)
cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
bounds = tuple((0, 1) for _ in range(N))
res = minimize(lambda w: np.sqrt(w @ cov @ w), w_eq,
               method="SLSQP", bounds=bounds, constraints=cons)
w_mvp = res.x
ret_mvp = w_mvp @ mu
vol_mvp = np.sqrt(w_mvp @ cov @ w_mvp)
sharpe_mvp = (ret_mvp - rf) / vol_mvp

# Max Sharpe (Markowitz)
res = minimize(lambda w: -(w @ mu - rf) / np.sqrt(w @ cov @ w), w_eq,
               method="SLSQP", bounds=bounds, constraints=cons)
w_tan = res.x
ret_tan = w_tan @ mu
vol_tan = np.sqrt(w_tan @ cov @ w_tan)
sharpe_tan = (ret_tan - rf) / vol_tan

# Inverse Variance
w_iv = 1.0 / np.diag(cov)
w_iv /= w_iv.sum()
ret_iv = w_iv @ mu
vol_iv = np.sqrt(w_iv @ cov @ w_iv)
sharpe_iv = (ret_iv - rf) / vol_iv

print(f"\n{'Method':<22} {'Return':>8} {'Vol':>8} {'Sharpe':>8} {'Max Wt':>8} {'# Active':>8}")
print("-" * 60)
for name, w, r, v, s in [
    ("Equal Weight (1/N)", w_eq, ret_eq, vol_eq, sharpe_eq),
    ("Inverse Variance", w_iv, ret_iv, vol_iv, sharpe_iv),
    ("Min Variance (MVO)", w_mvp, ret_mvp, vol_mvp, sharpe_mvp),
    ("Max Sharpe (MVO)", w_tan, ret_tan, vol_tan, sharpe_tan),
    ("HRP (de Prado)", w_hrp, ret_hrp, vol_hrp, sharpe_hrp),
]:
    n_active = np.sum(w > 0.01)
    print(f"{name:<22} {r*100:>7.1f}% {v*100:>7.1f}% {s:>7.2f} {max(w)*100:>7.1f}% {n_active:>6}")

# ============================================================
# 6. Out-of-Sample Backtest (Rolling Window)
# ============================================================
print("\n" + "=" * 60)
print("Step 6: Rolling Backtest (12-month window)")
print("=" * 60)

window = 12  # 12-month estimation window
cumret = {"HRP": [1.0], "MVO_MVP": [1.0], "EqualWeight": [1.0], "InvVar": [1.0]}

for t in range(window, len(monthly)):
    est_data = monthly.iloc[t - window : t]
    cov_est = est_data.cov().values * 12
    corr_est = est_data.corr().values

    # Current month return
    ret_t = monthly.iloc[t].values

    # HRP
    dist_t = np.sqrt(0.5 * (1 - corr_est))
    np.fill_diagonal(dist_t, 0)
    link_t = linkage(squareform(dist_t), method="ward")
    sort_t = list(leaves_list(link_t))
    w_h = hrp_allocation(cov_est, sort_t)
    w_h /= w_h.sum()

    # MVP
    try:
        res_m = minimize(lambda w: np.sqrt(w @ cov_est @ w), w_eq,
                         method="SLSQP", bounds=bounds, constraints=cons)
        w_m = res_m.x
    except Exception:
        w_m = w_eq

    # Inverse Variance
    w_i = 1.0 / np.diag(cov_est)
    w_i /= w_i.sum()

    for name, w in [("HRP", w_h), ("MVO_MVP", w_m), ("EqualWeight", w_eq), ("InvVar", w_i)]:
        port_ret = w @ ret_t
        cumret[name].append(cumret[name][-1] * (1 + port_ret))

dates_bt = monthly.index[window - 1:]
print(f"Backtest period: {dates_bt[0]} to {dates_bt[-1]}")

for name in cumret:
    total_ret = cumret[name][-1] / cumret[name][0] - 1
    rets = np.diff(cumret[name]) / cumret[name][:-1]
    vol = np.std(rets) * np.sqrt(12)
    sr = (np.mean(rets) * 12 - rf) / vol if vol > 0 else 0
    print(f"  {name:<15} Total: {total_ret*100:>6.1f}%, Vol: {vol*100:>5.1f}%, Sharpe: {sr:.2f}")

# ============================================================
# 7. Comprehensive Plots
# ============================================================
fig, axes = plt.subplots(3, 2, figsize=(16, 16))
fig.suptitle("HRP: Hierarchical Risk Parity (de Prado 2016)", fontsize=14, fontweight="bold")

# (a) Dendrogram
ax = axes[0, 0]
dendrogram(link, labels=tickers, ax=ax, leaf_rotation=45, leaf_font_size=9,
           color_threshold=0.7 * max(link[:, 2]))
ax.set_title("Step 1: Hierarchical Clustering (Dendrogram)")
ax.set_ylabel("Distance")

# (b) Original vs Quasi-Diagonalized Covariance
ax = axes[0, 1]
# Show sorted correlation matrix
im = ax.imshow(corr_sorted, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(N))
ax.set_xticklabels(sorted_tickers, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(N))
ax.set_yticklabels(sorted_tickers, fontsize=8)
for i in range(N):
    for j in range(N):
        ax.text(j, i, f"{corr_sorted[i,j]:.1f}", ha="center", va="center", fontsize=7)
plt.colorbar(im, ax=ax)
ax.set_title("Step 2: Quasi-Diagonalized Correlation")

# (c) Weight comparison
ax = axes[1, 0]
x = np.arange(N)
width = 0.18
offsets = [-2, -1, 0, 1, 2]
names = ["EqualWt", "InvVar", "MVP", "MaxSharpe", "HRP"]
all_weights = [w_eq, w_iv, w_mvp, w_tan, w_hrp]
colors = ["#64748b", "#f59e0b", "#ef4444", "#3b82f6", "#10b981"]

for offset, name, w, color in zip(offsets, names, all_weights, colors):
    ax.bar(x + offset * width, w * 100, width, label=name, color=color, alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(tickers, fontsize=8)
ax.set_ylabel("Weight (%)")
ax.set_title("Step 3: Weight Comparison")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis="y")

# (d) HRP weight pie chart
ax = axes[1, 1]
sorted_pairs = sorted(zip(tickers, w_hrp), key=lambda x: -x[1])
labels_pie = [f"{t}\n{w*100:.1f}%" for t, w in sorted_pairs]
sizes = [w for _, w in sorted_pairs]
cmap = plt.cm.Set3(np.linspace(0, 1, N))
ax.pie(sizes, labels=labels_pie, colors=cmap, autopct="", startangle=90)
ax.set_title("HRP Allocation")

# (e) Backtest cumulative returns
ax = axes[2, 0]
bt_colors = {"HRP": "#10b981", "MVO_MVP": "#ef4444", "EqualWeight": "#64748b", "InvVar": "#f59e0b"}
for name, vals in cumret.items():
    ax.plot(dates_bt, vals, label=name, color=bt_colors[name], linewidth=1.5)
ax.set_title("Out-of-Sample Backtest (12mo rolling)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.tick_params(axis="x", rotation=45)
ax.set_ylabel("Cumulative Return ($1 invested)")

# (f) HRP algorithm summary
ax = axes[2, 1]
ax.axis("off")
summary = """
    HRP Algorithm (de Prado, 2016)
    ──────────────────────────────

    Step 1: Tree Clustering
    • Compute distance: d = √(½(1-ρ))
    • Hierarchical clustering → dendrogram
    • Groups similar assets together

    Step 2: Quasi-Diagonalization
    • Reorder covariance matrix by cluster
    • Similar assets become adjacent
    • Creates block-diagonal structure

    Step 3: Recursive Bisection
    • Split assets into 2 clusters
    • Allocate ∝ 1/cluster_variance
    • Recurse until individual assets

    Key Advantages over Markowitz:
    ✓ No covariance matrix inversion
    ✓ Stable weights (small input changes
      → small weight changes)
    ✓ Always fully diversified
    ✓ Works with singular covariance matrices
    ✓ Better out-of-sample performance
"""
ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
        va="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#f0fdf4", alpha=0.8))

plt.tight_layout()
plt.savefig("/mnt/nas/Class/2026/asset_pricing/10_hrp_hierarchical_risk_parity.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: 10_hrp_hierarchical_risk_parity.png")

# ============================================================
# 8. Summary
# ============================================================
print(f"""
{'='*60}
HRP SUMMARY
{'='*60}

Problem with Markowitz (Mean-Variance Optimization):
  • Inverts covariance matrix → amplifies estimation errors
  • Weights are highly sensitive to input changes
  • Often produces concentrated portfolios
  • Fails when N > T (more assets than observations)

HRP Solution (3 Steps):
  1. CLUSTER: Group similar assets via hierarchical clustering
  2. REORDER: Quasi-diagonalize the covariance matrix
  3. BISECT:  Allocate risk top-down through the tree

Mathematical Elegance:
  • No matrix inversion required
  • Uses ONLY variances and correlations (not inverse)
  • Weight allocation: α = 1 - V_L/(V_L + V_R)
  • Recursive: splits large problem into many small ones

Practical Benefits:
  • Stable: small changes in correlation → small changes in weights
  • Robust: works even with singular covariance matrices
  • Diversified: always uses all assets (no zero-weight solutions)
  • Fast: O(N² log N) vs O(N³) for matrix inversion
  • Out-of-sample: typically beats MVO in real backtests

When to Use:
  • Large universes (100+ assets)
  • Short estimation windows
  • When stability matters more than optimality
  • As a benchmark for more complex methods

Reference:
  López de Prado, M. (2016). "Building Diversified Portfolios that
  Outperform Out-of-Sample." Journal of Portfolio Management, 42(4), 59-69.
""")
