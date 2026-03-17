"""
07. Fama-French 3-Factor Model
================================
Goal: Extend CAPM with Size (SMB) and Value (HML) factors.

Theory:
  E[R_i] - R_f = β_MKT*(R_m - R_f) + β_SMB*SMB + β_HML*HML + α

  Three Factors:
    1. MKT (Market):  R_m - R_f  (same as CAPM)
    2. SMB (Small Minus Big): Return of small-cap minus large-cap stocks
       - Small stocks historically earn higher returns (size premium)
    3. HML (High Minus Low): Return of high B/M minus low B/M stocks
       - Value stocks (high book-to-market) outperform growth stocks (value premium)

  Factor Data Source:
    Kenneth French's Data Library (https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/)
    We retrieve it using the pandas_datareader library.

What Students Should Learn:
  - How to retrieve Fama-French factors
  - Multi-factor regression interpretation
  - Why some stocks have negative SMB or HML loadings
  - Comparing CAPM vs FF3: does R² improve?
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ============================================================
# 1. Retrieve Fama-French Factor Data
# ============================================================
print("=" * 60)
print("Fama-French 3-Factor Model")
print("=" * 60)

# Method: Download directly from Kenneth French's website
# The pandas_datareader library has a built-in reader
try:
    import pandas_datareader.data as web
    ff3 = web.DataReader("F-F_Research_Data_Factors", "famafrench", start="2019-01", end="2025-12")[0]
    print("Retrieved FF3 factors via pandas_datareader")
except ImportError:
    print("pandas_datareader not installed. Downloading CSV directly...")
    # Fallback: download directly from French's website
    import io
    import zipfile
    import urllib.request

    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
    response = urllib.request.urlopen(url)
    z = zipfile.ZipFile(io.BytesIO(response.read()))
    csv_name = z.namelist()[0]
    with z.open(csv_name) as f:
        lines = f.read().decode("utf-8").splitlines()

    # Find the monthly data section (first table, before annual)
    data_lines = []
    header_found = False
    for line in lines:
        if "Mkt-RF" in line and "SMB" in line:
            header_found = True
            continue
        if header_found:
            parts = line.strip().split(",")
            if len(parts) >= 5 and len(parts[0].strip()) == 6:  # YYYYMM format
                data_lines.append(parts[:5])
            elif len(parts[0].strip()) == 4:  # Annual section starts
                break

    ff3 = pd.DataFrame(data_lines, columns=["Date", "Mkt-RF", "SMB", "HML", "RF"])
    ff3["Date"] = pd.to_datetime(ff3["Date"].str.strip(), format="%Y%m")
    ff3.set_index("Date", inplace=True)
    ff3 = ff3.astype(float)
    ff3 = ff3.loc["2019":"2025"]
    print("Retrieved FF3 factors directly from Ken French's website")

# FF3 data is in percentage terms; convert index to period
print(f"\nFama-French Factor Data:")
print(f"  Period: {ff3.index[0]} to {ff3.index[-1]}")
print(f"  Columns: {list(ff3.columns)}")
print(f"\nFactor Summary Statistics (monthly %):")
print(ff3[["Mkt-RF", "SMB", "HML"]].describe().round(3))

# ============================================================
# 2. Download Stock Returns
# ============================================================
print("\n" + "=" * 60)
print("Step 2: Download Stock Returns")
print("=" * 60)

tickers = ["AAPL", "MSFT", "NVDA", "JNJ", "PG", "KO", "TSLA", "XOM", "JPM", "META"]
print(f"Stocks: {tickers}")

prices = yf.download(tickers, start="2019-01-01", end="2025-12-31", auto_adjust=True)["Close"]
if isinstance(prices.columns, pd.MultiIndex):
    prices.columns = prices.columns.droplevel(1)

# Monthly returns
monthly_prices = prices.resample("ME").last()
monthly_returns = monthly_prices.pct_change().dropna() * 100  # Convert to %

# Align dates with FF3 data
# Convert monthly_returns index to period for matching
monthly_returns.index = monthly_returns.index.to_period("M")
ff3.index = ff3.index.to_period("M") if not isinstance(ff3.index, pd.PeriodIndex) else ff3.index

common_idx = monthly_returns.index.intersection(ff3.index)
monthly_returns = monthly_returns.loc[common_idx]
ff3_aligned = ff3.loc[common_idx]

# Excess stock returns (subtract risk-free rate)
stock_excess = monthly_returns.sub(ff3_aligned["RF"], axis=0)

print(f"Aligned months: {len(common_idx)}")

# ============================================================
# 3. Run CAPM and FF3 Regressions Side by Side
# ============================================================
print("\n" + "=" * 60)
print("Step 3: CAPM vs Fama-French 3-Factor")
print("=" * 60)

results_capm = {}
results_ff3 = {}

print(f"\n{'':=<90}")
print(f"{'Ticker':<8} | {'--- CAPM ---':^30} | {'--- Fama-French 3 ---':^42}")
print(f"{'':8} | {'α':>6} {'β_MKT':>7} {'R²':>6} | {'α':>6} {'β_MKT':>7} {'β_SMB':>7} {'β_HML':>7} {'R²':>6}")
print(f"{'':=<90}")

for ticker in tickers:
    y = stock_excess[ticker].values

    # --- CAPM ---
    X_capm = sm.add_constant(ff3_aligned["Mkt-RF"].values)
    capm = sm.OLS(y, X_capm).fit()
    results_capm[ticker] = {
        "alpha": capm.params[0], "beta_mkt": capm.params[1],
        "r2": capm.rsquared,
    }

    # --- Fama-French 3 ---
    X_ff3 = sm.add_constant(ff3_aligned[["Mkt-RF", "SMB", "HML"]].values)
    ff3_model = sm.OLS(y, X_ff3).fit()
    results_ff3[ticker] = {
        "alpha": ff3_model.params[0],
        "beta_mkt": ff3_model.params[1],
        "beta_smb": ff3_model.params[2],
        "beta_hml": ff3_model.params[3],
        "r2": ff3_model.rsquared,
        "p_alpha": ff3_model.pvalues[0],
        "model": ff3_model,
    }

    sig = "*" if results_ff3[ticker]["p_alpha"] < 0.05 else " "
    print(f"{ticker:<8} | {results_capm[ticker]['alpha']:>6.2f} {results_capm[ticker]['beta_mkt']:>7.2f} "
          f"{results_capm[ticker]['r2']:>6.2f} | "
          f"{results_ff3[ticker]['alpha']:>5.2f}{sig} {results_ff3[ticker]['beta_mkt']:>7.2f} "
          f"{results_ff3[ticker]['beta_smb']:>7.2f} {results_ff3[ticker]['beta_hml']:>7.2f} "
          f"{results_ff3[ticker]['r2']:>6.2f}")

print(f"\n* = FF3 alpha significant at 5%")

# ============================================================
# 4. Detailed Interpretation
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Factor Loading Interpretation")
print("=" * 60)

for ticker in tickers:
    r = results_ff3[ticker]
    size = "small-cap tilt" if r["beta_smb"] > 0.2 else "large-cap tilt" if r["beta_smb"] < -0.2 else "neutral"
    value = "value tilt" if r["beta_hml"] > 0.2 else "growth tilt" if r["beta_hml"] < -0.2 else "neutral"
    print(f"  {ticker:<6} β_SMB={r['beta_smb']:>5.2f} ({size}), β_HML={r['beta_hml']:>5.2f} ({value})")

# ============================================================
# 5. Plots
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Fama-French 3-Factor Model", fontsize=16, fontweight="bold")

# (a) R² Comparison: CAPM vs FF3
ax = axes[0, 0]
x = np.arange(len(tickers))
width = 0.35
r2_capm = [results_capm[t]["r2"] for t in tickers]
r2_ff3 = [results_ff3[t]["r2"] for t in tickers]
ax.barh(x - width / 2, r2_capm, width, label="CAPM", color="#ef4444", alpha=0.7)
ax.barh(x + width / 2, r2_ff3, width, label="FF3", color="#3b82f6", alpha=0.7)
ax.set_yticks(x)
ax.set_yticklabels(tickers)
ax.set_xlabel("R²")
ax.set_title("R² Comparison: CAPM vs FF3")
ax.legend()
ax.grid(True, alpha=0.3, axis="x")

# (b) Factor loadings heatmap
ax = axes[0, 1]
loadings = pd.DataFrame({
    "β_MKT": [results_ff3[t]["beta_mkt"] for t in tickers],
    "β_SMB": [results_ff3[t]["beta_smb"] for t in tickers],
    "β_HML": [results_ff3[t]["beta_hml"] for t in tickers],
}, index=tickers)
im = ax.imshow(loadings.values, cmap="RdBu_r", aspect="auto", vmin=-1.5, vmax=2.5)
ax.set_xticks(range(3))
ax.set_xticklabels(["β_MKT", "β_SMB", "β_HML"])
ax.set_yticks(range(len(tickers)))
ax.set_yticklabels(tickers)
for i in range(len(tickers)):
    for j in range(3):
        ax.text(j, i, f"{loadings.values[i,j]:.2f}", ha="center", va="center", fontsize=9)
plt.colorbar(im, ax=ax)
ax.set_title("Factor Loadings Heatmap")

# (c) Alpha comparison
ax = axes[0, 2]
alphas_capm = [results_capm[t]["alpha"] * 12 for t in tickers]  # Annualized
alphas_ff3 = [results_ff3[t]["alpha"] * 12 for t in tickers]
ax.barh(x - width / 2, alphas_capm, width, label="CAPM α", color="#ef4444", alpha=0.7)
ax.barh(x + width / 2, alphas_ff3, width, label="FF3 α", color="#3b82f6", alpha=0.7)
ax.set_yticks(x)
ax.set_yticklabels(tickers)
ax.axvline(0, color="black", linewidth=1)
ax.set_xlabel("Alpha (% annual)")
ax.set_title("Alpha: CAPM vs FF3 (annualized)")
ax.legend()
ax.grid(True, alpha=0.3, axis="x")

# (d) Factor cumulative returns
ax = axes[1, 0]
factor_cumret = (1 + ff3_aligned[["Mkt-RF", "SMB", "HML"]] / 100).cumprod()
dates = factor_cumret.index.to_timestamp()
ax.plot(dates, factor_cumret["Mkt-RF"], label="MKT", color="#3b82f6", linewidth=1.5)
ax.plot(dates, factor_cumret["SMB"], label="SMB", color="#10b981", linewidth=1.5)
ax.plot(dates, factor_cumret["HML"], label="HML", color="#f59e0b", linewidth=1.5)
ax.set_title("Factor Cumulative Returns")
ax.legend()
ax.grid(True, alpha=0.3)
ax.tick_params(axis="x", rotation=45)

# (e) Factor correlation
ax = axes[1, 1]
corr = ff3_aligned[["Mkt-RF", "SMB", "HML"]].corr()
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(3))
ax.set_xticklabels(["MKT", "SMB", "HML"])
ax.set_yticks(range(3))
ax.set_yticklabels(["MKT", "SMB", "HML"])
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=12, fontweight="bold")
plt.colorbar(im, ax=ax)
ax.set_title("Factor Correlation Matrix")

# (f) Detailed regression for one stock
ax = axes[1, 2]
ticker_detail = "NVDA"
r = results_ff3[ticker_detail]
model = r["model"]
ax.text(0.05, 0.95, f"FF3 Regression: {ticker_detail}", transform=ax.transAxes,
        fontsize=13, fontweight="bold", va="top")
summary_text = (
    f"\nR_excess = α + β_MKT·MKT + β_SMB·SMB + β_HML·HML\n\n"
    f"α     = {r['alpha']:>7.3f}  (p={r['p_alpha']:.3f})\n"
    f"β_MKT = {r['beta_mkt']:>7.3f}  (market sensitivity)\n"
    f"β_SMB = {r['beta_smb']:>7.3f}  ({'small' if r['beta_smb']>0 else 'large'}-cap tilt)\n"
    f"β_HML = {r['beta_hml']:>7.3f}  ({'value' if r['beta_hml']>0 else 'growth'} tilt)\n\n"
    f"R²    = {r['r2']:.3f}\n"
    f"R²(CAPM) = {results_capm[ticker_detail]['r2']:.3f}\n\n"
    f"Interpretation:\n"
    f"  FF3 explains {r['r2']*100:.1f}% of {ticker_detail}'s variance\n"
    f"  vs CAPM's {results_capm[ticker_detail]['r2']*100:.1f}%"
)
ax.text(0.05, 0.82, summary_text, transform=ax.transAxes, fontsize=10,
        va="top", fontfamily="monospace")
ax.axis("off")

plt.tight_layout()
plt.savefig("/mnt/nas/Class/2026/deeplearning/07_fama_french_3factor.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: 07_fama_french_3factor.png")

# ============================================================
# 6. Summary
# ============================================================
print("\n" + "=" * 60)
print("FAMA-FRENCH 3-FACTOR SUMMARY")
print("=" * 60)
print(f"""
Model: R_i - R_f = α + β_MKT·(R_m - R_f) + β_SMB·SMB + β_HML·HML

Three Factors:
  MKT (Market Risk Premium): Compensation for bearing market risk
  SMB (Small Minus Big):     Small-cap stocks outperform large-cap
  HML (High Minus Low):      Value stocks outperform growth stocks

Factor Loading Interpretation:
  β_SMB > 0 → behaves like small-cap stock
  β_SMB < 0 → behaves like large-cap stock
  β_HML > 0 → behaves like value stock
  β_HML < 0 → behaves like growth stock

Key Findings:
  1. FF3 generally has higher R² than CAPM (more variance explained)
  2. Tech stocks (NVDA, AAPL) tend to have negative β_HML (growth tilt)
  3. Defensive stocks (KO, PG) tend to have low β_MKT
  4. If FF3 alpha is still significant → other factors may be needed
     (momentum, profitability → leads to FF5 model)

Data Source:
  Kenneth French Data Library
  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
""")
