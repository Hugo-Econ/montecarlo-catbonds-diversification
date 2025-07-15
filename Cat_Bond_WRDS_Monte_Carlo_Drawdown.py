# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 12:37:32 2025
@author: Hugo

This script analyzes historical Cat Bond and SPY returns,
fits distributions, evaluates them statistically, and runs
a Monte Carlo simulation of two portfolios:
  - 100% SPY
  - 90% SPY / 10% Cat Bonds

Results Summary
===============

🧪 Simulation Setup:
- 100 independent simulations
- Each simulation spans 12 months
- Returns drawn from fitted Student's t-distributions (Shown lower that it is a best match)

📊 Average Performance Over Simulations:
---------------------------------------
100% S&P 500:
- Final Wealth:         117.88%
- Annual Return:        16.71%
- Annual Volatility:    13.89%
- Sharpe Ratio:        122.22%
- Max Drawdown:          7.38%
- VaR (5%):             -4.45%

90% S&P 500 / 10% Cat Bonds:
- Final Wealth:         116.59%
- Annual Return:        15.57%
- Annual Volatility:    12.51%
- Sharpe Ratio:        126.08%
- Max Drawdown:          6.54%
- VaR (5%):             -3.97%

🏆 Outperformance Frequency (90/10 vs 100% SPY):
----------------------------------------------
- Final Wealth:         28% of (annual) simulations
- Annual Return:        20%
- Annual Volatility:     0%  (SPY always more volatile)
- Sharpe Ratio:         96%
- Max Drawdown:          1%  ✅ (90/10 had lower drawdown almost always)
- VaR (5%):             99%  ✅ (much better tail protection)

📌 Interpretation:
------------------
Adding just 10% of Cat Bonds:
- Slightly lowers return (~1% p.a.)
- Consistently reduces downside risk (lower max drawdown, better VaR)
- Sharpe ratio improves in **96%** of simulations

Conclusion:
-----------
A 90/10 blend improves risk-adjusted performance and significantly reduces tail risk — with only a modest drop in expected return.
"""
# -------------------------
# 📦 Imports
# -------------------------
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import skewnorm, t
# import wrds  # Uncomment if using WRDS

# -------------------------
# 📈 Load Cat Bond Returns (2006–2024)
# -------------------------


# Create DataFrame from dictionary
cat_bond_df = pd.DataFrame(cat_bond_data).T
cat_bond_df.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Convert to long format with datetime
cat_bond_df = cat_bond_df.stack().reset_index()
cat_bond_df.columns = ['Year', 'Month', 'Return']
cat_bond_df['Date'] = pd.to_datetime(cat_bond_df['Year'].astype(str) + '-' + cat_bond_df['Month'], format='%Y-%b')
cat_bond_df['Date'] = cat_bond_df['Date'].dt.to_period('M')
cat_bond_df['Return'] = cat_bond_df['Return'] / 100  # Convert % to decimal
cat_bond_df = cat_bond_df.set_index('Date')

# -------------------------
# 🧠 Load SPY Returns from WRDS
# -------------------------

# db = wrds.Connection()  # Uncomment to use WRDS

query_SPY = """
    SELECT msf.date, msf.ret
    FROM crsp.msf AS msf
    JOIN crsp.stocknames AS sn
        ON msf.permno = sn.permno
    WHERE sn.ticker = 'SPY'
      AND msf.date BETWEEN '2006-01-01' AND '2024-12-31'
"""

# Run the query
SPY_df = db.raw_sql(query_SPY)

# Format and clean
SPY_df['date'] = pd.to_datetime(SPY_df['date'])
SPY_df = SPY_df.set_index('date').to_period('M')   
SPY_df = SPY_df.rename(columns={'ret': 'SPY_Return'})

# -------------------------
# 🔗 Merge SPY and Cat Bonds
# -------------------------
merged_df = pd.merge(SPY_df, cat_bond_df, left_index=True, right_index=True)
merged_df.columns = ['SPY_Return', 'Year', 'Month', 'CatBond_Return']

# -------------------------
# 📊 Correlation
# -------------------------
correlation = merged_df[['SPY_Return', 'CatBond_Return']].corr().iloc[0, 1]
print(f"Correlation between SPY and Cat Bond returns: {correlation:.4f}")

# -------------------------
# 🔍 Explore Distributions
# -------------------------

# Cat Bond stats
cat_bond_returns = merged_df['CatBond_Return'].dropna().values
print("Cat Bond Return Stats:")
print("Median:", np.median(cat_bond_returns))
print("Mean:", np.mean(cat_bond_returns))
print("Std:", np.std(cat_bond_returns))
print("Skew:", stats.skew(cat_bond_returns))
print("Kurtosis:", stats.kurtosis(cat_bond_returns))

# SPY stats
sp500_returns = merged_df['SPY_Return'].dropna().values
print("\nSPY Return Stats:")
print("Median:", np.median(sp500_returns))
print("Mean:", np.mean(sp500_returns))
print("Std:", np.std(sp500_returns))
print("Skew:", stats.skew(sp500_returns))
print("Kurtosis:", stats.kurtosis(sp500_returns))

# Plot Cat Bond distribution
mu_cb, std_cb = stats.norm.fit(cat_bond_returns)
plt.figure(figsize=(10, 6))
plt.hist(cat_bond_returns, bins=30, density=True, alpha=0.6, label='Cat Bond Returns')
x = np.linspace(cat_bond_returns.min(), cat_bond_returns.max(), 100)
plt.plot(x, stats.norm.pdf(x, mu_cb, std_cb), 'r', label='Normal Fit')
plt.title("Cat Bond Return Distribution")
plt.legend()
plt.show()

# Plot SPY distribution
mu_spy, std_spy = stats.norm.fit(sp500_returns)
plt.figure(figsize=(10, 6))
plt.hist(sp500_returns, bins=30, density=True, alpha=0.6, label='SPY Returns')
x = np.linspace(sp500_returns.min(), sp500_returns.max(), 100)
plt.plot(x, stats.norm.pdf(x, mu_spy, std_spy), 'r', label='Normal Fit')
plt.title("SPY Return Distribution")
plt.legend()
plt.show()

# -------------------------
# ⚖️ Fit Alternative Distributions
# -------------------------
normal_params = stats.norm.fit(cat_bond_returns)
t_params = stats.t.fit(cat_bond_returns)
skew_params = skewnorm.fit(cat_bond_returns)

# Plot distribution fits
x_vals = np.linspace(cat_bond_returns.min(), cat_bond_returns.max(), 1000)
plt.figure(figsize=(10, 6))
plt.hist(cat_bond_returns, bins=50, density=True, alpha=0.5, label='Empirical')
plt.plot(x_vals, stats.norm.pdf(x_vals, *normal_params), 'r', label='Normal')
plt.plot(x_vals, stats.t.pdf(x_vals, *t_params), 'g', label=f"t (df={t_params[0]:.1f})")
plt.plot(x_vals, skewnorm.pdf(x_vals, *skew_params), 'orange', label='SkewNorm')
plt.title("Distribution Fit - Cat Bonds")
plt.legend()
plt.show()

# K-S tests
print("Kolmogorov–Smirnov Test Results:")
print(f"Normal: D={stats.kstest(cat_bond_returns, 'norm', args=normal_params).statistic:.4f}")
print(f"t-dist: D={stats.kstest(cat_bond_returns, 't', args=t_params).statistic:.4f}")
print(f"SkewNorm: D={stats.kstest(cat_bond_returns, 'skewnorm', args=skew_params).statistic:.4f}")

# -------------------------
# 🎯 Fit Final t-Distributions
# -------------------------
params_spy_t = stats.t.fit(sp500_returns)
print(f"\nSPY t-dist fit: df={params_spy_t[0]:.2f}, loc={params_spy_t[1]:.4f}, scale={params_spy_t[2]:.4f}")

params_CatB_t = stats.t.fit(cat_bond_returns)
print(f"Cat Bond t-dist fit: df={params_CatB_t[0]:.2f}, loc={params_CatB_t[1]:.4f}, scale={params_CatB_t[2]:.4f}")

# -------------------------
# 🧪 Monte Carlo Simulation
# -------------------------

# 1. Extract fitted parameters
df_spy, loc_spy, scale_spy = params_spy_t
df_cb, loc_cb, scale_cb = params_CatB_t

# 2. Simulate 1,000 months of returns
n_months = 1000
rf_monthly = 0.0005  # risk-free return ≈ 0.6% annual

sim_spy = t.rvs(df_spy, loc=loc_spy, scale=scale_spy, size=n_months)
sim_cb = t.rvs(df_cb, loc=loc_cb, scale=scale_cb, size=n_months)

# 3. Define portfolios
r_100_spy = sim_spy
r_90_10 = 0.9 * sim_spy + 0.1 * sim_cb

# 4. Store returns in DataFrame
df_sim = pd.DataFrame({
    'SPY_100': r_100_spy,
    'SPY_90_CB_10': r_90_10
})

# 5. Compute cumulative wealth
wealth_df = (1 + df_sim).cumprod()

# 6. Portfolio metrics function
def compute_metrics(returns, rf=rf_monthly):
    ann_ret = np.mean(returns) * 12
    ann_vol = np.std(returns) * np.sqrt(12)
    sharpe = (ann_ret - rf * 12) / ann_vol
    wealth = (1 + returns).cumprod()
    drawdown = 1 - wealth / wealth.cummax()
    return {
        'Annual Return': ann_ret,
        'Annual Volatility': ann_vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': drawdown.max(),
        'VaR (5%)': np.percentile(returns, 5)
    }

# 7. Compute metrics
metrics_spy = compute_metrics(df_sim['SPY_100'])
metrics_90_10 = compute_metrics(df_sim['SPY_90_CB_10'])

# 8. Print results
print("\n🔍 Portfolio Performance Metrics")
print("100% S&P 500:")
for k, v in metrics_spy.items():
    print(f"{k}: {v:.2%}")

print("\n90% S&P 500 / 10% Cat Bonds:")
for k, v in metrics_90_10.items():
    print(f"{k}: {v:.2%}")

# 9. Plot cumulative wealth
plt.figure(figsize=(10, 6))
plt.plot(wealth_df['SPY_100'], label='100% S&P 500')
plt.plot(wealth_df['SPY_90_CB_10'], label='90% SPY / 10% Cat Bonds')
plt.title('Cumulative Wealth (Monte Carlo Simulated)')
plt.xlabel('Months')
plt.ylabel('Portfolio Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#100 sims


# -----------------------------
# 2. Simulation Settings
# -----------------------------
n_months = 12           # Each simulation lasts 100 months (~8.3 years)
n_sims = 100             # Number of Monte Carlo simulations
rf_monthly = 0.0005      # Monthly risk-free rate (~0.6% annual)

# -----------------------------
# 3. Metric Computation Function
# -----------------------------
def compute_metrics(returns, rf=rf_monthly):
    ann_ret = np.mean(returns) * 12
    ann_vol = np.std(returns) * np.sqrt(12)
    sharpe = (ann_ret - rf * 12) / ann_vol
    wealth = pd.Series((1 + returns).cumprod())
    drawdown = 1 - wealth / wealth.cummax()
    return {
        'Final Wealth': wealth.iloc[-1],
        'Annual Return': ann_ret,
        'Annual Volatility': ann_vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': drawdown.max(),
        'VaR (5%)': np.percentile(returns, 5)
    }

# -----------------------------
# 4. Run Simulations
# -----------------------------
metrics_list_spy = []
metrics_list_90_10 = []

for _ in range(n_sims):
    # Simulate monthly returns
    sim_spy = t.rvs(df_spy, loc=loc_spy, scale=scale_spy, size=n_months)
    sim_cb = t.rvs(df_cb, loc=loc_cb, scale=scale_cb, size=n_months)
    
    # Portfolio 1: 100% SPY
    metrics_list_spy.append(compute_metrics(sim_spy))
    
    # Portfolio 2: 90% SPY / 10% Cat Bonds
    mixed_returns = 0.9 * sim_spy + 0.1 * sim_cb
    metrics_list_90_10.append(compute_metrics(mixed_returns))

# Convert results to DataFrames
metrics_df_spy = pd.DataFrame(metrics_list_spy)
metrics_df_90_10 = pd.DataFrame(metrics_list_90_10)


# Compare performance per simulation
comparison = metrics_df_90_10 > metrics_df_spy

# Calculate how often 90/10 outperformed 100% SPY
outperformance_ratio = comparison.mean()

# Display results
print("✅ How often 90/10 outperformed 100% SPY (out of 100 simulations):")
print(outperformance_ratio.apply(lambda x: f"{x:.0%}"))

# -----------------------------
# 5. Plot Final Wealth Distribution
# -----------------------------
plt.figure(figsize=(10, 6))
plt.hist(metrics_df_spy['Final Wealth'], bins=20, alpha=0.6, label='100% S&P 500')
plt.hist(metrics_df_90_10['Final Wealth'], bins=20, alpha=0.6, label='90% SPY / 10% Cat Bonds')
plt.title('Distribution of Final Portfolio Wealth\n(100 Simulations, 100 Months Each)')
plt.xlabel('Final Portfolio Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 6. Plot Distributions of Other Metrics
# -----------------------------
metrics_to_plot = ['Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Max Drawdown', 'VaR (5%)']

for metric in metrics_to_plot:
    plt.figure(figsize=(10, 5))
    plt.hist(metrics_df_spy[metric], bins=20, alpha=0.6, label='100% S&P 500')
    plt.hist(metrics_df_90_10[metric], bins=20, alpha=0.6, label='90% SPY / 10% Cat Bonds')
    plt.title(f'Distribution of {metric} Across Simulations')
    plt.xlabel(metric)
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -----------------------------
# 7. Summary Statistics
# -----------------------------
print("📊 Average Metrics Over 100 Simulations:\n")

print("100% S&P 500:")
print(metrics_df_spy.mean().apply(lambda x: f"{x:.2%}"))

print("\n90% S&P 500 / 10% Cat Bonds:")
print(metrics_df_90_10.mean().apply(lambda x: f"{x:.2%}"))