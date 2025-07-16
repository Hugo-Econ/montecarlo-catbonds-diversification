# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 12:37:32 2025
@author: Hugo

This script analyzes historical SPY, AGG, and Cat Bond (Swiss Re Index) returns, 
fits appropriate distributions, and runs Monte Carlo simulations on four portfolio allocations:
  - 100% S&P 500
  - 80% S&P 500 / 20% Cat Bonds
  - 80% S&P 500 / 20% AGG Bonds
  - 80% S&P 500 / 10% Cat Bonds / 10% AGG Bonds

Methodology
===========
- Monthly return data from 2006 to 2024
- Distribution fitting:
    • SPY & Cat Bonds: Student's t-distribution
    • AGG Bonds: Normal distribution
- Simulated 1,000 paths over 1-year and 10-year horizons
- The draws account for the covariance matrix
- Cat Bonds adjusted for a 1.5% annual ETF fee
- Metrics: Final Wealth, Annual Return, Volatility, Sharpe Ratio, Max Drawdown, VaR (5%)

Results Summary
===============

📊 Median 1-Year Performance (1,000 Simulations, with Fee):
-----------------------------------------------------------
100% S&P 500:
- Final Wealth:         114.65%
- Annual Return:        14.65%
- Annual Volatility:    13.88%
- Sharpe Ratio:        101.09%
- Max Drawdown:          7.21%
- VaR (5%):             -4.45%

80% S&P 500 / 20% Cat Bonds:
- Final Wealth:         112.52%
- Annual Return:        12.52%
- Annual Volatility:    11.13%
- Sharpe Ratio:        105.43%
- Max Drawdown:          5.64%
- VaR (5%):             -3.50%

80% S&P 500 / 20% AGG Bonds:
- Final Wealth:         112.00%
- Annual Return:        12.00%
- Annual Volatility:    11.19%
- Sharpe Ratio:        104.76%
- Max Drawdown:          5.70%
- VaR (5%):             -3.51%

📊 Median 10-Year Performance (1,000 Simulations):
--------------------------------------------------
100% S&P 500:
- Final Wealth:         386.63%
- Annual Return:        14.48%
- Annual Volatility:    15.21%
- Sharpe Ratio:         91.38%
- Max Drawdown:         20.93%
- VaR (5%):             -5.52%

80% S&P 500 / 20% Cat Bonds:
- Final Wealth:         330.09%
- Annual Return:        12.68%
- Annual Volatility:    12.20%
- Sharpe Ratio:        100.03%
- Max Drawdown:         16.10%
- VaR (5%):             -4.34%

80% S&P 500 / 20% AGG Bonds:
- Final Wealth:         317.95%
- Annual Return:        12.26%
- Annual Volatility:    12.20%
- Sharpe Ratio:         96.45%
- Max Drawdown:         16.59%
- VaR (5%):             -4.39%

🏆 SPY 80 / Cat Bonds 20 vs SPY 80 / AGG 20 (with 1.5% fee):
-----------------------------------------------------------
Outperformance Frequency (out of 1,000 simulations):
- Final Wealth:         56%
- Annual Return:        56%
- Sharpe Ratio:         56%
- Max Drawdown:         46%
- VaR (5%):             54%

📌 Interpretation:
------------------
Cat Bonds are an excellent source of uncorrelated returns (SPY–CatBond corr ≈ 0.25).
Before applying a the 1.5% fee:
- Cat Bonds **consistently outperform AGG Bonds** across most metrics
- They improve Sharpe ratio and reduce drawdown risk
- In a 20% allocation, Cat Bonds offer a **clear diversification edge**
- Once the fee is included the advantages vanishes and become near identical to the typical 80/20. 
- The 80/10/10 portfolio showed surprinsingly limited benefit, despite a 15% correlation between Cat Bonds and AGG, 
  and a ~25% correlation with the equity market.

Conclusion:
-----------
While 100% equities still deliver the highest returns, introducing Cat Bonds can improve portfolio efficiency. 
However, the management fee is an hindrance; their performance after fees are mitigated and suggests that 
wider adoption — and lower-cost access — could make them a valuable addition to long-term portfolios.
"""

# -------------------------
# 📦 Imports
# -------------------------
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import skewnorm, t, norm
from scipy.linalg import cholesky
from numpy.random import default_rng
# import wrds  # Uncomment if using WRDS

# -------------------------
# 📈 Load Cat Bond Returns (2006–2024)
# -------------------------
#cat_bond_data = See Below.


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

##SPY
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

##Bonds
query_AGG = """
    SELECT msf.date, msf.ret
    FROM crsp.msf AS msf
    JOIN crsp.stocknames AS sn
        ON msf.permno = sn.permno
    WHERE sn.ticker = 'AGG'
      AND msf.date BETWEEN '2006-01-01' AND '2024-12-31'
"""

# Run the query
AGG_df = db.raw_sql(query_AGG)

# Format and clean
AGG_df['date'] = pd.to_datetime(AGG_df['date'])
AGG_df = AGG_df.set_index('date').to_period('M')   
AGG_df = AGG_df.rename(columns={'ret': 'AGG_Return'})

# -------------------------
# 🔗 Merge SPY and Cat Bonds
# -------------------------
merged_df = pd.merge(SPY_df, cat_bond_df, left_index=True, right_index=True)
merged_df = pd.merge(merged_df, AGG_df, left_index=True, right_index=True)
merged_df.columns = ['SPY_Return', 'Year', 'Month', 'CatBond_Return','AGG_Return']

# -------------------------
# 📊 Correlation
# -------------------------
correlation_matrix = merged_df[['SPY_Return', 'CatBond_Return', 'AGG_Return']].corr()

# Display
print("📊 Correlation Matrix:")
print(correlation_matrix.round(4))
# -------------------------
# 🔍 Explore Distributions
# -------------------------

# Bond stats
bond_returns = merged_df['AGG_Return'].dropna().values
print("Cat Bond Return Stats:")
print("Median:", np.median(bond_returns))
print("Mean:", np.mean(bond_returns))
print("Std:", np.std(bond_returns))
print("Skew:", stats.skew(bond_returns))
print("Kurtosis:", stats.kurtosis(bond_returns))

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

# Bonds (Normal)
mu_bond, std_bond = stats.norm.fit(bond_returns)

plt.figure(figsize=(10, 6))
plt.hist(bond_returns, bins=30, density=True, alpha=0.6, label='Empirical')
x = np.linspace(bond_returns.min(), bond_returns.max(), 100)
plt.plot(x, stats.norm.pdf(x, mu_bond, std_bond), 'r', label='Normal Fit')
plt.title('AGG (Bond ETF) Return Distribution')
plt.legend()
plt.grid(True)
plt.show()


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

mu_bond, std_bond = norm.fit(bond_returns)

# Print fitted parameters
print(f"AGG normal fit: mean={mu_bond:.4f}, std={std_bond:.4f}")

# -------------------------
# 🧪 Monte Carlo Simulation
# -------------------------

# %% 1000 1-year simulation 
# -------------------------------------
# 🎯 Use fitted distribution parameters
# -------------------------------------

# SPY (t-distribution)
df_spy, loc_spy, scale_spy = params_spy_t

# Cat Bonds (t-distribution)
df_cb, loc_cb, scale_cb = params_CatB_t

# AGG (normal distribution)
mu_bond, std_bond = norm.fit(bond_returns)

# -------------------------------------
# ⚙️ Simulation settings
# -------------------------------------
n_months = 12       # Simulate 1-year forward
n_sims = 1000       # Run 1000 simulations
rf_monthly = 0.0005 # Monthly risk-free rate

# -------------------------------------
# 📊 Metric computation function
# -------------------------------------
def compute_metrics(returns, rf=rf_monthly):
    ann_ret = (np.prod(1 + returns))**(12 / len(returns)) - 1
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

# -------------------------------------
# 🔁 Run simulations
# -------------------------------------
results = {
    'SPY_100': [],
    'SPY_80_CB_20': [],
    'SPY_80_AGG_20': [],
    'SPY_80_CB_10_AGG_10': []
}
# Cholesky decomposition
L = cholesky(cor_matrix, lower=True)

# RNG
rng = default_rng()

for _ in range(n_sims):
    # Step 1: Generate uncorrelated standard normals
    z = rng.standard_normal(size=(3, n_months))  # 3 assets × months

    # Step 2: Apply Cholesky to induce correlation
    correlated_z = L @ z  # Shape: (3, n_months)

    # Step 3: Transform each series to their respective distributions
    sim_spy = t.ppf(norm.cdf(correlated_z[0]), df=df_spy) * scale_spy + loc_spy
    sim_cb = t.ppf(norm.cdf(correlated_z[1]), df=df_cb) * scale_cb + loc_cb_net  # with fee
    sim_bond = correlated_z[2] * std_bond + mu_bond  # normal

    # Step 4: Portfolio combinations
    results['SPY_100'].append(compute_metrics(sim_spy))
    results['SPY_80_CB_20'].append(compute_metrics(0.8 * sim_spy + 0.2 * sim_cb))
    results['SPY_80_AGG_20'].append(compute_metrics(0.8 * sim_spy + 0.2 * sim_bond))
    results['SPY_80_CB_10_AGG_10'].append(compute_metrics(0.8 * sim_spy + 0.1 * sim_cb + 0.1 * sim_bond))

# -------------------------------------
# 📦 Convert to DataFrames
# -------------------------------------
df_metrics = {k: pd.DataFrame(v) for k, v in results.items()}

print("📊 Median 1-year Metrics Over 1000 Simulations\n")

# Loop through each portfolio and print median values
for name in ['SPY_100', 'SPY_80_CB_20', 'SPY_80_AGG_20', 'SPY_80_CB_10_AGG_10']:
    print(f"{name.replace('_', ' ')}:")
    print(df_metrics[name].median().apply(lambda x: f"{x:.2%}"))
    print()


# %% 1000 10-year simulation 
# -------------------------------------
# ⚙️ Simulation settings
# -------------------------------------
n_months = 12*10       # Simulate 1-year forward
n_sims = 1000       # Run 1000 simulations
rf_monthly = 0.0005 # Monthly risk-free rate

# -------------------------------------
# 📊 Metric computation function
# -------------------------------------
def compute_metrics(returns, rf=rf_monthly):
    ann_ret = (np.prod(1 + returns))**(12 / len(returns)) - 1
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

# -------------------------------------
# 🔁 Run simulations
# -------------------------------------
results_10 = {
    'SPY_100': [],
    'SPY_80_CB_20': [],
    'SPY_80_AGG_20': [],
    'SPY_80_CB_10_AGG_10': []
}

for _ in range(n_sims):
    # Step 1: Generate uncorrelated standard normals
    z = rng.standard_normal(size=(3, n_months))  # 3 assets × months

    # Step 2: Apply Cholesky to induce correlation
    correlated_z = L @ z  # Shape: (3, n_months)

    # Step 3: Transform each series to their respective distributions
    sim_spy = t.ppf(norm.cdf(correlated_z[0]), df=df_spy) * scale_spy + loc_spy
    sim_cb = t.ppf(norm.cdf(correlated_z[1]), df=df_cb) * scale_cb + loc_cb_net  # with fee
    sim_bond = correlated_z[2] * std_bond + mu_bond  # normal

    # Step 4: Portfolio combinations
    results_10['SPY_100'].append(compute_metrics(sim_spy))
    results_10['SPY_80_CB_20'].append(compute_metrics(0.8 * sim_spy + 0.2 * sim_cb))
    results_10['SPY_80_AGG_20'].append(compute_metrics(0.8 * sim_spy + 0.2 * sim_bond))
    results_10['SPY_80_CB_10_AGG_10'].append(compute_metrics(0.8 * sim_spy + 0.1 * sim_cb + 0.1 * sim_bond))


# -------------------------------------
# 📦 Convert to DataFrames
# -------------------------------------
df_metrics_10 = {k: pd.DataFrame(v) for k, v in results_10.items()}

print("📊 Median Metrics for 10 years Over 1000 Simulations\n")

# Loop through each portfolio and print median values
for name in ['SPY_100', 'SPY_80_CB_20', 'SPY_80_AGG_20', 'SPY_80_CB_10_AGG_10']:
    print(f"{name.replace('_', ' ')}:")
    print(df_metrics_10[name].median().apply(lambda x: f"{x:.2%}"))
    print()

# -------------------------------------
# Frequency Comparison
# -------------------------------------
##80/20 Cat Bonds Vs. Agg Bonds
# Define the two portfolios
cb_port = df_metrics_10['SPY_80_CB_20']
agg_port = df_metrics_10['SPY_80_AGG_20']

# Compare metrics (row-by-row, same metric)
comparison = cb_port > agg_port  # DataFrame: True where CB outperforms

# Calculate how often CB outperformed AGG on each metric
dominance_ratio = comparison.mean()

print("✅ How often SPY 80 / Cat Bonds 20 outperformed SPY 80 / AGG 20 (out of 1000 simulations):")
print(dominance_ratio.apply(lambda x: f"{x:.0%}"))


# %% 1000 1-year simulation, WITH FEE
# -------------------------------------
# 🎯 Use fitted distribution parameters
# -------------------------------------


monthly_fee = (1 + 0.015)**(1/12) - 1  # ≈ 0.00124
loc_cb_net  = loc_cb - monthly_fee

# -------------------------------------
# ⚙️ Simulation settings
# -------------------------------------
n_months = 12       # Simulate 1-year forward
n_sims = 1000       # Run 1000 simulations
rf_monthly = 0.0005 # Monthly risk-free rate

# -------------------------------------
# 📊 Metric computation function
# -------------------------------------
def compute_metrics(returns, rf=rf_monthly):
    ann_ret = (np.prod(1 + returns))**(12 / len(returns)) - 1
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

# -------------------------------------
# 🔁 Run simulations
# -------------------------------------
results_fee = {
    'SPY_100': [],
    'SPY_80_CB_20': [],
    'SPY_80_AGG_20': [],
    'SPY_80_CB_10_AGG_10': []
}

for _ in range(n_sims):
    # Step 1: Generate uncorrelated standard normals
    z = rng.standard_normal(size=(3, n_months))  # 3 assets × months

    # Step 2: Apply Cholesky to induce correlation
    correlated_z = L @ z  # Shape: (3, n_months)

    # Step 3: Transform each series to their respective distributions
    sim_spy = t.ppf(norm.cdf(correlated_z[0]), df=df_spy) * scale_spy + loc_spy
    sim_cb = t.ppf(norm.cdf(correlated_z[1]), df=df_cb) * scale_cb + loc_cb_net  # with fee
    sim_bond = correlated_z[2] * std_bond + mu_bond  # normal

    # Step 4: Portfolio combinations
    results_fee['SPY_100'].append(compute_metrics(sim_spy))
    results_fee['SPY_80_CB_20'].append(compute_metrics(0.8 * sim_spy + 0.2 * sim_cb))
    results_fee['SPY_80_AGG_20'].append(compute_metrics(0.8 * sim_spy + 0.2 * sim_bond))
    results_fee['SPY_80_CB_10_AGG_10'].append(compute_metrics(0.8 * sim_spy + 0.1 * sim_cb + 0.1 * sim_bond))

# -------------------------------------
# 📦 Convert to DataFrames
# -------------------------------------
df_metrics_fee = {k: pd.DataFrame(v) for k, v in results_fee.items()}

print("📊 Median 1-year Metrics Over 1000 Simulations **WITH 1.5% FEE** \n")

# Loop through each portfolio and print median values
for name in ['SPY_100', 'SPY_80_CB_20', 'SPY_80_AGG_20','SPY_80_CB_10_AGG_10']:
    print(f"{name.replace('_', ' ')}:")
    print(df_metrics_fee[name].median().apply(lambda x: f"{x:.2%}"))
    print()
    
# -------------------------------------
# Frequency Comparison
# -------------------------------------
##80/20 Cat Bonds Vs. Agg Bonds
# Define the two portfolios
cb_port = df_metrics_fee['SPY_80_CB_20']
agg_port = df_metrics_fee['SPY_80_AGG_20']

# Compare metrics (row-by-row, same metric)
comparison = cb_port > agg_port  # DataFrame: True where CB outperforms

# Calculate how often CB outperformed AGG on each metric
dominance_ratio = comparison.mean()

print("✅ How often SPY 80 / Cat Bonds 20 outperformed SPY 80 / AGG 20 **WITH FEE**:")
print(dominance_ratio.apply(lambda x: f"{x:.0%}"))

## 80/20 Cat Bonds vs. 80/10/10 Cat Bonds + AGG
# Define the two portfolios
cb_port = df_metrics_fee['SPY_80_CB_20']
agg_port = df_metrics_fee['SPY_80_CB_10_AGG_10']

# Compare metrics (row-by-row, same metric)
comparison = cb_port > agg_port  # DataFrame: True where CB outperforms

# Calculate how often CB outperformed AGG on each metric
dominance_ratio = comparison.mean()

print("✅ How often SPY 80 / Cat Bonds 20 outperformed SPY 80 / 10% Cat + 10% AGG (with fee):")
print(dominance_ratio.apply(lambda x: f"{x:.0%}"))



# -------------------------
# 📈 Load Cat Bond Returns (2006–2024)
# -------------------------
cat_bond_data = {
    2024: [1.17, 1.1, 0.76, 0.5, 0.01, 0.76, 1.34, 1.69, 1.9, 0.82, 1.04, 1.32],
    2023: [1.11, 0.84, 1.14, 1.12, 1.26, 1.29, 0.81, 1.48, 1.31, 1.42, 0.74, 0.57],
    2022: [0.25, 0.08, 0.03, 0.00, 0.07, -0.09, 0.24, 0.80, -6.88, 0.97, 1.23, 1.35],
    2021: [0.13, -0.6, 0.26, 0.27, 0.57, 0.31, 0.18, -0.43, -0.47, 0.2, 0.4, 0.23],
    2020: [0.61, 0.32, -0.7, -0.16, 0.12, 0.67, 0.69, 0.69, 1.07, -0.08, -0.04, 0.24],
    2019: [0.52, -0.01, -0.08, -0.69, -1.13, 0.27, 0.41, 0.09, 1.42, -0.39, 0.22, 0.3],
    2018: [0.54, 0.08, -0.24, -0.28, 0.19, 0.27, 0.61, 0.45, -0.08, -0.81, -3.68, -0.97],
    2017: [0.36, 0.32, 0.21, 0.15, 0.19, 0.4, 0.56, -0.31, -8.61, 0.4, 0.68, 0.27],
    2016: [0.21, 0.53, 0.4, 0.4, 0.04, 0.26, 0.41, 0.86, 1.03, 0.42, 0.31, 0.18],
    2015: [0.28, 0.28, 0.25, 0.31, 0.12, 0.21, 0.34, 0.2, 0.11, 0.09, 0.15, 0.17],
    2014: [0.37, 0.32, 0.29, 0.25, 0.27, 0.31, 0.29, 0.28, 0.24, 0.22, 0.18, 0.24],
    2013: [1.5, 0.7, 0.89, 0.95, 0.68, 0.45, 0.71, 0.73, 0.78, 1.05, 0.62, 0.41],
    2012: [0.85, 0.45, 0.38, 0.33, 0.05, 0.19, 0.52, 0.75, 0.58, 0.15, 0.45, 0.3],
    2011: [0.65, 0.45, 0.15, 0.35, 0.25, 0.22, 0.29, 0.15, -0.35, 0.25, 0.15, 0.42],
    2010: [0.45, 0.38, 0.52, 0.48, 0.35, 0.42, 0.55, 0.62, 0.58, 0.45, 0.32, 0.41],
    2009: [0.35, 0.28, 0.45, 0.52, 0.48, 0.55, 0.65, 0.72, 0.68, 0.58, 0.45, 0.52],
    2008: [0.45, 0.38, 0.25, -0.15, 0.22, 0.18, 0.25, -0.52, -1.85, -3.25, -0.85, -0.45],
    2007: [1.25, 0.85, 0.95, 0.78, 0.82, 0.68, 0.72, 0.55, 0.65, 0.85, 0.62, 0.58],
    2006: [0.85, 0.65, 0.72, 0.68, 0.55, 0.62, 0.75, 0.82, 0.78, 0.65, 0.58, 0.72]
}

cor_matrix = np.array([
    [1.0,        0.254011, 0.246423],
    [0.254011,   1.0,      0.150656],
    [0.246423,   0.150656, 1.0]
])