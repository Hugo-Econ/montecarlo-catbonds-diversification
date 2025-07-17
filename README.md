# SPY, AGG, and Cat Bond Portfolio Motecarlo Simulation

This script analyzes historical returns and runs Monte Carlo simulations for four portfolio allocations:
- 100% S&P 500  
- 80% S&P 500 / 20% Cat Bonds  
- 80% S&P 500 / 20% AGG Bonds  
- 80% S&P 500 / 10% Cat Bonds / 10% AGG Bonds  

## Key Takeaways
- Cat Bonds (pre-fee) outperform AGG across most metrics
- Offer diversification (corr ≈ 0.25 with SPY)
- Post-fee, benefits largely offset
- 80/10/10 mix provides limited advantage
- Fee reduction could make Cat Bonds a valuable addition

## Methodology
- Monthly returns (2006–2024)
- Distributions:
  - SPY & Cat Bonds: Student’s t
  - AGG Bonds: Normal
- 1,000 simulations over 1- and 10-year horizons
- Covariance-based random draws
- 1.5% annual ETF fee applied to Cat Bonds
- Metrics: Final Wealth, Annual Return, Volatility, Sharpe, Max Drawdown, VaR (5%)

## Results Summary

### Median 1-Year Returns (after fee)
| Portfolio                     | Wealth | Return | Volatility | Sharpe | Drawdown | VaR (5%) |
|------------------------------|--------|--------|------------|--------|-----------|----------|
| 100% SPY                     | 114.7% | 14.7%  | 13.9%      | 101.1% | 7.2%      | -4.5%    |
| 80/20 Cat Bonds              | 112.5% | 12.5%  | 11.1%      | 105.4% | 5.6%      | -3.5%    |
| 80/20 AGG                    | 112.0% | 12.0%  | 11.2%      | 104.8% | 5.7%      | -3.5%    |

### Median 10-Year Returns
| Portfolio                     | Wealth | Return | Volatility | Sharpe | Drawdown | VaR (5%) |
|------------------------------|--------|--------|------------|--------|-----------|----------|
| 100% SPY                     | 386.6% | 14.5%  | 15.2%      | 91.4%  | 20.9%     | -5.5%    |
| 80/20 Cat Bonds              | 330.1% | 12.7%  | 12.2%      | 100.0% | 16.1%     | -4.3%    |
| 80/20 AGG                    | 318.0% | 12.3%  | 12.2%      | 96.5%  | 16.6%     | -4.4%    |

### Head-to-Head: SPY 80 / Cat 20 vs SPY 80 / AGG 20
| Metric         | Cat Outperformance Rate |
|----------------|--------------------------|
| Final Wealth   | 56%                      |
| Annual Return  | 56%                      |
| Sharpe Ratio   | 56%                      |
| Max Drawdown   | 46%                      |
| VaR (5%)       | 54%                      |

