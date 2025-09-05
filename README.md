# Technical Indicator Strategy Fitter

This repository contains a technical indicator strategy fitter, designed to fit indicator-based trend-following strategies to historical price data of a chosen asset. I initially developed these strategies through manual parameter fitting, which was a slow and inefficient process. The fitter was created as a solution to this, dramatically improving the speed of strategy development through automation.

Strategies are fit using a combination of Bayesian Optimisation and a custom step-wise search method, optimised relative to a custom scoring system known as the 'Slap Score'. The slap score assigns points to various strategy performance metrics based on pre-defined buckets, summing them into a single score.

A report is included in the repository, showcasing how the fitter was used to produce a portfolio of strategies.

Repository Contents:
- `Report.ipynb` – Main report
- `code/` – Simplified version of the strategy fitter code.
  - `Optimisers.py` – Backtest function and optimisation engine
  - `Indicators.py` – Technical indicator code, strategy logic and aggregation
  - `PerformanceMetrics.py` –  Data handling and performance metric calculation functions
  - `RobustnessTesting.py` –  Parameter robustness testing functions
  - `Plots.py` – Plotting functions
- `images/` – Figures used in the report
- `README.md`

The code contained in this repository is a simplified version of the fitter. If you are interested in seeing the full code, please contact me via email at william.keir9@gmail.com

## Key Findings:

The strategy fitter was used to produced a portfolio of strategies with:
- Out-of-Sample Sharpe ratio: 1.39
- Out-of-Sample Annualised Returns: 72.10%
