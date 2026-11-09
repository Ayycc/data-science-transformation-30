# Day 04: Grid Load Forecasting with Probabilistic Backtesting

**Tier I - Advanced** | **Domain:** Energy & utilities | [Folder on GitHub](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-04-energy-load-forecasting-conformal/) | [README](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-04-energy-load-forecasting-conformal/README.md)

- **Key metric:** day-ahead MASE (best model): **0.445** (seasonal-naive MASE 0.874)
- **Data:** synthetic fallback (public source blocked); public sources attempted: EIA-930; reachable: none
- **Tests (static):** 20 functions, 21 cases
- **Lines of code:** 1,237 Python source, 220 tests, 0 SQL
- **CI:** success
- **Tools:** coreforecast, holidays, lightgbm, mapie, matplotlib, numpy, pandas, plotly, scikit-learn, scipy, statsforecast, statsmodels, utilsforecast

## Caveats

- 7 of 120 region x horizon cells outside 87-93% coverage (criterion met overall)
