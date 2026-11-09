# Day 02: Card Fraud Detection at Speed with Polars + LightGBM

**Tier I - Advanced** | **Domain:** Financial services | [Folder on GitHub](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-02-card-fraud-polars-lightgbm/) | [README](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-02-card-fraud-polars-lightgbm/README.md)

- **Key metric:** test PR-AUC (LightGBM): **0.961** (test-month fraud cost at chosen threshold $41,073)
- **Data:** synthetic by design
- **Tests (static):** 18 functions, 20 cases
- **Lines of code:** 1,461 Python source, 212 tests, 0 SQL
- **CI:** success
- **Tools:** lightgbm, matplotlib, numpy, optuna, pandas, polars, pyarrow, scikit-learn, scipy, shap

## Caveats

- Fraud rate 0.243% instead of the brief's ~0.3%
