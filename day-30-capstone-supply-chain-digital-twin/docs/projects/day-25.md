# Day 25: Privacy-Preserving Analytics: Differential Privacy and Synthetic Data

**Tier IV - Distinguished** | **Domain:** Census & health statistics | [Folder on GitHub](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-25-privacy-dp-synthetic-data/) | [README](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-25-privacy-dp-synthetic-data/README.md)

- **Key metric:** TSTR AUC, DP synthetic data (eps=1): **0.830** (AUC on real data 0.875)
- **Data:** synthetic fallback (public source blocked); public sources attempted: UCI Adult; reachable: none
- **Tests (static):** 20 functions, 20 cases (20 stated in README)
- **Lines of code:** 2,156 Python source, 198 tests, 0 SQL
- **CI:** success
- **Tools:** anonymeter, diffprivlib, matplotlib, numba, numpy, pandas, scikit-learn, scipy

## Caveats

- SDV/CTGAN need PyTorch (CPU index blocked): copula and CTGAN-style GAN re-implemented
