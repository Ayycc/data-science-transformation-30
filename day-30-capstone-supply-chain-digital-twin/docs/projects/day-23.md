# Day 23: Observability Analytics: Multivariate Anomaly Detection and Root Cause

**Tier III - Principal** | **Domain:** Technology operations | [Folder on GitHub](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-23-sre-anomaly-root-cause/) | [README](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-23-sre-anomaly-root-cause/README.md)

- **Key metric:** alert volume reduction (raw alerts to incidents): **88.2%** (raw threshold alerts 730)
- **Data:** synthetic + public supplement; public sources attempted: NAB; reachable: NAB
- **Tests (static):** 18 functions, 18 cases
- **Lines of code:** 1,324 Python source, 175 tests, 0 SQL
- **CI:** success
- **Tools:** causal-learn, matplotlib, networkx, numpy, pandas, pyod, ruptures, scikit-learn, scipy, statsmodels

## Caveats

- Two low-severity error bursts missed
