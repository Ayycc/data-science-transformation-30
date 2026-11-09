# Day 12: Did the Programme Work? Causal Inference with Double ML

**Tier II - Expert** | **Domain:** Public policy & workforce | [Folder on GitHub](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-12-policy-causal-inference-dml/) | [README](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-12-policy-causal-inference-dml/README.md)

- **Key metric:** DML earnings ATE: **$2,243** (95% CI low $2,111)
- **Data:** synthetic + public supplement; public sources attempted: LaLonde/NSW; reachable: LaLonde/NSW
- **Tests (static):** 19 functions, 19 cases
- **Lines of code:** 1,676 Python source, 155 tests, 0 SQL
- **CI:** success
- **Tools:** dowhy, econml, linearmodels, matplotlib, networkx, numpy, pandas, scikit-learn, scipy, statsmodels

## Acceptance criteria not met

- 95% CI covers the true effect (employment outcome): CI [0.1135, 0.1437] misses truth 0.1121 on the committed seed
