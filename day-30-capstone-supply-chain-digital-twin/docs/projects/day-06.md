# Day 06: Experimentation Engine: CUPED, Sequential and Bayesian Tests

**Tier I - Advanced** | **Domain:** Digital product & marketing | [Folder on GitHub](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-06-experimentation-platform-cuped/) | [README](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-06-experimentation-platform-cuped/README.md)

- **Key metric:** mSPRT false-positive rate under daily peeking: **1.35%** (naive peeking FPR 25.9%)
- **Data:** synthetic by design
- **Tests (static):** 47 functions, 51 cases (51 stated in README)
- **Lines of code:** 1,576 Python source, 399 tests, 0 SQL
- **CI:** success
- **Tools:** matplotlib, numpy, pandas, scipy, statsmodels, streamlit

## Acceptance criteria not met

- CUPED variance reduction >= 30% on correlated metrics: revenue reached 24.1% (engagement 65.7%)
