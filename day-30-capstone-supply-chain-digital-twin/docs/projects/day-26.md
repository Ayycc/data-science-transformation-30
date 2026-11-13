# Day 26: Responsible Credit Decisioning: Fairness, Explainability and Model Governance

**Tier IV - Distinguished** | **Domain:** Lending | [Folder on GitHub](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-26-lending-fairness-model-governance/) | [README](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-26-lending-fairness-model-governance/README.md)

- **Key metric:** min adverse impact ratio (chosen model): **0.776** (unmitigated baseline 0.599)
- **Data:** synthetic fallback (public source blocked); public sources attempted: HMDA, UCI German Credit; reachable: none
- **Tests (static):** 18 functions, 18 cases (18 stated in README)
- **Lines of code:** 1,714 Python source, 198 tests, 0 SQL
- **CI:** success
- **Tools:** fairlearn, lightgbm, matplotlib, numba, numpy, pandas, scikit-learn, scipy, shap

## Acceptance criteria not met

- adverse impact ratio >= 0.80 for every group: best mitigated model 0.776; flagged not ready
