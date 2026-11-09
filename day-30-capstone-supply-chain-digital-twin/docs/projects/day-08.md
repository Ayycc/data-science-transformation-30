# Day 08: Telecom Retention: Survival Analysis + Uplift Modelling

**Tier II - Expert** | **Domain:** Telecommunications | [Folder on GitHub](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-08-telecom-churn-survival-uplift/) | [README](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-08-telecom-churn-survival-uplift/README.md)

- **Key metric:** uplift-policy net value (true): **$500,314** (churn-score policy net value $359,254)
- **Data:** synthetic by design
- **Tests (static):** 15 functions, 15 cases (15 stated in README)
- **Lines of code:** 1,173 Python source, 181 tests, 0 SQL
- **CI:** success
- **Tools:** causalml, lifelines, matplotlib, numpy, pandas, pyyaml, scikit-learn, scikit-survival, scipy

## Caveats

- Uplift random forest and DR policy tree did not beat the churn score (X-learner did)
