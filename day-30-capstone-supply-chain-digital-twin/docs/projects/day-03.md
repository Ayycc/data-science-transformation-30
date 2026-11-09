# Day 03: Hospital Readmission Risk on an OMOP-style Model

**Tier I - Advanced** | **Domain:** Healthcare | [Folder on GitHub](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-03-healthcare-readmission-omop/) | [README](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-03-healthcare-readmission-omop/README.md)

- **Key metric:** test AUROC (logistic): **0.782** (LACE baseline AUROC 0.707)
- **Data:** synthetic fallback (public source blocked); public sources attempted: Synthea; reachable: none
- **Tests (static):** 37 functions, 37 cases
- **Lines of code:** 1,985 Python source, 315 tests, 158 SQL
- **CI:** success
- **Tools:** duckdb, matplotlib, numpy, pandas, pyarrow, pydantic, scikit-learn, scipy

## Caveats

- Secondary GBM calibration-slope CI upper bound 1.217 above the 1.2 bound (primary model inside)
