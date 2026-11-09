# Day 16: Insurance Claims Severity: Full MLOps Lifecycle

**Tier III - Principal** | **Domain:** Insurance | [Folder on GitHub](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-16-insurance-claims-mlops/) | [README](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-16-insurance-claims-mlops/README.md)

- **Key metric:** \|A/E - 1\| after retraining: **4.3%** (\|A/E - 1\| static model 11.1%)
- **Data:** synthetic fallback (public source blocked); public sources attempted: OpenML freMTPL2; reachable: none
- **Tests (static):** 22 functions, 36 cases (36 stated in README)
- **Lines of code:** 1,526 Python source, 284 tests, 0 SQL
- **CI:** success
- **Tools:** evidently, fastapi, httpx, lightgbm, matplotlib, mlflow, numpy, pandas, pyarrow, pydantic, scikit-learn, scipy, uvicorn

## Caveats

- Docker build and GitHub Actions deploy not run end to end
