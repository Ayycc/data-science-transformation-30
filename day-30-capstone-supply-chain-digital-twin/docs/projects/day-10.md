# Day 10: Air Quality Intelligence with Dagster Software-Defined Assets

**Tier II - Expert** | **Domain:** Environment & climate | [Folder on GitHub](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-10-air-quality-dagster-assets/) | [README](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-10-air-quality-dagster-assets/README.md)

- **Key metric:** alert precision (pipeline): **0.966** (false alerts withheld on unit-bug day 130)
- **Data:** synthetic fallback (public source blocked); public sources attempted: OpenAQ; reachable: none
- **Tests (static):** 20 functions, 20 cases
- **Lines of code:** 1,901 Python source, 345 tests, 0 SQL
- **CI:** success
- **Tools:** dagster, dagster-duckdb, dagster-webserver, duckdb, matplotlib, numpy, pandas, pandera, plotly, pyarrow

## Caveats

- Five drifting sensors missed on the last day
