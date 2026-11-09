# Day 17: Point-in-Time Correct Feature Store for Ride-Hailing ETA

**Tier III - Principal** | **Domain:** Mobility marketplace | [Folder on GitHub](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-17-ridehail-feature-store-pit/) | [README](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-17-ridehail-feature-store-pit/README.md)

- **Key metric:** naive join offline-to-production MAE gap: **26.9%** (PIT model production MAE (min) 2.159)
- **Data:** synthetic fallback (public source blocked); public sources attempted: NYC TLC; reachable: none
- **Tests (static):** 16 functions, 16 cases (16 stated in README)
- **Lines of code:** 1,257 Python source, 209 tests, 0 SQL
- **CI:** success
- **Tools:** dill, duckdb, feast, ibis-framework, lightgbm, matplotlib, numpy, pandas, pyarrow
