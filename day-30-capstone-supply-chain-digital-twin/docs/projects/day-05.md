# Day 05: City 311 Data Quality and Service Hotspots

**Tier I - Advanced** | **Domain:** Public sector | [Folder on GitHub](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-05-civic-311-data-quality-hotspots/) | [README](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-05-civic-311-data-quality-hotspots/README.md)

- **Key metric:** share of rows quarantined: **8.7%** (defect classes fully caught 15)
- **Data:** synthetic fallback (public source blocked); public sources attempted: Chicago 311, NYC 311, SF 311; reachable: none
- **Tests (static):** 19 functions, 19 cases
- **Lines of code:** 1,879 Python source, 230 tests, 0 SQL
- **CI:** success
- **Tools:** duckdb, geopandas, great-expectations, h3, matplotlib, numpy, pandas, pandera, pyproj, rapidfuzz, scipy, shapely

## Caveats

- One storm day missed on total volume
