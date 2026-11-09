# Day 15: Bike-Share Demand and Fleet Rebalancing

**Tier II - Expert** | **Domain:** Urban mobility & transport | [Folder on GitHub](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-15-bikeshare-geospatial-rebalancing/) | [README](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-15-bikeshare-geospatial-rebalancing/README.md)

- **Key metric:** stockout-hour reduction vs fixed routes: **45.9%** (station-hour MAE (LightGBM) 0.663)
- **Data:** public data (headline); public sources attempted: Capital Bikeshare, NOAA ISD, GBFS, OSM; reachable: Capital Bikeshare, NOAA ISD
- **Tests (static):** 13 functions, 14 cases (14 stated in README)
- **Lines of code:** 1,742 Python source, 205 tests, 0 SQL
- **CI:** success
- **Tools:** folium, geopandas, h3, lightgbm, matplotlib, networkx, numpy, ortools, osmnx, pandas, pillow, pyogrio, pyproj, scikit-learn, scipy, shapely

## Caveats

- GBFS and OSM unreachable: dock capacities estimated, straight-line drive times
