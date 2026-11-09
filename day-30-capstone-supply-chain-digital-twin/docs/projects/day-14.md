# Day 14: Streaming Predictive Maintenance for Manufacturing IoT

**Tier II - Expert** | **Domain:** Manufacturing | [Folder on GitHub](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-14-iot-streaming-anomaly-detection/) | [README](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-14-iot-streaming-anomaly-detection/README.md)

- **Key metric:** warning precision: **0.912** (warning recall 0.877)
- **Data:** synthetic by design; public sources attempted: NASA C-MAPSS; reachable: NASA C-MAPSS
- **Tests (static):** 20 functions, 22 cases (22 stated in README)
- **Lines of code:** 1,702 Python source, 256 tests, 0 SQL
- **CI:** success
- **Tools:** duckdb, matplotlib, narwhals, numpy, pandas, plotly, river, scipy

## Caveats

- Bytewax and a live Kafka broker not run (asyncio processor used, as the brief allows)
