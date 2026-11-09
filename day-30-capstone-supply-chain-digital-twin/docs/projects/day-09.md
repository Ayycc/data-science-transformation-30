# Day 09: Clickstream Sessionisation at Scale with PySpark + Delta Lake

**Tier II - Expert** | **Domain:** Media & e-commerce | [Folder on GitHub](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-09-clickstream-spark-delta/) | [README](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-09-clickstream-spark-delta/README.md)

- **Key metric:** exact session recovery rate: **99.995%** (AQE skew-join speed-up 1.76x)
- **Data:** synthetic by design
- **Tests (static):** 17 functions, 17 cases
- **Lines of code:** 1,358 Python source, 224 tests, 0 SQL
- **CI:** success
- **Tools:** chispa, delta-spark, matplotlib, numpy, pandas, pyarrow, pyspark

## Caveats

- Structured Streaming / change-data-feed stretch goals not attempted
