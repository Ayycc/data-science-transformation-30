{% docs __overview__ %}
# Retail Medallion Lakehouse (SYNTHETIC data)

Bronze (raw Parquet landed in DuckDB `bronze.*`) -> Silver (`silver.*`: typing, dedup,
timezone normalisation, late-arrival handling) -> Gold (`gold.*`: star schema and marts).
`audit.*` holds the revenue reconciliation and the data-quality scorecard. The product
dimension is SCD Type 2 via the `snap_product_catalog` dbt snapshot.
{% enddocs %}
