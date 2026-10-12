# SCD Type 2 demonstration: a renamed and re-coded SKU

Product `1449` (SYNTHETIC) is the best-selling product whose SKU code *and* name changed. Rows below come from `gold.dim_product` (built from the `snap_product_catalog` dbt snapshot after replaying 25 monthly catalog extracts).

| product_sk | version_number | sku_code | product_name | list_price | valid_from | valid_to | is_current | captured_from_extract |
|---|---|---|---|---|---|---|---|---|
| 1449001 | 1 | SKU-101449 | Granite Bay Compact Sweater | 29.99 | 2022-04-09 20:12:39 | 2026-03-03 14:40:18 | False | 2024-10-01 00:00:00 |
| 1449002 | 2 | SKU-901449 | Granite Bay Sweater 2.0 | 29.99 | 2026-03-03 14:40:18 | 9999-12-31 00:00:00 | True | 2026-04-01 00:00:00 |

Order lines joined to each version (`gold.fct_order_lines.product_sk`, matched on the UTC sale time falling in `[valid_from, valid_to)`). `mismatches` counts lines whose receipt SKU code or list price differs from the version they were assigned to:

| version_number | sku_code | product_name | order_lines | first_sale_utc | last_sale_utc | mismatches |
|---|---|---|---|---|---|---|
| 1 | SKU-101449 | Granite Bay Compact Sweater | 17795 | 2024-10-01 07:41:24 | 2026-03-03 14:16:20 | 0 |
| 2 | SKU-901449 | Granite Bay Sweater 2.0 | 7045 | 2026-03-03 15:31:56 | 2026-10-01 04:32:07 | 0 |

Across all 1,996,860 fact lines, receipt SKU/price vs assigned version mismatches: **0** (enforced by the dbt test `assert_order_lines_match_product_version_at_sale`).
