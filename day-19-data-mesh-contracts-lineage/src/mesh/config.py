"""Shared constants and paths. Everything is seeded and anchored to fixed dates so runs are reproducible."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
SAMPLE = DATA / "sample"
WAREHOUSE = DATA / "warehouse"
CONTRACTS = ROOT / "contracts"
DBT = ROOT / "dbt"
RESULTS = ROOT / "results"

SEED = 20261029
START = datetime(2026, 7, 1)
N_DAYS = 90
# The moment SLOs are evaluated: 06:00 on the morning after the last simulated day.
AS_OF = START + timedelta(days=N_DAYS, hours=6)

DOMAINS = ("orders", "payments", "logistics")
PRODUCTS = {"orders": "fct_orders", "payments": "fct_payments", "logistics": "fct_shipments"}
DBT_PROJECTS = ("orders", "payments", "logistics", "analytics")

OL_NAMESPACE = "duckdb://mesh-warehouse"
OL_FILE_NAMESPACE = "file://data/raw"
OL_JOB_NAMESPACE = "data-mesh"
OL_PRODUCER = "https://github.com/Ayycc/data-science-transformation-30/tree/main/day-19-data-mesh-contracts-lineage"
