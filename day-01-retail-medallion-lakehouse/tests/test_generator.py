import hashlib
import json
from pathlib import Path

import pandas as pd

from retail_lakehouse.config import SAMPLE, SAMPLE_DIR, Scale
from retail_lakehouse.generate import generate

TINY = Scale(name="tiny", n_lines=1500, n_customers=200, n_products=60, n_stores=120)


def _digest(folder: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(folder.rglob("*.parquet")):
        h.update(f.name.encode())
        h.update(pd.read_parquet(f).to_csv(index=False).encode())
    return h.hexdigest()


def test_generator_is_deterministic(tmp_path: Path) -> None:
    a = generate(tmp_path / "a", TINY, seed=7)
    b = generate(tmp_path / "b", TINY, seed=7)
    assert a == b
    assert _digest(tmp_path / "a") == _digest(tmp_path / "b")


def test_committed_sample_is_reproducible(tmp_path: Path) -> None:
    regenerated = generate(tmp_path / "s", SAMPLE)
    committed = json.loads((SAMPLE_DIR / "manifest.json").read_text())
    assert regenerated == committed


def test_sample_contains_injected_mess() -> None:
    m = json.loads((SAMPLE_DIR / "manifest.json").read_text())
    for key in ["duplicate_rows_injected", "late_pos_orders_injected", "training_lines_injected",
                "unmapped_sku_lines_injected", "pos_rows_with_clock_drift", "sku_recodes", "renames",
                "price_changes", "return_duplicate_rows_injected", "customer_duplicate_rows_injected"]:  # fmt: skip
        assert m[key] > 0, key


def test_sample_is_small() -> None:
    total = sum(f.stat().st_size for f in SAMPLE_DIR.rglob("*") if f.is_file())
    assert total <= 1_000_000


def test_pos_export_is_messy_text() -> None:
    pos = pd.concat(pd.read_parquet(f) for f in sorted((SAMPLE_DIR / "pos_order_lines").glob("*.parquet")))
    assert pd.api.types.is_string_dtype(pos["unit_price"]) and pd.api.types.is_string_dtype(pos["txn_ts"])
    assert pos["order_line_id"].duplicated().any()  # re-sent lines
    assert (pos["sku"].dropna() != pos["sku"].dropna().str.upper().str.strip()).any()  # casing / spaces
