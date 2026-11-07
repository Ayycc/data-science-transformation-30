import ibis
import pandas as pd
import pyarrow as pa
import pytest

from legacy28.generator import as_of_close
from legacy28.io import extracts_to_arrow
from legacy28.legacy import run_legacy, sas_round_div
from legacy28.modern import COMPAT, TARGET, attribution_steps, half_away_div, half_up_div, quarterly_report, run_modern
from legacy28.reconcile import KEYS, diff_cells


@pytest.mark.parametrize("close_q", [3, 8])
def test_compat_equals_legacy(extracts, close_q):
    ex = as_of_close(extracts, close_q)
    legacy = run_legacy(ex, close_q)
    modern = run_modern(extracts_to_arrow(ex), close_q, COMPAT)
    for t in KEYS:
        assert diff_cells(legacy[t], modern[t], KEYS[t]).empty, t


@pytest.mark.parametrize("cfg_name", ["COMPAT", "D4", "D2", "D3", "D1", "D5", "D6"])
def test_same_ibis_code_identical_on_duckdb_and_polars(sources, cfg_name):
    cfg = dict(attribution_steps())[cfg_name]
    a = run_modern(sources, 8, cfg, "duckdb")
    b = run_modern(sources, 8, cfg, "polars")
    for t in KEYS:
        pd.testing.assert_frame_equal(a[t], b[t])


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_integer_rounding_matches_legacy_on_both_backends(backend):
    nums = [-7, -5, -3, -1, 0, 1, 3, 5, 7, 10_001, 14_999, 15_000, 25_000]
    dens = [2, 2, 2, 2, 2, 2, 2, 2, 2, 10_000, 10_000, 10_000, 10_000]
    t = ibis.memtable(pa.table({"n": nums, "d": dens}))
    out = ibis.polars.connect() if backend == "polars" else ibis.duckdb.connect()
    res = out.to_pyarrow(t.select(h=half_away_div(t.n, t.d))).column("h").to_pylist()
    assert res == [sas_round_div(n, d) for n, d in zip(nums, dens)]
    pos = out.to_pyarrow(t.filter(t.n >= 0).select(h=half_up_div(t.n, t.d))).column("h").to_pylist()
    assert pos == [sas_round_div(n, d) for n, d in zip(nums, dens) if n >= 0]


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_d6_zero_earned_loss_ratio(backend):
    """A cell with claims but no earned premium: legacy IFERROR -> 0, target -> NULL."""
    prem = ibis.memtable(
        pa.table({"txn_id": [1], "quarter": [5], "policy_id": [1], "lob": ["HOME"], "region": ["EAST"], "written_cents": [100], "earned_ucents": [1_000_000]})
    )
    clm = ibis.memtable(
        pa.table(
            {
                "claim_id": [1],
                "quarter": [5],
                "policy_id": [2],
                "lob": ["MOTOR"],
                "region": ["EAST"],
                "paid_cents": [50],
                "case_reserve_cents": [10],
                "incurred_cum_cents": [60],
                "incurred_cents": [60],
            }
        )
    )
    con = ibis.polars.connect() if backend == "polars" else ibis.duckdb.connect()
    for cfg, expected in [(COMPAT, 0), (TARGET, None)]:
        r = con.to_pyarrow(quarterly_report(prem, clm, cfg)).to_pandas()
        motor = r[r.lob == "MOTOR"].iloc[0]
        assert pd.isna(motor.loss_ratio_bp) if expected is None else motor.loss_ratio_bp == expected
        assert r[r.lob == "HOME"].iloc[0].loss_ratio_bp == 0


def test_target_reallocates_but_preserves_premium(sources):
    a = run_modern(sources, 8, COMPAT)["report"]
    b = run_modern(sources, 8, TARGET)["report"]
    assert int(a.written_premium_cents.sum()) == int(b.written_premium_cents.sum())
    assert abs(int(a.earned_premium_cents.sum()) - int(b.earned_premium_cents.sum())) <= len(a) + 0.5 * len(sources["premium_txns"]) * 8
    assert int(b.incurred_cents.sum()) >= int(a.incurred_cents.sum())  # D3 restores paid that legacy dropped
    assert b.loc[b.lob == "OTHER", "earned_premium_cents"].sum() < a.loc[a.lob == "OTHER", "earned_premium_cents"].sum()  # D4
