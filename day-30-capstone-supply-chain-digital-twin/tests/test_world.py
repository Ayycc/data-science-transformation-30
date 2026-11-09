import gzip

import numpy as np
import pandas as pd

from twin30 import world
from twin30.config import N_CATS, N_STORES, geometry


def test_world_is_deterministic():
    a = world.simulate_world(seed=5, n_days=56)
    b = world.simulate_world(seed=5, n_days=56)
    assert np.array_equal(a.sales, b.sales)
    assert a.pos == b.pos


def test_sales_never_exceed_demand_and_stockouts_are_consistent(small_run):
    assert (small_run.sales <= small_run.demand).all()
    lost = small_run.demand > small_run.sales
    assert np.array_equal(lost, small_run.stockout)
    assert (small_run.store_oh >= 0).all() and (small_run.dc_oh >= -1e-9).all()


def test_store_inventory_balance(small_run):
    """on_hand[t] = on_hand[t-1] + arrivals[t] - sales[t] for every store x category."""
    arrivals = np.zeros_like(small_run.sales)
    cats = 3
    for _ship_day, arrive_day, _dc, s, *units in small_run.shipments:
        if 0 <= arrive_day < arrivals.shape[0]:
            arrivals[arrive_day, s] += np.array(units[:cats])
    k = 5  # skip the first days, which can receive trucks dispatched during the (unrecorded) burn-in
    lhs = small_run.store_oh[k:]
    rhs = small_run.store_oh[k - 1 : -1] + arrivals[k:] - small_run.sales[k:]
    assert np.allclose(lhs, rhs)


def test_promo_calendar_never_runs_two_weeks_in_a_row(small_run):
    weeks = small_run.promo[::7]
    assert not ((weeks[1:] > 0) & (weeks[:-1] > 0)).any()


def test_history_tables_hold_only_observable_columns(small_tables):
    assert set(small_tables) == {"sales", "shipments", "purchase_orders", "dc_inventory", "outages"}
    assert "demand" not in small_tables["sales"].columns
    s = small_tables["sales"]
    assert len(s) == 112 * N_STORES * N_CATS
    po = small_tables["purchase_orders"]
    assert (po.receipt_day > po.dispatch_day).all() and (po.dispatch_day >= po.order_day).all()


def test_transit_times_respect_geometry(small_tables):
    geo = geometry()
    sh = small_tables["shipments"]
    lag = sh.arrive_day - sh.ship_day
    assert lag.min() >= 1 and lag.max() <= geo.transit.max() + 1


def test_committed_sample_matches_generator(tmp_path):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    run = world.simulate_world()
    world.write_sample(run, tmp_path)
    for name in ("sales", "shipments", "purchase_orders", "dc_inventory", "outages"):
        committed = pd.read_csv(root / "data" / "sample" / f"{name}.csv.gz")
        fresh = pd.read_csv(tmp_path / f"{name}.csv.gz")
        pd.testing.assert_frame_equal(committed, fresh)
    total = sum(
        (root / "data" / "sample" / f"{n}.csv.gz").stat().st_size
        for n in ("sales", "shipments", "purchase_orders", "dc_inventory", "outages")
    )
    assert total <= 1_000_000
    with gzip.open(root / "data" / "sample" / "sales.csv.gz", "rt") as f:
        assert f.readline().startswith("day,date,store,category,units_sold")
