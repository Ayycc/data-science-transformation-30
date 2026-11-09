import numpy as np
import pandas as pd

from twin30 import network_opt
from twin30.config import HISTORICAL_ASSIGNMENT
from twin30.scenarios import cvar, risk_summary


def _units(scale=1.0):
    u = np.tile(np.array([400_000.0, 140_000.0, 35_000.0]), (12, 1)) * scale
    return u


def test_milp_respects_single_sourcing_and_capacity():
    sol = network_opt.solve(_units(), allow_east=True)
    assert sol["status"] == "OPTIMAL" and len(sol["assignment"]) == 12
    load = _units().sum(axis=1) / 364
    from twin30.config import FINANCE

    for d in range(3):
        assert load[np.array(sol["assignment"]) == d].sum() <= FINANCE.dc_capacity_units_per_day[d] + 1e-6


def test_milp_without_east_never_uses_it_and_is_no_worse_than_history():
    sol = network_opt.solve(_units(), allow_east=False)
    assert 2 not in sol["assignment"]
    assert sol["objective"] <= network_opt.evaluate(_units(), HISTORICAL_ASSIGNMENT) + 1e-6


def test_milp_does_not_open_east_when_volume_is_tiny():
    sol = network_opt.solve(_units(0.01), allow_east=True)
    assert sol["open"][2] is False


def test_milp_forced_east_serves_east_stores_from_east():
    sol = network_opt.solve(_units(), force_east=True)
    assert sol["open"][2] and all(a == 2 for a in sol["assignment"][8:])


def test_cvar_matches_hand_calculation():
    x = np.arange(1, 101, dtype=float)
    assert cvar(x, 0.05) == np.mean([1, 2, 3, 4, 5])
    assert cvar(x, 0.01) == 1.0


def test_risk_summary_pairs_by_seed():
    base = pd.DataFrame(
        {
            "operating_profit": [10.0, 20, 30, 40],
            "revenue": 1.0,
            "fill_rate": 0.96,
            "in_stock_rate": 0.95,
            "avg_dc_inventory_units": 1,
            "avg_store_inventory_units": 1,
        }
    )
    alt = base.assign(operating_profit=base.operating_profit + 5)
    s = risk_summary(alt, base)
    assert s["delta_profit_vs_baseline_mean"] == 5 and s["prob_beats_baseline"] == 1.0
    assert s["delta_cvar5_vs_baseline"] == 5
