from dataclasses import replace

import numpy as np

from twin30 import kpis, promo
from twin30.calibrate import sample_outages
from twin30.config import HISTORICAL_ASSIGNMENT, Decisions
from twin30.twin import Scenario, StartState, TwinSim, simulate


def _scen(dec=None, mode="none", forced=(), horizon=56):
    z = np.zeros((horizon, 3))
    return Scenario(dec or Decisions(), 56, horizon, z, z, mode, forced_outages=list(forced))


def test_twin_is_deterministic_for_a_seed(cal, oracle_posterior, snapshot):
    st = StartState.from_snapshot(snapshot)
    a = simulate(cal, oracle_posterior, _scen(), st, seed=7)
    b = simulate(cal, oracle_posterior, _scen(), st, seed=7)
    assert a == b


def test_zero_demand_means_no_sales_and_no_stockouts(cal, oracle_posterior, snapshot):
    st = StartState.from_snapshot(snapshot)
    sim = TwinSim(cal, oracle_posterior, _scen(), st, np.random.default_rng(1), 0)
    sim.demand[:] = 0
    led = sim.run()
    assert led.sales.sum() == 0 and led.stockout.sum() == 0


def test_twin_store_inventory_balance(cal, oracle_posterior, snapshot):
    st = StartState.from_snapshot(snapshot)
    sim = TwinSim(cal, oracle_posterior, _scen(), st, np.random.default_rng(2), 1)
    led = sim.run()
    shipped = led.ship.sum()
    pipeline_in = sum(v.sum() for _, v in st.store_pipeline)
    # everything that started on hand or in transit, plus shipments, is sold, on hand or still in transit
    lhs = st.store_on_hand.sum() + pipeline_in + shipped
    rhs = led.sales.sum() + sim.store_oh.sum() + sim.in_transit.sum()
    assert np.allclose(sim.store_oh, led.store_oh[-1])
    assert np.isclose(lhs, rhs)


def test_forced_outage_delays_offshore_receipts(cal, oracle_posterior, snapshot):
    st = StartState.from_snapshot(snapshot)
    calm = TwinSim(cal, oracle_posterior, _scen(), st, np.random.default_rng(3), 0)
    calm.run()
    hit = TwinSim(cal, oracle_posterior, _scen(forced=[(0, 0, 56)]), st, np.random.default_rng(3), 0)
    hit.run()
    assert hit.receipts[0].sum() < calm.receipts[0].sum()


def test_more_safety_stock_raises_fill_rate(cal, oracle_posterior, snapshot):
    st = StartState.from_snapshot(snapshot)
    lo = [simulate(cal, oracle_posterior, _scen(replace(Decisions(), store_z=0.5)), st, seed=s)["fill_rate"] for s in range(3)]
    hi = [simulate(cal, oracle_posterior, _scen(replace(Decisions(), store_z=3.0)), st, seed=s)["fill_rate"] for s in range(3)]
    assert np.mean(hi) > np.mean(lo)


def test_opening_dc_east_cuts_outbound_freight(cal, oracle_posterior, snapshot):
    st = StartState.from_snapshot(snapshot)
    east = replace(Decisions(), open_dc_east=True, assignment=(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2))
    a = simulate(cal, oracle_posterior, _scen(), st, seed=4)
    b = simulate(cal, oracle_posterior, _scen(east), st, seed=4)
    assert b["outbound_transport_cost"] < a["outbound_transport_cost"]
    assert b["dc_fixed_cost"] > a["dc_fixed_cost"]


def test_kpis_on_a_hand_built_ledger():
    T = 364
    z = np.zeros((T, 12, 3))
    sales = z.copy()
    sales[:, 0, 0] = 10
    led = kpis.Ledger(
        sales=sales,
        stockout=z,
        store_oh=z,
        dc_oh=np.zeros((T, 3, 3)),
        ship=z,
        assign=np.array(HISTORICAL_ASSIGNMENT),
        promo=np.zeros((T, 3)),
        receipts=np.zeros((2, 3)),
        open_dcs=np.array([True, True, False]),
    )
    k = kpis.compute(led)
    assert k["units_sold"] == 3640
    assert np.isclose(k["revenue"], 3640 * 4.0)
    assert np.isclose(k["gross_margin"], 3640 * (4.0 - 2.6))
    assert np.isclose(k["operating_profit"], k["gross_margin"] - 3_600_000)
    assert k["in_stock_rate"] == 1.0


def test_outage_sampler_uses_calibrated_rates(cal):
    rng = np.random.default_rng(0)
    n = [len(sample_outages(cal, 364, rng)) for _ in range(400)]
    expected = sum(a / b for a, b in zip(cal.outage_shape, cal.outage_rate))
    assert abs(np.mean(n) - expected) < 0.35


def test_targeted_promo_rule_skips_unprofitable_weeks_and_never_repeats():
    value = np.full((52, 2), -1.0)
    value[[10, 11, 30], 0] = [5, 4, 3]
    w = promo.targeted_rule(value, [3, 3])
    assert w[:, 1].sum() == 0  # negative value everywhere: no promos
    assert w[10, 0] == 1 and w[11, 0] == 0 and w[30, 0] == 1
