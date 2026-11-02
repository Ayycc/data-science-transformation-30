import numpy as np

from sreanomaly.generate import METRICS, SimConfig, generate
from sreanomaly.topology import SERVICES, call_graph, upstream_depths


def test_shapes_and_determinism(small_tel):
    tel = small_tel
    assert tel.values.shape == (5 * 1440, len(SERVICES), len(METRICS))
    again = generate(SimConfig(n_days=5, n_incidents=12, n_concurrent=2, n_surges=1))
    np.testing.assert_array_equal(tel.values, again.values)
    assert np.isfinite(tel.values).all()
    assert (tel.values[..., 1] >= 0).all() and (tel.values[..., 2] <= 100).all()


def test_topology_is_20_service_dag():
    g = call_graph()
    assert g.number_of_nodes() == 20
    assert upstream_depths(g, "user-db")["web"] == 3


def test_incident_ledger(small_tel):
    inc = small_tel.incidents
    assert len(inc) == 12
    assert (inc["start"] >= 2 * 1440).all()  # training window is incident-free
    assert inc["concurrent"].sum() == 4  # 2 pairs
    for r in inc.itertuples():
        assert r.affected.split(";")[0] == r.root
    kinds = set(small_tel.deploys["kind"])
    assert {"bad", "benign"} <= kinds


def test_fault_propagates_to_callers_with_lag(small_tel):
    tel = small_tel
    inc = tel.incidents
    r = inc[(inc["n_affected"] >= 3) & (inc["fault"].isin(["latency_regression", "cpu_saturation", "queue_backlog"]))].iloc[0]
    callers = [s for s in r.affected.split(";") if s != r.root]
    j = tel.services.index(callers[0])
    before = np.median(tel.values[r.start - 60 : r.start - 5, j, 0])
    during = np.median(tel.values[r.start + 5 : r.end, j, 0])
    assert during > 1.1 * before
