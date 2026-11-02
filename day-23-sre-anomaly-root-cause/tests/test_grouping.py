import math

import pandas as pd

from sreanomaly.evaluate import detection_metrics, match_groups
from sreanomaly.grouping import group_alerts, summarise_groups
from sreanomaly.topology import call_graph


def _ep(rows):
    return pd.DataFrame(rows, columns=["service", "start", "end", "peak"])


def test_topological_vs_temporal_grouping():
    g = call_graph()
    ep = _ep([("order-db", 100, 120, 3), ("order", 101, 120, 2), ("checkout", 103, 118, 2), ("user-db", 104, 110, 2)])
    topo = group_alerts(ep, g, window=10, max_hops=2)
    assert topo["group"].nunique() == 2  # user-db is 4 hops from order-db/order/checkout chain -> separate
    temporal = group_alerts(ep, g, window=10, max_hops=math.inf)
    assert temporal["group"].nunique() == 1


def test_window_splits_far_apart_alerts():
    g = call_graph()
    ep = _ep([("order", 100, 105, 2), ("order", 200, 205, 2)])
    assert group_alerts(ep, g)["group"].nunique() == 2


def test_matching_and_metrics():
    g = call_graph()
    ep = _ep([("order-db", 100, 120, 3), ("order", 101, 120, 2), ("search", 500, 503, 2)])
    summ = summarise_groups(group_alerts(ep, g))
    inc = pd.DataFrame([dict(incident_id=0, start=98, end=125, affected="order-db;order;checkout;api-gateway;mobile-bff;web")])
    m = match_groups(summ, inc)
    d = detection_metrics(m, inc)
    assert d["precision"] == 0.5 and d["recall"] == 1.0 and d["delay_median_min"] == 2.0
