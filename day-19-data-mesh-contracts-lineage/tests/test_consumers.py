from mesh import consumers


def test_versioned_products_shield_dashboards():
    r = consumers.replay()
    assert r["naive_broken_dashboard_days"] > 0
    assert r["mesh_broken_dashboard_days"] == 0


def test_referenced_columns():
    assert consumers.referenced_columns("select a, sum(b) from t where c = 1 group by 1") == {"a", "b", "c"}
