import copy

from mesh import contracts as C
from mesh.config import PRODUCTS


def test_all_contracts_valid():
    for d in PRODUCTS:
        for c in C.history(d):
            assert C.validate(c) == [], (d, c["version"])


def test_current_is_latest_history():
    for d in PRODUCTS:
        assert C.current(d) == C.history(d)[-1]


def test_dbt_yaml_generated_from_contract():
    for d in PRODUCTS:
        assert C.dbt_yaml_path(d).read_text() == C.render_dbt_yaml(C.current(d))


def test_schema_rejects_missing_owner_and_bad_physical_name():
    c = copy.deepcopy(C.current("orders"))
    c["team"]["members"] = [{"username": "x", "role": "engineer"}]
    assert C.validate(c)
    c = copy.deepcopy(C.current("orders"))
    C.obj(c)["physicalName"] = "fct_orders_v1"  # major is 2
    assert any("must end with _v2" in e for e in C.validate(c))


def test_dual_published_v1_is_rendered_as_dbt_version():
    text = C.render_dbt_yaml(C.current("orders"))
    assert "deprecation_date: '2026-10-31'" in text and "order_total" in text
