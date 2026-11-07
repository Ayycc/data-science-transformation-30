import importlib
from pathlib import Path

from legacy28.rules import DIFFERENCES, LINEAGE, RULES, rule_map, topo_order

ROOT = Path(__file__).resolve().parents[1]


def test_dependency_graph_is_acyclic_and_complete():
    order = topo_order()
    assert sorted(order) == sorted(r.rid for r in RULES)
    pos = {r: i for i, r in enumerate(order)}
    for r in RULES:
        assert all(pos[d] < pos[r.rid] for d in r.depends_on)


def test_every_rule_points_at_real_legacy_and_modern_code():
    mod = importlib.import_module("legacy28.modern")
    for r in RULES:
        path = r.legacy_source.split(":")[0].split(" ")[0]
        assert (ROOT / "legacy" / path).exists(), r.rid
        for impl in r.modern_impl.split(" + "):
            assert hasattr(mod, impl.split(".")[1]), impl


def test_every_difference_is_owned_by_a_rule_and_lineage_is_valid():
    owned = {d for r in RULES for d in r.differences}
    assert owned == {d.did for d in DIFFERENCES}
    rm = rule_map()
    for col, (_, rules) in LINEAGE.items():
        assert all(r in rm for r in rules), col
