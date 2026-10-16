from __future__ import annotations

from civic311 import categories as C
from civic311.config import REVIEWED_MAPPING


def test_normalise():
    assert C.normalise("  STREET LT OUT ") == "street light out"
    assert C.normalise("Sewer Cave-In Insp") == "sewer cave in inspection"


def test_auto_mapping_never_wrong_on_catalog():
    cat = C.variant_catalog()
    p = cat.merge(C.propose_mapping(cat.raw_value), on="raw_value")
    auto = p[p.method.isin(["exact", "normalised", "fuzzy_auto"])]
    assert (auto.proposed_category == auto.true_category).all()
    assert not p[p.kind == "junk"].method.isin(["exact", "normalised", "fuzzy_auto"]).any()


def test_reviewed_table_covers_review_queue_and_is_correct():
    cat = C.variant_catalog()
    reviewed = C.load_reviewed(REVIEWED_MAPPING)
    m = C.build_mapping(cat.raw_value, reviewed).merge(cat, on="raw_value")
    assert m.category.isna().sum() == (m.kind == "junk").sum()
    real = m[m.kind != "junk"]
    assert (real.category == real.true_category).all()
    assert set(reviewed.decision) <= {"accept", "override", "reject"}


def test_unreviewed_value_is_left_unmapped():
    m = C.build_mapping(["Hole in road", "Tree Trim"], None)
    assert m.set_index("raw_value").category.to_dict() == {"Hole in road": None, "Tree Trim": "Tree Trim"}
