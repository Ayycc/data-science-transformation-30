from legacy28.modern import attribution_steps, run_modern
from legacy28.reconcile import KEYS, affected_sets, check_step, compare_summary, datacompy_compare


def _steps(sources):
    return [(sid, run_modern(sources, 8, cfg)) for sid, cfg in attribution_steps()]


def test_datacompy_compat_matches_legacy(legacy_out, sources):
    compat = run_modern(sources, 8, dict(attribution_steps())["COMPAT"])
    for t in KEYS:
        s = compare_summary(datacompy_compare(legacy_out[t], compat[t], KEYS[t]))
        assert s["matches"] and s["unequal_cells"] == 0 and s["rows_only_legacy"] == 0, t


def test_every_step_difference_is_explained(final, sources):
    aff = affected_sets(final, 8)
    outs = _steps(sources)
    changed = 0
    for (_, prev), (sid, cur) in zip(outs, outs[1:]):
        r = check_step(sid, prev, cur, aff)
        assert sum(r.unexplained.values()) == 0, (sid, r.examples)
        changed += sum(r.changed_cells.values())
    assert changed > 0


def test_unintended_change_is_flagged(final, sources):
    """A bug slipped into the modern code (here: one claim's paid amount changed) must show up as unexplained."""
    aff = affected_sets(final, 8)
    outs = dict(_steps(sources))
    prev, cur = outs["D4"], {k: v.copy() for k, v in outs["D2"].items()}
    cur["claims_detail"].loc[0, "paid_cents"] += 1
    r = check_step("D2", prev, cur, aff)
    assert r.unexplained["claims_detail"] == 1


def test_wrong_attribution_is_flagged(final, sources):
    """Changes that belong to D1 (rounding) do not satisfy D5's predicate."""
    aff = affected_sets(final, 8)
    outs = dict(_steps(sources))
    r = check_step("D5", outs["D3"], outs["D1"], aff)
    assert sum(r.unexplained.values()) > 0
