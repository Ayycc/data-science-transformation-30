import copy

import yaml

import check_contracts
from mesh import contracts as C
from mesh import diff, scenarios


def _write(tmp_path, name, c):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(c, sort_keys=False))
    return p


def test_deliberate_breaking_change_is_caught_by_ci(tmp_path, capsys):
    """Acceptance: rename a column, bump only the minor -> the CI script exits non-zero."""
    old = C.current("payments")
    new = copy.deepcopy(old)
    C.columns(new)["amount"]["name"] = "amount_usd"
    new["version"] = "1.3.0"
    rc = check_contracts.main(["--old", str(_write(tmp_path, "old.yaml", old)), "--new", str(_write(tmp_path, "new.yaml", new))])
    out = capsys.readouterr().out
    assert rc == 1
    assert "column_removed" in out and "major" in out


def test_major_bump_with_dual_publish_passes(tmp_path):
    old = C.current("payments")
    new = copy.deepcopy(old)
    C.columns(new)["amount"]["name"] = "amount_usd"
    new["version"] = "2.0.0"
    C.obj(new)["physicalName"] = "fct_payments_v2"
    new["customProperties"] = [
        {
            "property": "dualPublish",
            "value": [{"version": old["version"], "physicalName": "fct_payments_v1", "deprecationDate": "2027-01-31"}],
        }
    ]
    rc = check_contracts.main(["--old", str(_write(tmp_path, "o.yaml", old)), "--new", str(_write(tmp_path, "n.yaml", new))])
    assert rc == 0


def test_release_history_passes_gate():
    assert all(r["passed"] for r in scenarios.replay_history())


def test_all_labelled_scenarios_classified_correctly():
    bad = [r for r in scenarios.run_suite() if not r["correct"]]
    assert not bad, bad


def test_widening_rules():
    assert diff._widened("integer", "bigint")
    assert diff._widened("decimal(12,2)", "decimal(18,2)")
    assert not diff._widened("decimal(12,2)", "decimal(12,0)")
    assert not diff._widened("bigint", "integer")


def test_working_tree_passes_git_mode():
    report = check_contracts.check_git("HEAD")
    assert report["passed"], report
