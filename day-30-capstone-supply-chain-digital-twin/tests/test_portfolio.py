import os
import subprocess
import sys
from pathlib import Path

import pytest

from portfolio import report, scan

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
HAVE_REPO = len(scan.discover(REPO)) >= 30
needs_repo = pytest.mark.skipif(not HAVE_REPO, reason="sibling day-* folders not present (folder checked out alone)")


def test_parse_requirements_handles_markers_extras_urls_and_index_options():
    text = """# comment
--extra-index-url https://example.org/simple
numpy==2.4.6
scipy==1.18.1; python_version >= "3.12"
scipy==1.17.1; python_version < "3.12"  # 1.18 has no 3.11 wheels
feast[duckdb]==0.50.0
PyYAML==6.0.3
en_core_web_md @ https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl
"""
    reqs = scan.parse_requirements(text)
    names = [r["name"] for r in reqs]
    assert names == ["numpy", "scipy", "scipy", "feast", "pyyaml", "en-core-web-md"]
    assert reqs[1]["marker"] == 'python_version >= "3.12"' and reqs[2]["version"] == "1.17.1"
    assert reqs[5]["url"] and reqs[5]["version"] == "3.8.0"


def test_static_test_count_multiplies_literal_parametrize(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text(
        "import pytest\n"
        "def test_a():\n    pass\n"
        "@pytest.mark.parametrize('x', [1, 2, 3])\ndef test_b(x):\n    pass\n"
        "@pytest.mark.parametrize('x', CASES)\ndef test_c(x):\n    pass\n"
        "class TestK:\n    def test_d(self):\n        pass\n    def helper(self):\n        pass\n"
        "def helper():\n    pass\n"
    )
    (tmp_path / "tests" / "conftest.py").write_text("def test_not_counted():\n    pass\n")
    c = scan.count_tests(tmp_path)
    assert c == {"test_files": 1, "test_functions": 4, "test_cases_static": 6, "parametrize_unknown": 1}


def test_loc_counts_code_tests_and_sql_but_skips_data(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("# c\nx = 1\n\ny = 2\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("def test():\n    assert True\n")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "big.py").write_text("x = 1\n" * 50)
    (tmp_path / "m.sql").write_text("-- c\nselect 1\nfrom t\n")
    assert scan.count_loc(tmp_path) == {"python_code": 2, "python_tests": 2, "sql": 2, "py_files": 2}


def test_readme_facts_parses_title_tier_domain_and_limitations():
    text = "# Day 07: Thing\n\n**Tier I - Advanced** | **Domain:** Retail | x\n\n## Limitations\n\n- a\n- b\n\n## How to run\n\nmake test   # 12 pytest tests\n"
    f = scan.readme_facts(text)
    assert f["title"] == "Thing" and f["tier"] == "Tier I - Advanced" and f["domain"] == "Retail"
    assert f["limitations_bullets"] == 2 and f["readme_stated_tests"] == 12


def test_formatting():
    assert report.fmt(0.9609, "f3") == "0.961"
    assert report.fmt(0.5, "pct_frac") == "50.0%"
    assert report.fmt(124647650.9, "usd") == "$124,647,651"
    assert report.fmt(0.0, "usd") == "$0.00"


@pytest.fixture(scope="module")
def repo_scan():
    return scan.scan_repo(REPO, ROOT / "data" / "sample" / "ci_snapshot.json")


@needs_repo
def test_scan_covers_all_30_days_and_registry_is_consistent(repo_scan):
    days = repo_scan["days"]
    assert [d.num for d in days] == [f"{i:02d}" for i in range(1, 31)]
    problems = {d.num: d.problems for d in days if d.problems}
    assert problems == {}, problems  # every metric path resolves and every evidence quote is in its README
    for d in days:
        assert d.has_metrics_json and d.metric.get("value") is not None, d.num
        assert d.tools and d.tests["test_functions"] > 0, d.num
        assert d.title and d.tier.startswith("Tier"), d.num


@needs_repo
def test_known_unmet_criteria_and_offline_llm_days_are_reported(repo_scan):
    days = {d.num: d for d in repo_scan["days"]}
    for n in ("06", "11", "12", "21", "24", "26", "29"):
        assert days[n].unmet, n
    assert abs(days["26"].metric["value"] - 0.776) < 0.001
    assert {n for n, d in days.items() if d.offline_llm} == {"20", "21", "29"}
    s = report.summarise(repo_scan["days"], repo_scan["ci_meta"])
    assert s["projects"] == 30 and s["unmet_criteria"] >= 7
    assert set(s["public_used_days"]) == {11, 12, 15, 23}


@needs_repo
def test_report_has_a_row_for_every_day(repo_scan):
    s = report.summarise(repo_scan["days"], repo_scan["ci_meta"])
    md = report.render_report(repo_scan["days"], s)
    for d in repo_scan["days"]:
        assert f"| {d.num} | [{d.title}](../{d.folder}/)" in md
    assert "Acceptance criteria that were not met" in md and "Test results: how they were obtained" in md


@needs_repo
def test_committed_report_is_reproducible(repo_scan):
    s = report.summarise(repo_scan["days"], repo_scan["ci_meta"])
    md = report.render_report(repo_scan["days"], s)
    assert (ROOT / "PORTFOLIO_REPORT.md").read_text() == md


@needs_repo
def test_mkdocs_site_builds_offline(repo_scan, tmp_path):
    s = report.summarise(repo_scan["days"], repo_scan["ci_meta"])
    md = report.render_report(repo_scan["days"], s)
    report.write_docs(repo_scan["days"], s, md, tmp_path / "docs", tmp_path / "mkdocs.yml")
    env = dict(os.environ)
    # any network access would fail: point every proxy at a closed local port
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY"):
        env[k] = "http://127.0.0.1:9"
    env.pop("NO_PROXY", None)
    env.pop("no_proxy", None)
    subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "--quiet", "-f", str(tmp_path / "mkdocs.yml")],
        check=True,
        cwd=tmp_path,
        env=env,
        timeout=120,
    )
    site = tmp_path / "site"
    assert (site / "index.html").exists() and (site / "portfolio-report.html").exists()
    assert len(list((site / "projects").glob("day-*.html"))) == 30
