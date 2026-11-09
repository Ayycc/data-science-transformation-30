"""Scan all day-* folders and write PORTFOLIO_REPORT.md, results/portfolio/*, docs/ and mkdocs.yml.

Offline: reads committed files only (CI status from data/sample/ci_snapshot.json).
Usage: python scripts/build_portfolio.py [--build-site]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from portfolio import report, scan  # noqa: E402


def main() -> None:
    res = scan.scan_repo(ROOT.parent, ROOT / "data" / "sample" / "ci_snapshot.json")
    days = res["days"]
    s = report.summarise(days, res["ci_meta"])
    if s["problems"]:
        raise SystemExit(f"registry problems: {s['problems']}")
    md = report.render_report(days, s)
    (ROOT / "PORTFOLIO_REPORT.md").write_text(md)
    report.write_tools_matrix(days, ROOT / "results" / "portfolio" / "tools_matrix.csv")
    report.write_summary(s, ROOT / "results" / "portfolio" / "summary.json")
    report.write_docs(days, s, md, ROOT / "docs", ROOT / "mkdocs.yml")
    print(
        f"portfolio: {s['projects']} projects, {s['distinct_tools_excluding_test_lint']} tools, "
        f"{s['test_functions_static']} test functions, {s['loc_python_code'] + s['loc_python_tests'] + s['loc_sql']:,} LOC, "
        f"{s['unmet_criteria']} unmet criteria"
    )
    if "--build-site" in sys.argv:
        subprocess.run(
            [sys.executable, "-m", "mkdocs", "build", "--strict", "--quiet", "-f", str(ROOT / "mkdocs.yml")], check=True, cwd=ROOT
        )
        print("docs site built in site/ (git-ignored)")


if __name__ == "__main__":
    main()
