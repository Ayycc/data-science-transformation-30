"""CI gate: fail the build when a contract changes in a breaking way without a major version bump.

Usage
-----
    python scripts/check_contracts.py --base origin/main        # compare working tree with a git ref
    python scripts/check_contracts.py --old a.yaml --new b.yaml  # compare two contract files

Checks (git mode, for each domain):
1. the current contract validates against ``contracts/odcs-lite.schema.json`` + semantic rules;
2. versus the base ref: semver gate (``mesh.diff.gate``);
3. released history files (``contracts/*/history/*``) are immutable - edits are rejected;
4. the current contract equals its own history file (every release is recorded);
5. the generated dbt ``_contract.yml`` is in sync with the contract.

Exit code 0 = all checks pass, 1 = at least one failure. A JSON report is written with ``--report``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mesh import contracts as C  # noqa: E402
from mesh import diff  # noqa: E402
from mesh.config import CONTRACTS, PRODUCTS, ROOT  # noqa: E402


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def _show(ref: str, path: Path) -> str | None:
    top = Path(_git("rev-parse", "--show-toplevel").stdout.strip())
    rel = path.resolve().relative_to(top)
    r = _git("show", f"{ref}:{rel.as_posix()}")
    return r.stdout if r.returncode == 0 else None


def check_pair(old: dict, new: dict) -> dict:
    return diff.gate(old, new).to_dict()


def check_git(base: str) -> dict:
    if _git("rev-parse", "--verify", base).returncode != 0:
        return {"base": base, "skipped": f"git ref {base!r} not found", "passed": True, "domains": []}
    out = {"base": base, "domains": [], "passed": True}
    for domain in PRODUCTS:
        cur_path = C.current_path(domain)
        cur = C.load(cur_path)
        entry: dict = {"domain": domain, "version": cur["version"], "errors": []}
        entry["errors"] += [f"schema: {e}" for e in C.validate(cur)]
        old_text = _show(base, cur_path)
        if old_text is None:
            entry["gate"] = "new data product (no base contract)"
        else:
            g = diff.gate(C.loads(old_text), cur)
            entry["gate"] = g.to_dict()
            entry["errors"] += g.errors
        for hf in sorted((CONTRACTS / domain / "history").glob("*.odcs.yaml")):
            base_text = _show(base, hf)
            if base_text is not None and base_text != hf.read_text():
                entry["errors"].append(f"released history file edited: {hf.name} (history is immutable)")
        hist = CONTRACTS / domain / "history" / f"{PRODUCTS[domain]}-{cur['version']}.odcs.yaml"
        if not hist.exists() or C.load(hist) != cur:
            entry["errors"].append(f"current contract {cur['version']} must be recorded verbatim as {hist.relative_to(ROOT)}")
        yml = C.dbt_yaml_path(domain)
        if not yml.exists() or yml.read_text() != C.render_dbt_yaml(cur):
            entry["errors"].append(f"{yml.relative_to(ROOT)} out of sync with contract (run `make contracts`)")
        entry["passed"] = not entry["errors"]
        out["passed"] &= entry["passed"]
        out["domains"].append(entry)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=None, help="git ref to compare against (git mode)")
    ap.add_argument("--old", type=Path)
    ap.add_argument("--new", type=Path)
    ap.add_argument("--report", type=Path)
    a = ap.parse_args(argv)
    if a.old and a.new:
        report = check_pair(C.load(a.old), C.load(a.new))
    else:
        report = check_git(a.base or "HEAD")
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(report, indent=2))
    ok = report["passed"]
    if "domains" in report:
        for d in report["domains"]:
            print(f"[{'PASS' if d['passed'] else 'FAIL'}] {d['domain']} {d['version']}")
            for e in d["errors"]:
                print(f"    - {e}")
        if report.get("skipped"):
            print(f"[SKIP] {report['skipped']}")
    else:
        print(
            f"[{'PASS' if ok else 'FAIL'}] {report['domain']} {report['old_version']} -> {report['new_version']} (required bump: {report['required_bump']})"
        )
        for c in report["changes"]:
            print(f"    {c['level']:<5} {c['kind']:<24} {c['path']}  {c['detail']}")
        for e in report["errors"]:
            print(f"    ERROR {e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
