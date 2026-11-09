"""Scan every day-* folder of the repository and collect auditable facts for the retrospective.

All facts are read from committed files: README.md, requirements.txt, results/metrics.json,
tests/*.py and source files. Nothing is re-run. Test counts are static (AST), not executed.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "registry.yaml"
DAY_RE = re.compile(r"^day-(\d{2})-(.+)$")
SKIP_DIRS = {
    "data",
    "results",
    "docs",
    "site",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "target",
    "dbt_packages",
    "logs",
    "mlruns",
}
DEV_TOOLS = {"pytest", "pytest-cov", "coverage", "ruff"}

# tool -> capability family (used for the coverage matrix); unknown packages land in "other"
FAMILIES: dict[str, list[str]] = {
    "dataframes & storage": [
        "numpy",
        "pandas",
        "polars",
        "pyarrow",
        "duckdb",
        "pyspark",
        "delta-spark",
        "ibis-framework",
        "pyiceberg",
        "sqlalchemy",
        "xarray",
        "dask",
        "zarr",
        "numcodecs",
        "narwhals",
    ],
    "SQL & transformation": ["dbt-core", "dbt-duckdb", "sqlglot", "datacompy"],
    "machine learning": [
        "scikit-learn",
        "lightgbm",
        "optuna",
        "umap-learn",
        "pynndescent",
        "implicit",
        "mapie",
        "pyod",
        "river",
        "threadpoolctl",
        "joblib",
    ],
    "deep learning, NLP & retrieval": ["torch", "faiss-cpu", "spacy", "en-core-web-md", "rank-bm25"],
    "statistics & forecasting": [
        "scipy",
        "statsmodels",
        "statsforecast",
        "coreforecast",
        "utilsforecast",
        "holidays",
        "lifelines",
        "scikit-survival",
        "ruptures",
        "linearmodels",
    ],
    "causal inference": ["dowhy", "econml", "causalml", "causal-learn"],
    "Bayesian modelling": ["pymc", "pymc-marketing", "pytensor", "arviz"],
    "optimisation & simulation": ["ortools", "pulp", "simpy"],
    "graph analytics": ["networkx", "python-igraph", "pyvis"],
    "geospatial": ["geopandas", "shapely", "pyproj", "pyogrio", "h3", "osmnx", "rioxarray", "rasterio", "folium"],
    "data quality, contracts & lineage": ["pandera", "great-expectations", "jsonschema", "pydantic", "openlineage-python", "chispa"],
    "orchestration, MLOps & serving": [
        "dagster",
        "dagster-webserver",
        "dagster-duckdb",
        "mlflow",
        "evidently",
        "feast",
        "fastapi",
        "uvicorn",
        "httpx",
        "dill",
        "cloudpickle",
    ],
    "privacy & responsible AI": ["diffprivlib", "anonymeter", "fairlearn", "shap"],
    "LLM & documents": ["anthropic", "pdfplumber", "reportlab", "rapidfuzz"],
    "visualisation, apps & docs": ["matplotlib", "plotly", "streamlit", "pillow", "mkdocs"],
    "utilities": ["pyyaml", "numba", "llvmlite", "faker"],
}
FAMILY_OF = {t: f for f, ts in FAMILIES.items() for t in ts}


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirements(text: str) -> list[dict]:
    """Parse pinned requirement lines: handles comments, extras, markers, URLs and index options."""
    out = []
    for raw in text.splitlines():
        line = raw.split(" #")[0].strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        marker = None
        if ";" in line:
            line, marker = (p.strip() for p in line.split(";", 1))
        if " @ " in line:
            name, url = (p.strip() for p in line.split(" @ ", 1))
            m = re.search(r"-(\d[\w.]*)-py", url)
            out.append({"name": normalise(name), "version": m.group(1) if m else None, "marker": marker, "url": True})
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)(\[[^\]]*\])?\s*(==|>=|<=|~=|!=|>|<)?\s*([^\s,]+)?", line)
        if not m:
            continue
        out.append(
            {
                "name": normalise(m.group(1)),
                "version": m.group(4) if m.group(3) == "==" else None,
                "pinned": m.group(3) == "==",
                "marker": marker,
                "url": False,
            }
        )
    return out


def count_loc(folder: Path) -> dict[str, int]:
    """Non-blank, non-comment lines in .py (code vs tests) and .sql files, skipping data/results."""
    res = {"python_code": 0, "python_tests": 0, "sql": 0, "py_files": 0}
    for p in sorted(folder.rglob("*")):
        rel = p.relative_to(folder)
        if any(part in SKIP_DIRS for part in rel.parts[:-1]) or not p.is_file():
            continue
        if p.suffix == ".py":
            n = sum(1 for ln in p.read_text(errors="ignore").splitlines() if ln.strip() and not ln.strip().startswith("#"))
            key = "python_tests" if rel.parts[0] == "tests" else "python_code"
            res[key] += n
            res["py_files"] += 1
        elif p.suffix == ".sql":
            res["sql"] += sum(1 for ln in p.read_text(errors="ignore").splitlines() if ln.strip() and not ln.strip().startswith("--"))
    return res


def _param_cases(dec: ast.expr) -> int | None:
    """Number of cases for @pytest.mark.parametrize(names, <literal list>), else None."""
    if not (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr == "parametrize"):
        return None
    if len(dec.args) >= 2 and isinstance(dec.args[1], (ast.List, ast.Tuple)):
        return len(dec.args[1].elts)
    return -1  # parametrized with a computed iterable: count unknown


def count_tests(folder: Path) -> dict[str, int]:
    """Static count of pytest test functions and their parametrized cases (no test is executed)."""
    funcs = cases = unknown = files = 0
    tdir = folder / "tests"
    if not tdir.exists():
        return {"test_files": 0, "test_functions": 0, "test_cases_static": 0, "parametrize_unknown": 0}
    for p in sorted(tdir.rglob("*.py")):
        if not (p.name.startswith("test_") or p.name.endswith("_test.py")):
            continue
        files += 1
        tree = ast.parse(p.read_text(errors="ignore"))
        nodes: list[ast.AST] = list(tree.body)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                nodes.extend(node.body)
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                funcs += 1
                mult = 1
                for d in node.decorator_list:
                    k = _param_cases(d)
                    if k is None:
                        continue
                    if k < 0:
                        unknown += 1
                    else:
                        mult *= max(k, 1)
                cases += mult
    return {"test_files": files, "test_functions": funcs, "test_cases_static": cases, "parametrize_unknown": unknown}


def readme_sections(text: str) -> dict[str, str]:
    parts = re.split(r"^## +", text, flags=re.M)
    out = {}
    for p in parts[1:]:
        head, _, body = p.partition("\n")
        out[head.strip().lower()] = body
    return out


def readme_facts(text: str) -> dict:
    h1 = re.search(r"^# (.+)$", text, flags=re.M)
    title = h1.group(1).strip() if h1 else ""
    title = re.sub(r"^Day \d+: *", "", title)
    tier = re.search(r"\*\*(Tier [IV]+ - [A-Za-z]+)\*\*", text)
    domain = re.search(r"\*\*Domain:\*\* *([^|\n]+)", text)
    sec = readme_sections(text)
    lim = next((v for k, v in sec.items() if k.startswith("limitation")), "")
    bullets = [ln for ln in lim.splitlines() if re.match(r"^- ", ln)]
    how = next((v for k, v in sec.items() if k.startswith("how to run")), "")
    stated = None
    for ln in how.splitlines():
        if "make test" in ln:
            m = re.search(r"(\d[\d,]*) (?:pytest )?tests", ln)
            if m:
                stated = int(m.group(1).replace(",", ""))
                break
    return {
        "title": title,
        "tier": tier.group(1) if tier else "",
        "domain": domain.group(1).strip() if domain else "",
        "limitations_bullets": len(bullets),
        "readme_stated_tests": stated,
    }


def get_path(obj, path: list):
    for k in path:
        obj = obj[k]
    return obj


@dataclass
class DayFacts:
    num: str
    folder: str
    title: str
    tier: str
    domain: str
    requirements: list[dict]
    tools: list[str]
    loc: dict[str, int]
    tests: dict[str, int]
    readme_stated_tests: int | None
    limitations_bullets: int
    metric: dict
    context: dict | None
    provenance: dict
    unmet: list[dict]
    caveats: list[dict]
    offline_llm: bool
    ci: dict | None
    has_metrics_json: bool
    problems: list[str] = field(default_factory=list)


def load_registry(path: Path = REGISTRY) -> dict:
    return yaml.safe_load(path.read_text())["days"]


def discover(repo: Path) -> list[Path]:
    return sorted(p for p in repo.iterdir() if p.is_dir() and DAY_RE.match(p.name))


def scan_day(folder: Path, reg: dict, ci: dict) -> DayFacts:
    num = DAY_RE.match(folder.name).group(1)
    entry = reg.get(num, {})
    problems: list[str] = []
    readme = (folder / "README.md").read_text() if (folder / "README.md").exists() else ""
    facts = readme_facts(readme)
    req_p = folder / "requirements.txt"
    reqs = parse_requirements(req_p.read_text()) if req_p.exists() else []
    tools = sorted({r["name"] for r in reqs})
    mpath = folder / "results" / "metrics.json"
    metrics = json.loads(mpath.read_text()) if mpath.exists() else None

    def resolve(spec: dict | None) -> dict | None:
        if not spec:
            return None
        out = dict(spec)
        try:
            out["value"] = get_path(metrics, spec["path"]) if metrics is not None else None
        except (KeyError, IndexError, TypeError):
            out["value"] = None
            problems.append(f"metric path {spec['path']} not found")
        return out

    for item in [entry.get("provenance", {})] + entry.get("unmet", []) + entry.get("caveats", []):
        ev = item.get("evidence")
        if ev and ev not in readme:
            problems.append(f"evidence not in README: {ev!r}")
    unmet = list(entry.get("unmet", []))
    if metrics is not None and num == "30":
        acc = metrics.get("acceptance", {})
        if acc and not acc.get("twin_baseline_kpis_within_5pct", True):
            unmet.append(
                {"criterion": "twin reproduces baseline KPIs within 5%", "result": f"max error {acc.get('max_abs_kpi_error_pct'):.2f}%"}
            )
        if acc and not acc.get("at_least_3_scenarios", True):
            unmet.append({"criterion": ">= 3 strategic scenarios", "result": str(acc.get("strategic_scenarios_compared_with_uncertainty"))})
    return DayFacts(
        num=num,
        folder=folder.name,
        title=facts["title"],
        tier=facts["tier"],
        domain=facts["domain"],
        requirements=reqs,
        tools=tools,
        loc=count_loc(folder),
        tests=count_tests(folder),
        readme_stated_tests=facts["readme_stated_tests"],
        limitations_bullets=facts["limitations_bullets"],
        metric=resolve(entry.get("metric")) or {},
        context=resolve(entry.get("context")),
        provenance=entry.get("provenance", {"category": "unknown", "attempted": [], "reachable": []}),
        unmet=unmet,
        caveats=entry.get("caveats", []),
        offline_llm=bool(entry.get("offline_llm", False)),
        ci=ci.get(folder.name),
        has_metrics_json=metrics is not None,
        problems=problems,
    )


def scan_repo(repo: Path, ci_snapshot: Path | None = None, registry: Path = REGISTRY) -> dict:
    reg = load_registry(registry)
    ci = {}
    ci_meta = None
    if ci_snapshot and ci_snapshot.exists():
        snap = json.loads(ci_snapshot.read_text())
        ci = snap.get("days", {})
        ci_meta = {k: snap.get(k) for k in ("fetched_at", "source", "runs_scanned")}
    days = [scan_day(f, reg, ci) for f in discover(repo)]
    return {"days": days, "ci_meta": ci_meta}
