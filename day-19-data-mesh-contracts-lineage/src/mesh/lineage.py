"""Emit OpenLineage RunEvents (with column-level lineage) for ingestion, every dbt model and SLO checks.

Column lineage is derived by parsing each dbt model's *compiled* SQL with sqlglot, qualified
against the live DuckDB catalog, so it reflects what actually ran. Events are written to a JSONL
file (OpenLineage ``FileTransport`` format). If ``OPENLINEAGE_URL`` is set (for example a local
Marquez, see ``lineage/marquez/``), the same events are also sent with the OpenLineage HTTP client.
No lineage server is needed to run or test this project.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

import sqlglot
from openlineage.client.event_v2 import InputDataset, Job, OutputDataset, Run, RunEvent, RunState
from openlineage.client.facet_v2 import (
    column_lineage_dataset as cl,
)
from openlineage.client.facet_v2 import (
    data_quality_assertions_dataset as dq,
)
from openlineage.client.facet_v2 import (
    dataset_version_dataset,
    documentation_dataset,
    job_type_job,
    ownership_dataset,
    schema_dataset,
    sql_job,
)
from openlineage.client.serde import Serde
from sqlglot import exp
from sqlglot.lineage import lineage as sqlglot_lineage

from mesh import contracts as C
from mesh.config import AS_OF, DOMAINS, OL_FILE_NAMESPACE, OL_JOB_NAMESPACE, OL_NAMESPACE, OL_PRODUCER, SEED
from mesh.warehouse import RAW_TABLE

_AGG = (exp.AggFunc,)
_UUID_NS = uuid.UUID(int=SEED)


def _run_id(job: str) -> str:
    return str(uuid.uuid5(_UUID_NS, job))


def _schema_facet(cols: dict[str, str]) -> schema_dataset.SchemaDatasetFacet:
    return schema_dataset.SchemaDatasetFacet(fields=[schema_dataset.SchemaDatasetFacetFields(name=c, type=t) for c, t in cols.items()])


def column_lineage(sql: str, out_cols: list[str], schema: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """For each output column: list of {dataset, field, subtype} source columns (direct lineage)."""
    out: dict[str, list[dict[str, str]]] = {}
    parsed = sqlglot.parse_one(sql, dialect="duckdb")
    for col in out_cols:
        try:
            node = sqlglot_lineage(col, parsed, schema=schema, dialect="duckdb")
        except Exception:  # noqa: BLE001 - a column without traceable lineage (e.g. count(*)) is simply skipped
            out[col] = []
            continue
        root_expr = node.expression
        if root_expr.find(*_AGG):
            subtype = "AGGREGATION"
        elif isinstance(root_expr, exp.Alias) and isinstance(root_expr.this, exp.Column) and root_expr.this.name == col:
            subtype = "IDENTITY"
        elif isinstance(root_expr, exp.Column):
            subtype = "IDENTITY"
        elif (
            isinstance(root_expr, exp.Alias)
            and isinstance(root_expr.this, exp.Cast)
            and isinstance(root_expr.this.this, exp.Column)
            and root_expr.this.this.name == col
        ):
            subtype = "IDENTITY"  # same-name type cast
        else:
            subtype = "TRANSFORMATION"
        seen: set[tuple[str, str]] = set()
        for n in node.walk():
            if isinstance(n.expression, exp.Table):
                t = n.expression
                ds = ".".join(p for p in (t.catalog, t.db, t.name) if p)
                field = n.name.split(".")[-1]
                if (ds, field) not in seen:
                    seen.add((ds, field))
                    out.setdefault(col, []).append({"dataset": ds, "field": field, "subtype": subtype})
        out.setdefault(col, [])
    return out


def indirect_lineage(sql: str, schema: dict[str, Any]) -> list[dict[str, str]]:
    """Columns of *physical* tables that affect every output row: JOIN keys, FILTERs, GROUP BYs, WINDOWs.

    Emitted as the OpenLineage ``columnLineage.dataset`` list (INDIRECT transformations).
    Columns reached only through a CTE/subquery alias are not resolved (see README limitations).
    """
    from sqlglot.optimizer.qualify import qualify

    q = qualify(sqlglot.parse_one(sql, dialect="duckdb"), schema=schema, dialect="duckdb", validate_qualify_columns=False)
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for sel in q.find_all(exp.Select):
        alias_map: dict[str, str] = {}
        for src in [sel.args.get("from_") or sel.args.get("from")] + list(sel.args.get("joins") or []):
            t = getattr(src, "this", None)
            if isinstance(t, exp.Table) and t.catalog:
                alias_map[t.alias_or_name] = ".".join([t.catalog, t.db, t.name])

        def collect(node: exp.Expression | None, subtype: str, alias_map: dict[str, str] = alias_map) -> None:
            if node is None:
                return
            for c in node.find_all(exp.Column):
                if c.table in alias_map and (alias_map[c.table], c.name, subtype) not in seen:
                    seen.add((alias_map[c.table], c.name, subtype))
                    out.append({"dataset": alias_map[c.table], "field": c.name, "subtype": subtype})

        for j in sel.args.get("joins") or []:
            collect(j.args.get("on"), "JOIN")
        collect(sel.args.get("where"), "FILTER")
        collect(sel.args.get("group"), "GROUP_BY")
        for w in sel.find_all(exp.Window):
            collect(w, "WINDOW")
    return out


def _cl_facet(
    lin: dict[str, list[dict[str, str]]], namespace_of, indirect: list[dict[str, str]] | None = None
) -> cl.ColumnLineageDatasetFacet:
    fields = {}
    for col, srcs in lin.items():
        if not srcs:
            continue
        fields[col] = cl.Fields(
            inputFields=[
                cl.InputField(
                    namespace=namespace_of(s["dataset"]),
                    name=s["dataset"],
                    field=s["field"],
                    transformations=[cl.Transformation(type="DIRECT", subtype=s["subtype"])],
                )
                for s in srcs
            ]
        )
    ds_level = [
        cl.InputField(
            namespace=namespace_of(s["dataset"]),
            name=s["dataset"],
            field=s["field"],
            transformations=[cl.Transformation(type="INDIRECT", subtype=s["subtype"])],
        )
        for s in indirect or []
    ]
    return cl.ColumnLineageDatasetFacet(fields=fields, dataset=ds_level)


def _ns(dataset: str) -> str:
    return OL_FILE_NAMESPACE if dataset.endswith(".csv") else OL_NAMESPACE


def _events(job_name: str, t0, inputs, outputs, job_facets) -> list[RunEvent]:
    job = Job(namespace=OL_JOB_NAMESPACE, name=job_name, facets=job_facets)
    run = Run(runId=_run_id(job_name))
    return [
        RunEvent(
            eventType=RunState.START, eventTime=t0.isoformat() + "Z", run=run, job=job, inputs=inputs, outputs=outputs, producer=OL_PRODUCER
        ),
        RunEvent(
            eventType=RunState.COMPLETE,
            eventTime=(t0 + timedelta(seconds=1)).isoformat() + "Z",
            run=run,
            job=job,
            inputs=inputs,
            outputs=outputs,
            producer=OL_PRODUCER,
        ),
    ]


def build_events(manifests: dict[str, dict[str, Any]], schema: dict[str, Any], slo_reports: list[dict[str, Any]]) -> list[RunEvent]:
    events: list[RunEvent] = []
    t = AS_OF
    contracts_by_iface = {}
    for d in DOMAINS:
        c = C.current(d)
        for phys, ic, dep in C.published_interfaces(c):
            contracts_by_iface[f"{d}.main.{phys}"] = (ic, dep, c)

    def cols_of(ds: str) -> dict[str, str]:
        db, sch, tbl = ds.split(".")
        return schema.get(db, {}).get(sch, {}).get(tbl, {})

    def out_facets(ds: str, lin, indirect=None) -> dict[str, Any]:
        f: dict[str, Any] = {"schema": _schema_facet(cols_of(ds)), "columnLineage": _cl_facet(lin, _ns, indirect)}
        if ds in contracts_by_iface:
            ic, dep, cur = contracts_by_iface[ds]
            f["version"] = dataset_version_dataset.DatasetVersionDatasetFacet(datasetVersion=ic["version"])
            f["ownership"] = ownership_dataset.OwnershipDatasetFacet(owners=[ownership_dataset.Owner(name=C.owner(cur), type="owner")])
            doc = ic["description"]["purpose"] + (f" DEPRECATED: retired after {dep}." if dep else "")
            f["documentation"] = documentation_dataset.DocumentationDatasetFacet(description=doc)
        return f

    # 1) ingestion: raw files -> <domain>.raw.<table> (identity column lineage)
    for d in DOMAINS:
        tbl = RAW_TABLE[d]
        ds_out = f"{d}.raw.{tbl}"
        ocols = cols_of(ds_out)
        file_ds = f"{d}/{d}_*.csv"
        fcols = {c: "VARCHAR" for c in ocols if c != "_source_file"}
        lin = {c: [{"dataset": file_ds, "field": c, "subtype": "IDENTITY"}] for c in fcols}
        inputs = [InputDataset(namespace=OL_FILE_NAMESPACE, name=file_ds, facets={"schema": _schema_facet(fcols)})]
        outputs = [OutputDataset(namespace=OL_NAMESPACE, name=ds_out, facets=out_facets(ds_out, lin))]
        jf = {"jobType": job_type_job.JobTypeJobFacet(processingType="BATCH", integration="DUCKDB", jobType="INGEST")}
        events += _events(f"{d}.ingest_raw", t, inputs, outputs, jf)
        t += timedelta(seconds=5)

    # 2) dbt models of every project, in manifest dependency order (domains first, consumer last)
    for project, m in manifests.items():
        nodes = {**m["nodes"], **m["sources"]}

        def rel(uid: str, nodes: dict[str, Any] = nodes) -> str:
            return nodes[uid]["relation_name"].replace('"', "")

        models = [n for n in m["nodes"].values() if n["resource_type"] == "model"]
        order = _topo(models)
        for n in order:
            ds_out = rel(n["unique_id"])
            sql = n.get("compiled_code") or ""
            lin = column_lineage(sql, list(cols_of(ds_out)), schema)
            indirect = indirect_lineage(sql, schema)
            in_ds = sorted({rel(u) for u in n["depends_on"]["nodes"] if u in nodes})
            inputs = [InputDataset(namespace=OL_NAMESPACE, name=i, facets={"schema": _schema_facet(cols_of(i))}) for i in in_ds]
            outputs = [OutputDataset(namespace=OL_NAMESPACE, name=ds_out, facets=out_facets(ds_out, lin, indirect))]
            jf = {
                "jobType": job_type_job.JobTypeJobFacet(processingType="BATCH", integration="DBT", jobType="MODEL"),
                "sql": sql_job.SQLJobFacet(query=sql, dialect="duckdb"),
            }
            vsuffix = f".v{n['version']}" if n.get("version") is not None else ""
            events += _events(f"{project}.{n['name']}{vsuffix}", t, inputs, outputs, jf)
            t += timedelta(seconds=5)

    # 3) SLO checks as data-quality assertions on each published data product
    for r in slo_reports:
        ds = f"{r['domain']}.main.{r['data_product']}"
        assertions = [
            dq.Assertion(assertion=f"sla_{c['sla']}", success=c["status"] == "PASS", expected=str(c["target"]), actual=str(c["observed"]))
            for c in r["checks"]
        ]
        inputs = [
            InputDataset(
                namespace=OL_NAMESPACE,
                name=ds,
                facets={"schema": _schema_facet(cols_of(ds))},
                inputFacets={"dataQualityAssertions": dq.DataQualityAssertionsDatasetFacet(assertions=assertions)},
            )
        ]
        jf = {"jobType": job_type_job.JobTypeJobFacet(processingType="BATCH", integration="PYTHON", jobType="SLO_CHECK")}
        events += _events(f"{r['domain']}.slo_check", t, inputs, [], jf)
        t += timedelta(seconds=5)
    return events


def _topo(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {m["unique_id"]: m for m in models}
    done: list[str] = []

    def visit(uid: str) -> None:
        if uid in done or uid not in by_id:
            return
        for d in by_id[uid]["depends_on"]["nodes"]:
            visit(d)
        done.append(uid)

    for uid in sorted(by_id):
        visit(uid)
    return [by_id[u] for u in done]


def write_events(events: list[RunEvent], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for e in events:
            f.write(json.dumps(Serde.to_dict(e), sort_keys=True) + "\n")
    if os.environ.get("OPENLINEAGE_URL"):
        from openlineage.client import OpenLineageClient

        client = OpenLineageClient()  # configured from OPENLINEAGE_URL / OPENLINEAGE_API_KEY
        for e in events:
            client.emit(e)
    return len(events)


def read_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
