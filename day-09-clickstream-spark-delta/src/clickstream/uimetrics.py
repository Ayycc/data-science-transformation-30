"""Read stage/task and SQL metrics from the Spark UI REST API (the same numbers the Spark UI shows)."""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from pyspark.sql import SparkSession

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # localhost: never use a proxy


class SparkUI:
    def __init__(self, spark: SparkSession):
        self.sc = spark.sparkContext
        self.base = (self.sc.uiWebUrl or "").rstrip("/")
        self.app = self.sc.applicationId

    @property
    def available(self) -> bool:
        if not self.base:
            return False
        try:
            self._get("/jobs?status=succeeded")
            return True
        except Exception:
            return False

    def _get(self, path: str) -> Any:
        url = f"{self.base}/api/v1/applications/{self.app}{path}"
        with _OPENER.open(url, timeout=30) as r:
            return json.loads(r.read().decode())

    def stages_for_group(self, group: str) -> list[dict]:
        """Per-stage task statistics for all jobs run under ``sc.setJobGroup(group, ...)``."""
        jobs = [j for j in self._get("/jobs") if j.get("jobGroup") == group]
        out = []
        for sid in sorted({s for j in jobs for s in j["stageIds"]}):
            for att in self._get(f"/stages/{sid}"):
                if att.get("status") != "COMPLETE":
                    continue
                q = self._get(f"/stages/{sid}/{att['attemptId']}/taskSummary?quantiles=0.5,1.0")
                rt = q.get("executorRunTime", [0, 0])
                out.append(
                    {
                        "stage_id": sid,
                        "name": att.get("name", "")[:60],
                        "num_tasks": att["numTasks"],
                        "median_task_ms": round(rt[0], 1),
                        "max_task_ms": round(rt[1], 1),
                        "total_task_ms": att.get("executorRunTime"),
                        "shuffle_read_mb": round(att.get("shuffleReadBytes", 0) / 2**20, 2),
                        "input_records": att.get("inputRecords"),
                    }
                )
        return out

    def sql_scan_metrics(self, description: str) -> dict:
        """Sum the 'files read' style metrics of every scan node in SQL executions with this description."""
        res: dict[str, float] = {}
        offset = 0
        while True:
            page = self._get(f"/sql?details=true&planDescription=false&offset={offset}&length=200")
            if not page:
                break
            for ex in page:
                if ex.get("description") != description:
                    continue
                for node in ex.get("nodes", []):
                    if not node.get("nodeName", "").startswith("Scan"):
                        continue
                    for m in node.get("metrics", []):
                        if m["name"] in ("number of files read", "number of output rows", "size of files read"):
                            res[m["name"]] = res.get(m["name"], 0.0) + _num(m["value"])
            offset += len(page)
        return res


def _num(v: str) -> float:
    """Parse '1,234' / '12.3 MiB' / 'total (min, med, max)\\n12.3 MiB (...)' into a number (MiB for sizes)."""
    s = v.split("\n")[-1] if "\n" in v else v
    s = s.split("(")[0].strip().replace(",", "")
    parts = s.split()
    try:
        x = float(parts[0])
    except (ValueError, IndexError):
        return 0.0
    unit = parts[1] if len(parts) > 1 else ""
    scale = {"B": 1 / 2**20, "KiB": 1 / 1024, "MiB": 1.0, "GiB": 1024.0}.get(unit, 1.0)
    return x * scale
