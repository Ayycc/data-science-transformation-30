"""Storage resources. Assets depend on the abstract interfaces; the deployment picks local or cloud.

``AQ_STORAGE=local`` (default) reads the landing zone from disk and writes reports to ``results/``.
``AQ_STORAGE=s3`` reads/writes the same layout in an S3 bucket via boto3 (not exercised offline).
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pandas as pd
from dagster import ConfigurableResource


class LandingStorage(ConfigurableResource):
    """Where sensor batches land: ``landing/<YYYY-MM-DD>/batch_NNN.parquet`` + ``reference/stations.parquet``."""

    def list_partitions(self) -> list[str]:
        raise NotImplementedError

    def list_batches(self, partition: str) -> list[tuple[str, int]]:
        """(name, size_bytes) of every batch file of one partition, sorted by name."""
        raise NotImplementedError

    def read_batch(self, partition: str, name: str) -> pd.DataFrame:
        raise NotImplementedError

    def read_registry(self) -> pd.DataFrame:
        raise NotImplementedError

    def fingerprint(self, partition: str) -> str:
        """Content fingerprint of a partition's landing files; changes when a late batch lands."""
        batches = self.list_batches(partition)
        if not batches:
            return ""
        return hashlib.sha1("|".join(f"{n}:{s}" for n, s in batches).encode()).hexdigest()[:12]


class LocalLandingStorage(LandingStorage):
    base_dir: str
    # Partitions whose first read raises a simulated transient error (exercises the RetryPolicy).
    transient_failures: list[str] = []

    def _root(self) -> Path:
        return Path(self.base_dir)

    def list_partitions(self) -> list[str]:
        d = self._root() / "landing"
        return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []

    def list_batches(self, partition: str) -> list[tuple[str, int]]:
        d = self._root() / "landing" / partition
        if not d.exists():
            return []
        return [(p.name, p.stat().st_size) for p in sorted(d.glob("batch_*.parquet"))]

    def read_batch(self, partition: str, name: str) -> pd.DataFrame:
        if partition in self.transient_failures:
            marker = self._root() / ".transient" / partition
            if not marker.exists():
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.touch()
                raise ConnectionError(f"simulated transient storage timeout reading {partition}/{name}")
        return pd.read_parquet(self._root() / "landing" / partition / name)

    def read_registry(self) -> pd.DataFrame:
        return pd.read_parquet(self._root() / "reference" / "stations.parquet")


class S3LandingStorage(LandingStorage):
    """Same layout under ``s3://bucket/prefix``. Requires ``boto3`` and credentials from the environment."""

    bucket: str
    prefix: str = "air-quality"
    endpoint_url: str | None = None

    def _client(self):
        import boto3  # optional dependency, only needed for the cloud deployment

        return boto3.client("s3", endpoint_url=self.endpoint_url)

    def _list(self, prefix: str) -> list[dict]:
        out: list[dict] = []
        for page in self._client().get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix):
            out.extend(page.get("Contents", []))
        return out

    def list_partitions(self) -> list[str]:
        keys = [o["Key"] for o in self._list(f"{self.prefix}/landing/")]
        return sorted({k.split("/")[-2] for k in keys if k.endswith(".parquet")})

    def list_batches(self, partition: str) -> list[tuple[str, int]]:
        objs = self._list(f"{self.prefix}/landing/{partition}/")
        return sorted((o["Key"].rsplit("/", 1)[-1], int(o["Size"])) for o in objs if o["Key"].endswith(".parquet"))

    def _read(self, key: str) -> pd.DataFrame:
        body = self._client().get_object(Bucket=self.bucket, Key=key)["Body"].read()
        return pd.read_parquet(io.BytesIO(body))

    def read_batch(self, partition: str, name: str) -> pd.DataFrame:
        return self._read(f"{self.prefix}/landing/{partition}/{name}")

    def read_registry(self) -> pd.DataFrame:
        return self._read(f"{self.prefix}/reference/stations.parquet")


class ReportStore(ConfigurableResource):
    """Where published reports (markdown, CSV, figures) are written."""

    def write_text(self, name: str, text: str) -> str:
        raise NotImplementedError

    def write_bytes(self, name: str, data: bytes) -> str:
        raise NotImplementedError

    def write_csv(self, name: str, df: pd.DataFrame) -> str:
        return self.write_text(name, df.to_csv(index=False))


class LocalReportStore(ReportStore):
    base_dir: str

    def _path(self, name: str) -> Path:
        p = Path(self.base_dir) / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write_text(self, name: str, text: str) -> str:
        p = self._path(name)
        p.write_text(text)
        return str(p)

    def write_bytes(self, name: str, data: bytes) -> str:
        p = self._path(name)
        p.write_bytes(data)
        return str(p)


class S3ReportStore(ReportStore):
    bucket: str
    prefix: str = "air-quality/results"
    endpoint_url: str | None = None

    def write_bytes(self, name: str, data: bytes) -> str:
        import boto3

        key = f"{self.prefix}/{name}"
        boto3.client("s3", endpoint_url=self.endpoint_url).put_object(Bucket=self.bucket, Key=key, Body=data)
        return f"s3://{self.bucket}/{key}"

    def write_text(self, name: str, text: str) -> str:
        return self.write_bytes(name, text.encode())
