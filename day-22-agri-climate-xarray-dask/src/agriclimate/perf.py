"""Performance instrumentation: memory sampling, task-graph size, chunk-size experiments, and the out-of-core run.

The out-of-core run executes in a child process whose address space is hard-capped with RLIMIT_AS, so "larger
than available memory" is enforced by the kernel rather than asserted. Only Dask's local threaded scheduler
is used: no distributed cluster, no worker processes.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

import dask
import numpy as np


def rss_mb() -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return float("nan")


class MemorySampler:
    """Background thread sampling this process's RSS every `interval` seconds."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append((time.perf_counter() - self._t0, rss_mb()))
            self._stop.wait(self.interval)

    def __enter__(self) -> MemorySampler:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()

    @property
    def peak_mb(self) -> float:
        return max((m for _, m in self.samples), default=rss_mb())

    def mark(self) -> float:
        return time.perf_counter() - self._t0


def graph_size(*arrays) -> dict:
    """Number of tasks in the Dask graph before and after Dask's graph optimisation (culling + fusion)."""
    raw = sum(len(a.__dask_graph__()) for a in arrays)
    opt = dask.optimize(*arrays)
    return {"tasks_raw": int(raw), "tasks_optimized": int(sum(len(a.__dask_graph__()) for a in opt))}


def chunk_info(da) -> dict:
    cs = da.data.chunksize
    return {
        "chunk_shape": [int(c) for c in cs],
        "chunk_mb": float(np.prod(cs) * da.dtype.itemsize / 2**20),
        "n_chunks": int(np.prod(da.data.numblocks)),
        "array_mb": float(da.nbytes / 2**20),
    }


CHUNKINGS: dict[str, dict | str] = {
    "tiny": {"time": 73, "lat": 20, "lon": 20},
    "stored": "stored",
    "large": {"time": 1460, "lat": 80, "lon": 120},
}


def benchmark(spec, zarr_path: Path, work_dir: Path, workers: int, seed: int, runs: list[tuple[str, str]], cap_gb: float = 64.0) -> list[dict]:
    """Each (chunking, method) GDD benchmark in a fresh child process, so memory numbers are not polluted by the parent."""
    out = []
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    for chunking, method in runs:
        r = run_stage(
            "bench", spec, zarr_path, Path(work_dir) / f"bench_{chunking}_{method}.json", cap_gb, workers, seed,
            extra=["--chunking", chunking, "--method", method],
        )  # fmt: skip
        r.pop("rss_trace", None)
        r["peak_rss_increase_mb"] = r["peak_rss_mb"] - r["rss_before_compute_mb"]
        out.append(r)
    return out


# ---------------- out-of-core run (child process) ----------------


def _child(args: argparse.Namespace) -> None:
    """One capped stage: 'write' the cube, attempt a naive in-memory 'load', or run the chunked 'compute'."""
    cap = int(args.cap_gb * 2**30)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    # Under an address-space cap, each thread's default 8 MB stack reservation counts; keep them small and few.
    threading.stack_size(4 * 2**20)
    import zarr

    zarr.config.set({"threading.max_workers": args.workers})
    from .config import CubeSpec
    from .generate import write_cube
    from .indices import annual_precip, compute_climate_features, open_cube

    spec = CubeSpec(args.n_lat, args.n_lon, tuple(range(args.y0, args.y1 + 1)), args.ct, args.cl, args.cn)
    report: dict = {"stage": args.stage, "cap_gb": args.cap_gb, "num_workers": args.workers}
    with MemorySampler(0.1) as ms:
        t0 = time.perf_counter()
        if args.stage == "write":
            report |= write_cube(spec, Path(args.zarr), args.seed)
            report["status"] = "succeeded"
        elif args.stage == "load":
            ds = open_cube(args.zarr)
            report["cube_gb_uncompressed"] = float(ds.nbytes / 2**30)
            try:
                with dask.config.set(scheduler="threads", num_workers=args.workers):
                    ds.load()
                report["status"] = "succeeded"
            except MemoryError as exc:
                report["status"] = "MemoryError"
                report["message"] = str(exc)[:160]
        elif args.stage == "compute":
            ds = open_cube(args.zarr)
            report["cube_gb_uncompressed"] = float(ds.nbytes / 2**30)
            report["chunk"] = chunk_info(ds["tmax"])
            feats, _, tasks = compute_climate_features(ds, num_workers=args.workers)
            annual_p = annual_precip(ds)
            tasks["precip_annual"] = int(len(annual_p.data.__dask_graph__()))
            with dask.config.set(scheduler="threads", num_workers=args.workers):
                annual_p = annual_p.compute()
            report["graph_tasks_per_feature"] = tasks
            report["result_mb"] = float((feats.nbytes + annual_p.nbytes) / 2**20)
            report["mean_gdd"] = float(feats["gdd"].mean())
            report["mean_heat_days"] = float(feats["heat_days"].mean())
            report["mean_precip_jja_mm"] = float(feats["precip_jja"].mean())
            report["mean_precip_annual_mm"] = float(annual_p.mean())
            report["status"] = "succeeded"
        elif args.stage == "bench":
            from .indices import daily_gdd, seasonal_sum, seasonal_sum_resample

            ds = open_cube(args.zarr, chunks=CHUNKINGS[args.chunking])
            g = daily_gdd(ds["tmax"], ds["tmin"])
            arr = (seasonal_sum if args.method == "dot" else seasonal_sum_resample)(g, (4, 9))
            report |= chunk_info(ds["tmax"]) | graph_size(arr.data) | {"chunking": args.chunking, "method": args.method}
            report["rss_before_compute_mb"] = rss_mb()
            t0 = time.perf_counter()
            with dask.config.set(scheduler="threads", num_workers=args.workers):
                res = arr.compute()
            report["checksum"] = float(res.sum())
        report["wall_s"] = time.perf_counter() - t0
    # Sampled RSS only: getrusage's ru_maxrss is inherited from the parent across fork/exec on Linux.
    report["peak_rss_mb"] = float(ms.peak_mb)
    step = max(1, len(ms.samples) // 400)
    report["rss_trace"] = [(round(t, 2), round(m, 1)) for t, m in ms.samples[::step]]
    Path(args.out).write_text(json.dumps(report, indent=1))


def run_stage(
    stage: str, spec, zarr_path: Path, out_json: Path, cap_gb: float, workers: int, seed: int, timeout: int = 1500, extra: list[str] | None = None
) -> dict:
    """Run one stage in a fresh child process with RLIMIT_AS = cap_gb and return its report."""
    env = dict(os.environ)
    # Fixed mmap threshold so freed chunk buffers go back to the OS and RSS tracks live data.
    env.update(
        {
            "MALLOC_ARENA_MAX": "2",
            "MALLOC_MMAP_THRESHOLD_": "262144",
            "MALLOC_TRIM_THRESHOLD_": "262144",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    src = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable, "-W", "ignore", "-m", "agriclimate.perf",
        "--stage", stage,
        "--zarr", str(zarr_path),
        "--out", str(out_json),
        "--cap-gb", str(cap_gb),
        "--workers", str(workers),
        "--seed", str(seed),
        "--n-lat", str(spec.n_lat),
        "--n-lon", str(spec.n_lon),
        "--y0", str(spec.years[0]),
        "--y1", str(spec.years[-1]),
        "--ct", str(spec.chunk_time),
        "--cl", str(spec.chunk_lat),
        "--cn", str(spec.chunk_lon),
    ] + (extra or [])  # fmt: skip
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"out-of-core stage {stage} failed ({proc.returncode}):\n{proc.stderr[-3000:]}")
    return json.loads(Path(out_json).read_text())


def run_out_of_core(spec, zarr_path: Path, work_dir: Path, cap_gb: float, workers: int, seed: int) -> dict:
    """Write a cube bigger than the cap, show a naive load fails, then compute the indices chunk by chunk."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for stage in ("write", "load", "compute"):
        out[stage] = run_stage(stage, spec, zarr_path, work_dir / f"ooc_{stage}.json", cap_gb, workers, seed)
    cube_gb = out["compute"]["cube_gb_uncompressed"]
    out["summary"] = {
        "cap_gb": cap_gb,
        "cube_gb_uncompressed": cube_gb,
        "cube_to_cap_ratio": cube_gb / cap_gb,
        "disk_gb": out["write"]["disk_bytes"] / 2**30,
        "naive_load_status": out["load"]["status"],
        "chunked_compute_status": out["compute"]["status"],
        "chunked_peak_rss_mb": out["compute"]["peak_rss_mb"],
        "chunked_peak_rss_fraction_of_cube": out["compute"]["peak_rss_mb"] / (cube_gb * 1024),
        "write_wall_s": out["write"]["wall_s"],
        "compute_wall_s": out["compute"]["wall_s"],
        "write_peak_rss_mb": out["write"]["peak_rss_mb"],
    }
    return out


def _parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["write", "load", "compute", "bench"], required=True)
    p.add_argument("--chunking", default="stored", choices=list(CHUNKINGS))
    p.add_argument("--method", default="dot", choices=["dot", "resample"])
    for a in ("--zarr", "--out"):
        p.add_argument(a, required=True)
    p.add_argument("--cap-gb", type=float, required=True)
    for a in ("--workers", "--seed", "--n-lat", "--n-lon", "--y0", "--y1", "--ct", "--cl", "--cn"):
        p.add_argument(a, type=int, required=True)
    return p.parse_args(argv)


if __name__ == "__main__":
    _child(_parse())
