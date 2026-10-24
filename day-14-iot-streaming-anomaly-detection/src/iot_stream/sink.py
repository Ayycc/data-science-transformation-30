"""DuckDB sinks: metrics logging, and an exactly-once (transactional checkpoint) demo.

Exactly-once here means *effectively once* for the sink table: window rows, the source offset and
the windower's state are committed in ONE DuckDB transaction. After a crash the job restores the
state and offset from the last commit and replays from there, so every window row lands exactly
once. The naive at-least-once sink (rows written as they fire, offset checkpointed separately)
duplicates the rows fired between the last checkpoint and the crash.
"""

from __future__ import annotations

import pickle
from collections.abc import Iterator

import duckdb
import pandas as pd

from .config import StreamConfig
from .simulate import SensorStream
from .windowing import EventTimeWindower

WINDOW_COLS = ["machine", "window_end", "vib_count", "vib_mean", "temp_count", "temp_mean", "cur_count", "cur_mean", "vib_max"]


class SimulatedCrash(RuntimeError):
    pass


def _rows(records: list[dict]) -> list[tuple]:
    return [tuple(r[c] for c in WINDOW_COLS) for r in records]


def _batches(stream: SensorStream, batch_size: int, start: int) -> Iterator[tuple[int, tuple]]:
    for i in range(start, len(stream), batch_size):
        j = min(i + batch_size, len(stream))
        yield i, (stream.machine[i:j].tolist(), stream.sensor[i:j].tolist(), stream.event_time[i:j].tolist(), stream.value[i:j].tolist())


def init_sink(con: duckdb.DuckDBPyConnection, with_pk: bool) -> None:
    pk = ", PRIMARY KEY (machine, window_end)" if with_pk else ""
    con.execute("DROP TABLE IF EXISTS sink_windows")
    con.execute("DROP TABLE IF EXISTS sink_checkpoint")
    con.execute(
        "CREATE TABLE sink_windows (machine INTEGER, window_end DOUBLE, vib_count INTEGER, vib_mean DOUBLE, temp_count INTEGER,"
        f" temp_mean DOUBLE, cur_count INTEGER, cur_mean DOUBLE, vib_max DOUBLE{pk})"
    )
    con.execute("CREATE TABLE sink_checkpoint (id INTEGER PRIMARY KEY, next_offset BIGINT, state BLOB)")


def run_transactional(
    con: duckdb.DuckDBPyConnection, stream: SensorStream, cfg: StreamConfig, checkpoint_every: int, crash_after_batch: int | None = None
) -> int:
    """Process (or resume) the stream with transactional checkpoints. Returns number of batches processed in this attempt."""
    row = con.execute("SELECT next_offset, state FROM sink_checkpoint WHERE id = 1").fetchone()
    if row is None:
        w, start = EventTimeWindower(cfg.slide_s, cfg.sliding_size_s, cfg.allowed_lateness_s), 0
    else:
        start, w = int(row[0]), pickle.loads(row[1])
    buffer: list[dict] = []
    done = 0

    def commit(next_offset: int) -> None:
        con.execute("BEGIN TRANSACTION")
        if buffer:
            con.executemany(f"INSERT INTO sink_windows VALUES ({', '.join('?' * len(WINDOW_COLS))})", _rows(buffer))
        con.execute("INSERT OR REPLACE INTO sink_checkpoint VALUES (1, ?, ?)", [next_offset, pickle.dumps(w)])
        con.execute("COMMIT")
        buffer.clear()

    for off, (m, s, et, v) in _batches(stream, cfg.batch_size, start):
        w.process_batch(m, s, et, v, offset=off)
        buffer.extend(w.advance())
        done += 1
        if crash_after_batch is not None and done == crash_after_batch:
            raise SimulatedCrash(f"crash after batch {done} (offset {off})")
        if done % checkpoint_every == 0:
            commit(off + len(m))
    buffer.extend(w.flush())
    commit(len(stream))
    return done


def run_at_least_once(
    con: duckdb.DuckDBPyConnection, stream: SensorStream, cfg: StreamConfig, checkpoint_every: int, crash_after_batch: int | None = None
) -> int:
    """Naive sink: rows are written as soon as windows fire; offset+state checkpoint separately."""
    row = con.execute("SELECT next_offset, state FROM sink_checkpoint WHERE id = 1").fetchone()
    if row is None:
        w, start = EventTimeWindower(cfg.slide_s, cfg.sliding_size_s, cfg.allowed_lateness_s), 0
    else:
        start, w = int(row[0]), pickle.loads(row[1])
    done = 0
    ins = f"INSERT INTO sink_windows VALUES ({', '.join('?' * len(WINDOW_COLS))})"
    for off, (m, s, et, v) in _batches(stream, cfg.batch_size, start):
        w.process_batch(m, s, et, v, offset=off)
        fired = w.advance()
        if fired:
            con.executemany(ins, _rows(fired))
        done += 1
        if crash_after_batch is not None and done == crash_after_batch:
            raise SimulatedCrash(f"crash after batch {done}")
        if done % checkpoint_every == 0:
            con.execute("INSERT OR REPLACE INTO sink_checkpoint VALUES (1, ?, ?)", [off + len(m), pickle.dumps(w)])
    fired = w.flush()
    if fired:
        con.executemany(ins, _rows(fired))
    return done


def exactly_once_demo(stream: SensorStream, cfg: StreamConfig, checkpoint_every: int = 4, crash_after_batch: int = 7) -> dict:
    """Clean run vs crash+restart for both sink designs; compare against the clean result."""
    out: dict = {"checkpoint_every_batches": checkpoint_every, "crash_after_batch": crash_after_batch, "batch_size": cfg.batch_size}
    con = duckdb.connect()
    init_sink(con, with_pk=True)
    run_transactional(con, stream, cfg, checkpoint_every)
    clean = con.execute("SELECT * FROM sink_windows ORDER BY machine, window_end").df()
    out["clean_rows"] = len(clean)

    for name, runner, pk in (("transactional", run_transactional, True), ("at_least_once", run_at_least_once, False)):
        con = duckdb.connect()
        init_sink(con, with_pk=pk)
        try:
            runner(con, stream, cfg, checkpoint_every, crash_after_batch)
        except SimulatedCrash:
            pass
        runner(con, stream, cfg, checkpoint_every)  # restart from the last checkpoint
        got = con.execute("SELECT * FROM sink_windows ORDER BY machine, window_end").df()
        dup = int(con.execute("SELECT COUNT(*) - COUNT(DISTINCT (machine, window_end)) FROM sink_windows").fetchone()[0])
        uniq = got.drop_duplicates(["machine", "window_end"]).reset_index(drop=True)
        out[name] = {
            "rows_after_restart": len(got),
            "duplicate_rows": dup,
            "missing_rows": int(len(clean) - len(uniq)),
            "identical_to_clean_run": bool(len(got) == len(clean) and _frames_equal(got, clean)),
        }
    return out


def _frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    a = a.sort_values(["machine", "window_end"]).reset_index(drop=True)
    b = b.sort_values(["machine", "window_end"]).reset_index(drop=True)
    return bool(((a.fillna(-1) - b.fillna(-1)).abs() < 1e-9).all().all())
