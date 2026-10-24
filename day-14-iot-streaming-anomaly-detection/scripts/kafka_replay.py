"""OPTIONAL Kafka/Redpanda path (not used by `make run` or tests).

produce: replay the committed SYNTHETIC sample to topic ``sensor-events`` in arrival order (key = machine).
consume: poll micro-batches, feed the same EventTimeWindower, and write fired windows to DuckDB; Kafka
         offsets are committed only after the DuckDB transaction commits (at-least-once delivery, and the
         primary key makes re-inserted rows idempotent). Windower state is NOT checkpointed here, so a
         restart can emit partial windows for the panes that were open at the crash. The full exactly-once
         pattern (rows + source offset + operator state in one transaction, seek to the stored offset on
         restart) is implemented and tested broker-free in ``iot_stream.sink``. This script was not run in
         the reported build (no broker available).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import duckdb  # noqa: E402

from iot_stream.config import StreamConfig  # noqa: E402
from iot_stream.data import load_sample  # noqa: E402
from iot_stream.sink import WINDOW_COLS, init_sink  # noqa: E402
from iot_stream.windowing import EventTimeWindower  # noqa: E402

TOPIC, BROKERS = "sensor-events", "localhost:9092"


def produce() -> None:
    from confluent_kafka import Producer

    s = load_sample()
    p = Producer({"bootstrap.servers": BROKERS, "enable.idempotence": True, "linger.ms": 20})
    for m, sn, et, v in zip(s.machine.tolist(), s.sensor.tolist(), s.event_time.tolist(), s.value.tolist(), strict=True):
        p.produce(TOPIC, key=str(m), value=json.dumps({"m": m, "s": sn, "t": et, "v": v}))
        p.poll(0)
    p.flush()
    print(f"produced {len(s):,} SYNTHETIC events to {TOPIC}")


def consume(db: str = "data/kafka_sink.duckdb") -> None:
    from confluent_kafka import Consumer

    cfg = StreamConfig()
    c = Consumer({"bootstrap.servers": BROKERS, "group.id": "day14-windower", "enable.auto.commit": False, "auto.offset.reset": "earliest"})
    c.subscribe([TOPIC])
    con = duckdb.connect(db)
    init_sink(con, with_pk=True)
    w = EventTimeWindower(cfg.slide_s, cfg.sliding_size_s, cfg.allowed_lateness_s)
    idle = 0
    ins = f"INSERT OR IGNORE INTO sink_windows VALUES ({', '.join('?' * len(WINDOW_COLS))})"
    while idle < 10:
        msgs = c.consume(num_messages=cfg.batch_size, timeout=1.0)
        if not msgs:
            idle += 1
            continue
        idle = 0
        ev = [json.loads(mm.value()) for mm in msgs if mm.error() is None]
        w.process_batch([e["m"] for e in ev], [e["s"] for e in ev], [e["t"] for e in ev], [e["v"] for e in ev])
        fired = w.advance()
        con.execute("BEGIN TRANSACTION")
        if fired:
            con.executemany(ins, [tuple(r[k] for k in WINDOW_COLS) for r in fired])
        con.execute("COMMIT")
        c.commit(asynchronous=False)  # offsets only after the sink commit
    fired = w.flush()
    if fired:
        con.executemany(ins, [tuple(r[k] for k in WINDOW_COLS) for r in fired])
    print(w.counters(), con.execute("SELECT COUNT(*) FROM sink_windows").fetchone())
    c.close()


if __name__ == "__main__":
    {"produce": produce, "consume": consume}[sys.argv[1] if len(sys.argv) > 1 else "consume"]()
