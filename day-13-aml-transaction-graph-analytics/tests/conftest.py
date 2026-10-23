from __future__ import annotations

import pandas as pd
import pytest

from aml_graph.config import HOUR, SMALL
from aml_graph.simulate import simulate


@pytest.fixture(scope="session")
def small_bank():
    return simulate(SMALL)


def make_tx(rows: list[tuple]) -> pd.DataFrame:
    """rows: (src, dst, hours, amount)."""
    df = pd.DataFrame(rows, columns=["src", "dst", "h", "amount"])
    df["ts"] = df["h"] * HOUR
    df["tx_id"] = range(len(df))
    df["is_laundering"] = False
    return df[["tx_id", "src", "dst", "ts", "amount", "is_laundering"]]


@pytest.fixture
def toy_tx() -> pd.DataFrame:
    rows = []
    # cycle 0 -> 1 -> 2 -> 0 (10h apart, small fee)
    rows += [(0, 1, 0, 50_000), (1, 2, 10, 49_000), (2, 0, 20, 48_000)]
    # mule chain 10 -> 11 -> 12 -> 13 -> 14
    rows += [(10, 11, 100, 20_000), (11, 12, 105, 19_500), (12, 13, 110, 19_000), (13, 14, 115, 18_600)]
    # scatter-gather 20 -> {21,22,23,24} -> 25
    for i, m in enumerate((21, 22, 23, 24)):
        rows += [(20, m, 200 + i, 10_000), (m, 25, 210 + i, 9_500)]
    # fan-in {30..35} -> 36 -> 37
    for i, s in enumerate(range(30, 36)):
        rows.append((s, 36, 300 + i, 9_000))
    rows.append((36, 37, 320, 50_000))
    # fan-out 40 -> 41 -> {42..47}
    rows.append((40, 41, 400, 60_000))
    for i, r in enumerate(range(42, 48)):
        rows.append((41, r, 401 + i, 9_000))
    # noise: slow hop (beyond 72 h) and a big amount jump should not form flow links
    rows += [(50, 51, 500, 10_000), (51, 52, 600, 9_900), (53, 54, 700, 10_000), (54, 55, 701, 2_000)]
    return make_tx(rows)
