import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
logging.getLogger("sqlglot").setLevel(logging.ERROR)

from semsql import generator  # noqa: E402
from semsql.guardrails import Guardrails, open_readonly  # noqa: E402
from semsql.semantic import SemanticLayer  # noqa: E402
from semsql.tools import Toolbox  # noqa: E402


@pytest.fixture(scope="session")
def tables():
    return generator.generate(generator.GenConfig(n_accounts=400))


@pytest.fixture(scope="session")
def db_path(tmp_path_factory, tables):
    p = tmp_path_factory.mktemp("wh") / "warehouse.duckdb"
    generator.write_duckdb(tables, p)
    return p


@pytest.fixture(scope="session")
def layer():
    return SemanticLayer()


@pytest.fixture(scope="session")
def guard(layer):
    return Guardrails(layer.pii_columns, layer.max_rows)


@pytest.fixture()
def con(db_path):
    c = open_readonly(db_path)
    yield c
    c.close()


@pytest.fixture()
def toolbox(layer, con):
    return Toolbox(layer, con)
