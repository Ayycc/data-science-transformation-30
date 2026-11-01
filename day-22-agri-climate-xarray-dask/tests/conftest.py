import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agriclimate.config import RunConfig  # noqa: E402
from agriclimate.generate import write_cube  # noqa: E402
from agriclimate.indices import open_cube  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return RunConfig.small()


@pytest.fixture(scope="session")
def cube_path(tmp_path_factory, cfg):
    p = tmp_path_factory.mktemp("cube") / "cube.zarr"
    write_cube(cfg.cube, p, cfg.seed)
    return p


@pytest.fixture()
def cube(cube_path):
    return open_cube(str(cube_path))
