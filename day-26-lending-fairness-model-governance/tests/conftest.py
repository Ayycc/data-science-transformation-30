import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit26.data import SYNTH_SPEC, CreditData, generate_applications, split_data  # noqa: E402
from credit26.models import GBMConfig, train_gbm  # noqa: E402


@pytest.fixture(scope="session")
def data():
    return CreditData(generate_applications(n=5000, seed=7), SYNTH_SPEC, "synthetic_generator")


@pytest.fixture(scope="session")
def parts(data):
    return split_data(data)


@pytest.fixture(scope="session")
def mono_model(parts):
    return train_gbm("m", parts.train, SYNTH_SPEC.numeric, SYNTH_SPEC, GBMConfig(monotone=True, n_estimators=120))
