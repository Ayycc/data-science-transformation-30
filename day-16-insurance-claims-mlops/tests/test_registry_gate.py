from __future__ import annotations

import numpy as np
import pytest

from claims_mlops import config as C
from claims_mlops import data as D
from claims_mlops import models as M
from claims_mlops import registry as R
from claims_mlops.modelcard import write_model_card


def _cand(dev=80.0, gini=0.3, ae=1.0, null=90.0):
    return R.Candidate("run", "9", "gbm", {"tweedie_deviance": dev, "gini": gini, "ae_ratio": ae, "null_deviance": null})


@pytest.mark.parametrize(
    ("cand", "champ", "passed"),
    [
        (_cand(dev=80.0), {"tweedie_deviance": 81.0}, True),
        (_cand(dev=81.0), {"tweedie_deviance": 81.0}, False),  # must strictly beat the champion
        (_cand(dev=82.0), {"tweedie_deviance": 81.0}, False),
        (_cand(dev=80.0, ae=1.2), {"tweedie_deviance": 81.0}, False),  # miscalibrated
        (_cand(dev=80.0, gini=0.1), {"tweedie_deviance": 81.0}, False),  # no segmentation power
        (_cand(dev=85.0), None, True),  # no champion: compare with null model
        (_cand(dev=91.0), None, False),
    ],
)
def test_gate_rules(cand, champ, passed):
    d = R.gate(cand, champ, "1" if champ else None, C.GatePolicy())
    assert d.passed is passed


def test_registry_promotes_only_when_gate_passes(tmp_path, small_book):
    client = R.setup(tmp_path / "mlruns")
    assert C.tracking_uri(tmp_path / "mlruns").startswith("sqlite:///")
    train, valid = D.split_months(small_book, range(1, 11)), D.split_months(small_book, [11, 12])

    good_model = M.fit(M.build_glm(), train)
    good = R.log_candidate(good_model, "glm", {"alpha": 1e-3}, train, valid)
    d1 = R.gate(good, None, None, C.GatePolicy())
    assert d1.passed and R.apply_decision(client, d1) == "promoted"
    assert R.champion_version(client) == good.version

    weak_model = M.fit(M.build_glm(features=["Area"]), train)
    weak = R.log_candidate(weak_model, "glm", {"features": "Area"}, train, valid)
    ver, cm = R.evaluate_champion(client, valid)
    d2 = R.gate(weak, cm, ver, C.GatePolicy())
    assert not d2.passed and R.apply_decision(client, d2) == "rejected"
    assert R.champion_version(client) == good.version  # champion untouched

    table = {r["version"]: r for r in R.registry_table(client)}
    assert table[int(good.version)]["stage"] == "Production" and "champion" in table[int(good.version)]["aliases"]
    assert table[int(weak.version)]["stage"] == "Archived" and table[int(weak.version)]["gate"] == "failed"

    loaded = R.load_version()  # models:/<name>@champion
    assert np.allclose(loaded.predict(valid.head(50)), good_model.predict(valid.head(50)))

    card = write_model_card(client, good.version, tmp_path / "card.md")
    assert card["stage"] == "Production" and card["gate_tags"]["gate"] == "passed"
    assert "Model card" in (tmp_path / "card.md").read_text()

    out = R.export_champion(tmp_path / "export")
    assert (out / "MLmodel").exists()
