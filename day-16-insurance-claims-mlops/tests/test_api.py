"""API contract tests (FastAPI TestClient, in-process; no server)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from claims_mlops import config as C
from claims_mlops.api import MAX_BATCH, create_app

POLICY = {
    "VehPower": 7,
    "VehAge": 3,
    "DrivAge": 42,
    "BonusMalus": 50,
    "Density": 1200,
    "Area": "D",
    "VehBrand": "B12",
    "VehGas": "Diesel",
    "Region": "R11",
    "Exposure": 0.5,
}


@pytest.fixture
def client(glm):
    with TestClient(create_app(glm, "v-test")) as c:
        yield c


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready")
    assert r.status_code == 200 and r.json() == {"ready": True, "model_version": "v-test"}
    assert client.get("/model").json()["features"] == C.FEATURES


def test_not_ready_without_model(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAIMS_MODEL_URI", str(tmp_path / "missing"))
    with TestClient(create_app()) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/ready").status_code == 503
        assert c.post("/score", json=POLICY).status_code == 503


def test_single_score_matches_model(client, glm):
    r = client.post("/score", json={**POLICY, "policy_id": "p1"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"policy_id", "pure_premium", "expected_cost", "model_version"}
    direct = glm.predict(pd.DataFrame([POLICY]))[0]
    assert body["pure_premium"] == pytest.approx(direct)
    assert body["expected_cost"] == pytest.approx(direct * 0.5)
    assert body["policy_id"] == "p1" and body["model_version"] == "v-test"


def test_exposure_defaults_to_one(client):
    p = {k: v for k, v in POLICY.items() if k != "Exposure"}
    body = client.post("/score", json=p).json()
    assert body["expected_cost"] == pytest.approx(body["pure_premium"])


def test_batch_matches_single(client, sample):
    recs = sample[C.FEATURES + ["Exposure"]].head(200).to_dict(orient="records")
    r = client.post("/score/batch", json={"policies": recs})
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 200 and len(body["scores"]) == 200
    single = client.post("/score", json=recs[7]).json()
    assert body["scores"][7]["pure_premium"] == pytest.approx(single["pure_premium"])
    assert body["total_expected_cost"] == pytest.approx(sum(s["expected_cost"] for s in body["scores"]))


@pytest.mark.parametrize(
    "patch",
    [
        {"DrivAge": 16},  # under age
        {"Area": "Z"},  # unknown enum
        {"VehGas": "Electric"},
        {"Region": "Paris"},  # pattern
        {"Exposure": 0},  # must be > 0
        {"Exposure": 1.5},
        {"BonusMalus": 20},
        {"Density": "dense"},  # type
        {"unexpected_field": 1},  # extra fields forbidden
    ],
)
def test_validation_rejects_bad_payloads(client, patch):
    r = client.post("/score", json={**POLICY, **patch})
    assert r.status_code == 422


def test_missing_field_rejected(client):
    p = {k: v for k, v in POLICY.items() if k != "DrivAge"}
    assert client.post("/score", json=p).status_code == 422


def test_batch_limits(client):
    assert client.post("/score/batch", json={"policies": []}).status_code == 422
    assert client.post("/score/batch", json={"policies": [POLICY] * (MAX_BATCH + 1)}).status_code == 422
    bad = [POLICY, {**POLICY, "Area": "Q"}]
    r = client.post("/score/batch", json={"policies": bad})
    assert r.status_code == 422 and r.json()["detail"][0]["loc"][:3] == ["body", "policies", 1]


def test_shadow_scoring_summary(glm, small_book):
    from claims_mlops import data as D
    from claims_mlops import models as M

    shadow = M.fit(M.build_glm(features=["DrivAge", "Area"]), D.split_months(small_book, range(1, 13)))
    with TestClient(create_app(glm, "v1", shadow, "v0")) as c:
        assert c.get("/shadow/summary").json()["n"] == 0
        c.post("/score/batch", json={"policies": [POLICY, {**POLICY, "DrivAge": 20}, {**POLICY, "Area": "A"}]})
        s = c.get("/shadow/summary").json()
        assert s["enabled"] and s["n"] == 3 and s["shadow_version"] == "v0"
        assert np.isfinite(s["mean_ratio_shadow_to_champion"])
    with TestClient(create_app(glm, "v1")) as c:
        assert c.get("/shadow/summary").json() == {"enabled": False}
