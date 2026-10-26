"""FastAPI scoring service for the champion pure-premium model.

Run locally (after ``make run`` has exported the champion):
    uvicorn claims_mlops.api:app --port 8000
The model is loaded from ``CLAIMS_MODEL_URI`` (an exported MLflow model directory or a ``models:/`` URI);
default ``models/champion``. An optional ``CLAIMS_SHADOW_MODEL_URI`` enables shadow scoring.
"""

from __future__ import annotations

import os
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from claims_mlops import config as C

Area = Literal["A", "B", "C", "D", "E", "F"]
Brand = Literal["B1", "B2", "B3", "B4", "B5", "B6", "B10", "B11", "B12", "B13", "B14"]
Gas = Literal["Regular", "Diesel"]
MAX_BATCH = 10_000


class Policy(BaseModel):
    """One motor TPL policy. Ranges follow freMTPL2's documented domains."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "VehPower": 7,
                "VehAge": 3,
                "DrivAge": 42,
                "BonusMalus": 50,
                "Density": 1200,
                "Area": "D",
                "VehBrand": "B12",
                "VehGas": "Diesel",
                "Region": "R11",
                "Exposure": 1.0,
            }
        },
    )

    VehPower: Annotated[int, Field(ge=1, le=20)]
    VehAge: Annotated[int, Field(ge=0, le=100)]
    DrivAge: Annotated[int, Field(ge=18, le=100)]
    BonusMalus: Annotated[int, Field(ge=50, le=350)]
    Density: Annotated[int, Field(ge=1, le=30_000)]
    Area: Area
    VehBrand: Brand
    VehGas: Gas
    Region: Annotated[str, Field(pattern=r"^R\d{2}$")]
    Exposure: Annotated[float, Field(gt=0, le=1.0)] = 1.0
    policy_id: str | None = None


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policies: Annotated[list[Policy], Field(min_length=1, max_length=MAX_BATCH)]


class Score(BaseModel):
    policy_id: str | None
    pure_premium: float = Field(description="Expected claim cost per policy-year (EUR)")
    expected_cost: float = Field(description="pure_premium x Exposure (EUR)")


class ScoreResponse(Score):
    model_version: str


class BatchResponse(BaseModel):
    model_version: str
    n: int
    scores: list[Score]
    total_expected_cost: float


class ModelState:
    def __init__(self) -> None:
        self.model: Any = None
        self.version: str = "unloaded"
        self.uri: str | None = None
        self.shadow: Any = None
        self.shadow_version: str | None = None
        self.shadow_log: deque[tuple[float, float]] = deque(maxlen=100_000)
        self.loaded_at: float | None = None
        self.error: str | None = None


def _load(uri: str):
    import mlflow.sklearn

    return mlflow.sklearn.load_model(uri)


def _version_from(uri: str) -> str:
    p = Path(uri)
    vf = p / "VERSION"
    if vf.exists():
        return vf.read_text().strip()
    return uri


def create_app(model: Any = None, model_version: str = "test", shadow_model: Any = None, shadow_version: str | None = None) -> FastAPI:
    """App factory. Tests inject a fitted model; production loads from CLAIMS_MODEL_URI at startup."""
    state = ModelState()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if model is not None:
            state.model, state.version, state.loaded_at = model, model_version, time.time()
            state.shadow, state.shadow_version = shadow_model, shadow_version
        else:
            uri = os.environ.get("CLAIMS_MODEL_URI", str(C.MODEL_EXPORT))
            state.uri = uri
            try:
                state.model, state.version, state.loaded_at = _load(uri), _version_from(uri), time.time()
            except Exception as exc:  # stay alive; readiness reports not-ready
                state.error = f"{type(exc).__name__}: {exc}"
            suri = os.environ.get("CLAIMS_SHADOW_MODEL_URI")
            if suri:
                try:
                    state.shadow, state.shadow_version = _load(suri), _version_from(suri)
                except Exception as exc:
                    state.error = (state.error or "") + f" shadow: {exc}"
        yield

    app = FastAPI(title="Claims pure-premium scoring", version="1.0.0", lifespan=lifespan)
    app.state.model_state = state

    def _predict(policies: list[Policy], log_shadow: bool = True) -> tuple[np.ndarray, np.ndarray]:
        if state.model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        X = pd.DataFrame([p.model_dump(exclude={"policy_id"}) for p in policies])
        pp = np.asarray(state.model.predict(X[C.FEATURES]), dtype=float)
        if state.shadow is not None and log_shadow:
            sp = np.asarray(state.shadow.predict(X[C.FEATURES]), dtype=float)
            state.shadow_log.extend(zip(pp.tolist(), sp.tolist(), strict=True))
        return pp, X["Exposure"].to_numpy(float)

    @app.get("/health")
    def health() -> dict:
        """Liveness: the process is up."""
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict:
        """Readiness: a model is loaded and can score a canary policy."""
        if state.model is None:
            raise HTTPException(status_code=503, detail={"ready": False, "error": state.error})
        canary = Policy(
            VehPower=6, VehAge=5, DrivAge=40, BonusMalus=50, Density=500, Area="C", VehBrand="B1", VehGas="Regular", Region="R11"
        )
        pp, _ = _predict([canary], log_shadow=False)
        if not np.isfinite(pp).all():
            raise HTTPException(status_code=503, detail={"ready": False, "error": "non-finite canary score"})
        return {"ready": True, "model_version": state.version}

    @app.get("/model")
    def model_info() -> dict:
        return {
            "model_name": C.MODEL_NAME,
            "model_version": state.version,
            "uri": state.uri,
            "shadow_version": state.shadow_version,
            "features": C.FEATURES,
        }

    @app.post("/score", response_model=ScoreResponse)
    def score(policy: Policy) -> ScoreResponse:
        pp, exp = _predict([policy])
        return ScoreResponse(
            policy_id=policy.policy_id, pure_premium=float(pp[0]), expected_cost=float(pp[0] * exp[0]), model_version=state.version
        )

    @app.post("/score/batch", response_model=BatchResponse)
    def score_batch(req: BatchRequest) -> BatchResponse:
        pp, exp = _predict(req.policies)
        cost = pp * exp
        scores = [
            Score(policy_id=p.policy_id, pure_premium=float(a), expected_cost=float(b))
            for p, a, b in zip(req.policies, pp, cost, strict=True)
        ]
        return BatchResponse(model_version=state.version, n=len(scores), scores=scores, total_expected_cost=float(cost.sum()))

    @app.get("/shadow/summary")
    def shadow_summary(request: Request) -> dict:
        """Champion vs shadow comparison over all traffic scored since startup."""
        if state.shadow is None:
            return {"enabled": False}
        if not state.shadow_log:
            return {"enabled": True, "n": 0, "shadow_version": state.shadow_version}
        a = np.array(state.shadow_log)
        return {
            "enabled": True,
            "shadow_version": state.shadow_version,
            "n": int(len(a)),
            "mean_champion": float(a[:, 0].mean()),
            "mean_shadow": float(a[:, 1].mean()),
            "mean_ratio_shadow_to_champion": float(a[:, 1].sum() / a[:, 0].sum()),
            "rank_corr": float(pd.Series(a[:, 0]).corr(pd.Series(a[:, 1]), method="spearman")) if len(a) > 2 else None,
        }

    return app


app = create_app()
