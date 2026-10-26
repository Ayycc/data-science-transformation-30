"""MLflow tracking + model registry helpers and the champion/challenger promotion gate.

The tracking and registry backend is a local SQLite file (``mlruns/mlflow.db``) with file artefacts
under ``mlruns/artifacts``: no tracking server is needed.
"""

from __future__ import annotations

import json
import logging
import tempfile
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature

from claims_mlops import config as C
from claims_mlops import models as M

for _name in ("mlflow.sklearn", "mlflow.store.db.utils", "mlflow.models.model", "alembic"):
    logging.getLogger(_name).setLevel(logging.ERROR)

CHAMPION, CHALLENGER = "champion", "challenger"


def setup(root: Path | None = None, experiment: str = C.EXPERIMENT) -> MlflowClient:
    """Point MLflow at a local SQLite store and ensure the experiment exists."""
    base = root or C.MLRUNS
    uri = C.tracking_uri(base)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        art = base / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(experiment, artifact_location=art.resolve().as_uri())
    mlflow.set_experiment(experiment)
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


@dataclass
class Candidate:
    run_id: str
    version: str
    family: str
    metrics: dict
    params: dict = field(default_factory=dict)


def log_candidate(
    model: M.BalancedModel,
    family: str,
    params: dict,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    tags: dict | None = None,
    artifacts: dict[str, Path] | None = None,
    model_name: str = C.MODEL_NAME,
) -> Candidate:
    """Log params, validation metrics, artefacts and the model; register a new model version."""
    metrics = M.evaluate(model, valid)
    metrics["null_deviance"] = M.null_deviance(train, valid)
    metrics["balance_factor"] = float(model.balance_factor_)
    run_tags = {"family": family, "train_months": f"{train.month.min()}-{train.month.max()}", **(tags or {})}
    with mlflow.start_run(run_name=f"{family}-m{train.month.max():02d}", tags=run_tags) as run:
        mlflow.log_params({**params, "family": family, "n_train": len(train), "tweedie_power": C.TWEEDIE_POWER})
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
        X = valid[C.FEATURES].head(200)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sig = infer_signature(X, model.predict(X))
            info = mlflow.sklearn.log_model(
                model,
                name="model",
                signature=sig,
                input_example=valid[C.FEATURES].head(3),
                registered_model_name=model_name,
                serialization_format="cloudpickle",
            )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "validation_metrics.json"
            p.write_text(json.dumps(metrics, indent=2))
            mlflow.log_artifact(str(p))
        for _, path in (artifacts or {}).items():
            mlflow.log_artifact(str(path))
    version = str(info.registered_model_version)
    client = MlflowClient()
    client.set_model_version_tag(model_name, version, "family", family)
    client.set_registered_model_alias(model_name, CHALLENGER, version)
    return Candidate(run.info.run_id, version, family, metrics, params)


def champion_version(client: MlflowClient, model_name: str = C.MODEL_NAME) -> str | None:
    try:
        return str(client.get_model_version_by_alias(model_name, CHAMPION).version)
    except Exception:
        return None


def load_version(version: str | None = None, model_name: str = C.MODEL_NAME, alias: str = CHAMPION):
    uri = f"models:/{model_name}/{version}" if version else f"models:/{model_name}@{alias}"
    return mlflow.sklearn.load_model(uri)


@dataclass
class GateDecision:
    passed: bool
    candidate_version: str
    champion_version: str | None
    checks: dict
    candidate_metrics: dict
    champion_metrics: dict | None

    def to_dict(self) -> dict:
        return asdict(self)


def gate(candidate: Candidate, champion_metrics: dict | None, champion_ver: str | None, policy: C.GatePolicy) -> GateDecision:
    """Challenger must beat the champion (or, if none, the null model) on the same validation data."""
    cm = candidate.metrics
    lo, hi = policy.ae_band
    if champion_metrics is None:
        ref_dev, ref_name = cm["null_deviance"], "null model"
    else:
        ref_dev, ref_name = champion_metrics["tweedie_deviance"], f"champion v{champion_ver}"
    gain = 1 - cm["tweedie_deviance"] / ref_dev
    checks = {
        "deviance_gain": {
            "value": gain,
            "threshold": policy.min_rel_deviance_gain,
            "vs": ref_name,
            "pass": gain > policy.min_rel_deviance_gain,
        },
        "gini_floor": {"value": cm["gini"], "threshold": policy.min_gini, "pass": cm["gini"] >= policy.min_gini},
        "ae_band": {"value": cm["ae_ratio"], "threshold": [lo, hi], "pass": lo <= cm["ae_ratio"] <= hi},
    }
    passed = all(c["pass"] for c in checks.values())
    return GateDecision(passed, candidate.version, champion_ver, checks, cm, champion_metrics)


def apply_decision(client: MlflowClient, decision: GateDecision, model_name: str = C.MODEL_NAME) -> str:
    """Automatic promotion only if the gate passed; otherwise the candidate is archived and tagged."""
    v = decision.candidate_version
    client.set_model_version_tag(model_name, v, "gate", "passed" if decision.passed else "failed")
    client.set_model_version_tag(model_name, v, "gate_checks", json.dumps({k: c["pass"] for k, c in decision.checks.items()}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # registry stages are deprecated in MLflow 3 but still supported
        if decision.passed:
            client.transition_model_version_stage(model_name, v, "Production", archive_existing_versions=True)
            client.set_registered_model_alias(model_name, CHAMPION, v)
            action = "promoted"
        else:
            client.transition_model_version_stage(model_name, v, "Archived")
            action = "rejected"
    try:
        client.delete_registered_model_alias(model_name, CHALLENGER)
    except Exception:
        pass
    return action


def evaluate_champion(client: MlflowClient, valid: pd.DataFrame, model_name: str = C.MODEL_NAME) -> tuple[str | None, dict | None]:
    ver = champion_version(client, model_name)
    if ver is None:
        return None, None
    return ver, M.evaluate(load_version(ver, model_name), valid)


def registry_table(client: MlflowClient, model_name: str = C.MODEL_NAME) -> list[dict]:
    rows = []
    for mv in sorted(client.search_model_versions(f"name='{model_name}'"), key=lambda m: int(m.version)):
        run = client.get_run(mv.run_id)
        aliases = list(getattr(client.get_model_version(model_name, mv.version), "aliases", []) or [])
        rows.append(
            {
                "version": int(mv.version),
                "stage": mv.current_stage,
                "aliases": aliases,
                "family": mv.tags.get("family"),
                "gate": mv.tags.get("gate"),
                "run_id": mv.run_id,
                "train_months": run.data.tags.get("train_months"),
                "trigger": run.data.tags.get("trigger", ""),
                "val_tweedie_deviance": run.data.metrics.get("val_tweedie_deviance"),
                "val_gini": run.data.metrics.get("val_gini"),
                "val_ae_ratio": run.data.metrics.get("val_ae_ratio"),
            }
        )
    return rows


def export_champion(dest: Path = C.MODEL_EXPORT, model_name: str = C.MODEL_NAME) -> Path:
    """Copy the champion's MLflow model directory locally (used by the Docker image; git-ignored)."""
    import shutil

    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{model_name}@{CHAMPION}", dst_path=str(dest))
    return dest
