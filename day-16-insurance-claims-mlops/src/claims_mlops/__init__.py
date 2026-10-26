"""Day 16: Insurance claims severity (pure premium) with a full MLOps lifecycle."""

import os

# Keep MLflow fully local and quiet: no telemetry calls, no agent hints.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("DO_NOT_TRACK", "true")
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

__version__ = "1.0.0"
