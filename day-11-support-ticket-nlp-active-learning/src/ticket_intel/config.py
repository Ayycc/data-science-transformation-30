"""Paths, data-source constants and run presets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLE_DIR = DATA_DIR / "sample"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

SEED = 20261021

# Public corpus: Bitext customer-support intent dataset (CDLA-Sharing-1.0), pinned to a commit.
BITEXT_COMMIT = "72ea2203180d14a416b579c657033ecd21e59a11"
BITEXT_FILE = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
BITEXT_URL = f"https://raw.githubusercontent.com/bitext/customer-support-llm-chatbot-training-dataset/{BITEXT_COMMIT}/data/{BITEXT_FILE}"
# The same file is also published on Hugging Face (tried first when reachable).
BITEXT_HF_URL = f"https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/resolve/main/{BITEXT_FILE}"

SPACY_MODEL = "en_core_web_md"
ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class RunConfig:
    """All knobs of one end-to-end run."""

    name: str = "full"
    seed: int = SEED
    max_per_intent: int | None = None  # cap tickets per intent (None = all)
    synthetic_per_intent: int = 700  # used only on the synthetic fallback path
    test_frac: float = 0.20
    pii_frac: float = 0.35  # share of tickets that get injected greeting/contact/signature PII
    multilingual_frac: float = 0.08  # share of tickets with a non-English snippet
    budget_frac: float = 0.02  # active-learning label budget as a share of the unlabelled pool
    init_frac: float = 0.001  # random seed set
    batch_frac: float = 0.0005  # labels queried per round
    al_repeats: int = 5
    abstain_target_acc: float = 0.95
    holdout_intents: tuple[str, ...] = ("check_cancellation_fee", "newsletter_subscription")
    holdout_intents_synthetic: tuple[str, ...] = ("sso_configuration", "api_rate_limit")
    emerging_conf_threshold: float = 0.5
    umap_neighbors: int = 15
    min_cluster_size: int = 40
    min_cluster_size_emerging: int = 15
    extra: dict = field(default_factory=dict)


FULL = RunConfig()
TINY = RunConfig(
    name="tiny",
    max_per_intent=60,
    synthetic_per_intent=60,
    budget_frac=0.10,
    init_frac=0.03,
    batch_frac=0.02,
    al_repeats=2,
    min_cluster_size=10,
    min_cluster_size_emerging=5,
)
