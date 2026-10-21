import json

from ticket_intel.config import TINY
from ticket_intel.pipeline import run


def test_tiny_end_to_end(tmp_path):
    m = run(TINY, tmp_path, source="synthetic", write_sample=False, n_jobs=2)
    assert m["data"]["path"] == "synthetic" and m["data"]["n_intents"] == 15
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["classification"]["tfidf_full_labels"]["macro_f1"] > 0.8
    assert set(saved["active_learning"]["by_features"]) == {"tfidf", "emb"}
    for f in ["learning_curve_tfidf.png", "confusion_al_2pct.png", "topics_umap.png", "abstention.png", "pii_recall.png"]:
        assert (tmp_path / "figures" / f).exists()
    assert (tmp_path / "report.md").read_text().startswith("# Support ticket")
