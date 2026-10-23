from __future__ import annotations

import json

import pandas as pd

from aml_graph.config import SAMPLE_DIR, SMALL, ModelConfig
from aml_graph.data import load_ibm_csv
from aml_graph.pipeline import run


def test_pipeline_end_to_end_small(tmp_path):
    m = run(SMALL, mcfg=ModelConfig(n_bootstrap=20, n_estimators=100), out_dir=tmp_path, try_public=False, n_alerts=2)
    assert (tmp_path / "metrics.json").exists() and (tmp_path / "report.md").exists()
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["data"]["source"] == "synthetic"
    assert saved["acceptance"]["each_typology_recall_reported"]
    assert saved["acceptance"]["alerts_have_explainable_subgraph"]
    htmls = list((tmp_path / "alerts").glob("alert_*.html"))
    assert len(htmls) == 2
    assert "Top reasons" in htmls[0].read_text()
    for fs in m["models"]["feature_sets"].values():
        assert 0 <= fs["pr_auc"] <= 1


def test_committed_sample_matches_generator(small_bank):
    files = list(SAMPLE_DIR.iterdir())
    assert sum(f.stat().st_size for f in files) <= 1_000_000
    tx = pd.read_csv(SAMPLE_DIR / "transactions.csv.gz")
    assert len(tx) == len(small_bank.transactions)
    assert tx["is_laundering"].sum() == small_bank.transactions["is_laundering"].sum()


def test_ibm_loader(tmp_path):
    p = tmp_path / "HI-Small_Trans.csv"
    p.write_text(
        "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
        "2022/09/01 00:20,10,8000EBD30,10,8000EBD30,3697.34,US Dollar,3697.34,US Dollar,Reinvestment,0\n"
        "2022/09/01 00:25,3208,8000F4580,1,8000F5340,0.01,US Dollar,0.01,US Dollar,Cheque,1\n"
    )
    bank = load_ibm_csv(p)
    assert len(bank.transactions) == 2
    assert bank.accounts["is_laundering"].sum() == 2
    assert bank.transactions["ts"].min() == 0
