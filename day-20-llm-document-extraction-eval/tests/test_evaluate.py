import numpy as np
import pandas as pd

from extract_eval.evaluate import (
    IsotonicCalibrator,
    choose_threshold,
    classify_error,
    compare,
    ece,
    match_line_items,
    prf,
    routing_summary,
)


def test_compare_semantics():
    assert compare("supplier", "ACME Metals, Inc", "Acme Metals Inc.") == (True, True)
    assert compare("supplier", "Acme Metals Incorporated", "Acme Metals Inc.") == (False, True)
    assert compare("invoice_number", "INV-1", "INV-2") == (False, False)
    assert compare("total", 10.004, 10.0)[0] and not compare("total", 10.01, 10.0)[0]
    assert compare("po_number", None, None) == (True, True)
    assert compare("po_number", "PO-1", None) == (False, False)


def test_line_item_matching():
    gold = [
        {"description": "Widget A", "quantity": 1, "unit_price": 2.0, "amount": 2.0},
        {"description": "Gadget", "quantity": 2, "unit_price": 5.0, "amount": 10.0},
    ]
    pred = [
        {"description": "widget a", "quantity": 1, "unit_price": 2.0, "amount": 2.0},
        {"description": "Gadget", "quantity": 1, "unit_price": 10.0, "amount": 10.0},
        {"description": "Freight", "quantity": 1, "unit_price": 3.0, "amount": 3.0},
    ]
    m = match_line_items(pred, gold)
    assert m == {"tp": 2, "tp_strict": 1, "n_pred": 3, "n_gold": 2}
    assert abs(prf(m["tp"], m["n_pred"], m["n_gold"])["f1"] - 0.8) < 1e-9


def test_ece_and_calibration():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, 5000)
    correct = rng.uniform(0, 1, 5000) < conf
    assert ece(conf, correct) < 0.03
    assert ece(np.full(5000, 0.99), correct) > 0.4
    df = pd.DataFrame({"field": "f", "confidence": np.full(5000, 0.9), "correct": correct})
    cal = IsotonicCalibrator().fit(df)
    assert abs(cal.transform(df).mean() - correct.mean()) < 1e-6


def test_threshold_and_routing():
    conf = np.array([0.99] * 50 + [0.5] * 50)
    correct = np.array([True] * 50 + [True] * 25 + [False] * 25)
    t = choose_threshold(conf, correct, 0.99)
    assert t == 0.99
    df = pd.DataFrame({"doc_id": np.repeat(np.arange(10), 10), "conf_cal": conf, "correct": correct, "valid_record": True})
    r = routing_summary(df, "conf_cal", t)
    assert r["auto_accept_rate"] == 0.5 and r["errors_leaked"] == 0 and r["errors_caught_by_review"] == 25


def test_taxonomy():
    meta = {
        "template": "msa_prose",
        "distractors": {"total_value": [500.0]},
        "scanned": False,
        "derived": {"expiration_date": True},
    }
    assert classify_error("total_value", 500.0, 100.0, meta) == "distractor_confusion"
    assert classify_error("total_value", 1.0, 100.0, meta) == "number_format"
    assert classify_error("expiration_date", "2026-01-01", "2026-02-01", meta) == "derived_value_error"
    assert classify_error("effective_date", "2026-04-03", "2026-03-04", meta) == "date_format"
    assert classify_error("supplier", None, "X", {**meta, "template": "order_form_unseen"}) == "unseen_layout"
    assert classify_error("supplier", None, "X", {**meta, "scanned": True}) == "ocr_noise"
