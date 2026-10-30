import copy

from extract_eval.generator import generate
from extract_eval.schemas import validate_record

INVOICE = {
    "invoice_number": "INV-2026-00001",
    "po_number": None,
    "supplier": "Acme Metals Inc.",
    "customer": "Beta Foods Ltd",
    "invoice_date": "2026-01-10",
    "due_date": "2026-02-09",
    "currency": "USD",
    "subtotal": 30.0,
    "tax": 2.5,
    "total": 32.5,
    "payment_terms_days": 30,
    "line_items": [{"description": "Widget", "quantity": 3, "unit_price": 10.0, "amount": 30.0}],
}


def test_valid_invoice_passes():
    model, errs = validate_record("invoice", INVOICE)
    assert model is not None and errs == []


def test_arithmetic_and_dates_are_enforced():
    for mutate, needle in [
        (lambda d: d.update(total=99.0), "subtotal"),
        (lambda d: d.update(due_date="2025-12-01"), "due_date"),
        (lambda d: d["line_items"][0].update(amount=31.0), "line amount"),
        (lambda d: d.update(subtotal=31.0, total=33.5), "line items sum"),
        (lambda d: d.update(currency="JPY"), "currency"),
        (lambda d: d.update(unexpected="x"), "unexpected"),
        (lambda d: d.update(line_items=[]), "line_items"),
    ]:
        d = copy.deepcopy(INVOICE)
        mutate(d)
        model, errs = validate_record("invoice", d)
        assert model is None and any(needle in e for e in errs), (needle, errs)


def test_contract_consistency_rules():
    c = {
        "supplier": "A Co.",
        "customer": "B Ltd",
        "effective_date": "2026-01-01",
        "expiration_date": "2025-01-01",
        "currency": "EUR",
        "auto_renewal": False,
    }
    assert validate_record("contract", c)[0] is None
    c["expiration_date"] = "2027-01-01"
    assert validate_record("contract", c)[0] is not None
    c["renewal_term_months"] = 12  # renewal term without auto-renewal is inconsistent
    assert validate_record("contract", c)[0] is None


def test_every_gold_label_is_schema_valid():
    for d in generate(200):
        model, errs = validate_record(d.doc_type, d.gold)
        assert model is not None, (d.doc_id, errs)
