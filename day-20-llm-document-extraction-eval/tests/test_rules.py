from extract_eval.extractors.base import detect_doc_type
from extract_eval.extractors.rules import RulesExtractor
from extract_eval.pdf_text import DocText

R, S = RulesExtractor("rules"), RulesExtractor("strict")


def test_money_parsing():
    assert R.parse_money("$1,234.56") == (1234.56, "USD")
    assert R.parse_money("1.234,56 EUR") == (1234.56, "EUR")
    assert R.parse_money("£ 12.345.678,90")[0] == 12345678.90
    assert R.parse_money("1.234.56")[0] == 1234.56  # OCR flipped a thousands separator
    assert S.parse_money("1.234,56 EUR")[0] != 1234.56  # strict baseline only knows US format


def test_date_parsing():
    for s in ["2026-03-04", "03/04/2026", "March 4, 2026", "4 March 2026", "the 4th day of March, 2026", "04-Mar-2026"]:
        assert R.parse_date(s).isoformat() == "2026-03-04", s
    assert R.parse_date("4 Mrach 2026").isoformat() == "2026-03-04"  # fuzzy month in rules mode
    assert S.parse_date("4 Mrach 2026") is None


CONTRACT = """MASTER SERVICES AGREEMENT
Agreement No. MSA-2025-0042
This Master Services Agreement (the "Agreement") is entered into as of March 4, 2025 (the "Effective Date") by and
between ACME METALS INC., a Delaware corporation ("Supplier"), and Beta Foods, Limited ("Customer").
This Agreement supersedes the prior agreement between the parties dated January 2, 2023.
1. Term. This Agreement commences on the Effective Date and continues for an initial term of twenty-four (24) months.
2. Fees. Customer shall pay Supplier fees up to a total contract value of $120,000.00, exclusive of applicable taxes.
3. Payment. Undisputed invoices are payable within thirty (30) days of receipt of a correct invoice.
4. Renewal. This Agreement shall not renew automatically; any extension requires a written amendment signed by both parties. Either party may terminate for convenience on sixty (60) days' written notice.
5. Limitation of Liability. Each party's aggregate liability shall not exceed $240,000.00.
6. Governing Law. This Agreement shall be governed by the laws of the State of New York.
"""


def test_rules_contract_extraction_handles_distractors_and_derivation():
    r = R.extract(DocText("X", CONTRACT))
    f = r.fields
    assert r.doc_type == "contract"
    assert f["supplier"] == "Acme Metals Inc." and f["customer"] == "Beta Foods Ltd"
    assert f["effective_date"] == "2025-03-04" and f["expiration_date"] == "2027-03-04"
    assert f["total_value"] == 120000.0 and f["currency"] == "USD" and f["payment_terms_days"] == 30
    assert f["auto_renewal"] is False and f["renewal_notice_days"] is None and f["governing_law"] == "New York"


def test_strict_baseline_falls_for_distractors():
    f = S.extract(DocText("X", CONTRACT)).fields
    assert f["auto_renewal"] is True  # "not renew automatically" contains "automatic"
    assert f["renewal_notice_days"] == 60  # termination notice mistaken for renewal notice
    assert f["payment_terms_days"] is None


def test_invoice_text_table_and_derived_due_date():
    text = """NORTHWIND LOGISTICS LLC
1 Road
TAX INVOICE
Invoice # SI-2026-00042 Date: 2026-01-10
Order Date: 2026-01-01
Customer: Beta Foods Ltd
Item Qty Rate Amount
Widget A 3 10.00 30.00
Widget B 2 1,000.50 2,001.00
Net amount 2,031.00
VAT 20% 406.20
Invoice total USD 2,437.20
Previous balance 999.00 (already paid, not included above)
Payment due within 30 days."""
    r = R.extract(DocText("Y", text))
    assert r.doc_type == "invoice" and detect_doc_type(text) == "invoice"
    f = r.fields
    assert f["invoice_number"] == "SI-2026-00042" and f["invoice_date"] == "2026-01-10" and f["due_date"] == "2026-02-09"
    assert (f["subtotal"], f["tax"], f["total"]) == (2031.0, 406.2, 2437.2)
    assert [i["amount"] for i in r.line_items] == [30.0, 2001.0]
    assert r.confidence["total"] > 0.9
