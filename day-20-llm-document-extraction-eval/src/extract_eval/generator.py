"""Seeded SYNTHETIC supplier contracts and invoices with gold JSON labels.

Every document is SYNTHETIC: company names, amounts and clauses are drawn from word lists with a fixed seed.
Variety is deliberate so that extraction is not trivial:

* 3 contract templates (prose MSA, key-value cover sheet, procurement letter with reversed party order)
  and 3 invoice templates (ruled table, unruled text table, compact "qty x item @ price" lines).
* Mixed date styles (ISO, US numeric, "March 3, 2026", "3 March 2026", "the 3rd day of March, 2026"),
  US and European number formats, currency symbols vs codes.
* Distractor values that a naive pattern grabs (liability caps, superseded-agreement dates, order dates,
  balance-forward amounts, termination notice periods in non-renewing contracts).
* Derived fields (expiration from "initial term of N months", due date from payment terms).
* "Scanned-style" noise on ~25% of documents: OCR-like character confusions (0/O, 1/l, 5/S, 8/B, rn/m),
  merged and split tokens and decimal-separator flips, injected into the PDF text layer.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

SEED = 20261030

ADJ = [
    "Northwind",
    "Bluepeak",
    "Ironbridge",
    "Silverline",
    "Crescent",
    "Granite",
    "Harbor",
    "Juniper",
    "Keystone",
    "Lakeside",
    "Meridian",
    "Oakridge",
    "Pinecrest",
    "Redwood",
    "Summit",
    "Tidewater",
    "Vanguard",
    "Westgate",
    "Alder",
    "Brightwater",
    "Cobalt",
    "Driftwood",
    "Evergreen",
    "Foxglove",
    "Highland",
    "Larkspur",
    "Maple",
    "Orion",
    "Quarry",
    "Sterling",
    "Thornbury",
    "Upland",
    "Willow",
    "Zephyr",
    "Ashford",
    "Beacon",
]
NOUN = [
    "Logistics",
    "Industrial",
    "Analytics",
    "Components",
    "Packaging",
    "Facilities",
    "Medical Supply",
    "Software",
    "Engineering",
    "Foods",
    "Energy",
    "Print Services",
    "Security",
    "Consulting",
    "Metals",
    "Electronics",
    "Fleet Services",
    "Office Supply",
    "Laboratories",
    "Textiles",
]
SUFFIX = ["Inc.", "LLC", "Ltd", "GmbH", "Corp.", "plc", "Group", "Co."]
SUFFIX_LONG = {"Inc.": "Incorporated", "Corp.": "Corporation", "Ltd": "Limited", "Co.": "Company"}
LAWS = ["New York", "Delaware", "California", "Texas", "England and Wales", "Ireland", "Germany", "Ontario"]
LAW_PHRASE = {
    "New York": ["the State of New York", "New York"],
    "Delaware": ["the State of Delaware", "Delaware"],
    "California": ["the State of California"],
    "Texas": ["the State of Texas"],
    "England and Wales": ["England and Wales"],
    "Ireland": ["Ireland", "the Republic of Ireland"],
    "Germany": ["the Federal Republic of Germany", "Germany"],
    "Ontario": ["the Province of Ontario"],
}
STATES = ["Delaware", "New York", "Texas", "California", "Nevada"]
ITEMS = [
    "Steel bracket 40mm",
    "Pallet wrap film",
    "Nitrile gloves (box of 100)",
    "Hydraulic hose assembly",
    "Server maintenance hours",
    "On-site technician visit",
    "Printer toner cartridge",
    "Corrugated carton 12x12",
    "LED panel light 60W",
    "Safety helmet class E",
    "Data cleansing service",
    "Cloud storage (TB-month)",
    "Forklift rental (day)",
    "Laboratory reagent kit",
    "Industrial adhesive 5L",
    "Copper cable 2.5mm (100m)",
    "Consulting day rate",
    "Freight surcharge",
    "Calibration service",
    "Thermal labels roll",
    "Filter cartridge",
    "Ergonomic office chair",
    "Training workshop seat",
    "Bearing kit 6204",
    "Disinfectant concentrate 10L",
]
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
NUM_WORDS = {
    0: "zero",
    7: "seven",
    10: "ten",
    12: "twelve",
    14: "fourteen",
    15: "fifteen",
    24: "twenty-four",
    30: "thirty",
    36: "thirty-six",
    45: "forty-five",
    60: "sixty",
    90: "ninety",
    120: "one hundred twenty",
}
CURRENCY_SYMBOL = {"USD": "$", "EUR": "€", "GBP": "£"}

OCR_CONFUSIONS = {
    "0": "O",
    "O": "0",
    "1": "l",
    "l": "1",
    "I": "l",
    "5": "S",
    "S": "5",
    "8": "B",
    "B": "8",
    "m": "rn",
    "e": "c",
    "a": "o",
    ".": ",",
    ",": ".",
}


@dataclass
class SyntheticDoc:
    doc_id: str
    doc_type: str
    gold: dict[str, Any]
    meta: dict[str, Any]
    blocks: list[Any] = field(default_factory=list)  # str lines, or ("table", rows) for ruled tables

    @property
    def text(self) -> str:
        out: list[str] = []
        for b in self.blocks:
            if isinstance(b, tuple):
                out.extend(" | ".join(r) for r in b[1])
            else:
                out.append(b)
        return "\n".join(out)


def add_months(d: date, months: int) -> date:
    m = d.month - 1 + months
    y, m = d.year + m // 12, m % 12 + 1
    days = [31, 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return date(y, m, min(d.day, days[m - 1]))


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def fmt_date(d: date, style: str) -> str:
    return {
        "iso": d.isoformat(),
        "us": f"{d.month:02d}/{d.day:02d}/{d.year}",
        "long": f"{MONTHS[d.month - 1]} {d.day}, {d.year}",
        "dmy": f"{d.day} {MONTHS[d.month - 1]} {d.year}",
        "legal": f"the {_ordinal(d.day)} day of {MONTHS[d.month - 1]}, {d.year}",
        "abbr": f"{d.day:02d}-{MONTHS[d.month - 1][:3]}-{d.year}",
    }[style]


def fmt_num(x: float, style: str) -> str:
    s = f"{x:,.2f}"
    if style == "eu":
        s = s.replace(",", "_").replace(".", ",").replace("_", ".")
    return s


def fmt_money(x: float, cur: str, style: str, rng: random.Random) -> str:
    n = fmt_num(x, style)
    if style == "eu":
        return f"{n} {cur}" if rng.random() < 0.5 else f"{CURRENCY_SYMBOL[cur]} {n}"
    return f"{CURRENCY_SYMBOL[cur]}{n}" if rng.random() < 0.6 else f"{cur} {n}"


def words_num(n: int) -> str:
    return f"{NUM_WORDS[n]} ({n})" if n in NUM_WORDS else str(n)


def company(rng: random.Random, used: set[str]) -> str:
    while True:
        name = f"{rng.choice(ADJ)} {rng.choice(NOUN)} {rng.choice(SUFFIX)}"
        if name not in used:
            used.add(name)
            return name


def name_variant(name: str, rng: random.Random) -> tuple[str, bool]:
    """Surface form of a party name; True if it differs from the canonical gold form."""
    base, suf = name.rsplit(" ", 1)
    r = rng.random()
    if suf in SUFFIX_LONG and r < 0.2:
        return f"{base}, {SUFFIX_LONG[suf]}", True
    if r < 0.3:
        return name.upper(), False  # case-only difference counts as exact after normalisation
    return name, False


def ocr_noise(text: str, rate: float, rng: random.Random) -> str:
    toks = text.split(" ")
    out: list[str] = []
    for t in toks:
        if t and rng.random() < rate:
            op = rng.random()
            if op < 0.65:
                idx = [i for i, c in enumerate(t) if c in OCR_CONFUSIONS]
                if idx:
                    i = rng.choice(idx)
                    t = t[:i] + OCR_CONFUSIONS[t[i]] + t[i + 1 :]
            elif op < 0.82 and out:
                out[-1] = out[-1] + t  # merged tokens (lost space)
                continue
            elif len(t) > 3:
                i = rng.randint(1, len(t) - 2)
                t = t[:i] + " " + t[i:]
        out.append(t)
    return " ".join(out)


def _terms_phrase(days: int, rng: random.Random) -> str:
    if days == 0:
        return rng.choice(["upon receipt of invoice", "due on receipt"])
    return rng.choice(
        [
            f"within {words_num(days)} days of receipt of a correct invoice",
            f"Net {days}",
            f"net {days} days from the invoice date",
            f"2/10 net {days}" if days >= 30 else f"Net {days}",
        ]
    )


def _renewal_clause(auto: bool, months: int | None, notice: int | None, rng: random.Random) -> tuple[str, int | None]:
    if auto:
        term = {
            12: rng.choice(["one (1) year", "twelve (12) months", "12-month"]),
            24: rng.choice(["twenty-four (24) months", "two (2) year", "24-month"]),
            36: rng.choice(["thirty-six (36) months", "three (3) year"]),
        }[months]
        return rng.choice(
            [
                f"This Agreement shall automatically renew for successive {term} periods unless either party gives at least {words_num(notice)} days' prior written notice of non-renewal.",
                f"Upon expiry the Agreement renews automatically for additional {term} terms. Notice of non-renewal must be served no later than {notice} days before the end of the then-current term.",
            ]
        ), None
    distract = rng.choice([30, 60, 90])
    return rng.choice(
        [
            f"This Agreement shall not renew automatically; any extension requires a written amendment signed by both parties. Either party may terminate for convenience on {words_num(distract)} days' written notice.",
            f"The parties may renew this Agreement by mutual written agreement only. Either party may terminate for convenience upon {distract} days' notice.",
        ]
    ), distract


def make_contract(i: int, rng: random.Random, used: set[str]) -> SyntheticDoc:
    template = rng.choices(["msa_prose", "cover_sheet", "procurement_letter", "order_form_unseen"], [0.27, 0.27, 0.26, 0.2])[0]
    supplier, customer = company(rng, used), company(rng, used)
    cur = rng.choices(["USD", "EUR", "GBP"], [0.6, 0.25, 0.15])[0]
    num_style = "eu" if (cur == "EUR" and rng.random() < 0.6) else "us"
    eff = date(2024, 1, 1) + timedelta(days=rng.randint(0, 1000))
    term_months = rng.choice([12, 24, 36, 48])
    explicit_exp = rng.random() < 0.5
    exp = add_months(eff, term_months)
    value = round(rng.choice([rng.uniform(20_000, 250_000), rng.uniform(250_000, 4_000_000)]), -2) if rng.random() < 0.8 else None
    terms = rng.choice([0, 15, 30, 30, 45, 60, 90])
    terms_present = rng.random() < 0.92
    auto = rng.random() < 0.55
    r_months = rng.choice([12, 24, 36]) if auto else None
    notice = rng.choice([30, 60, 90]) if auto else None
    law = rng.choice(LAWS)
    law_present = rng.random() < 0.9
    number = f"{rng.choice(['MSA', 'SA', 'PA', 'CTR'])}-{eff.year}-{rng.randint(100, 9999):04d}" if rng.random() < 0.85 else None
    dstyle = (
        rng.choice(["iso", "us", "long", "dmy", "legal", "abbr"])
        if template != "cover_sheet"
        else rng.choice(["iso", "us", "long", "abbr"])
    )
    sup_s, sup_var = name_variant(supplier, rng)
    cus_s, cus_var = name_variant(customer, rng)
    cap = round((value or rng.uniform(50_000, 1_000_000)) * rng.choice([1.0, 1.5, 2.0]) + rng.choice([0, 5000, 25000]), -2)
    prior = eff - timedelta(days=rng.randint(200, 1500))
    signed = eff - timedelta(days=rng.randint(1, 20))
    renewal_text, distract_notice = _renewal_clause(auto, r_months, notice, rng)

    def money(x: float) -> str:
        return fmt_money(x, cur, num_style, rng)

    distractors: dict[str, Any] = {"effective_date": [prior.isoformat(), signed.isoformat()], "total_value": [cap]}
    if distract_notice:
        distractors["renewal_notice_days"] = [distract_notice]
    term_text = (
        f"until {fmt_date(exp, dstyle)}"
        if explicit_exp
        else f"for an initial term of {words_num(term_months) if term_months in NUM_WORDS else term_months} months"
    )

    lines: list[Any] = []
    if template == "msa_prose":
        lines += ["MASTER SERVICES AGREEMENT"]
        if number:
            lines += [f"Agreement No. {number}"]
        lines += [
            f'This Master Services Agreement (the "Agreement") is entered into as of {fmt_date(eff, dstyle)} (the "Effective Date") by and between {sup_s}, a {rng.choice(STATES)} corporation ("Supplier"), and {cus_s} ("Customer").',
            f"This Agreement supersedes the prior agreement between the parties dated {fmt_date(prior, dstyle)}.",
            f"1. Term. This Agreement commences on the Effective Date and continues {term_text}.",
        ]
        if value is not None:
            lines += [
                f"2. Fees. Customer shall pay Supplier fees up to a total contract value of {money(value)}, exclusive of applicable taxes."
            ]
        else:
            lines += ["2. Fees. Fees are set out in each Statement of Work issued under this Agreement."]
        if terms_present:
            lines += [f"3. Payment. Undisputed invoices are payable {_terms_phrase(terms, rng)}."]
        lines += [
            f"4. Renewal. {renewal_text}",
            f"5. Limitation of Liability. Each party's aggregate liability shall not exceed {money(cap)}.",
        ]
        if law_present:
            lines += [f"6. Governing Law. This Agreement shall be governed by the laws of {rng.choice(LAW_PHRASE[law])}."]
        lines += [f"Signed on behalf of each party on {fmt_date(signed, dstyle)}."]
    elif template == "cover_sheet":
        lines += ["SUPPLY AGREEMENT - COVER SHEET"]
        if number:
            lines += [f"Contract Ref: {number}"]
        lines += [
            f"Supplier: {sup_s}",
            f"Buyer: {cus_s}",
            f"Date of Signature: {fmt_date(signed, dstyle)}",
            f"Commencement Date: {fmt_date(eff, dstyle)}",
        ]
        lines += [f"Expiry Date: {fmt_date(exp, dstyle)}" if explicit_exp else f"Initial Term: {term_months} months"]
        if value is not None:
            lines += [f"Contract Value: {money(value)}"]
        lines += [f"Liability Cap: {money(cap)}"]
        if terms_present:
            lines += [f"Payment Terms: {'Due on receipt' if terms == 0 else f'Net {terms}'}"]
        lines += [f"Auto-Renewal: {'Yes - ' + str(r_months) + ' month periods' if auto else 'No'}"]
        if auto:
            lines += [f"Non-Renewal Notice: {notice} days"]
        if law_present:
            lines += [f"Governing Law: {law}"]
        lines += [
            "GENERAL TERMS",
            renewal_text,
            f"This cover sheet amends and restates the supply terms dated {fmt_date(prior, dstyle)}.",
        ]
    elif template == "order_form_unseen":  # held out: the rules were written before this layout existed
        lines += [f"ORDER FORM {number or ''}".strip(), "Commercial terms are set out in the grid below."]
        rows = [
            ["Term", "Details"],
            ["Provider", sup_s],
            ["Client", cus_s],
            ["Start", fmt_date(eff, dstyle)],
            [
                "End" if explicit_exp else "Duration",
                fmt_date(exp, dstyle) if explicit_exp else f"{term_months} months from Start",
            ],
            ["Fees (max)", money(value) if value is not None else "per rate card"],
            ["Liability ceiling", money(cap)],
        ]
        if terms_present:
            rows.append(["Invoicing", "payable on receipt" if terms == 0 else f"payable {terms} days after invoice"])
        rows.append(
            ["Rollover", f"rolls over every {r_months} months unless cancelled {notice} days ahead" if auto else "no rollover"]
        )
        if law_present:
            rows.append(["Jurisdiction", law])
        lines += [
            ("table", rows),
            f"Order signed {fmt_date(signed, dstyle)}; prior order dated {fmt_date(prior, dstyle)} is closed.",
        ]
    else:  # procurement_letter: customer named first
        lines += ["PROCUREMENT AGREEMENT"]
        lines += [
            f'This Procurement Agreement is made effective {fmt_date(eff, dstyle)} between {cus_s} (the "Purchaser") and {sup_s} (the "Vendor").'
        ]
        if number:
            lines += [f"Purchaser reference: {number}."]
        lines += [f"The Vendor shall supply the goods and services described in Schedule A {term_text}."]
        if value is not None:
            lines += [f"The Purchaser's total commitment under this Agreement shall be a maximum of {money(value)}."]
        if terms_present:
            lines += [f"The Purchaser will settle invoices {_terms_phrase(terms, rng)}."]
        lines += [renewal_text, f"The Vendor's total liability is capped at {money(cap)}."]
        if law_present:
            lines += [f"This Agreement is subject to the laws of {rng.choice(LAW_PHRASE[law])}."]
        lines += [f"Executed on {fmt_date(signed, dstyle)}; it replaces the letter of intent dated {fmt_date(prior, dstyle)}."]

    gold = {
        "doc_type": "contract",
        "contract_number": number,
        "supplier": supplier,
        "customer": customer,
        "effective_date": eff.isoformat(),
        "expiration_date": exp.isoformat(),
        "total_value": value,
        "currency": cur,
        "payment_terms_days": terms if terms_present else None,
        "auto_renewal": auto,
        "renewal_term_months": r_months,
        "renewal_notice_days": notice,
        "governing_law": law if law_present else None,
    }
    meta = {
        "template": template,
        "date_style": dstyle,
        "number_style": num_style,
        "distractors": distractors,
        "derived": {"expiration_date": not explicit_exp},
        "name_variant": {"supplier": sup_var, "customer": cus_var},
    }
    return SyntheticDoc(f"C{i:04d}", "contract", gold, meta, lines)


def make_invoice(i: int, rng: random.Random, used: set[str]) -> SyntheticDoc:
    template = rng.choices(["ruled_table", "text_table", "compact_lines", "eu_layout_unseen"], [0.27, 0.27, 0.26, 0.2])[0]
    supplier, customer = company(rng, used), company(rng, used)
    cur = rng.choices(["USD", "EUR", "GBP"], [0.6, 0.25, 0.15])[0]
    num_style = "eu" if (cur == "EUR" and rng.random() < 0.5) else "us"
    inv_date = date(2025, 1, 1) + timedelta(days=rng.randint(0, 600))
    terms = rng.choice([0, 15, 30, 30, 45, 60])
    due = inv_date + timedelta(days=terms)
    items = []
    for desc in rng.sample(ITEMS, rng.randint(1, 7)):
        qty = float(rng.choice([1, 2, 3, 4, 5, 6, 8, 10, 12, 20, 24, 50, 100]))
        price = round(rng.uniform(2, 2500), 2)
        items.append({"description": desc, "quantity": qty, "unit_price": price, "amount": round(qty * price, 2)})
    subtotal = round(sum(x["amount"] for x in items), 2)
    rate, tax_label = rng.choice(
        [(0.20, "VAT 20%"), (0.0825, "Sales Tax (8.25%)"), (0.19, "MwSt 19%"), (0.05, "GST 5%"), (0.0, "Tax exempt")]
    )
    tax = round(subtotal * rate, 2)
    total = round(subtotal + tax, 2)
    number = f"{rng.choice(['INV', 'IN', 'SI'])}-{inv_date.year}-{rng.randint(1, 99999):05d}"
    po = f"PO-{rng.randint(10000, 99999)}" if rng.random() < 0.6 else None
    order_date = inv_date - timedelta(days=rng.randint(3, 40))
    balance = round(rng.uniform(100, 20000), 2)
    dstyle = rng.choice(["iso", "us", "long", "dmy", "abbr"])
    sup_s, sup_var = name_variant(supplier, rng)
    cus_s, cus_var = name_variant(customer, rng)

    def n(x: float) -> str:
        return fmt_num(x, num_style)

    def money(x: float) -> str:
        return fmt_money(x, cur, num_style, rng)

    show_due = template not in ("text_table", "eu_layout_unseen") or rng.random() < 0.3
    terms_present = (template != "ruled_table" or rng.random() < 0.7) or not show_due
    distractors: dict[str, Any] = {"invoice_date": [order_date.isoformat()], "total": [balance], "subtotal": [balance]}
    blocks: list[Any] = []
    terms_txt = "Due upon receipt" if terms == 0 else rng.choice([f"Net {terms}", f"Payment due within {terms} days"])
    if template == "ruled_table":
        blocks += [
            "INVOICE",
            f"From: {sup_s}",
            f"Bill To: {cus_s}",
            f"Invoice No: {number}",
            f"Invoice Date: {fmt_date(inv_date, dstyle)}",
            f"Due Date: {fmt_date(due, dstyle)}",
        ]
        if po:
            blocks += [f"PO Number: {po}"]
        blocks += [f"Currency: {cur}"]
        rows = [["Description", "Qty", "Unit Price", "Amount"]] + [
            [x["description"], f"{x['quantity']:g}", n(x["unit_price"]), n(x["amount"])] for x in items
        ]
        blocks += [("table", rows)]
        blocks += [f"Subtotal: {money(subtotal)}", f"{tax_label}: {money(tax)}", f"Total Due: {money(total)}"]
        if terms_present:
            blocks += [f"Terms: {terms_txt}"]
    elif template == "text_table":
        blocks += [
            sup_s.upper(),
            f"{rng.randint(10, 999)} {rng.choice(ADJ)} Road, Suite {rng.randint(1, 50)}",
            "TAX INVOICE",
            f"Invoice # {number}    Date: {fmt_date(inv_date, dstyle)}",
            f"Order Date: {fmt_date(order_date, dstyle)}",
            f"Customer: {cus_s}",
        ]
        if po:
            blocks += [f"Your Ref: {po}"]
        if show_due:
            blocks += [f"Payment Due: {fmt_date(due, dstyle)}"]
        blocks += ["Item                                   Qty      Rate        Amount"]
        for x in items:
            blocks += [f"{x['description']:<38} {x['quantity']:>5g}  {n(x['unit_price']):>10}  {n(x['amount']):>12}"]
        blocks += [
            f"Net amount {n(subtotal)}",
            f"{tax_label} {n(tax)}",
            f"Invoice total {cur} {n(total)}",
            f"Previous balance {n(balance)} (already paid, not included above)",
            terms_txt + ".",
        ]
    elif template == "eu_layout_unseen":  # held out: the rules were written before this layout existed
        blocks += [
            "RECHNUNG / BILL",
            f"Seller {sup_s}",
            f"Buyer {cus_s}",
            f"Document number {number}",
            f"Issue date {fmt_date(inv_date, dstyle)}",
        ]
        if show_due:
            blocks += [f"Payment deadline {fmt_date(due, dstyle)}"]
        if po:
            blocks += [f"Order reference {po} of {fmt_date(order_date, dstyle)}"]
        rows = [["Pos", "Article", "Quantity", "Price", "Line total"]] + [
            [str(k + 1), x["description"], f"{x['quantity']:g}", n(x["unit_price"]), n(x["amount"])] for k, x in enumerate(items)
        ]
        blocks += [
            ("table", rows),
            f"Net total {cur} {n(subtotal)}",
            f"{tax_label} {cur} {n(tax)}",
            f"Gross total {cur} {n(total)}",
        ]
        if terms_present:
            blocks += ["Payable immediately." if terms == 0 else f"Payable {terms} days net."]
    else:
        blocks += [
            f"Statement & Invoice from {sup_s}",
            f"To: {cus_s}",
            f"Ref: {number.replace('-', '/')}",
            f"Issued {fmt_date(inv_date, dstyle)}",
        ]
        if po:
            blocks += [f"Purchase order {po}"]
        if show_due:
            blocks += [f"Please pay by {fmt_date(due, dstyle)}"]
        blocks += [f"Balance forward: {money(balance)}"]
        for x in items:
            blocks += [f"{x['quantity']:g} x {x['description']} @ {n(x['unit_price'])} = {n(x['amount'])}"]
        blocks += [
            f"Sub-total {money(subtotal)}",
            f"{tax_label}: {money(tax)}",
            f"Amount Due: {money(total)}",
            f"Terms: {terms_txt}",
        ]
        number = number.replace("-", "/")

    gold = {
        "doc_type": "invoice",
        "invoice_number": number,
        "po_number": po,
        "supplier": supplier,
        "customer": customer,
        "invoice_date": inv_date.isoformat(),
        "due_date": due.isoformat(),
        "currency": cur,
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
        "payment_terms_days": terms if terms_present else None,
        "line_items": items,
    }
    meta = {
        "template": template,
        "date_style": dstyle,
        "number_style": num_style,
        "distractors": distractors,
        "derived": {"due_date": not show_due},
        "name_variant": {"supplier": sup_var, "customer": cus_var},
    }
    return SyntheticDoc(f"I{i:04d}", "invoice", gold, meta, blocks)


def apply_scan_noise(doc: SyntheticDoc, rng: random.Random, p_scanned: float) -> None:
    scanned = rng.random() < p_scanned
    rate = round(rng.uniform(0.03, 0.10), 3) if scanned else 0.0
    doc.meta["scanned"], doc.meta["noise_rate"] = scanned, rate
    if not scanned:
        return
    new: list[Any] = []
    for b in doc.blocks:
        if isinstance(b, tuple):
            new.append(("table", [[ocr_noise(c, rate, rng) for c in r] for r in b[1]]))
        else:
            new.append(ocr_noise(b, rate, rng))
    doc.blocks = new


def generate(n_docs: int = 500, seed: int = SEED, p_scanned: float = 0.25) -> list[SyntheticDoc]:
    rng = random.Random(seed)
    used: set[str] = set()
    docs: list[SyntheticDoc] = []
    for k in range(n_docs):
        doc = make_contract(k // 2 + 1, rng, used) if k % 2 == 0 else make_invoice(k // 2 + 1, rng, used)
        apply_scan_noise(doc, rng, p_scanned)
        docs.append(doc)
    return docs


def render_pdf(doc: SyntheticDoc, path: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

    scanned = doc.meta.get("scanned", False)
    font = "Courier" if scanned else "Helvetica"
    body = ParagraphStyle("b", fontName=font, fontSize=9.5, leading=12.5)
    head = ParagraphStyle("h", fontName=font + ("-Bold" if not scanned else ""), fontSize=13, leading=16)
    mono = ParagraphStyle("m", fontName="Courier", fontSize=8.5, leading=11)
    story: list[Any] = []
    for j, b in enumerate(doc.blocks):
        if isinstance(b, tuple):
            t = Table(b[1], hAlign="LEFT")
            t.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                        ("FONT", (0, 0), (-1, -1), font, 9),
                        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                    ]
                )
            )
            story += [Spacer(1, 4), t, Spacer(1, 4)]
        elif "   " in b and doc.meta["template"] == "text_table":
            story.append(Preformatted(b, mono))
        else:
            story.append(Paragraph(escape(b), head if j == 0 else body))
            story.append(Spacer(1, 3))
    SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=50,
        rightMargin=50,
        topMargin=50,
        bottomMargin=50,
        title=f"SYNTHETIC {doc.doc_type} {doc.doc_id}",
        author="synthetic generator",
        invariant=1,
    ).build(story)


def write_corpus(docs: list[SyntheticDoc], out_dir: Path) -> None:
    for sub in ("pdf", "text", "gold"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    for d in docs:
        render_pdf(d, out_dir / "pdf" / f"{d.doc_id}.pdf")
        (out_dir / "text" / f"{d.doc_id}.txt").write_text(d.text + "\n", encoding="utf-8")
        (out_dir / "gold" / f"{d.doc_id}.json").write_text(
            json.dumps({"doc_id": d.doc_id, "label": d.gold, "meta": d.meta, "synthetic": True}, indent=1) + "\n",
            encoding="utf-8",
        )
