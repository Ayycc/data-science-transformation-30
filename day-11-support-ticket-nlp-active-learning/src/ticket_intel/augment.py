"""Turn base ticket texts into realistic SYNTHETIC-PII tickets with gold PII spans.

Every name, email address and phone number produced here is fake and randomly generated.
The gold spans (character offsets) are what the redaction step is scored against.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

FIRST_NAMES = [
    "James",
    "Maria",
    "Wei",
    "Aisha",
    "Carlos",
    "Priya",
    "Olumide",
    "Sofia",
    "Kenji",
    "Fatima",
    "Liam",
    "Chloe",
    "Mateo",
    "Ananya",
    "Jonas",
    "Elena",
    "Tariq",
    "Mei",
    "Lucas",
    "Amara",
    "Noah",
    "Isabella",
    "Dmitri",
    "Yuki",
    "Omar",
    "Grace",
    "Rahul",
    "Ingrid",
    "Diego",
    "Zara",
    "Ethan",
    "Leila",
    "Hiroshi",
    "Nadia",
    "Samuel",
    "Camila",
    "Arjun",
    "Freya",
    "Kwame",
    "Lucia",
    "Daniel",
    "Hannah",
    "Mohammed",
    "Emma",
    "Javier",
    "Sara",
    "Andrei",
    "Chen",
    "Tomasz",
    "Ines",
]
LAST_NAMES = [
    "Smith",
    "Garcia",
    "Zhang",
    "Khan",
    "Okafor",
    "Rossi",
    "Nakamura",
    "Muller",
    "Patel",
    "Silva",
    "Johnson",
    "Dubois",
    "Kowalski",
    "Ivanova",
    "Haddad",
    "Andersen",
    "Moreno",
    "Tanaka",
    "Osei",
    "Fernandez",
    "Brown",
    "Nguyen",
    "Schmidt",
    "Kim",
    "Lopez",
    "Martin",
    "Ahmed",
    "Rodriguez",
    "Wilson",
    "Novak",
    "Costa",
    "Ali",
    "Jensen",
    "Murphy",
    "Singh",
    "Popescu",
    "Lindqvist",
    "Mensah",
    "Ortega",
    "Fischer",
    "Watanabe",
    "Hughes",
    "Bianchi",
    "Reyes",
    "Sato",
    "Clarke",
]
EMAIL_DOMAINS = ["example.com", "mail.example.org", "corp.example.net", "inbox.test", "acme-widgets.example"]

PHONE_FORMATS = [
    "({a}) 555-{d4}",
    "{a}-555-{d4}",
    "{a}.555.{d4}",
    "+1 {a} 555 {d4}",
    "+1-{a}-555-{d4}",
    "+44 20 7946 {d4}",
    "020 7946 {d4}",
    "+49 30 {d6}",
    "+33 1 {d2} {d2b} {d2c} {d2d}",
    "+91 98{d3} {d5}",
    "+61 4{d2} {d3} {d3b}",
]

ACCOUNT_TYPES = ["Premium", "Business", "Free", "Pro", "Family", "Gold", "Standard", "Enterprise"]
CITIES = ["Madrid", "Toronto", "Lagos", "Osaka", "Berlin", "Austin", "Lyon", "Pune", "Melbourne", "Leeds"]
COUNTRIES = ["Spain", "Canada", "Nigeria", "Japan", "Germany", "France", "India", "Australia", "Mexico"]
CURRENCIES = ["$", "€", "£"]

GREETINGS = [
    "Hi, this is {name}. ",
    "Hello, my name is {name}. ",
    "Hi there, {name} here. ",
    "Good morning, I am {name}. ",
    "Hey, {name} from the finance team. ",
    "Dear support, I'm {name}. ",
]
CONTACTS = [
    " You can reach me at {email}.",
    " My email is {email}.",
    " Please call me back on {phone}.",
    " My number is {phone}.",
    " Contact: {email} / {phone}.",
    " Text me at {phone} or email {email}.",
]
SIGNATURES = [
    "\n\nThanks,\n{name}",
    "\n\nRegards,\n{name}",
    "\n-- {name}",
    " Cheers, {first}",
    "\n\nBest,\n{name}\n{phone}",
    " Sent by {name} ({email})",
]
SNIPPETS_PRE = ["Hola, ", "Bonjour, ", "Hallo, ", "Olá, ", "Ciao, ", "Buenas tardes, "]
SNIPPETS_POST = [
    " Gracias de antemano.",
    " Merci beaucoup.",
    " Danke im Voraus.",
    " Obrigado!",
    " Grazie mille.",
    " Por favor, es urgente.",
    " C'est urgent, svp.",
]

PLACEHOLDER = re.compile(r"\{\{([^}]+)\}\}")


class _Builder:
    """Concatenate segments while tracking character offsets of PII segments."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.spans: list[tuple[int, int, str]] = []
        self.pos = 0

    def add(self, text: str, pii: str | None = None) -> None:
        if pii is not None:
            self.spans.append((self.pos, self.pos + len(text), pii))
        self.parts.append(text)
        self.pos += len(text)

    def add_template(self, template: str, values: dict[str, tuple[str, str | None]]) -> None:
        """`template` uses {key}; each value is (text, pii_type or None)."""
        for chunk in re.split(r"(\{\w+\})", template):
            m = re.fullmatch(r"\{(\w+)\}", chunk)
            if m:
                text, pii = values[m.group(1)]
                self.add(text, pii)
            elif chunk:
                self.add(chunk)

    def build(self) -> tuple[str, list[tuple[int, int, str]]]:
        return "".join(self.parts), list(self.spans)


def _digits(rng: np.random.Generator, n: int) -> str:
    return "".join(str(int(x)) for x in rng.integers(0, 10, size=n))


def fake_phone(rng: np.random.Generator) -> str:
    fmt = PHONE_FORMATS[int(rng.integers(len(PHONE_FORMATS)))]
    return fmt.format(
        a=str(int(rng.integers(201, 990))),
        d2=_digits(rng, 2),
        d2b=_digits(rng, 2),
        d2c=_digits(rng, 2),
        d2d=_digits(rng, 2),
        d3=_digits(rng, 3),
        d3b=_digits(rng, 3),
        d4=_digits(rng, 4),
        d5=_digits(rng, 5),
        d6=_digits(rng, 6),
    )


def fake_person(rng: np.random.Generator) -> tuple[str, str]:
    first = FIRST_NAMES[int(rng.integers(len(FIRST_NAMES)))]
    last = LAST_NAMES[int(rng.integers(len(LAST_NAMES)))]
    return first, last


def fake_email(rng: np.random.Generator, first: str, last: str) -> str:
    style = int(rng.integers(4))
    local = [
        f"{first}.{last}",
        f"{first[0]}{last}",
        f"{first}_{last}{int(rng.integers(1, 99))}",
        f"{last}.{first[0]}",
    ][style].lower()
    return f"{local}@{EMAIL_DOMAINS[int(rng.integers(len(EMAIL_DOMAINS)))]}"


def _fill_placeholder(key: str, rng: np.random.Generator) -> tuple[str, str | None]:
    key = key.strip().lower()
    if key == "person name":
        first, last = fake_person(rng)
        return f"{first} {last}", "NAME"
    if key == "order number":
        return (f"#{_digits(rng, 6)}" if rng.random() < 0.5 else f"ORD-{_digits(rng, 6)}"), None
    if key == "invoice number":
        return f"INV-{_digits(rng, 5)}", None
    if key in ("account type", "account category"):
        return ACCOUNT_TYPES[int(rng.integers(len(ACCOUNT_TYPES)))], None
    if key == "refund amount":
        return f"{int(rng.integers(5, 900))}.{_digits(rng, 2)}", None
    if key == "currency symbol":
        return CURRENCIES[int(rng.integers(len(CURRENCIES)))], None
    if key == "delivery city":
        return CITIES[int(rng.integers(len(CITIES)))], None
    if key == "delivery country":
        return COUNTRIES[int(rng.integers(len(COUNTRIES)))], None
    return key.title(), None


def augment_ticket(
    base: str, rng: np.random.Generator, pii_frac: float, multilingual_frac: float
) -> tuple[str, list[tuple[int, int, str]], bool]:
    """Return (text, gold PII spans, has_snippet)."""
    b = _Builder()
    first, last = fake_person(rng)
    name = f"{first} {last}"
    email = fake_email(rng, first, last)
    phone = fake_phone(rng)
    vals = {
        "name": (name, "NAME"),
        "first": (first, "NAME"),
        "email": (email, "EMAIL"),
        "phone": (phone, "PHONE"),
    }
    add_pii = rng.random() < pii_frac
    snippet = rng.random() < multilingual_frac
    if snippet and rng.random() < 0.5:
        b.add(SNIPPETS_PRE[int(rng.integers(len(SNIPPETS_PRE)))])
        snippet_post = False
    else:
        snippet_post = snippet
    parts = [] if not add_pii else list(rng.choice(["greet", "contact", "sign"], size=int(rng.integers(1, 4)), replace=False))
    if "greet" in parts:
        b.add_template(GREETINGS[int(rng.integers(len(GREETINGS)))], vals)
    pos = 0
    for m in PLACEHOLDER.finditer(base):
        b.add(base[pos : m.start()])
        text, pii = _fill_placeholder(m.group(1), rng)
        b.add(text, pii)
        pos = m.end()
    b.add(base[pos:])
    if "contact" in parts:
        b.add_template(CONTACTS[int(rng.integers(len(CONTACTS)))], vals)
    if snippet_post:
        b.add(SNIPPETS_POST[int(rng.integers(len(SNIPPETS_POST)))])
    if "sign" in parts:
        b.add_template(SIGNATURES[int(rng.integers(len(SIGNATURES)))], vals)
    text, spans = b.build()
    return text, spans, snippet


def augment_frame(base: pd.DataFrame, seed: int, pii_frac: float, multilingual_frac: float) -> pd.DataFrame:
    """`base` has columns text, intent, category. Returns tickets with gold spans (JSON)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, r in enumerate(base.itertuples(index=False)):
        text, spans, snip = augment_ticket(r.text, rng, pii_frac, multilingual_frac)
        rows.append(
            {
                "ticket_id": f"T{i:06d}",
                "text": text,
                "intent": r.intent,
                "category": r.category,
                "has_snippet": snip,
                "pii_spans": json.dumps(spans),
            }
        )
    return pd.DataFrame(rows)
