"""SYNTHETIC regulation corpus and gold Q&A generator (seeded).

The corpus imitates the structure of a real regulation: Parts -> Sections (with headings) -> lettered clauses.
Every section regulates one topic ("customer complaints", "vendor access logs", ...) through five clause types
(deadline, retention, penalty, approval, exemption), so clauses in different sections are near-duplicates that
differ only in their topic and their facts. That makes retrieval genuinely hard: the right clause must match both
the topic and the kind of obligation asked about.

Terminology varies the way it does in real drafting: each mention of a topic qualifier, topic noun, actor,
authority or verb is drawn from a synonym set (canonical form with probability 0.55). Section headings always use
the canonical form. Gold questions come in two styles:

* ``lexical``    - the question reuses the exact words of the supporting clause;
* ``paraphrase`` - the question frame is reworded and synonym-bearing words are swapped for variants *not*
                   used in the supporting clause (topic words with p=0.6, at least one always; generic words p=0.7).

All text is SYNTHETIC. It is not a real regulation and must not be relied on as legal guidance.
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass, field

SEED = 20261031
TITLE = "Model Data Protection and Conduct Regulation (SYNTHETIC)"

QUALIFIERS: dict[str, list[str]] = {
    "customer": ["customer", "client", "consumer"],
    "employee": ["employee", "staff", "personnel"],
    "vendor": ["vendor", "supplier", "contractor"],
    "payment": ["payment", "transaction", "remittance"],
    "marketing": ["marketing", "promotional", "advertising"],
    "biometric": ["biometric", "fingerprint", "facial-recognition"],
    "location": ["location", "geolocation", "positioning"],
    "health": ["health", "medical", "clinical"],
    "credit": ["credit", "lending", "loan"],
    "account": ["account", "deposit", "savings"],
}
NOUNS: dict[str, list[str]] = {
    "records": ["records", "files", "documentation"],
    "complaints": ["complaints", "grievances", "disputes"],
    "consent forms": ["consent forms", "authorisations", "permission slips"],
    "transfer agreements": ["transfer agreements", "data-sharing contracts", "export arrangements"],
    "risk assessments": ["risk assessments", "impact evaluations", "hazard reviews"],
    "access logs": ["access logs", "audit trails", "usage journals"],
}
GENERIC: dict[str, list[str]] = {
    "actor": ["regulated entity", "covered institution", "licensed firm"],
    "authority": ["Supervisory Authority", "Commission", "regulator"],
    "notify": ["notify", "inform", "alert"],
    "retain": ["retain", "keep", "preserve"],
    "event": ["security incident", "data breach", "unauthorised disclosure"],
    "approved": ["approved", "signed off", "endorsed"],
    "contravenes": ["contravenes", "breaches", "violates"],
}
OFFICERS = [
    "Chief Compliance Officer",
    "Data Protection Officer",
    "Chief Risk Officer",
    "board audit committee",
    "Head of Internal Audit",
]
FREQUENCIES = ["annually", "semi-annually", "quarterly", "every 24 months"]
TRIGGERS = [
    "discovery of the incident",
    "receipt of a written request",
    "the end of the reporting quarter",
    "becoming aware of the event",
]
RET_TRIGGERS = ["creation", "the last transaction", "account closure", "the end of the business relationship"]
CLAUSE_TYPES = ["deadline", "retention", "penalty", "approval", "exemption"]
LETTERS = "abcdefgh"

QUESTION_FRAMES: dict[str, dict[str, list[str]]] = {
    "deadline": {
        "lexical": [
            "Within how many days must a {actor} {notify} the {authority} of a {event} affecting {q} {n}?",
            "What is the deadline for a {actor} to {notify} the {authority} of any {event} affecting {q} {n}?",
        ],
        "paraphrase": [
            "How quickly does a {actor} have to {notify} the {authority} after a {event} involving {q} {n}?",
            "If a {event} hits {q} {n}, how long does the {actor} have to {notify} the {authority}?",
        ],
    },
    "retention": {
        "lexical": [
            "For how many years must a {actor} {retain} {q} {n}?",
            "What is the minimum period for which {q} {n} must be {retained}?",
        ],
        "paraphrase": [
            "How long does a {actor} need to {retain} its {q} {n}?",
            "What is the shortest time {q} {n} can be {retained} before disposal?",
        ],
    },
    "penalty": {
        "lexical": [
            "What civil penalty applies to a {actor} that {contravenes} the requirements on {q} {n}?",
            "What is the maximum civil penalty for each violation of the rules on {q} {n}?",
        ],
        "paraphrase": [
            "How large a fine can a {actor} face when it {contravenes} the rules about {q} {n}?",
            "What sanction per violation can be imposed over {q} {n} non-compliance?",
        ],
    },
    "approval": {
        "lexical": [
            "Who must review and approve {q} {n}, and how often?",
            "By whom must {q} {n} be reviewed and {approved}?",
        ],
        "paraphrase": [
            "Which officer is responsible for signing off {q} {n}?",
            "Whose sign-off do {q} {n} need, and at what interval?",
        ],
    },
    "exemption": {
        "lexical": [
            "Which firms are exempt from the requirements on {q} {n}?",
            "When do the requirements of the section on {q} {n} not apply to a {actor}?",
        ],
        "paraphrase": [
            "Are small firms excused from the {q} {n} rules?",
            "Is there a size threshold below which a {actor} can ignore the {q} {n} obligations?",
        ],
    },
}

OUT_OF_SCOPE_OBJECTS = [
    "annual income tax returns",
    "building permits for garages",
    "commercial fishing licences",
    "aircraft maintenance logs",
    "restaurant hygiene inspections",
    "school enrolment forms",
    "vehicle emissions certificates",
    "wine import labels",
    "firearm registrations",
    "patent renewal fees",
    "jury duty summons",
    "parking permits",
]
OUT_OF_SCOPE_FRAMES = [
    "What is the deadline for filing {x}?",
    "Who must approve {x}?",
    "What is the maximum penalty for late {x}?",
    "How long must {x} be kept?",
]


@dataclass
class Clause:
    clause_id: str  # e.g. "§3.2(b)"
    section_id: str  # e.g. "3.2"
    part: int
    heading: str
    letter: str
    text: str
    clause_type: str  # deadline | retention | penalty | approval | exemption | general | definition
    topic: str = ""  # canonical "qualifier noun"
    answer: str = ""  # the fact a question about this clause should return
    variants: dict[str, str] = field(default_factory=dict)  # synonym-set key -> variant used in this clause


@dataclass
class GoldQuestion:
    qid: str
    question: str
    style: str  # lexical | paraphrase | out_of_scope
    clause_type: str
    gold_clause_ids: list[str]
    answer: str
    in_scope: bool
    split: str  # dev | test


def _pick(rng: random.Random, options: list[str], p_canonical: float = 0.55) -> str:
    if rng.random() < p_canonical:
        return options[0]
    return rng.choice(options[1:])


def _other(rng: random.Random, options: list[str], used: str) -> str:
    rest = [o for o in options if o != used]
    return rng.choice(rest)


def _a(word: str) -> str:
    return ("an " if word[0].lower() in "aeiou" else "a ") + word


def _render_clause(rng: random.Random, ctype: str, qual: str, noun: str) -> tuple[str, str, dict[str, str]]:
    v = {
        "q": _pick(rng, QUALIFIERS[qual]),
        "n": _pick(rng, NOUNS[noun]),
        **{k: _pick(rng, opts) for k, opts in GENERIC.items()},
    }
    if ctype == "deadline":
        days = rng.choice([3, 5, 7, 10, 14, 15, 20, 21, 30, 45, 60, 72, 90])
        daytype = rng.choice(["business", "calendar"])
        answer = f"{days} {daytype} days"
        text = (
            f"{_a(v['actor']).capitalize()} shall {v['notify']} the {v['authority']} of any {v['event']} affecting "
            f"{v['q']} {v['n']} within {answer} of {rng.choice(TRIGGERS)}."
        )
    elif ctype == "retention":
        years = rng.choice([2, 3, 4, 5, 6, 7, 8, 10, 12, 15])
        answer = f"{years} years"
        text = (
            f"{_a(v['actor']).capitalize()} shall {v['retain']} all {v['q']} {v['n']} for a period of not less than "
            f"{answer} from the date of {rng.choice(RET_TRIGGERS)}."
        )
    elif ctype == "penalty":
        amount = rng.choice([5, 10, 15, 20, 25, 40, 50, 75, 100, 150, 250, 500]) * 1000
        answer = f"${amount:,}"
        text = (
            f"{_a(v['actor']).capitalize()} that {v['contravenes']} the requirements on {v['q']} {v['n']} in this "
            f"section is liable to a civil penalty of up to {answer} for each violation."
        )
    elif ctype == "approval":
        officer = rng.choice(OFFICERS)
        freq = rng.choice(FREQUENCIES)
        answer = f"the {officer}"
        text = f"All {v['q']} {v['n']} shall be reviewed and {v['approved']} by the {officer} at least {freq}."
        answer = f"the {officer}, {freq}"
    elif ctype == "exemption":
        k = rng.choice([50, 100, 250, 500, 1000, 2500, 5000, 10000])
        answer = f"fewer than {k:,}"
        text = (
            f"The requirements of this section on {v['q']} {v['n']} do not apply to {_a(v['actor'])} that serves "
            f"{answer} data subjects in a calendar year."
        )
    else:  # pragma: no cover - guarded by CLAUSE_TYPES
        raise ValueError(ctype)
    return text, answer, v


def generate_corpus(seed: int = SEED, n_parts: int = 10, sections_per_part: int = 5) -> list[Clause]:
    rng = random.Random(seed)
    clauses: list[Clause] = []
    # Part 1: scope and definitions (no questions are asked about it; it is realistic distractor text and it
    # states the synonym equivalences, as definitions sections in real regulations do).
    defs = [
        (
            "1.1",
            "Scope",
            "general",
            [
                f"This Regulation applies to every {GENERIC['actor'][0]}, {GENERIC['actor'][1]} and {GENERIC['actor'][2]} that processes personal data in the course of business.",
                f"Nothing in this Regulation limits the powers of the {GENERIC['authority'][0]} under any other enactment.",
            ],
        )
    ]
    def_lines = []
    for opts in QUALIFIERS.values():
        def_lines.append(f"A reference to {opts[0]} data includes {opts[1]} data and {opts[2]} data.")
    for n, opts in NOUNS.items():
        def_lines.append(f'"{n.capitalize()}" includes {opts[1]} and {opts[2]}.')
    def_lines.append(f'"{GENERIC["event"][0].capitalize()}" includes any {GENERIC["event"][1]} or {GENERIC["event"][2]}.')
    defs.append(("1.2", "Definitions", "definition", def_lines))
    for sid, heading, ctype, lines in defs:
        for i, line in enumerate(lines):
            letter = LETTERS[i] if i < len(LETTERS) else f"{LETTERS[i // len(LETTERS) - 1]}{LETTERS[i % len(LETTERS)]}"
            clauses.append(Clause(f"§{sid}({letter})", sid, 1, heading, letter, line, ctype))

    quals = list(QUALIFIERS)
    rng.shuffle(quals)
    for p_idx, qual in enumerate(quals[:n_parts], start=2):
        nouns = rng.sample(list(NOUNS), sections_per_part)
        for s_idx, noun in enumerate(nouns, start=1):
            sid = f"{p_idx}.{s_idx}"
            heading = f"{qual.capitalize()} {noun}"
            types = CLAUSE_TYPES[:]
            rng.shuffle(types)
            for c_idx, ctype in enumerate(types):
                text, answer, variants = _render_clause(rng, ctype, qual, noun)
                letter = LETTERS[c_idx]
                clauses.append(
                    Clause(f"§{sid}({letter})", sid, p_idx, heading, letter, text, ctype, f"{qual} {noun}", answer, variants)
                )
            letter = LETTERS[len(types)]
            general = f"A {_pick(rng, GENERIC['actor'])} shall make the policies required by this section available to the {_pick(rng, GENERIC['authority'])} on request."
            clauses.append(Clause(f"§{sid}({letter})", sid, p_idx, heading, letter, general, "general", f"{qual} {noun}"))
    return clauses


def _fill(frame: str, vals: dict[str, str]) -> str:
    out = frame.format(**vals)
    out = re.sub(r"\ba ([aeiouAEIOU])", r"an \1", out)
    return out[0].upper() + out[1:]


def _retained(word: str) -> str:
    return {"retain": "retained", "keep": "kept", "preserve": "preserved"}[word]


def generate_questions(clauses: list[Clause], seed: int = SEED, test_fraction: float = 0.5) -> list[GoldQuestion]:
    """Two gold questions per answerable clause (one per style) plus out-of-scope questions.

    The dev/test split is by *section*, so a test question never shares its supporting section with a dev question.
    """
    rng = random.Random(seed + 1)
    sections = sorted({c.section_id for c in clauses if c.clause_type in CLAUSE_TYPES})
    rng.shuffle(sections)
    n_test = int(round(len(sections) * test_fraction))
    test_sections = set(sections[:n_test])
    questions: list[GoldQuestion] = []
    for c in clauses:
        if c.clause_type not in CLAUSE_TYPES:
            continue
        qual, noun = _split_topic(c.topic)
        split = "test" if c.section_id in test_sections else "dev"
        for style in ("lexical", "paraphrase"):
            if style == "lexical":
                vals = {"q": c.variants["q"], "n": c.variants["n"], **{k: c.variants[k] for k in GENERIC}}
            else:
                # Users reuse some of the regulation's words but not all: each topic word is swapped with
                # probability 0.6 (at least one of the two is always swapped), each generic word with 0.7.
                swap_q = rng.random() < 0.6
                swap_n = rng.random() < 0.6
                if not (swap_q or swap_n):
                    swap_q, swap_n = (True, False) if rng.random() < 0.5 else (False, True)
                vals = {
                    "q": _other(rng, QUALIFIERS[qual], c.variants["q"]) if swap_q else c.variants["q"],
                    "n": _other(rng, NOUNS[noun], c.variants["n"]) if swap_n else c.variants["n"],
                    **{k: _other(rng, GENERIC[k], c.variants[k]) if rng.random() < 0.7 else c.variants[k] for k in GENERIC},
                }
            vals["retained"] = _retained(vals["retain"])
            frame = rng.choice(QUESTION_FRAMES[c.clause_type][style])
            qid = f"Q{len(questions) + 1:04d}"
            questions.append(GoldQuestion(qid, _fill(frame, vals), style, c.clause_type, [c.clause_id], c.answer, True, split))
    oos = [f.format(x=x) for x in OUT_OF_SCOPE_OBJECTS for f in OUT_OF_SCOPE_FRAMES]
    rng.shuffle(oos)
    for i, q in enumerate(oos):
        qid = f"Q{len(questions) + 1:04d}"
        questions.append(GoldQuestion(qid, q, "out_of_scope", "none", [], "", False, "test" if i % 2 == 0 else "dev"))
    return questions


def _split_topic(topic: str) -> tuple[str, str]:
    qual, noun = topic.split(" ", 1)
    return qual, noun


def render_document(clauses: list[Clause]) -> str:
    """Flatten the corpus into one plain-text document (used by fixed-size chunking)."""
    lines = [TITLE, ""]
    last_section = None
    for c in clauses:
        if c.section_id != last_section:
            lines += ["", f"§ {c.section_id} {c.heading}"]
            last_section = c.section_id
        lines.append(f"({c.letter}) {c.text}")
    return "\n".join(lines) + "\n"


def to_records(items: list) -> list[dict]:
    return [asdict(x) for x in items]
