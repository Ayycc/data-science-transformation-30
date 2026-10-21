"""Seeded SYNTHETIC SaaS ticket generator (fallback when the public corpus is unreachable).

15 intents, templated phrasing with slot fillers, plus keyboard-style typos. PII and
multilingual snippets are added afterwards by `augment.py`, exactly as for the public corpus.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TEMPLATES: dict[str, tuple[str, list[str]]] = {
    "login_issue": (
        "ACCESS",
        [
            "i cant log in to my account since {when}",
            "login keeps failing with an error on the {surface}",
            "the sign in page just spins and never loads",
            "locked out of the dashboard after the {when} update",
            "getting invalid credentials even though my login is correct",
        ],
    ),
    "password_reset": (
        "ACCESS",
        [
            "how do i reset my password",
            "the password reset email never arrives",
            "forgot my password, need a reset link",
            "reset link says expired when i click it",
            "need to change my password asap",
        ],
    ),
    "sso_configuration": (
        "ACCESS",
        [
            "how do we set up SAML single sign on with {idp}",
            "SSO with {idp} fails with an assertion error",
            "need help configuring okta style sso for our workspace",
            "can we enforce sso for all users in the org",
            "the {idp} metadata upload for sso is rejected",
        ],
    ),
    "billing_overcharge": (
        "BILLING",
        [
            "i was charged twice this month",
            "my card was billed more than the plan price",
            "why is there an extra charge on my {when} invoice",
            "you overcharged us for seats we removed",
            "double charge on my credit card please fix",
        ],
    ),
    "refund_request": (
        "BILLING",
        [
            "i want a refund for the {plan} plan",
            "please refund my last payment",
            "can i get my money back, the tool did not work for us",
            "requesting a refund for the annual subscription",
            "how long does a refund take to arrive",
        ],
    ),
    "invoice_request": (
        "BILLING",
        [
            "where can i download my invoice for {when}",
            "need a VAT invoice with our company details",
            "please send me the receipt for the {plan} plan",
            "our accountant needs all invoices from last year",
            "how do i add a PO number to the invoice",
        ],
    ),
    "cancel_subscription": (
        "ACCOUNT",
        [
            "i want to cancel my subscription",
            "how do i stop the auto renewal",
            "please cancel our {plan} plan at the end of the term",
            "cancel my account billing, we no longer need it",
            "how to end the trial without being charged",
        ],
    ),
    "upgrade_plan": (
        "ACCOUNT",
        [
            "how do i upgrade to the {plan} plan",
            "we need more seats, can we move to {plan}",
            "what does the {plan} tier include and how to switch",
            "upgrade our workspace to annual billing",
            "can i add 10 more users to my plan",
        ],
    ),
    "account_deletion": (
        "ACCOUNT",
        [
            "delete my account and all my data",
            "i want to close my account permanently",
            "please remove my profile under gdpr",
            "how do i erase our workspace completely",
            "request for account deletion and data removal",
        ],
    ),
    "bug_report": (
        "PRODUCT",
        [
            "the {feature} button does nothing when clicked",
            "found a bug: {feature} crashes on save",
            "error 500 when opening {feature}",
            "the {surface} shows a blank screen after the update",
            "{feature} duplicates every record i create",
        ],
    ),
    "feature_request": (
        "PRODUCT",
        [
            "it would be great if {feature} supported dark mode",
            "please add bulk edit to {feature}",
            "feature request: export {feature} to excel",
            "could you add keyboard shortcuts for {feature}",
            "we would love a mobile version of {feature}",
        ],
    ),
    "performance_slow": (
        "PRODUCT",
        [
            "the {surface} is extremely slow today",
            "{feature} takes over a minute to load",
            "pages time out constantly since {when}",
            "everything is lagging, is there an outage",
            "search is really slow on large projects",
        ],
    ),
    "data_export": (
        "DATA",
        [
            "how do i export all my data to csv",
            "need a full backup of our workspace",
            "the export of {feature} is incomplete",
            "can i download my project history as json",
            "export job has been pending for hours",
        ],
    ),
    "integration_error": (
        "DATA",
        [
            "the {app} integration stopped syncing",
            "webhook to {app} returns 401",
            "cannot connect {app}, oauth fails",
            "{app} sync creates duplicate tasks",
            "our zapier style automation with {app} broke",
        ],
    ),
    "api_rate_limit": (
        "DATA",
        [
            "we keep hitting the api rate limit",
            "getting 429 too many requests from the api",
            "can you raise our api quota",
            "what is the rate limit per api key",
            "api throttling is blocking our nightly import",
        ],
    ),
}

SLOTS = {
    "when": ["yesterday", "monday", "last week", "this morning", "march", "the weekend"],
    "surface": ["web app", "mobile app", "desktop client", "admin console", "dashboard"],
    "idp": ["Okta", "Azure AD", "Google Workspace", "OneLogin", "Ping"],
    "plan": ["Pro", "Business", "Enterprise", "Starter", "Team"],
    "feature": ["reports", "the calendar", "kanban board", "invoicing", "time tracking", "the editor"],
    "app": ["Slack", "Salesforce", "Jira", "HubSpot", "Google Drive", "GitHub"],
}

PREFIXES = ["", "", "", "hi, ", "hello, ", "urgent: ", "quick question - ", "help! "]
SUFFIXES = ["", "", "", " thanks", " please help", " asap", "?", "!!"]
KEYBOARD = "abcdefghijklmnopqrstuvwxyz"


def add_typos(text: str, rng: np.random.Generator, rate: float = 0.03) -> str:
    """Random character drop / duplicate / swap / substitution on letters."""
    chars = list(text)
    out: list[str] = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if c.isalpha() and rng.random() < rate:
            op = int(rng.integers(4))
            if op == 0:
                i += 1
                continue
            if op == 1:
                out.extend([c, c])
            elif op == 2 and i + 1 < len(chars):
                out.extend([chars[i + 1], c])
                i += 2
                continue
            else:
                out.append(KEYBOARD[int(rng.integers(26))])
        else:
            out.append(c)
        i += 1
    return "".join(out)


def generate_base(per_intent: int, seed: int) -> pd.DataFrame:
    """Return a SYNTHETIC base frame with columns text, intent, category."""
    rng = np.random.default_rng(seed)
    rows = []
    for intent, (category, templates) in TEMPLATES.items():
        for _ in range(per_intent):
            t = templates[int(rng.integers(len(templates)))]
            filled = t.format(**{k: v[int(rng.integers(len(v)))] for k, v in SLOTS.items()})
            text = PREFIXES[int(rng.integers(len(PREFIXES)))] + filled + SUFFIXES[int(rng.integers(len(SUFFIXES)))]
            rows.append({"text": add_typos(text, rng), "intent": intent, "category": category})
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
