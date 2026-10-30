"""Pydantic target schemas for supplier contracts and invoices.

These are the *downstream* contracts: nothing reaches the accepted store unless it validates here.
Cross-field rules (totals add up, dates in order, renewal fields consistent) live in model validators,
so a hallucinated or mis-parsed value that breaks arithmetic is rejected rather than silently written.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

Currency = Literal["USD", "EUR", "GBP"]
Money = Annotated[float, Field(ge=0, le=1e9)]
TOL = 0.011  # one cent plus float slack

CONTRACT_FIELDS: list[str] = [
    "contract_number",
    "supplier",
    "customer",
    "effective_date",
    "expiration_date",
    "total_value",
    "currency",
    "payment_terms_days",
    "auto_renewal",
    "renewal_term_months",
    "renewal_notice_days",
    "governing_law",
]
INVOICE_FIELDS: list[str] = [
    "invoice_number",
    "po_number",
    "supplier",
    "customer",
    "invoice_date",
    "due_date",
    "currency",
    "subtotal",
    "tax",
    "total",
    "payment_terms_days",
]
FIELDS_BY_TYPE: dict[str, list[str]] = {"contract": CONTRACT_FIELDS, "invoice": INVOICE_FIELDS}
DATE_FIELDS = {"effective_date", "expiration_date", "invoice_date", "due_date"}
MONEY_FIELDS = {"total_value", "subtotal", "tax", "total"}
INT_FIELDS = {"payment_terms_days", "renewal_term_months", "renewal_notice_days"}
BOOL_FIELDS = {"auto_renewal"}
PARTY_FIELDS = {"supplier", "customer"}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class LineItem(_Strict):
    description: str = Field(min_length=2, max_length=200)
    quantity: float = Field(gt=0, le=1e6)
    unit_price: Money
    amount: Money

    @model_validator(mode="after")
    def _amount_is_qty_times_price(self) -> LineItem:
        if abs(round(self.quantity * self.unit_price, 2) - self.amount) > TOL:
            raise ValueError(f"line amount {self.amount} != quantity {self.quantity} x unit_price {self.unit_price}")
        return self


class ContractRecord(_Strict):
    doc_type: Literal["contract"] = "contract"
    contract_number: str | None = Field(default=None, max_length=40)
    supplier: str = Field(min_length=2, max_length=120)
    customer: str = Field(min_length=2, max_length=120)
    effective_date: date
    expiration_date: date | None = None
    total_value: Money | None = None
    currency: Currency
    payment_terms_days: int | None = Field(default=None, ge=0, le=180)
    auto_renewal: bool
    renewal_term_months: int | None = Field(default=None, ge=1, le=120)
    renewal_notice_days: int | None = Field(default=None, ge=0, le=365)
    governing_law: str | None = Field(default=None, max_length=80)

    @model_validator(mode="after")
    def _consistency(self) -> ContractRecord:
        if self.expiration_date is not None and self.expiration_date <= self.effective_date:
            raise ValueError("expiration_date must be after effective_date")
        if self.supplier.casefold() == self.customer.casefold():
            raise ValueError("supplier and customer must differ")
        if not self.auto_renewal and self.renewal_term_months is not None:
            raise ValueError("renewal_term_months given but auto_renewal is false")
        return self


class InvoiceRecord(_Strict):
    doc_type: Literal["invoice"] = "invoice"
    invoice_number: str = Field(min_length=3, max_length=40)
    po_number: str | None = Field(default=None, max_length=40)
    supplier: str = Field(min_length=2, max_length=120)
    customer: str = Field(min_length=2, max_length=120)
    invoice_date: date
    due_date: date
    currency: Currency
    subtotal: Money
    tax: Money
    total: Money
    payment_terms_days: int | None = Field(default=None, ge=0, le=180)
    line_items: list[LineItem] = Field(min_length=1, max_length=50)

    @field_validator("invoice_number")
    @classmethod
    def _has_digit(cls, v: str) -> str:
        if not any(c.isdigit() for c in v):
            raise ValueError("invoice_number must contain a digit")
        return v

    @model_validator(mode="after")
    def _arithmetic(self) -> InvoiceRecord:
        if self.due_date < self.invoice_date:
            raise ValueError("due_date before invoice_date")
        if abs(self.subtotal + self.tax - self.total) > TOL:
            raise ValueError(f"subtotal {self.subtotal} + tax {self.tax} != total {self.total}")
        items = round(sum(li.amount for li in self.line_items), 2)
        if abs(items - self.subtotal) > TOL:
            raise ValueError(f"line items sum {items} != subtotal {self.subtotal}")
        return self


MODEL_BY_TYPE: dict[str, type[BaseModel]] = {"contract": ContractRecord, "invoice": InvoiceRecord}


def validate_record(doc_type: str, fields: dict[str, Any]) -> tuple[BaseModel | None, list[str]]:
    """Validate an extracted dict. Returns (model, []) on success or (None, error messages)."""
    model = MODEL_BY_TYPE.get(doc_type)
    if model is None:
        return None, [f"unknown doc_type {doc_type!r}"]
    payload = {k: v for k, v in fields.items() if k != "doc_type"}
    try:
        return model.model_validate({"doc_type": doc_type, **payload}), []
    except ValidationError as exc:
        errs = []
        for e in exc.errors():
            loc = ".".join(str(p) for p in e["loc"]) or "<record>"
            errs.append(f"{loc}: {e['msg']}")
        return None, errs
