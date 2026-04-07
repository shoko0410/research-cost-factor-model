"""Deterministic KR fundamentals transformer for DART-like payloads."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime
from ..schema.contracts import canonical_fundamental


class UnmappedCorpCodeError(ValueError):
    """Raised when a KR corp code cannot be resolved to a stock code."""


def _clean_text(value: object) -> str:
    return str(value or "").strip()


def _to_iso_date(value: object, *, field_name: str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()

    text = _clean_text(value)
    if not text:
        raise ValueError(f"{field_name} is required")

    if len(text) == 8 and text.isdigit():
        return f"{text[0:4]}-{text[4:6]}-{text[6:8]}"

    if "/" in text:
        text = text.replace("/", "-")

    return date.fromisoformat(text).isoformat()


def build_corp_code_stock_code_map(
    rows: Iterable[Mapping[str, object]],
    *,
    corp_code_field: str = "corp_code",
    stock_code_field: str = "stock_code",
) -> dict[str, str]:
    """Create a deterministic corp_code -> stock_code lookup map."""

    mapping: dict[str, str] = {}
    for row in rows:
        corp_code = _clean_text(row.get(corp_code_field))
        stock_code = _clean_text(row.get(stock_code_field))
        if not corp_code or not stock_code:
            continue
        mapping[corp_code] = stock_code
    return mapping


def _resolve_stock_code(
    row: Mapping[str, object],
    *,
    corp_code_to_stock_code: Mapping[str, str],
    strict_unmapped_corp_code: bool,
    warnings: list[str] | None,
) -> str | None:
    stock_code = _clean_text(row.get("stock_code"))
    if stock_code:
        return stock_code

    corp_code = _clean_text(row.get("corp_code"))
    mapped = _clean_text(corp_code_to_stock_code.get(corp_code))
    if mapped:
        return mapped

    message = f"unmapped corp_code: {corp_code or '<missing>'}"
    if strict_unmapped_corp_code:
        raise UnmappedCorpCodeError(message)
    if warnings is not None:
        warnings.append(message)
    return None


def _first_present(*values: object) -> object:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def transform_kr_dart_fundamentals(
    raw_rows: Iterable[Mapping[str, object]],
    *,
    corp_code_to_stock_code: Mapping[str, str] | None = None,
    strict_unmapped_corp_code: bool = True,
    warnings: list[str] | None = None,
) -> list[dict[str, object]]:
    """Transform KR DART-like rows into canonical fundamental records."""

    mapping = corp_code_to_stock_code or {}
    canonical_records: list[dict[str, object]] = []

    for row in raw_rows:
        stock_code = _resolve_stock_code(
            row,
            corp_code_to_stock_code=mapping,
            strict_unmapped_corp_code=strict_unmapped_corp_code,
            warnings=warnings,
        )
        if stock_code is None:
            continue

        filing_date = _to_iso_date(row.get("filing_date") or row.get("rcept_dt"), field_name="filing_date")
        period_end_source = row.get("period_end") or row.get("bsns_year_end") or row.get("thstrm_dt")
        as_of_source = row.get("as_of_date") or filing_date

        record_payload: dict[str, object] = {
            "as_of_date": _to_iso_date(as_of_source, field_name="as_of_date"),
            "security_id": f"KRX:{stock_code}",
            "country": "kr",
            "period_end": _to_iso_date(period_end_source, field_name="period_end"),
            "filing_date": filing_date,
            "rd_expense": _first_present(row.get("rd_expense"), row.get("rnd_expense")),
            "sales_ttm": _first_present(row.get("sales_ttm"), row.get("revenue_ttm"), row.get("sales")),
            "effective_from": filing_date,
            "effective_to": "",
            "is_current": True,
        }
        canonical_records.append(canonical_fundamental(record_payload))

    return canonical_records


__all__ = [
    "UnmappedCorpCodeError",
    "build_corp_code_stock_code_map",
    "transform_kr_dart_fundamentals",
]
