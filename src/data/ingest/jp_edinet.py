"""JP fundamentals ingestion parser for EDINET-like payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import cast

from ..schema import canonical_fundamental


RawRow = Mapping[str, object]
CanonicalRecord = dict[str, object]


@dataclass(frozen=True)
class IngestFailure:
    """Deterministic failure payload for malformed rows."""

    row_index: int
    message: str


def parse_jp_edinet_fundamentals(
    payload: Mapping[str, object] | Sequence[object],
    *,
    as_of_date: object | None = None,
    default_exchange: str = "TSE",
) -> tuple[list[CanonicalRecord], list[IngestFailure]]:
    """Parse EDINET-like payload into canonical fundamentals records."""

    rows = _extract_rows(payload)
    records: list[CanonicalRecord] = []
    failures: list[IngestFailure] = []

    for row_index, row_obj in enumerate(rows):
        if not isinstance(row_obj, Mapping):
            failures.append(IngestFailure(row_index=row_index, message="row must be a mapping"))
            continue

        try:
            row = _as_raw_row(cast(Mapping[object, object], row_obj))
            canonical_input = _normalize_row(
                row,
                as_of_date=as_of_date,
                default_exchange=default_exchange,
            )
            records.append(canonical_fundamental(canonical_input))
        except (TypeError, ValueError) as exc:
            failures.append(IngestFailure(row_index=row_index, message=str(exc)))

    return records, failures


def _extract_rows(payload: object) -> Sequence[object]:
    if isinstance(payload, Mapping):
        for key in ("results", "rows", "records", "xbrl_rows", "data"):
            if key in payload:
                rows = payload[key]
                if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
                    raise ValueError(f"{key} must be a sequence of rows")
                return rows
        raise ValueError("payload mapping must include one of: results, rows, records, xbrl_rows, data")

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return payload
    raise ValueError("payload must be a mapping or a sequence of rows")


def _as_raw_row(row_obj: Mapping[object, object]) -> RawRow:
    if not all(isinstance(key, str) for key in row_obj):
        raise ValueError("row keys must be text")
    return cast(RawRow, row_obj)


def _normalize_row(
    row: RawRow,
    *,
    as_of_date: object | None,
    default_exchange: str,
) -> CanonicalRecord:
    filing_date = _normalize_iso_date(
        _pick(
            row,
            "filing_date",
            "filingDate",
            "submit_date",
            "submitDate",
            "submitDateTime",
            "submit_datetime",
        ),
        "filing_date",
    )
    period_end = _normalize_iso_date(
        _pick(
            row,
            "period_end",
            "periodEnd",
            "currentFiscalYearEndDateDEI",
            "fiscal_year_end",
        ),
        "period_end",
    )

    stock_code = _normalize_stock_code(
        _pick(row, "jp_code", "stock_code", "code", "secCode", "securityCode")
    )
    exchange = _normalize_exchange(_pick(row, "exchange", "market", required=False), default=default_exchange)
    security_id = _normalize_security_id(
        _pick(row, "security_id", "securityId", required=False),
        stock_code=stock_code,
        exchange=exchange,
    )

    rd_expense = _normalize_number(
        _pick(
            row,
            "rd_expense",
            "research_and_development_expense",
            "researchDevelopmentExpense",
            "ResearchAndDevelopmentExpenses",
        ),
        "rd_expense",
    )
    sales_ttm = _normalize_number(
        _pick(
            row,
            "sales_ttm",
            "net_sales_ttm",
            "sales",
            "NetSales",
            "RevenueIFRS",
        ),
        "sales_ttm",
    )

    effective_from = _normalize_iso_date(
        _pick(row, "effective_from", "effectiveFrom", required=False, default=filing_date),
        "effective_from",
    )
    effective_to_raw = _pick(row, "effective_to", "effectiveTo", required=False, default=None)
    effective_to = None if effective_to_raw in (None, "") else _normalize_iso_date(effective_to_raw, "effective_to")
    is_current = _normalize_bool(
        _pick(row, "is_current", "isCurrent", required=False, default=effective_to is None),
        "is_current",
    )
    as_of = _normalize_iso_date(
        as_of_date
        if as_of_date is not None
        else _pick(row, "as_of_date", "asOfDate", "snapshot_date", required=False, default=filing_date),
        "as_of_date",
    )

    return {
        "as_of_date": as_of,
        "security_id": security_id,
        "country": "jp",
        "period_end": period_end,
        "filing_date": filing_date,
        "rd_expense": rd_expense,
        "sales_ttm": sales_ttm,
        "effective_from": effective_from,
        "effective_to": effective_to,
        "is_current": is_current,
    }


def _pick(
    row: RawRow,
    *keys: str,
    required: bool = True,
    default: object | None = None,
) -> object:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    if required:
        raise ValueError(f"missing required field: {'/'.join(keys)}")
    return default


def _normalize_iso_date(value: object, field_name: str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{field_name} is required")
        if len(text) >= 10 and text[4] == "-" and text[7] == "-":
            try:
                return date.fromisoformat(text[:10]).isoformat()
            except ValueError:
                pass
        for fmt in ("%Y%m%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(text, fmt).date().isoformat()
            except ValueError:
                continue
    raise ValueError(f"{field_name} must be a valid date")


def _normalize_bool(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{field_name} must be boolean")


def _normalize_number(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if text.startswith("(") and text.endswith(")"):
            text = f"-{text[1:-1]}"
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be numeric") from exc
    raise ValueError(f"{field_name} must be numeric")


def _normalize_stock_code(value: object) -> str:
    if isinstance(value, int):
        digits = str(value)
    elif isinstance(value, str):
        digits = "".join(ch for ch in value if ch.isdigit())
    else:
        raise ValueError("stock_code must be text or integer")

    if not digits:
        raise ValueError("stock_code must contain digits")
    return digits.zfill(4) if len(digits) < 4 else digits


def _normalize_exchange(value: object, *, default: str) -> str:
    if value in (None, ""):
        return default
    if not isinstance(value, str):
        raise ValueError("exchange must be text")

    normalized = value.strip().upper()
    aliases = {
        "TOKYO": "TSE",
        "T": "TSE",
        "XTKS": "TSE",
    }
    return aliases.get(normalized, normalized)


def _normalize_security_id(value: object, *, stock_code: str, exchange: str) -> str:
    if value in (None, ""):
        return f"JP:{stock_code}:{exchange}"
    if not isinstance(value, str):
        raise ValueError("security_id must be text")

    text = value.strip().upper()
    if text.count(":") == 2:
        _country, code, venue = text.split(":")
        return f"JP:{_normalize_stock_code(code)}:{_normalize_exchange(venue, default=exchange)}"

    return f"JP:{_normalize_stock_code(text)}:{exchange}" if text.isdigit() else text
