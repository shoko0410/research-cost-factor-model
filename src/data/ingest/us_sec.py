"""Deterministic SEC fundamentals transformer for US R&D/Sales facts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from ..schema.contracts import canonical_fundamental


CanonicalRecord = dict[str, object]


def normalize_cik(value: str | int) -> str:
    """Return CIK as a 10-digit zero-padded string."""

    text = str(value).strip()
    if not text:
        raise ValueError("cik cannot be empty")
    if not text.isdigit():
        raise ValueError(f"cik must contain digits only: {value!r}")
    if len(text) > 10:
        raise ValueError(f"cik must be at most 10 digits: {value!r}")
    return text.zfill(10)


def handle_missing_sec_payload(
    *,
    security_id: str,
    payload_rows: Sequence[Mapping[str, object]] | None,
) -> tuple[list[str], list[str]]:
    """Return skip metadata for missing/empty SEC payload rows."""

    if payload_rows is None:
        return [security_id], [f"{security_id}: SEC response missing"]
    if len(payload_rows) == 0:
        return [security_id], [f"{security_id}: SEC payload contains no rows"]
    return [], []


def _require_rows_fields(row: Mapping[str, object], *, index: int) -> None:
    required = ("period_end", "filing_date", "rd_expense", "sales_ttm")
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"row {index} missing required fields: {', '.join(missing)}")


def transform_us_sec_fundamentals(
    *,
    security_id: str,
    cik: str | int,
    payload_rows: Sequence[Mapping[str, object]] | None,
    country: str = "US",
) -> tuple[list[CanonicalRecord], list[str], list[str]]:
    """Transform SEC-like rows into canonical fundamental records.

    Returns ``(records, skipped, warnings)`` for deterministic offline use.
    """

    _ = normalize_cik(cik)
    skipped, warnings = handle_missing_sec_payload(security_id=security_id, payload_rows=payload_rows)
    if skipped:
        return [], skipped, warnings

    assert payload_rows is not None
    normalized: list[CanonicalRecord] = []
    for index, row in enumerate(payload_rows):
        _require_rows_fields(row, index=index)
        filing_date = row["filing_date"]
        canonical = canonical_fundamental(
            {
                "security_id": security_id,
                "country": country,
                "period_end": row["period_end"],
                "filing_date": filing_date,
                "rd_expense": row["rd_expense"],
                "sales_ttm": row["sales_ttm"],
                "as_of_date": filing_date,
                "effective_from": filing_date,
                "effective_to": None,
                "is_current": True,
            }
        )
        normalized.append(canonical)

    normalized.sort(key=lambda record: (str(record["period_end"]), str(record["filing_date"])))
    return normalized, [], []


__all__ = ["handle_missing_sec_payload", "normalize_cik", "transform_us_sec_fundamentals"]
