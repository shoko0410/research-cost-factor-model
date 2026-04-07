"""Canonical data schema contracts and PIT validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
import math


RawRecord = Mapping[str, object]
CanonicalRecord = dict[str, object]


PIT_VIOLATION = "PIT_VIOLATION"


@dataclass(frozen=True)
class ContractViolation:
    """Deterministic validation error payload."""

    marker: str
    message: str
    record_index: int | None = None
    field: str | None = None
    value: object | None = None


def _coerce_iso_date(value: object, field_name: str) -> str:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError as exc:
            raise ValueError(f"{field_name} must be ISO date (YYYY-MM-DD): {value!r}") from exc
    raise ValueError(f"{field_name} must be ISO date string or date object")


def _coerce_bool(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"{field_name} must be boolean")


def _coerce_optional_iso_date(value: object, field_name: str) -> str | None:
    if value in (None, ""):
        return None
    return _coerce_iso_date(value, field_name)


def _coerce_float(value: object, field_name: str) -> float:
    if isinstance(value, (int, float)):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"{field_name} must be finite numeric")
        return parsed
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be numeric") from exc
        if not math.isfinite(parsed):
            raise ValueError(f"{field_name} must be finite numeric")
        return parsed
    raise ValueError(f"{field_name} must be numeric")


def _require_keys(record: RawRecord, required_keys: tuple[str, ...]) -> None:
    missing = [key for key in required_keys if key not in record]
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")


def _validate_temporal_window(effective_from: str, effective_to: str | None, is_current: bool) -> None:
    start = date.fromisoformat(effective_from)
    end = date.fromisoformat(effective_to) if effective_to else None
    if end is not None and end < start:
        raise ValueError("effective_to cannot be earlier than effective_from")
    if is_current and end is not None:
        raise ValueError("is_current=True requires effective_to to be null")
    if not is_current and end is None:
        raise ValueError("is_current=False requires non-null effective_to")


def _require_text(record: RawRecord, key: str) -> str:
    value = record[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be text")
    return value


def _require_date(record: RawRecord, key: str) -> str:
    return _coerce_iso_date(record[key], key)


def _require_optional_date(record: RawRecord, key: str) -> str | None:
    return _coerce_optional_iso_date(record[key], key)


def _require_bool(record: RawRecord, key: str) -> bool:
    return _coerce_bool(record[key], key)


def _require_float(record: RawRecord, key: str) -> float:
    return _coerce_float(record[key], key)


def canonical_constituent(record: RawRecord) -> CanonicalRecord:
    required = (
        "as_of_date",
        "security_id",
        "issuer_id",
        "ticker",
        "stock_code",
        "exchange",
        "country",
        "universe",
        "effective_from",
        "effective_to",
        "is_current",
    )
    _require_keys(record, required)

    effective_from = _require_date(record, "effective_from")
    effective_to = _require_optional_date(record, "effective_to")
    is_current = _require_bool(record, "is_current")
    _validate_temporal_window(effective_from=effective_from, effective_to=effective_to, is_current=is_current)

    return {
        "as_of_date": _coerce_iso_date(record["as_of_date"], "as_of_date"),
        "security_id": _require_text(record, "security_id"),
        "issuer_id": _require_text(record, "issuer_id"),
        "ticker": _require_text(record, "ticker"),
        "stock_code": _require_text(record, "stock_code"),
        "exchange": _require_text(record, "exchange"),
        "country": _require_text(record, "country").lower(),
        "universe": _require_text(record, "universe").lower(),
        "effective_from": effective_from,
        "effective_to": effective_to,
        "is_current": is_current,
    }


def canonical_fundamental(record: RawRecord) -> CanonicalRecord:
    required = (
        "as_of_date",
        "security_id",
        "country",
        "period_end",
        "filing_date",
        "rd_expense",
        "sales_ttm",
        "effective_from",
        "effective_to",
        "is_current",
    )
    _require_keys(record, required)

    effective_from = _require_date(record, "effective_from")
    effective_to = _require_optional_date(record, "effective_to")
    is_current = _require_bool(record, "is_current")
    _validate_temporal_window(effective_from=effective_from, effective_to=effective_to, is_current=is_current)

    return {
        "as_of_date": _coerce_iso_date(record["as_of_date"], "as_of_date"),
        "security_id": _require_text(record, "security_id"),
        "country": _require_text(record, "country").lower(),
        "period_end": _require_date(record, "period_end"),
        "filing_date": _require_date(record, "filing_date"),
        "rd_expense": _require_float(record, "rd_expense"),
        "sales_ttm": _require_float(record, "sales_ttm"),
        "effective_from": effective_from,
        "effective_to": effective_to,
        "is_current": is_current,
    }


def canonical_price(record: RawRecord) -> CanonicalRecord:
    required = (
        "as_of_date",
        "security_id",
        "country",
        "price_date",
        "close",
        "currency",
        "effective_from",
        "effective_to",
        "is_current",
    )
    _require_keys(record, required)

    effective_from = _require_date(record, "effective_from")
    effective_to = _require_optional_date(record, "effective_to")
    is_current = _require_bool(record, "is_current")
    _validate_temporal_window(effective_from=effective_from, effective_to=effective_to, is_current=is_current)

    return {
        "as_of_date": _coerce_iso_date(record["as_of_date"], "as_of_date"),
        "security_id": _require_text(record, "security_id"),
        "country": _require_text(record, "country").lower(),
        "price_date": _require_date(record, "price_date"),
        "close": _require_float(record, "close"),
        "currency": _require_text(record, "currency").upper(),
        "effective_from": effective_from,
        "effective_to": effective_to,
        "is_current": is_current,
    }


def canonical_benchmark(record: RawRecord) -> CanonicalRecord:
    required = (
        "as_of_date",
        "benchmark_id",
        "country",
        "benchmark_date",
        "close",
        "currency",
        "effective_from",
        "effective_to",
        "is_current",
    )
    _require_keys(record, required)

    effective_from = _require_date(record, "effective_from")
    effective_to = _require_optional_date(record, "effective_to")
    is_current = _require_bool(record, "is_current")
    _validate_temporal_window(effective_from=effective_from, effective_to=effective_to, is_current=is_current)

    return {
        "as_of_date": _coerce_iso_date(record["as_of_date"], "as_of_date"),
        "benchmark_id": _require_text(record, "benchmark_id"),
        "country": _require_text(record, "country").lower(),
        "benchmark_date": _require_date(record, "benchmark_date"),
        "close": _require_float(record, "close"),
        "currency": _require_text(record, "currency").upper(),
        "effective_from": effective_from,
        "effective_to": effective_to,
        "is_current": is_current,
    }


def canonical_fx(record: RawRecord) -> CanonicalRecord:
    required = (
        "as_of_date",
        "pair",
        "fx_date",
        "rate",
        "effective_from",
        "effective_to",
        "is_current",
    )
    _require_keys(record, required)

    effective_from = _require_date(record, "effective_from")
    effective_to = _require_optional_date(record, "effective_to")
    is_current = _require_bool(record, "is_current")
    _validate_temporal_window(effective_from=effective_from, effective_to=effective_to, is_current=is_current)

    return {
        "as_of_date": _coerce_iso_date(record["as_of_date"], "as_of_date"),
        "pair": _require_text(record, "pair").upper(),
        "fx_date": _require_date(record, "fx_date"),
        "rate": _require_float(record, "rate"),
        "effective_from": effective_from,
        "effective_to": effective_to,
        "is_current": is_current,
    }


def validate_fundamentals_pit(
    fundamentals: Sequence[RawRecord],
    rebalance_date: str | date,
) -> list[ContractViolation]:
    """Return PIT violations where filing_date is in the future."""

    rebalance = date.fromisoformat(_coerce_iso_date(rebalance_date, "rebalance_date"))
    violations: list[ContractViolation] = []
    for index, raw_record in enumerate(fundamentals):
        record = canonical_fundamental(raw_record)
        filing_value = record["filing_date"]
        if not isinstance(filing_value, str):
            raise ValueError("filing_date must be text after canonicalization")
        filing = date.fromisoformat(filing_value)
        if filing > rebalance:
            violations.append(
                ContractViolation(
                    marker=PIT_VIOLATION,
                    message="filing_date occurs after rebalance_date",
                    record_index=index,
                    field="filing_date",
                    value=filing_value,
                )
            )
    return violations


def assert_fundamentals_pit(
    fundamentals: Sequence[RawRecord],
    rebalance_date: str | date,
) -> None:
    """Raise ValueError with PIT marker if any violation exists."""

    violations = validate_fundamentals_pit(fundamentals=fundamentals, rebalance_date=rebalance_date)
    if not violations:
        return

    details = ", ".join(
        f"idx={violation.record_index} {violation.field}={violation.value}"
        for violation in violations
    )
    raise ValueError(f"{PIT_VIOLATION}: {len(violations)} violation(s); {details}")
