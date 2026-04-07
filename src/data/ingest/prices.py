"""Price ingestion normalization with deterministic source fallback."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from ..schema.contracts import canonical_price


RawRecord = Mapping[str, object]
CanonicalPriceRecord = dict[str, object]


def _normalize_source_name(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("source must be text")
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError("source cannot be empty")
    return normalized


def _normalize_quality_flag(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("quality_flag must be text")
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError("quality_flag cannot be empty")
    return normalized


@dataclass(frozen=True)
class PriceSourceSelection:
    """Selected source payload with deterministic fallback metadata."""

    source: str
    rows: tuple[RawRecord, ...]
    fallback_used: bool
    primary_source: str
    quality_flag: str


def select_price_source(
    rows_by_source: Mapping[str, Sequence[RawRecord]],
    source_priority: Sequence[str],
) -> PriceSourceSelection:
    """Pick the first source from priority list that has rows."""

    if not source_priority:
        raise ValueError("source_priority must contain at least one source")

    normalized_rows_by_source: dict[str, tuple[RawRecord, ...]] = {}
    for source_name, rows in rows_by_source.items():
        normalized_source = _normalize_source_name(source_name)
        if normalized_source in normalized_rows_by_source:
            raise ValueError(f"duplicate source after normalization: {normalized_source}")
        normalized_rows_by_source[normalized_source] = tuple(rows)

    normalized_priority = tuple(_normalize_source_name(source) for source in source_priority)
    primary_source = normalized_priority[0]

    for source_name in normalized_priority:
        rows = normalized_rows_by_source.get(source_name, ())
        if not rows:
            continue
        fallback_used = source_name != primary_source
        quality_flag = "ok" if not fallback_used else "fallback_source"
        return PriceSourceSelection(
            source=source_name,
            rows=rows,
            fallback_used=fallback_used,
            primary_source=primary_source,
            quality_flag=quality_flag,
        )

    priority_text = ", ".join(normalized_priority)
    raise ValueError(f"no price rows available for source priority: {priority_text}")


def normalize_price_row(
    raw_row: RawRecord,
    *,
    source: str,
    quality_flag: str,
) -> CanonicalPriceRecord:
    """Normalize one raw price row into canonical schema plus source metadata."""

    canonical_row = canonical_price(raw_row)
    normalized_source = _normalize_source_name(source)
    normalized_quality_flag = _normalize_quality_flag(quality_flag)
    return {
        "as_of_date": canonical_row["as_of_date"],
        "security_id": canonical_row["security_id"],
        "country": canonical_row["country"],
        "price_date": canonical_row["price_date"],
        "close": canonical_row["close"],
        "currency": canonical_row["currency"],
        "effective_from": canonical_row["effective_from"],
        "effective_to": canonical_row["effective_to"],
        "is_current": canonical_row["is_current"],
        "source": normalized_source,
        "quality_flag": normalized_quality_flag,
    }


def normalize_price_rows(
    raw_rows: Iterable[RawRecord],
    *,
    source: str,
    quality_flag: str,
) -> list[CanonicalPriceRecord]:
    """Normalize a collection of raw price rows with source metadata."""

    return [
        normalize_price_row(raw_row, source=source, quality_flag=quality_flag)
        for raw_row in raw_rows
    ]


def normalize_prices_with_source_fallback(
    rows_by_source: Mapping[str, Sequence[RawRecord]],
    source_priority: Sequence[str],
) -> tuple[list[CanonicalPriceRecord], PriceSourceSelection]:
    """Select source by priority, then normalize selected rows."""

    selection = select_price_source(rows_by_source=rows_by_source, source_priority=source_priority)
    normalized = normalize_price_rows(
        selection.rows,
        source=selection.source,
        quality_flag=selection.quality_flag,
    )
    return normalized, selection


__all__ = [
    "CanonicalPriceRecord",
    "PriceSourceSelection",
    "normalize_price_row",
    "normalize_price_rows",
    "normalize_prices_with_source_fallback",
    "select_price_source",
]
