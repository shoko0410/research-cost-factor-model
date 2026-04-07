"""Constituent SCD2 utilities and deterministic month-end cache builder."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date


class ConstituentSCD2Error(ValueError):
    """Base error for constituent SCD2 validation failures."""


class ConstituentOverlapError(ConstituentSCD2Error):
    """Raised when SCD2 windows overlap for a universe/security pair."""


def _to_date(value: object, field_name: str) -> date:
    if isinstance(value, date):
        return value
    if value is None:
        raise ConstituentSCD2Error(f"{field_name} is required")
    text = str(value).strip()
    if not text:
        raise ConstituentSCD2Error(f"{field_name} is required")
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ConstituentSCD2Error(f"{field_name} must be ISO date (YYYY-MM-DD)") from exc


def _to_optional_date(value: object, field_name: str) -> date | None:
    if value in (None, ""):
        return None
    return _to_date(value, field_name)


def _to_bool(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    raise ConstituentSCD2Error(f"{field_name} must be boolean")


def _to_text(value: object, field_name: str, *, lowercase: bool = False) -> str:
    if not isinstance(value, str):
        raise ConstituentSCD2Error(f"{field_name} must be text")
    text = value.strip()
    if not text:
        raise ConstituentSCD2Error(f"{field_name} cannot be empty")
    if lowercase:
        return text.lower()
    return text


@dataclass(frozen=True)
class ConstituentSCD2Row:
    """Canonical constituent SCD2 row with inclusive validity bounds."""

    universe: str
    security_id: str
    effective_from: date
    effective_to: date | None
    is_current: bool

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "ConstituentSCD2Row":
        required = ("universe", "security_id", "effective_from", "effective_to", "is_current")
        missing = [key for key in required if key not in row]
        if missing:
            raise ConstituentSCD2Error(f"missing required fields: {', '.join(missing)}")

        universe = _to_text(row["universe"], "universe", lowercase=True)
        security_id = _to_text(row["security_id"], "security_id")
        effective_from = _to_date(row["effective_from"], "effective_from")
        effective_to = _to_optional_date(row["effective_to"], "effective_to")
        is_current = _to_bool(row["is_current"], "is_current")

        if effective_to is not None and effective_to < effective_from:
            raise ConstituentSCD2Error("effective_to cannot be earlier than effective_from")
        if is_current and effective_to is not None:
            raise ConstituentSCD2Error("is_current=True requires effective_to to be null")
        if not is_current and effective_to is None:
            raise ConstituentSCD2Error("is_current=False requires non-null effective_to")

        return cls(
            universe=universe,
            security_id=security_id,
            effective_from=effective_from,
            effective_to=effective_to,
            is_current=is_current,
        )

    def is_active_on(self, as_of: date) -> bool:
        if as_of < self.effective_from:
            return False
        if self.effective_to is not None and as_of > self.effective_to:
            return False
        return True


def _canonicalize_rows(rows: Iterable[ConstituentSCD2Row | Mapping[str, object]]) -> tuple[ConstituentSCD2Row, ...]:
    canonical: list[ConstituentSCD2Row] = []
    for row in rows:
        if isinstance(row, ConstituentSCD2Row):
            canonical.append(row)
            continue
        canonical.append(ConstituentSCD2Row.from_mapping(row))
    return tuple(
        sorted(
            canonical,
            key=lambda item: (
                item.universe,
                item.security_id,
                item.effective_from,
                item.effective_to or date.max,
            ),
        )
    )


def assert_no_overlaps(rows: Iterable[ConstituentSCD2Row | Mapping[str, object]]) -> tuple[ConstituentSCD2Row, ...]:
    """Return canonical rows or raise ConstituentOverlapError on overlap."""

    canonical = _canonicalize_rows(rows)
    grouped: dict[tuple[str, str], list[ConstituentSCD2Row]] = {}
    for row in canonical:
        grouped.setdefault((row.universe, row.security_id), []).append(row)

    for (universe, security_id), windows in grouped.items():
        if len(windows) < 2:
            continue
        previous = windows[0]
        previous_end = previous.effective_to or date.max
        for current in windows[1:]:
            if current.effective_from <= previous_end:
                previous_effective_to = previous.effective_to.isoformat() if previous.effective_to else "null"
                current_effective_to = current.effective_to.isoformat() if current.effective_to else "null"
                message = (
                    "overlapping constituent windows detected for "
                    f"(universe={universe!r}, security_id={security_id!r}): "
                    f"[{previous.effective_from.isoformat()}, {previous_effective_to}] overlaps "
                    f"[{current.effective_from.isoformat()}, {current_effective_to}]"
                )
                raise ConstituentOverlapError(
                    message
                )
            previous = current
            previous_end = previous.effective_to or date.max

    return canonical


def _membership_for_date(
    rows: tuple[ConstituentSCD2Row, ...],
    *,
    universe: str,
    as_of: date,
) -> tuple[str, ...]:
    universe_key = _to_text(universe, "universe", lowercase=True)
    members = {
        row.security_id
        for row in rows
        if row.universe == universe_key and row.is_active_on(as_of)
    }
    return tuple(sorted(members))


def membership_for_date(
    rows: Iterable[ConstituentSCD2Row | Mapping[str, object]],
    *,
    universe: str,
    as_of_date: date | str,
) -> tuple[str, ...]:
    """Return deterministic security membership for a date."""

    canonical = assert_no_overlaps(rows)
    as_of = _to_date(as_of_date, "as_of_date")
    return _membership_for_date(canonical, universe=universe, as_of=as_of)


def _month_end(day: date) -> date:
    if day.month == 12:
        return date(day.year, 12, 31)
    first_next_month = date(day.year, day.month + 1, 1)
    return first_next_month.fromordinal(first_next_month.toordinal() - 1)


def _iter_month_ends(start: date, end: date) -> tuple[date, ...]:
    if end < start:
        return ()

    pointer = _month_end(start)
    if pointer < start:
        if pointer.month == 12:
            pointer = date(pointer.year + 1, 1, 31)
        else:
            pointer = _month_end(date(pointer.year, pointer.month + 1, 1))

    values: list[date] = []
    current = pointer
    while current <= end:
        values.append(current)
        if current.month == 12:
            current = date(current.year + 1, 1, 31)
        else:
            current = _month_end(date(current.year, current.month + 1, 1))
    return tuple(values)


def build_month_end_cache(
    rows: Iterable[ConstituentSCD2Row | Mapping[str, object]],
    *,
    universe: str,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
) -> dict[date, tuple[str, ...]]:
    """Materialize month-end membership snapshots from validated SCD2 rows."""

    canonical = assert_no_overlaps(rows)
    universe_key = _to_text(universe, "universe", lowercase=True)
    scoped = tuple(row for row in canonical if row.universe == universe_key)
    if not scoped:
        return {}

    start = _to_date(start_date, "start_date") if start_date is not None else min(row.effective_from for row in scoped)

    if end_date is not None:
        end = _to_date(end_date, "end_date")
    else:
        end = max((row.effective_to or row.effective_from) for row in scoped)

    cache: dict[date, tuple[str, ...]] = {}
    for month_end in _iter_month_ends(start, end):
        cache[month_end] = _membership_for_date(scoped, universe=universe_key, as_of=month_end)
    return cache


__all__ = [
    "ConstituentOverlapError",
    "ConstituentSCD2Error",
    "ConstituentSCD2Row",
    "assert_no_overlaps",
    "build_month_end_cache",
    "membership_for_date",
]
