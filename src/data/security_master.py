"""Security master mapping utilities for cross-market identifiers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime


class SecurityMasterError(ValueError):
    """Base error for security master mapping issues."""


class DuplicateActiveMappingError(SecurityMasterError):
    """Raised when a key has overlapping active mappings."""


def _normalize_market(market: str) -> str:
    return str(market).strip().upper()


def _normalize_key(value: str) -> str:
    return str(value).strip().upper()


def _to_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None:
        raise SecurityMasterError("valid_from is required")
    text = str(value).strip()
    if not text:
        raise SecurityMasterError("valid_from is required")
    return date.fromisoformat(text)


def _to_optional_date(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    return date.fromisoformat(text)


@dataclass(frozen=True)
class SecurityMapping:
    """One key->security mapping with validity bounds."""

    market: str
    key_type: str
    raw_key: str
    security_id: str
    valid_from: date
    valid_to: date | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.market, self.key_type, self.raw_key)

    def is_active_on(self, as_of: date) -> bool:
        if as_of < self.valid_from:
            return False
        if self.valid_to is not None and as_of > self.valid_to:
            return False
        return True


class SecurityMaster:
    """Deterministic resolver for canonical security identifiers."""

    def __init__(self, mappings: Iterable[SecurityMapping]):
        normalized = [
            SecurityMapping(
                market=_normalize_market(mapping.market),
                key_type=str(mapping.key_type).strip(),
                raw_key=_normalize_key(mapping.raw_key),
                security_id=str(mapping.security_id).strip(),
                valid_from=_to_date(mapping.valid_from),
                valid_to=_to_optional_date(mapping.valid_to),
            )
            for mapping in mappings
        ]
        self._mappings: tuple[SecurityMapping, ...] = tuple(
            sorted(
                normalized,
                key=lambda item: (
                    item.market,
                    item.key_type,
                    item.raw_key,
                    item.valid_from,
                    item.valid_to or date.max,
                    item.security_id,
                ),
            )
        )
        index: dict[tuple[str, str, str], list[SecurityMapping]] = {}
        for mapping in self._mappings:
            index.setdefault(mapping.key, []).append(mapping)
        self._index: dict[tuple[str, str, str], tuple[SecurityMapping, ...]] = {
            key: tuple(value) for key, value in index.items()
        }
        self._validate_collisions()

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Mapping[str, object]],
        *,
        key_fields: tuple[str, ...] = ("security_id", "issuer_id", "ticker", "stock_code"),
        market_field: str = "country",
        security_id_field: str = "security_id",
        valid_from_field: str = "effective_from",
        valid_to_field: str = "effective_to",
    ) -> "SecurityMaster":
        mappings: list[SecurityMapping] = []
        for row in rows:
            market_value = row.get(market_field)
            security_id_value = row.get(security_id_field)
            if market_value is None or security_id_value is None:
                continue
            market = _normalize_market(str(market_value))
            security_id = str(security_id_value).strip()
            valid_from = _to_date(row.get(valid_from_field))
            valid_to = _to_optional_date(row.get(valid_to_field))
            for key_type in key_fields:
                raw = row.get(key_type)
                if raw is None:
                    continue
                raw_text = str(raw).strip()
                if not raw_text:
                    continue
                mappings.append(
                    SecurityMapping(
                        market=market,
                        key_type=key_type,
                        raw_key=raw_text,
                        security_id=security_id,
                        valid_from=valid_from,
                        valid_to=valid_to,
                    )
                )
        return cls(mappings)

    def resolve(self, *, market: str, key_type: str, key_value: str, as_of: date | str) -> str | None:
        as_of_date = _to_date(as_of)
        key = (_normalize_market(market), str(key_type).strip(), _normalize_key(key_value))
        windows = self._index.get(key, ())
        for mapping in windows:
            if mapping.is_active_on(as_of_date):
                return mapping.security_id
        return None

    def _validate_collisions(self) -> None:
        for key, windows in self._index.items():
            if len(windows) < 2:
                continue
            sorted_windows = sorted(windows, key=lambda item: (item.valid_from, item.valid_to or date.max))
            prev = sorted_windows[0]
            for current in sorted_windows[1:]:
                prev_end = prev.valid_to or date.max
                if current.valid_from <= prev_end:
                    raise DuplicateActiveMappingError(
                        f"duplicate active mapping detected for key {key}: {prev.security_id} overlaps {current.security_id}"
                    )
                prev = current


__all__ = [
    "DuplicateActiveMappingError",
    "SecurityMapping",
    "SecurityMaster",
    "SecurityMasterError",
]
