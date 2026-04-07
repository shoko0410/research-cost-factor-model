"""FX ingestion normalization and KRW-base conversion helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime

from ..schema import canonical_fx


class FxIngestError(ValueError):
    """Base error for deterministic FX ingestion utilities."""


class DuplicateFxDateError(FxIngestError):
    """Raised when a pair contains duplicate observations for one date."""


class MissingFxRateError(FxIngestError):
    """Raised when the requested date is not available under policy."""


class UnsupportedCurrencyError(FxIngestError):
    """Raised when KRW-base conversion receives unsupported currency."""


def _to_iso_date(value: object, *, field_name: str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        raise FxIngestError(f"{field_name} is required")
    return date.fromisoformat(text).isoformat()


def _to_float(value: object, *, field_name: str) -> float:
    try:
        text = str(value).strip()
        return float(text)
    except (TypeError, ValueError) as exc:
        raise FxIngestError(f"{field_name} must be numeric") from exc


def detect_duplicate_fx_dates(rows: Iterable[Mapping[str, object]]) -> list[tuple[str, str]]:
    """Return sorted duplicate (pair, fx_date) keys found in rows."""

    observed: set[tuple[str, str]] = set()
    duplicates: set[tuple[str, str]] = set()
    for row in rows:
        pair = str(row["pair"]).upper()
        fx_date = _to_iso_date(row["fx_date"], field_name="fx_date")
        key = (pair, fx_date)
        if key in observed:
            duplicates.add(key)
            continue
        observed.add(key)
    return sorted(duplicates)


def normalize_fx_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Normalize raw FX rows into canonical records with temporal fields."""

    enriched_rows: list[dict[str, object]] = []
    for raw in rows:
        row = dict(raw)
        raw_fx_date = row.get("fx_date", row.get("as_of_date"))
        fx_date = _to_iso_date(raw_fx_date, field_name="fx_date")
        if "as_of_date" not in row:
            row["as_of_date"] = fx_date
        row["fx_date"] = fx_date
        if "effective_from" not in row:
            row["effective_from"] = fx_date
        if "effective_to" not in row:
            row["effective_to"] = ""
        if "is_current" not in row:
            row["is_current"] = True
        enriched_rows.append(row)

    duplicates = detect_duplicate_fx_dates(enriched_rows)
    if duplicates:
        joined = ", ".join([f"{pair}@{fx_date}" for pair, fx_date in duplicates])
        raise DuplicateFxDateError(f"duplicate FX rows detected: {joined}")

    canonical_rows = [canonical_fx(row) for row in enriched_rows]
    return sorted(
        canonical_rows,
        key=lambda row: (
            str(row["pair"]),
            str(row["fx_date"]),
            str(row["as_of_date"]),
        ),
    )


def fx_rate_on_date(
    rows: Iterable[Mapping[str, object]],
    *,
    pair: str,
    fx_date: date | datetime | str,
    missing_date_policy: str = "raise",
) -> float:
    """Resolve one FX rate by date with deterministic missing-date policy.

    Policies:
      - "raise": require exact date match
      - "previous": use latest prior available date for the pair
    """

    normalized_pair = str(pair).strip().upper()
    lookup_date = date.fromisoformat(_to_iso_date(fx_date, field_name="fx_date"))

    candidates: list[tuple[date, float]] = []
    for row in rows:
        if str(row["pair"]).upper() != normalized_pair:
            continue
        row_date = date.fromisoformat(_to_iso_date(row["fx_date"], field_name="fx_date"))
        candidates.append((row_date, _to_float(row["rate"], field_name="rate")))

    if not candidates:
        raise MissingFxRateError(f"pair not available: {normalized_pair}")

    date_to_rate = {row_date: rate for row_date, rate in candidates}
    direct_rate = date_to_rate.get(lookup_date)
    if direct_rate is not None:
        return direct_rate

    policy = str(missing_date_policy).strip().lower()
    if policy == "raise":
        raise MissingFxRateError(f"missing rate for {normalized_pair} on {lookup_date.isoformat()}")
    if policy != "previous":
        raise FxIngestError(f"unsupported missing_date_policy: {missing_date_policy}")

    prior_dates = [row_date for row_date, _ in candidates if row_date <= lookup_date]
    if not prior_dates:
        raise MissingFxRateError(
            f"missing rate for {normalized_pair} on {lookup_date.isoformat()} with no previous data"
        )
    nearest = max(prior_dates)
    return date_to_rate[nearest]


def fx_return_to_krw(
    *,
    currency: str,
    start_usd_krw: float,
    end_usd_krw: float,
    start_usd_jpy: float | None = None,
    end_usd_jpy: float | None = None,
) -> float:
    """Return FX-only return in KRW base for KRW/USD/JPY exposures."""

    normalized_currency = str(currency).strip().upper()
    if normalized_currency == "KRW":
        return 0.0
    if normalized_currency == "USD":
        return (float(end_usd_krw) / float(start_usd_krw)) - 1.0
    if normalized_currency == "JPY":
        if start_usd_jpy is None or end_usd_jpy is None:
            raise FxIngestError("USD/JPY start/end rates are required for JPY conversion")
        start_jpy_krw = float(start_usd_krw) / float(start_usd_jpy)
        end_jpy_krw = float(end_usd_krw) / float(end_usd_jpy)
        return (end_jpy_krw / start_jpy_krw) - 1.0
    raise UnsupportedCurrencyError(f"unsupported currency for KRW-base conversion: {currency}")


def convert_return_to_krw_base(
    *,
    local_return: float,
    currency: str,
    start_usd_krw: float,
    end_usd_krw: float,
    start_usd_jpy: float | None = None,
    end_usd_jpy: float | None = None,
) -> float:
    """Convert local-currency return to KRW base return."""

    fx_return = fx_return_to_krw(
        currency=currency,
        start_usd_krw=start_usd_krw,
        end_usd_krw=end_usd_krw,
        start_usd_jpy=start_usd_jpy,
        end_usd_jpy=end_usd_jpy,
    )
    return ((1.0 + float(local_return)) * (1.0 + fx_return)) - 1.0


def decompose_return_contribution(*, local_return: float, fx_return: float) -> dict[str, float]:
    """Provide additive decomposition for local and FX effects."""

    local_component = float(local_return)
    fx_component = float(fx_return)
    interaction = local_component * fx_component
    total = local_component + fx_component + interaction
    return {
        "local": local_component,
        "fx": fx_component,
        "interaction": interaction,
        "total": total,
    }


__all__ = [
    "DuplicateFxDateError",
    "FxIngestError",
    "MissingFxRateError",
    "UnsupportedCurrencyError",
    "convert_return_to_krw_base",
    "decompose_return_contribution",
    "detect_duplicate_fx_dates",
    "fx_rate_on_date",
    "fx_return_to_krw",
    "normalize_fx_rows",
]
