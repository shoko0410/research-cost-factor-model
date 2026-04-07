"""PIT feature assembly with eligibility and investability filters."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from statistics import median
from typing import cast
from ..data.constituents_scd2 import membership_for_date


DEFAULT_FILING_LAG_DAYS: dict[str, int] = {
    "US": 45,
    "KR": 60,
    "JP": 60,
}

DEFAULT_INVESTABILITY_THRESHOLD_KRW = 500_000_000.0

REASON_NO_ELIGIBLE_FUNDAMENTALS = "NO_ELIGIBLE_FUNDAMENTALS"
REASON_FILING_LAG_NOT_MET = "FILING_LAG_NOT_MET"
REASON_MISSING_RD_EXPENSE = "MISSING_RD_EXPENSE"
REASON_NON_POSITIVE_SALES_TTM = "NON_POSITIVE_SALES_TTM"
REASON_NO_PRICE_DATA = "NO_PRICE_DATA"
REASON_INVESTABILITY_BELOW_THRESHOLD = "INVESTABILITY_BELOW_THRESHOLD"
REASON_INVESTABILITY_DATA_ERROR = "INVESTABILITY_DATA_ERROR"


class PITAssemblerError(ValueError):
    """Base error for PIT assembly."""


class PITViolationError(PITAssemblerError):
    """Raised when a look-ahead (future filing) violation is detected."""


class MissingFxRateError(PITAssemblerError):
    """Raised when FX conversion cannot resolve a required rate."""


class InsufficientPriceDataError(PITAssemblerError):
    """Raised when investability cannot be computed due to sparse/invalid prices."""


@dataclass(frozen=True)
class AcceptedSecurity:
    """Accepted security payload for downstream feature/ranking stages."""

    security_id: str
    country: str
    period_end: str
    filing_date: str
    rd_expense: float
    sales_ttm: float | None
    median_traded_value_krw: float


@dataclass(frozen=True)
class RejectedSecurity:
    """Rejected security payload with deterministic rejection reasons."""

    security_id: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PITAssemblyResult:
    """Deterministic PIT assembly output."""

    as_of_date: str
    universe: str
    pit_violations: int
    accepted: tuple[AcceptedSecurity, ...]
    rejected: tuple[RejectedSecurity, ...]


@dataclass(frozen=True)
class _FundamentalRecord:
    security_id: str
    country: str
    period_end: date
    filing_date: date
    rd_expense: float | None
    sales_ttm: float | None
    effective_from: date
    effective_to: date | None

    def is_active_on(self, as_of: date) -> bool:
        if as_of < self.effective_from:
            return False
        if self.effective_to is not None and as_of > self.effective_to:
            return False
        return True


@dataclass(frozen=True)
class _PriceRecord:
    security_id: str
    price_date: date
    currency: str
    traded_value: float


def _to_date(value: object, field_name: str) -> date:
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        raise PITAssemblerError(f"{field_name} is required")
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise PITAssemblerError(f"{field_name} must be ISO date (YYYY-MM-DD)") from exc


def _to_optional_date(value: object, field_name: str) -> date | None:
    if value in (None, ""):
        return None
    return _to_date(value, field_name)


def _to_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise PITAssemblerError(f"{field_name} must be text")
    text = value.strip()
    if not text:
        raise PITAssemblerError(f"{field_name} cannot be empty")
    return text


def _to_optional_float(value: object, field_name: str) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise PITAssemblerError(f"{field_name} must be numeric") from exc


def _parse_fundamental(row: Mapping[str, object]) -> _FundamentalRecord:
    required = ("security_id", "country", "period_end", "filing_date")
    missing = [key for key in required if key not in row]
    if missing:
        raise PITAssemblerError(f"fundamental row missing required fields: {', '.join(missing)}")

    filing_date = _to_date(row["filing_date"], "filing_date")
    effective_from = _to_date(row.get("effective_from", filing_date), "effective_from")
    return _FundamentalRecord(
        security_id=_to_text(row["security_id"], "security_id"),
        country=_to_text(row["country"], "country").upper(),
        period_end=_to_date(row["period_end"], "period_end"),
        filing_date=filing_date,
        rd_expense=_to_optional_float(row.get("rd_expense"), "rd_expense"),
        sales_ttm=_to_optional_float(row.get("sales_ttm"), "sales_ttm"),
        effective_from=effective_from,
        effective_to=_to_optional_date(row.get("effective_to"), "effective_to"),
    )


def _parse_price(row: Mapping[str, object]) -> _PriceRecord:
    required = ("security_id", "price_date", "currency", "traded_value")
    missing = [key for key in required if key not in row]
    if missing:
        raise InsufficientPriceDataError(f"price row missing required fields: {', '.join(missing)}")
    traded_value = _to_optional_float(row.get("traded_value"), "traded_value")
    if traded_value is None:
        raise InsufficientPriceDataError("traded_value is required")
    if traded_value < 0:
        raise InsufficientPriceDataError("traded_value cannot be negative")
    return _PriceRecord(
        security_id=_to_text(row["security_id"], "security_id"),
        price_date=_to_date(row["price_date"], "price_date"),
        currency=_to_text(row["currency"], "currency").upper(),
        traded_value=traded_value,
    )


def _get_lag_days(country: str, lag_days: Mapping[str, int]) -> int:
    key = country.upper()
    if key not in lag_days:
        raise PITAssemblerError(f"missing filing lag policy for country: {country}")
    value = int(lag_days[key])
    if value < 0:
        raise PITAssemblerError(f"filing lag must be non-negative for country: {country}")
    return value


def _normalize_fx_rates(
    fx_rates: Mapping[str, object],
) -> dict[str, Mapping[date, float] | float]:
    normalized: dict[str, Mapping[date, float] | float] = {}
    for raw_pair, raw_value in fx_rates.items():
        pair = _to_text(raw_pair, "fx pair").upper()
        if isinstance(raw_value, Mapping):
            parsed_by_date: dict[date, float] = {}
            mapping_by_date = cast(Mapping[object, object], raw_value)
            for raw_date, raw_rate in mapping_by_date.items():
                rate_value = _to_optional_float(raw_rate, "fx_rate")
                if rate_value is None:
                    raise MissingFxRateError("fx_rate is required")
                parsed_by_date[_to_date(raw_date, "fx_date")] = rate_value
            normalized[pair] = parsed_by_date
            continue
        rate_value = _to_optional_float(raw_value, "fx_rate")
        if rate_value is None:
            raise MissingFxRateError("fx_rate is required")
        normalized[pair] = rate_value
    return normalized


def _lookup_fx_rate(normalized_fx_rates: Mapping[str, Mapping[date, float] | float], *, pair: str, fx_date: date) -> float:
    pair_key = pair.upper()
    raw_value = normalized_fx_rates.get(pair_key)
    if raw_value is None:
        raise MissingFxRateError(f"missing FX pair: {pair_key}")
    if isinstance(raw_value, Mapping):
        if fx_date not in raw_value:
            raise MissingFxRateError(f"missing FX rate for {pair_key} on {fx_date.isoformat()}")
        return float(raw_value[fx_date])
    return float(raw_value)


def _convert_to_krw(
    *,
    amount: float,
    currency: str,
    value_date: date,
    normalized_fx_rates: Mapping[str, Mapping[date, float] | float],
) -> float:
    if currency == "KRW":
        return amount
    if currency == "USD":
        return amount * _lookup_fx_rate(normalized_fx_rates, pair="USD/KRW", fx_date=value_date)
    if currency == "JPY":
        if "JPY/KRW" in normalized_fx_rates:
            return amount * _lookup_fx_rate(normalized_fx_rates, pair="JPY/KRW", fx_date=value_date)
        usd_krw = _lookup_fx_rate(normalized_fx_rates, pair="USD/KRW", fx_date=value_date)
        usd_jpy = _lookup_fx_rate(normalized_fx_rates, pair="USD/JPY", fx_date=value_date)
        return amount * (usd_krw / usd_jpy)
    raise MissingFxRateError(f"unsupported currency for KRW conversion: {currency}")


def compute_median_traded_value_krw(
    *,
    price_rows: Iterable[Mapping[str, object]],
    fx_rates: Mapping[str, object],
    min_observations: int = 1,
) -> float:
    """Compute median traded value in KRW equivalent from price rows."""

    if min_observations <= 0:
        raise InsufficientPriceDataError("min_observations must be at least 1")

    normalized_fx_rates = _normalize_fx_rates(fx_rates)
    parsed_prices = [_parse_price(row) for row in price_rows]
    if len(parsed_prices) < min_observations:
        raise InsufficientPriceDataError(
            f"insufficient price rows for median traded value: {len(parsed_prices)} < {min_observations}"
        )

    converted_values = [
        _convert_to_krw(
            amount=row.traded_value,
            currency=row.currency,
            value_date=row.price_date,
            normalized_fx_rates=normalized_fx_rates,
        )
        for row in parsed_prices
    ]
    return float(median(converted_values))


def assemble_pit_universe(
    *,
    universe: str,
    as_of_date: date | str,
    constituent_rows: Iterable[Mapping[str, object]],
    fundamental_rows: Iterable[Mapping[str, object]],
    price_rows: Iterable[Mapping[str, object]],
    fx_rates: Mapping[str, object],
    filing_lag_days: Mapping[str, int] | None = None,
    require_positive_sales_ttm: bool = True,
    investability_threshold_krw: float = DEFAULT_INVESTABILITY_THRESHOLD_KRW,
    investability_lookback_days: int = 20,
    investability_min_observations: int = 3,
) -> PITAssemblyResult:
    """Assemble PIT-eligible and investable universe for one rebalance date."""

    if investability_lookback_days <= 0:
        raise PITAssemblerError("investability_lookback_days must be positive")

    as_of = _to_date(as_of_date, "as_of_date")
    members = membership_for_date(constituent_rows, universe=universe, as_of_date=as_of)
    lag_days = {key.upper(): int(value) for key, value in (filing_lag_days or DEFAULT_FILING_LAG_DAYS).items()}

    parsed_fundamentals = [_parse_fundamental(row) for row in fundamental_rows]
    pit_violations = [row for row in parsed_fundamentals if row.filing_date > as_of]
    if pit_violations:
        raise PITViolationError(
            f"PIT violations must be zero; found {len(pit_violations)} rows with filing_date after {as_of.isoformat()}"
        )

    fundamentals_by_security: dict[str, list[_FundamentalRecord]] = {}
    for row in parsed_fundamentals:
        fundamentals_by_security.setdefault(row.security_id, []).append(row)
    for rows in fundamentals_by_security.values():
        rows.sort(key=lambda item: (item.filing_date, item.period_end, item.security_id))

    parsed_prices = [_parse_price(row) for row in price_rows]
    prices_by_security: dict[str, list[_PriceRecord]] = {}
    for row in parsed_prices:
        prices_by_security.setdefault(row.security_id, []).append(row)
    for rows in prices_by_security.values():
        rows.sort(key=lambda item: (item.price_date, item.security_id))

    accepted: list[AcceptedSecurity] = []
    rejected: list[RejectedSecurity] = []

    lookback_start = as_of - timedelta(days=investability_lookback_days - 1)
    for security_id in members:
        reasons: list[str] = []
        security_fundamentals = [
            row
            for row in fundamentals_by_security.get(security_id, [])
            if row.is_active_on(as_of) and row.filing_date <= as_of
        ]
        if not security_fundamentals:
            reasons.append(REASON_NO_ELIGIBLE_FUNDAMENTALS)
            rejected.append(RejectedSecurity(security_id=security_id, reasons=tuple(reasons)))
            continue

        lag_eligible: list[_FundamentalRecord] = []
        for row in security_fundamentals:
            lag = _get_lag_days(row.country, lag_days)
            if row.period_end + timedelta(days=lag) <= as_of:
                lag_eligible.append(row)

        if not lag_eligible:
            reasons.append(REASON_FILING_LAG_NOT_MET)
            rejected.append(RejectedSecurity(security_id=security_id, reasons=tuple(reasons)))
            continue

        selected = lag_eligible[-1]
        if selected.rd_expense is None:
            reasons.append(REASON_MISSING_RD_EXPENSE)
        if require_positive_sales_ttm and (selected.sales_ttm is None or selected.sales_ttm <= 0):
            reasons.append(REASON_NON_POSITIVE_SALES_TTM)
        if reasons:
            rejected.append(RejectedSecurity(security_id=security_id, reasons=tuple(reasons)))
            continue

        security_prices = [
            {
                "security_id": row.security_id,
                "price_date": row.price_date,
                "currency": row.currency,
                "traded_value": row.traded_value,
            }
            for row in prices_by_security.get(security_id, [])
            if lookback_start <= row.price_date <= as_of
        ]
        if not security_prices:
            reasons.append(REASON_NO_PRICE_DATA)
            rejected.append(RejectedSecurity(security_id=security_id, reasons=tuple(reasons)))
            continue

        try:
            median_traded_value_krw = compute_median_traded_value_krw(
                price_rows=security_prices,
                fx_rates=fx_rates,
                min_observations=investability_min_observations,
            )
        except (MissingFxRateError, InsufficientPriceDataError):
            reasons.append(REASON_INVESTABILITY_DATA_ERROR)
            rejected.append(RejectedSecurity(security_id=security_id, reasons=tuple(reasons)))
            continue

        if median_traded_value_krw < float(investability_threshold_krw):
            reasons.append(REASON_INVESTABILITY_BELOW_THRESHOLD)
            rejected.append(RejectedSecurity(security_id=security_id, reasons=tuple(reasons)))
            continue

        assert selected.rd_expense is not None
        assert selected.sales_ttm is not None
        accepted.append(
            AcceptedSecurity(
                security_id=security_id,
                country=selected.country,
                period_end=selected.period_end.isoformat(),
                filing_date=selected.filing_date.isoformat(),
                rd_expense=float(selected.rd_expense),
                sales_ttm=float(selected.sales_ttm),
                median_traded_value_krw=median_traded_value_krw,
            )
        )

    return PITAssemblyResult(
        as_of_date=as_of.isoformat(),
        universe=universe,
        pit_violations=0,
        accepted=tuple(sorted(accepted, key=lambda row: row.security_id)),
        rejected=tuple(sorted(rejected, key=lambda row: row.security_id)),
    )


__all__ = [
    "DEFAULT_FILING_LAG_DAYS",
    "DEFAULT_INVESTABILITY_THRESHOLD_KRW",
    "AcceptedSecurity",
    "InsufficientPriceDataError",
    "MissingFxRateError",
    "PITAssemblerError",
    "PITAssemblyResult",
    "PITViolationError",
    "RejectedSecurity",
    "assemble_pit_universe",
    "compute_median_traded_value_krw",
]
