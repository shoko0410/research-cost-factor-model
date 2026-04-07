"""R&D/Sales(TTM) factor computation and country-sleeve ranking.

Default behavior does not winsorize factor values. Set ``winsor_quantiles`` to a
``(lower, upper)`` tuple to clamp eligible factor values before ranking.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import floor
from typing import cast

REASON_ELIGIBLE = "ELIGIBLE"
REASON_MISSING_SALES_TTM = "MISSING_SALES_TTM"
REASON_NON_POSITIVE_SALES_TTM = "NON_POSITIVE_SALES_TTM"
REASON_MISSING_RD_EXPENSE = "MISSING_RD_EXPENSE"


@dataclass(frozen=True)
class RNDSalesTTMFactorRow:
    """Output row for one security's R&D/Sales(TTM) factor state."""

    security_id: str
    country: str
    rd_expense: float | None
    sales_ttm: float | None
    factor_value: float | None
    rank_in_country: int | None
    is_eligible: bool
    eligibility_reason: str


def _get_field(record: object, field_name: str) -> object:
    if isinstance(record, Mapping):
        if field_name in record:
            return record[field_name]
        raise ValueError(f"record missing required field: {field_name}")
    if hasattr(record, field_name):
        return cast(object, getattr(record, field_name))
    raise ValueError(f"record missing required field: {field_name}")


def _to_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be text")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} cannot be empty")
    return text


def _to_optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("numeric field must be parseable as float") from exc


def _validate_winsor_quantiles(winsor_quantiles: tuple[float, float] | None) -> tuple[float, float] | None:
    if winsor_quantiles is None:
        return None
    lower, upper = winsor_quantiles
    if not (0.0 <= lower <= upper <= 1.0):
        raise ValueError("winsor quantiles must satisfy 0.0 <= lower <= upper <= 1.0")
    return float(lower), float(upper)


def _compute_quantile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute quantile from empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * quantile
    low_index = floor(position)
    high_index = min(low_index + 1, len(sorted_values) - 1)
    low_value = sorted_values[low_index]
    high_value = sorted_values[high_index]
    if low_index == high_index:
        return low_value
    weight_high = position - low_index
    weight_low = 1.0 - weight_high
    return (weight_low * low_value) + (weight_high * high_value)


def winsorize_values(values: Iterable[float], *, lower_quantile: float, upper_quantile: float) -> tuple[float, ...]:
    """Clamp values to quantile bounds in deterministic order."""

    quantiles = _validate_winsor_quantiles((lower_quantile, upper_quantile))
    assert quantiles is not None
    lower_q, upper_q = quantiles

    raw_values = [float(value) for value in values]
    if not raw_values:
        return ()

    sorted_values = sorted(raw_values)
    lower_bound = _compute_quantile(sorted_values, lower_q)
    upper_bound = _compute_quantile(sorted_values, upper_q)
    return tuple(min(max(value, lower_bound), upper_bound) for value in raw_values)


def compute_rnd_sales_ttm_factor(
    accepted_records: Iterable[object],
    *,
    winsor_quantiles: tuple[float, float] | None = None,
) -> tuple[RNDSalesTTMFactorRow, ...]:
    """Compute R&D/Sales(TTM) and deterministic ranks by country sleeve."""

    normalized_quantiles = _validate_winsor_quantiles(winsor_quantiles)
    rows: list[RNDSalesTTMFactorRow] = []

    for record in accepted_records:
        security_id = _to_text(_get_field(record, "security_id"), "security_id")
        country = _to_text(_get_field(record, "country"), "country").upper()
        rd_expense = _to_optional_float(_get_field(record, "rd_expense"))
        sales_ttm = _to_optional_float(_get_field(record, "sales_ttm"))

        if rd_expense is None:
            rows.append(
                RNDSalesTTMFactorRow(
                    security_id=security_id,
                    country=country,
                    rd_expense=None,
                    sales_ttm=sales_ttm,
                    factor_value=None,
                    rank_in_country=None,
                    is_eligible=False,
                    eligibility_reason=REASON_MISSING_RD_EXPENSE,
                )
            )
            continue

        if sales_ttm is None:
            rows.append(
                RNDSalesTTMFactorRow(
                    security_id=security_id,
                    country=country,
                    rd_expense=rd_expense,
                    sales_ttm=None,
                    factor_value=None,
                    rank_in_country=None,
                    is_eligible=False,
                    eligibility_reason=REASON_MISSING_SALES_TTM,
                )
            )
            continue

        if sales_ttm <= 0.0:
            rows.append(
                RNDSalesTTMFactorRow(
                    security_id=security_id,
                    country=country,
                    rd_expense=rd_expense,
                    sales_ttm=sales_ttm,
                    factor_value=None,
                    rank_in_country=None,
                    is_eligible=False,
                    eligibility_reason=REASON_NON_POSITIVE_SALES_TTM,
                )
            )
            continue

        rows.append(
            RNDSalesTTMFactorRow(
                security_id=security_id,
                country=country,
                rd_expense=rd_expense,
                sales_ttm=sales_ttm,
                factor_value=rd_expense / sales_ttm,
                rank_in_country=None,
                is_eligible=True,
                eligibility_reason=REASON_ELIGIBLE,
            )
        )

    if normalized_quantiles is not None:
        eligible_indices = [index for index, row in enumerate(rows) if row.is_eligible and row.factor_value is not None]
        eligible_values = [rows[index].factor_value for index in eligible_indices]
        clamped_values = winsorize_values(
            [value for value in eligible_values if value is not None],
            lower_quantile=normalized_quantiles[0],
            upper_quantile=normalized_quantiles[1],
        )
        for index, clamped_value in zip(eligible_indices, clamped_values, strict=True):
            row = rows[index]
            rows[index] = RNDSalesTTMFactorRow(
                security_id=row.security_id,
                country=row.country,
                rd_expense=row.rd_expense,
                sales_ttm=row.sales_ttm,
                factor_value=clamped_value,
                rank_in_country=None,
                is_eligible=row.is_eligible,
                eligibility_reason=row.eligibility_reason,
            )

    by_country: dict[str, list[tuple[int, RNDSalesTTMFactorRow]]] = {}
    for index, row in enumerate(rows):
        if row.is_eligible and row.factor_value is not None:
            by_country.setdefault(row.country, []).append((index, row))

    for country_rows in by_country.values():
        ranked = sorted(country_rows, key=lambda pair: (-cast(float, pair[1].factor_value), pair[1].security_id))
        for rank, (index, row) in enumerate(ranked, start=1):
            rows[index] = RNDSalesTTMFactorRow(
                security_id=row.security_id,
                country=row.country,
                rd_expense=row.rd_expense,
                sales_ttm=row.sales_ttm,
                factor_value=row.factor_value,
                rank_in_country=rank,
                is_eligible=True,
                eligibility_reason=REASON_ELIGIBLE,
            )

    return tuple(sorted(rows, key=lambda row: (row.country, row.rank_in_country or 10**9, row.security_id)))


__all__ = [
    "REASON_ELIGIBLE",
    "REASON_MISSING_RD_EXPENSE",
    "REASON_MISSING_SALES_TTM",
    "REASON_NON_POSITIVE_SALES_TTM",
    "RNDSalesTTMFactorRow",
    "compute_rnd_sales_ttm_factor",
    "winsorize_values",
]
