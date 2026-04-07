"""Portfolio construction from ranked factor rows with sleeve constraints."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast

DEFAULT_MAX_HOLDINGS = 20
DEFAULT_MAX_SINGLE_NAME_WEIGHT = 0.08
DEFAULT_COUNTRY_TARGETS: tuple[tuple[str, float], ...] = (
    ("JP", 1.0 / 3.0),
    ("KR", 1.0 / 3.0),
    ("US", 1.0 / 3.0),
)
DEFAULT_COUNTRY_TOLERANCE = 0.02
DEFAULT_SECTOR_ACTIVE_BAND = 0.10
DEFAULT_TE_ACTIVE_L2_CAP = 0.08
DEFAULT_ALPHA_TILT_STRENGTH = 0.35
DEFAULT_MAX_ADV_PARTICIPATION = 0.10
DEFAULT_PORTFOLIO_VALUE_KRW = 10_000_000.0


@dataclass(frozen=True)
class PortfolioHolding:
    security_id: str
    country: str
    weight: float
    rank_in_country: int | None
    factor_value: float | None


@dataclass(frozen=True)
class PortfolioSelectionDiagnostics:
    requested_holdings: int
    selected_holdings: int
    available_eligible: int
    requested_country_counts: tuple[tuple[str, int], ...]
    available_country_counts: tuple[tuple[str, int], ...]
    selected_country_counts: tuple[tuple[str, int], ...]
    country_weights: tuple[tuple[str, float], ...]
    cash_weight: float


@dataclass(frozen=True)
class PortfolioConstructionResult:
    holdings: tuple[PortfolioHolding, ...]
    fallback_triggered: bool
    fallback_reasons: tuple[str, ...]
    jp_odd_lot_enabled: bool
    diagnostics: PortfolioSelectionDiagnostics


@dataclass(frozen=True)
class _RankedRow:
    security_id: str
    country: str
    factor_value: float | None
    rank_in_country: int | None
    sector: str
    benchmark_proxy_weight: float | None
    median_traded_value_krw: float | None


def _get_field(record: object, field_name: str) -> object:
    if isinstance(record, Mapping):
        if field_name in record:
            return record[field_name]
        raise ValueError(f"record missing required field: {field_name}")
    if hasattr(record, field_name):
        return cast(object, getattr(record, field_name))
    raise ValueError(f"record missing required field: {field_name}")


def _get_optional_field(record: object, field_name: str, default: object = None) -> object:
    if isinstance(record, Mapping):
        return record.get(field_name, default)
    return cast(object, getattr(record, field_name, default))


def _to_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be text")
    parsed = value.strip()
    if not parsed:
        raise ValueError(f"{field_name} cannot be empty")
    return parsed


def _to_optional_float(value: object, field_name: str) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc


def _to_optional_int(value: object, field_name: str) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive when provided")
    return parsed


def _to_optional_bool(value: object, field_name: str) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field_name} must be boolean when provided")


def _to_optional_non_negative_float(value: object, field_name: str) -> float | None:
    parsed = _to_optional_float(value, field_name)
    if parsed is None:
        return None
    if parsed < 0.0:
        raise ValueError(f"{field_name} must be non-negative when provided")
    return parsed


def _to_optional_upper_text(value: object, field_name: str) -> str | None:
    if value in (None, ""):
        return None
    parsed = _to_text(value, field_name)
    return parsed.upper()


def _to_float_value(value: object, field_name: str) -> float:
    try:
        return float(cast(float | int | str, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc


def _parse_ranked_rows(ranked_factor_rows: Iterable[object]) -> tuple[_RankedRow, ...]:
    parsed: list[_RankedRow] = []
    for record in ranked_factor_rows:
        is_eligible = _to_optional_bool(_get_field(record, "is_eligible"), "is_eligible")
        if is_eligible is False:
            continue
        row = _RankedRow(
            security_id=_to_text(_get_field(record, "security_id"), "security_id"),
            country=_to_text(_get_field(record, "country"), "country").upper(),
            factor_value=_to_optional_float(_get_field(record, "factor_value"), "factor_value"),
            rank_in_country=_to_optional_int(_get_field(record, "rank_in_country"), "rank_in_country"),
            sector=(_to_optional_upper_text(_get_optional_field(record, "sector"), "sector") or "UNKNOWN"),
            benchmark_proxy_weight=_to_optional_non_negative_float(
                _get_optional_field(record, "benchmark_proxy_weight"),
                "benchmark_proxy_weight",
            ),
            median_traded_value_krw=_to_optional_non_negative_float(
                _get_optional_field(record, "median_traded_value_krw"),
                "median_traded_value_krw",
            ),
        )
        if row.factor_value is None:
            continue
        parsed.append(row)
    return tuple(parsed)


def _requested_counts(*, max_holdings: int, countries: tuple[str, ...], target_weights: Mapping[str, float]) -> dict[str, int]:
    raw = {country: max_holdings * target_weights[country] for country in countries}
    floored = {country: int(raw[country]) for country in countries}
    remaining = max_holdings - sum(floored.values())
    remainders = sorted(countries, key=lambda country: (-((raw[country]) - floored[country]), country))
    for country in remainders[:remaining]:
        floored[country] += 1
    return floored


def _format_reasons(reasons: set[str]) -> tuple[str, ...]:
    return tuple(sorted(reasons))


def _safe_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(max(variance, 0.0))


def _rescale_with_caps(
    *,
    base_weights: Mapping[str, float],
    cap_weights: Mapping[str, float],
    target_total: float,
) -> dict[str, float]:
    if target_total <= 0.0:
        return {security_id: 0.0 for security_id in base_weights}

    weights = {security_id: max(0.0, float(weight)) for security_id, weight in base_weights.items()}
    caps = {security_id: max(0.0, float(cap_weights.get(security_id, 0.0))) for security_id in weights}
    total_capacity = sum(caps.values())
    target = min(target_total, total_capacity)
    if target <= 0.0:
        return {security_id: 0.0 for security_id in weights}

    result = {security_id: 0.0 for security_id in weights}
    active = {security_id for security_id in weights if caps[security_id] > 0.0}
    remaining_target = target

    while active and remaining_target > 1e-12:
        total_base = sum(weights[security_id] for security_id in active)
        if total_base <= 1e-12:
            equal = remaining_target / len(active)
            provisional = {security_id: equal for security_id in active}
        else:
            provisional = {
                security_id: remaining_target * (weights[security_id] / total_base)
                for security_id in active
            }

        saturated = [security_id for security_id, proposed in provisional.items() if proposed >= caps[security_id] - 1e-12]
        if not saturated:
            for security_id, proposed in provisional.items():
                result[security_id] += proposed
            remaining_target = 0.0
            break

        for security_id in saturated:
            room = max(0.0, caps[security_id] - result[security_id])
            result[security_id] += room
            remaining_target -= room
            active.remove(security_id)

    if remaining_target > 1e-12 and active:
        tail = remaining_target / len(active)
        for security_id in active:
            room = max(0.0, caps[security_id] - result[security_id])
            result[security_id] += min(room, tail)

    return result


def _blend_with_turnover_limit(
    *,
    target_weights: Mapping[str, float],
    previous_weights: Mapping[str, float],
    max_turnover: float,
) -> dict[str, float]:
    normalized_max_turnover = max(0.0, min(max_turnover, 2.0))
    ids = set(target_weights) | set(previous_weights)
    turnover = 0.5 * sum(abs(float(target_weights.get(security_id, 0.0)) - float(previous_weights.get(security_id, 0.0))) for security_id in ids)
    if turnover <= normalized_max_turnover + 1e-12 or turnover <= 1e-12:
        return {security_id: float(target_weights.get(security_id, 0.0)) for security_id in ids}

    blend = normalized_max_turnover / turnover
    return {
        security_id: float(previous_weights.get(security_id, 0.0))
        + blend * (float(target_weights.get(security_id, 0.0)) - float(previous_weights.get(security_id, 0.0)))
        for security_id in ids
    }


def construct_portfolio_with_constraints(
    ranked_factor_rows: Iterable[object],
    *,
    max_holdings: int = DEFAULT_MAX_HOLDINGS,
    max_single_name_weight: float = DEFAULT_MAX_SINGLE_NAME_WEIGHT,
    country_targets: tuple[tuple[str, float], ...] = DEFAULT_COUNTRY_TARGETS,
    country_tolerance: float = DEFAULT_COUNTRY_TOLERANCE,
    risk_controls: Mapping[str, object] | None = None,
) -> PortfolioConstructionResult:
    """Construct a long-only portfolio with QEPM-style robust risk controls."""

    if max_holdings <= 0:
        raise ValueError("max_holdings must be positive")
    if not (0.0 < max_single_name_weight <= 1.0):
        raise ValueError("max_single_name_weight must be in (0, 1]")
    if not (0.0 <= country_tolerance < 1.0):
        raise ValueError("country_tolerance must be in [0, 1)")
    if not country_targets:
        raise ValueError("country_targets cannot be empty")

    controls = {} if risk_controls is None else dict(risk_controls)
    te_active_l2_cap = _to_float_value(controls.get("te_active_l2_cap", DEFAULT_TE_ACTIVE_L2_CAP), "te_active_l2_cap")
    alpha_tilt_strength = _to_float_value(controls.get("alpha_tilt_strength", DEFAULT_ALPHA_TILT_STRENGTH), "alpha_tilt_strength")
    max_adv_participation = _to_float_value(controls.get("max_adv_participation", DEFAULT_MAX_ADV_PARTICIPATION), "max_adv_participation")
    portfolio_value_krw = _to_float_value(controls.get("portfolio_value_krw", DEFAULT_PORTFOLIO_VALUE_KRW), "portfolio_value_krw")
    max_turnover_raw = controls.get("max_turnover")
    max_turnover = _to_float_value(max_turnover_raw, "max_turnover") if max_turnover_raw is not None else None
    previous_weights_raw = controls.get("previous_weights", {})

    if te_active_l2_cap < 0.0:
        raise ValueError("te_active_l2_cap must be non-negative")
    if alpha_tilt_strength < 0.0:
        raise ValueError("alpha_tilt_strength must be non-negative")
    if max_adv_participation <= 0.0:
        raise ValueError("max_adv_participation must be positive")
    if portfolio_value_krw <= 0.0:
        raise ValueError("portfolio_value_krw must be positive")
    if max_turnover is not None and max_turnover < 0.0:
        raise ValueError("max_turnover must be non-negative when provided")
    if not isinstance(previous_weights_raw, Mapping):
        raise ValueError("previous_weights must be a mapping when provided")

    previous_weights: dict[str, float] = {}
    for key, value in previous_weights_raw.items():
        security_id = _to_text(str(key), "previous_weights.security_id")
        weight = _to_float_value(value, "previous_weights.weight")
        if weight < 0.0:
            raise ValueError("previous_weights must be non-negative")
        previous_weights[security_id] = weight

    normalized_targets: list[tuple[str, float]] = []
    for country, target in country_targets:
        country_key = _to_text(country, "country_targets.country").upper()
        normalized_target = float(target)
        if normalized_target < 0.0:
            raise ValueError("country target weights cannot be negative")
        normalized_targets.append((country_key, normalized_target))
    target_weight_sum = sum(weight for _, weight in normalized_targets)
    if abs(target_weight_sum - 1.0) > 1e-12:
        raise ValueError("country_targets weights must sum to 1.0")

    countries = tuple(country for country, _ in sorted(normalized_targets, key=lambda item: item[0]))
    target_weights = {country: weight for country, weight in normalized_targets}

    eligible_rows = _parse_ranked_rows(ranked_factor_rows)
    rows_by_country: dict[str, list[_RankedRow]] = {country: [] for country in countries}
    for row in eligible_rows:
        if row.country in rows_by_country:
            rows_by_country[row.country].append(row)

    for country in countries:
        rows_by_country[country].sort(
            key=lambda row: (
                row.rank_in_country if row.rank_in_country is not None else 10**9,
                -cast(float, row.factor_value),
                row.security_id,
            )
        )

    reasons: set[str] = set()
    requested = _requested_counts(max_holdings=max_holdings, countries=countries, target_weights=target_weights)
    selected: list[_RankedRow] = []
    selected_country_counts = {country: 0 for country in countries}

    for country in countries:
        picks = rows_by_country[country][: requested[country]]
        selected.extend(picks)
        selected_country_counts[country] = len(picks)
        if len(picks) < requested[country]:
            reasons.add(f"INSUFFICIENT_NAMES_{country}")

    if len(selected) < max_holdings:
        remaining_pool = [
            row
            for country in countries
            for row in rows_by_country[country][selected_country_counts[country] :]
        ]
        remaining_pool.sort(
            key=lambda row: (
                row.rank_in_country if row.rank_in_country is not None else 10**9,
                -cast(float, row.factor_value),
                row.country,
                row.security_id,
            )
        )
        need = max_holdings - len(selected)
        selected.extend(remaining_pool[:need])
        for row in remaining_pool[:need]:
            selected_country_counts[row.country] += 1

    if len(selected) < max_holdings:
        reasons.add("TOTAL_UNDER_MAX_HOLDINGS")

    selected.sort(key=lambda row: (row.country, row.rank_in_country if row.rank_in_country is not None else 10**9, row.security_id))

    country_caps = {country: selected_country_counts[country] * max_single_name_weight for country in countries}
    country_weights = {country: 0.0 for country in countries}

    for country in countries:
        if selected_country_counts[country] == 0:
            reasons.add(f"NO_SELECTED_NAMES_{country}")
            continue
        target = target_weights[country]
        cap = country_caps[country]
        if cap < target:
            reasons.add(f"COUNTRY_CAPACITY_BIND_{country}")
        country_weights[country] = min(target, cap)

    allocated = sum(country_weights.values())
    remaining_weight = max(0.0, 1.0 - allocated)
    if remaining_weight > 0.0:
        for country in countries:
            if remaining_weight <= 0.0:
                break
            spare = max(0.0, country_caps[country] - country_weights[country])
            if spare <= 0.0:
                continue
            increment = min(spare, remaining_weight)
            country_weights[country] += increment
            remaining_weight -= increment

    if remaining_weight > 1e-12:
        reasons.add("UNALLOCATED_CASH_DUE_TO_CAPACITY")

    lower_bounds = {country: max(0.0, target_weights[country] - country_tolerance) for country in countries}
    upper_bounds = {country: min(1.0, target_weights[country] + country_tolerance) for country in countries}
    for country in countries:
        if selected_country_counts[country] == 0:
            continue
        weight = country_weights[country]
        if weight < lower_bounds[country] - 1e-12 or weight > upper_bounds[country] + 1e-12:
            reasons.add(f"COUNTRY_TOLERANCE_UNMET_{country}")

    rows_by_country_selected: dict[str, list[_RankedRow]] = {country: [] for country in countries}
    for row in selected:
        rows_by_country_selected[row.country].append(row)

    final_weights: dict[str, float] = {}
    benchmark_weights: dict[str, float] = {}
    caps_by_security: dict[str, float] = {}

    for country in countries:
        country_rows = rows_by_country_selected[country]
        target_country_weight = country_weights[country]
        if not country_rows or target_country_weight <= 0.0:
            continue

        benchmark_seed: dict[str, float] = {}
        for row in country_rows:
            seed = row.benchmark_proxy_weight
            if seed is None or seed <= 0.0:
                seed = row.median_traded_value_krw if row.median_traded_value_krw is not None else 1.0
            benchmark_seed[row.security_id] = max(float(seed), 1.0)

        seed_sum = sum(benchmark_seed.values())
        benchmark_country = {
            security_id: value / seed_sum
            for security_id, value in benchmark_seed.items()
        }

        factor_values = [cast(float, row.factor_value) for row in country_rows]
        mean_factor = sum(factor_values) / len(factor_values)
        std_factor = _safe_std(factor_values)
        zscores = {
            row.security_id: ((cast(float, row.factor_value) - mean_factor) / std_factor if std_factor > 1e-12 else 0.0)
            for row in country_rows
        }

        raw_weights: dict[str, float] = {}
        country_caps_by_security: dict[str, float] = {}
        for row in country_rows:
            security_id = row.security_id
            bench = benchmark_country[security_id]
            score = zscores[security_id]
            raw = max(0.0, bench * (1.0 + alpha_tilt_strength * score))
            raw_weights[security_id] = raw

            liquidity = row.median_traded_value_krw
            adv_cap = max_single_name_weight
            if liquidity is not None and liquidity > 0.0:
                adv_cap = min(max_single_name_weight, (max_adv_participation * liquidity) / portfolio_value_krw)
                if adv_cap < max_single_name_weight - 1e-12:
                    reasons.add("ADV_CAP_BIND")
            cap = max(0.0, min(max_single_name_weight, adv_cap))
            if cap <= 0.0:
                reasons.add(f"ZERO_CAP_{country}")
            country_caps_by_security[security_id] = cap
            caps_by_security[security_id] = cap
            benchmark_weights[security_id] = target_country_weight * bench

        country_allocated = _rescale_with_caps(
            base_weights=raw_weights,
            cap_weights=country_caps_by_security,
            target_total=target_country_weight,
        )
        final_weights.update(country_allocated)

    if te_active_l2_cap > 0.0 and final_weights:
        active_l2 = math.sqrt(
            sum((final_weights.get(security_id, 0.0) - benchmark_weights.get(security_id, 0.0)) ** 2 for security_id in final_weights)
        )
        if active_l2 > te_active_l2_cap + 1e-12:
            reasons.add("TE_ACTIVE_L2_CAP_BIND")
            scale = te_active_l2_cap / active_l2
            te_adjusted_weights: dict[str, float] = {}
            for country in countries:
                country_rows = rows_by_country_selected[country]
                if not country_rows:
                    continue
                target_country_weight = country_weights[country]
                raw_country: dict[str, float] = {}
                caps_country: dict[str, float] = {}
                for row in country_rows:
                    security_id = row.security_id
                    bench = benchmark_weights.get(security_id, 0.0)
                    active_component = final_weights.get(security_id, 0.0) - bench
                    raw_country[security_id] = max(0.0, bench + active_component * scale)
                    caps_country[security_id] = caps_by_security.get(security_id, max_single_name_weight)
                te_adjusted_weights.update(
                    _rescale_with_caps(
                        base_weights=raw_country,
                        cap_weights=caps_country,
                        target_total=target_country_weight,
                    )
                )
            final_weights = te_adjusted_weights

    if max_turnover is not None and max_turnover >= 0.0 and final_weights and previous_weights:
        blended = _blend_with_turnover_limit(
            target_weights=final_weights,
            previous_weights=previous_weights,
            max_turnover=max_turnover,
        )
        if any(abs(blended.get(security_id, 0.0) - final_weights.get(security_id, 0.0)) > 1e-12 for security_id in final_weights):
            reasons.add("TURNOVER_CAP_BIND")
            turnover_adjusted_weights: dict[str, float] = {}
            for country in countries:
                country_rows = rows_by_country_selected[country]
                if not country_rows:
                    continue
                target_country_weight = country_weights[country]
                raw_country = {row.security_id: max(0.0, blended.get(row.security_id, 0.0)) for row in country_rows}
                caps_country = {row.security_id: caps_by_security.get(row.security_id, max_single_name_weight) for row in country_rows}
                turnover_adjusted_weights.update(
                    _rescale_with_caps(
                        base_weights=raw_country,
                        cap_weights=caps_country,
                        target_total=target_country_weight,
                    )
                )
            final_weights = turnover_adjusted_weights

    holdings: list[PortfolioHolding] = []
    for country in countries:
        country_rows = rows_by_country_selected[country]
        for row in country_rows:
            weight = float(final_weights.get(row.security_id, 0.0))
            if weight <= 0.0:
                continue
            holdings.append(
                PortfolioHolding(
                    security_id=row.security_id,
                    country=country,
                    weight=weight,
                    rank_in_country=row.rank_in_country,
                    factor_value=row.factor_value,
                )
            )

    holdings.sort(key=lambda row: (row.country, row.rank_in_country if row.rank_in_country is not None else 10**9, row.security_id))
    realized_country_weights = {country: 0.0 for country in countries}
    for holding in holdings:
        realized_country_weights[holding.country] += holding.weight
    cash_weight = max(0.0, 1.0 - sum(holding.weight for holding in holdings))
    if cash_weight < 1e-12:
        cash_weight = 0.0
    diagnostics = PortfolioSelectionDiagnostics(
        requested_holdings=max_holdings,
        selected_holdings=len(holdings),
        available_eligible=len(eligible_rows),
        requested_country_counts=tuple((country, requested[country]) for country in countries),
        available_country_counts=tuple((country, len(rows_by_country[country])) for country in countries),
        selected_country_counts=tuple((country, selected_country_counts[country]) for country in countries),
        country_weights=tuple((country, realized_country_weights[country]) for country in countries),
        cash_weight=cash_weight,
    )

    return PortfolioConstructionResult(
        holdings=tuple(holdings),
        fallback_triggered=bool(reasons),
        fallback_reasons=_format_reasons(reasons),
        jp_odd_lot_enabled=True,
        diagnostics=diagnostics,
    )


__all__ = [
    "DEFAULT_COUNTRY_TARGETS",
    "DEFAULT_COUNTRY_TOLERANCE",
    "DEFAULT_MAX_HOLDINGS",
    "DEFAULT_MAX_SINGLE_NAME_WEIGHT",
    "PortfolioConstructionResult",
    "PortfolioHolding",
    "PortfolioSelectionDiagnostics",
    "construct_portfolio_with_constraints",
]
