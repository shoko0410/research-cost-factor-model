"""Stable boundary contracts for Python and Rust kernel integration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
from dataclasses import dataclass
from datetime import date
from typing import Callable, TypeVar

from ..portfolio.constructor import (
    PortfolioConstructionResult,
    PortfolioHolding,
    PortfolioSelectionDiagnostics,
)
import math

from ..backtest.engine import (
    BacktestMetrics,
    BacktestResult,
    HoldingSnapshot,
    PeriodReturn,
    TradeLedgerEntry,
)


_PairValue = TypeVar("_PairValue")


@dataclass(frozen=True)
class RankingKernelRequest:
    factor_rows: tuple[Mapping[str, object], ...]
    accepted_rows: tuple[Mapping[str, object], ...]
    sector_by_security: dict[str, str]
    requested_counts_by_country: dict[str, int]
    factor_model: str
    sector_active_band: float
    use_size_stratification: bool


@dataclass(frozen=True)
class ConstructorKernelRequest:
    ranked_factor_rows: tuple[Mapping[str, object], ...]
    country_targets: tuple[tuple[str, float], ...] | None
    risk_controls: dict[str, object]


@dataclass(frozen=True)
class BacktestKernelRequest:
    rebalance_schedule: tuple[date | str, ...]
    prices: tuple[Mapping[str, object], ...]
    benchmark_series: tuple[Mapping[str, object], ...]
    fx_rates: tuple[Mapping[str, object], ...]
    portfolio_allocations: Mapping[object, Sequence[Mapping[str, object]]] | tuple[Mapping[str, object], ...]
    initial_nav_krw: float
    cost_tax_config: Mapping[str, object] | None = None


def _stable_record(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        attrs = getattr(value, "__dict__", None)
        if isinstance(attrs, Mapping):
            return {
                str(key): row_value
                for key, row_value in attrs.items()
                if not str(key).startswith("_")
            }
        raise ValueError("kernel boundary rows must be mapping-like")
    return {str(key): row_value for key, row_value in value.items()}


def normalize_ranking_request(
    *,
    factor_rows: Iterable[object],
    accepted_rows: Sequence[object],
    sector_by_security: Mapping[str, str],
    requested_counts_by_country: Mapping[str, int],
    factor_model: str,
    sector_active_band: float,
    use_size_stratification: bool,
) -> RankingKernelRequest:
    return RankingKernelRequest(
        factor_rows=tuple(_stable_record(row) for row in factor_rows),
        accepted_rows=tuple(_stable_record(row) for row in accepted_rows),
        sector_by_security={str(key): str(value) for key, value in sector_by_security.items()},
        requested_counts_by_country={str(key): int(value) for key, value in requested_counts_by_country.items()},
        factor_model=str(factor_model),
        sector_active_band=float(sector_active_band),
        use_size_stratification=bool(use_size_stratification),
    )


def normalize_constructor_request(
    *,
    ranked_factor_rows: Iterable[object],
    country_targets: tuple[tuple[str, float], ...] | None,
    risk_controls: Mapping[str, object],
) -> ConstructorKernelRequest:
    return ConstructorKernelRequest(
        ranked_factor_rows=tuple(_stable_record(row) for row in ranked_factor_rows),
        country_targets=country_targets,
        risk_controls={str(key): value for key, value in risk_controls.items()},
    )


def serialize_ranking_request(request: RankingKernelRequest) -> str:
    payload = {
        "factor_rows": list(request.factor_rows),
        "accepted_rows": list(request.accepted_rows),
        "sector_by_security": dict(request.sector_by_security),
        "requested_counts_by_country": dict(request.requested_counts_by_country),
        "factor_model": request.factor_model,
        "sector_active_band": request.sector_active_band,
        "use_size_stratification": request.use_size_stratification,
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, allow_nan=False)


def decode_ranking_result(payload: object) -> list[dict[str, object]]:
    parsed_payload: object = payload
    if isinstance(payload, str):
        parsed_payload = json.loads(payload)

    if not isinstance(parsed_payload, Sequence) or isinstance(parsed_payload, (str, bytes, bytearray)):
        raise ValueError("ranking result payload must be sequence")

    parsed_rows: list[dict[str, object]] = []
    for index, raw_row in enumerate(parsed_payload):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"ranking result rows[{index}] must be mapping")
        row = {str(key): value for key, value in raw_row.items()}

        parsed_rows.append(
            {
                "security_id": _to_text(row.get("security_id"), f"rows[{index}].security_id"),
                "country": _to_text(row.get("country"), f"rows[{index}].country"),
                "factor_value": _to_float_value(row.get("factor_value"), f"rows[{index}].factor_value"),
                "base_factor_value": _to_float_value(
                    row.get("base_factor_value"),
                    f"rows[{index}].base_factor_value",
                ),
                "sector": _to_text(row.get("sector"), f"rows[{index}].sector"),
                "median_traded_value_krw": _to_non_negative_float(
                    row.get("median_traded_value_krw"),
                    f"rows[{index}].median_traded_value_krw",
                ),
                "rd_expense": _to_float_value(row.get("rd_expense"), f"rows[{index}].rd_expense"),
                "sales_ttm": _to_optional_float(row.get("sales_ttm"), f"rows[{index}].sales_ttm"),
                "size_bucket": _to_text(row.get("size_bucket"), f"rows[{index}].size_bucket"),
                "rank_in_country": _to_optional_int(row.get("rank_in_country"), f"rows[{index}].rank_in_country"),
                "is_eligible": _to_bool(row.get("is_eligible"), f"rows[{index}].is_eligible"),
            }
        )
    return parsed_rows


def serialize_constructor_request(request: ConstructorKernelRequest) -> str:
    payload = {
        "ranked_factor_rows": list(request.ranked_factor_rows),
        "country_targets": list(request.country_targets) if request.country_targets is not None else None,
        "risk_controls": dict(request.risk_controls),
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, allow_nan=False)


def _to_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be text")
    parsed = value.strip()
    if not parsed:
        raise ValueError(f"{field_name} cannot be empty")
    return parsed


def _to_float_value(value: object, field_name: str) -> float:
    return _as_float(value, field_name=field_name)


def _to_int_value(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{field_name} must be integer")
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError as exc:
            raise ValueError(f"{field_name} must be integer") from exc
    raise ValueError(f"{field_name} must be integer")


def _to_bool(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field_name} must be boolean")


def _to_non_negative_float(value: object, field_name: str) -> float:
    parsed = _to_float_value(value, field_name)
    if parsed < 0.0:
        raise ValueError(f"{field_name} must be non-negative")
    return parsed


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


def _to_optional_float(value: object, field_name: str) -> float | None:
    if value in (None, ""):
        return None
    return _to_float_value(value, field_name)


def _to_country_pairs(
    value: object,
    field_name: str,
    *,
    value_cast: Callable[[object, str], _PairValue],
) -> tuple[tuple[str, _PairValue], ...]:
    if not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a sequence of pairs")
    parsed: list[tuple[str, _PairValue]] = []
    for index, pair in enumerate(value):
        if not isinstance(pair, Sequence) or len(pair) != 2:
            raise ValueError(f"{field_name}[{index}] must be a (country, value) pair")
        country = _to_text(pair[0], f"{field_name}[{index}].country")
        parsed.append((country, value_cast(pair[1], f"{field_name}[{index}].value")))
    return tuple(parsed)


def deserialize_constructor_result(payload: object) -> PortfolioConstructionResult:
    if isinstance(payload, PortfolioConstructionResult):
        return payload

    parsed_payload: Mapping[str, object]
    if isinstance(payload, str):
        loaded = json.loads(payload)
        if not isinstance(loaded, Mapping):
            raise ValueError("constructor result payload must be object-like")
        parsed_payload = {str(key): value for key, value in loaded.items()}
    elif isinstance(payload, Mapping):
        parsed_payload = {str(key): value for key, value in payload.items()}
    else:
        raise ValueError("constructor result payload must be mapping or JSON string")

    holdings_raw = parsed_payload.get("holdings")
    if not isinstance(holdings_raw, Sequence):
        raise ValueError("constructor result holdings must be sequence")
    holdings: list[PortfolioHolding] = []
    for index, row in enumerate(holdings_raw):
        if not isinstance(row, Mapping):
            raise ValueError(f"holdings[{index}] must be mapping")
        holdings.append(
            PortfolioHolding(
                security_id=_to_text(row.get("security_id"), f"holdings[{index}].security_id"),
                country=_to_text(row.get("country"), f"holdings[{index}].country"),
                weight=_to_non_negative_float(row.get("weight"), f"holdings[{index}].weight"),
                rank_in_country=_to_optional_int(row.get("rank_in_country"), f"holdings[{index}].rank_in_country"),
                factor_value=_to_optional_float(row.get("factor_value"), f"holdings[{index}].factor_value"),
            )
        )

    diagnostics_raw = parsed_payload.get("diagnostics")
    if not isinstance(diagnostics_raw, Mapping):
        raise ValueError("constructor result diagnostics must be mapping")

    requested_country_counts = _to_country_pairs(
        diagnostics_raw.get("requested_country_counts"),
        "diagnostics.requested_country_counts",
        value_cast=_to_int_value,
    )
    available_country_counts = _to_country_pairs(
        diagnostics_raw.get("available_country_counts"),
        "diagnostics.available_country_counts",
        value_cast=_to_int_value,
    )
    selected_country_counts = _to_country_pairs(
        diagnostics_raw.get("selected_country_counts"),
        "diagnostics.selected_country_counts",
        value_cast=_to_int_value,
    )
    country_weights = _to_country_pairs(
        diagnostics_raw.get("country_weights"),
        "diagnostics.country_weights",
        value_cast=lambda value, field_name: _to_float_value(value, field_name),
    )

    diagnostics = PortfolioSelectionDiagnostics(
        requested_holdings=_to_int_value(diagnostics_raw.get("requested_holdings"), "diagnostics.requested_holdings"),
        selected_holdings=_to_int_value(diagnostics_raw.get("selected_holdings"), "diagnostics.selected_holdings"),
        available_eligible=_to_int_value(diagnostics_raw.get("available_eligible"), "diagnostics.available_eligible"),
        requested_country_counts=requested_country_counts,
        available_country_counts=available_country_counts,
        selected_country_counts=selected_country_counts,
        country_weights=country_weights,
        cash_weight=_to_non_negative_float(diagnostics_raw.get("cash_weight"), "diagnostics.cash_weight"),
    )

    fallback_reasons_raw = parsed_payload.get("fallback_reasons")
    if not isinstance(fallback_reasons_raw, Sequence):
        raise ValueError("constructor result fallback_reasons must be sequence")
    fallback_reasons = tuple(
        _to_text(reason, f"fallback_reasons[{index}]")
        for index, reason in enumerate(fallback_reasons_raw)
    )

    return PortfolioConstructionResult(
        holdings=tuple(holdings),
        fallback_triggered=_to_bool(parsed_payload.get("fallback_triggered"), "fallback_triggered"),
        fallback_reasons=fallback_reasons,
        jp_odd_lot_enabled=_to_bool(parsed_payload.get("jp_odd_lot_enabled"), "jp_odd_lot_enabled"),
        diagnostics=diagnostics,
    )


def normalize_backtest_request(
    *,
    rebalance_schedule: Sequence[date | str],
    prices: Sequence[Mapping[str, object]],
    benchmark_series: Sequence[Mapping[str, object]],
    fx_rates: Sequence[Mapping[str, object]],
    portfolio_allocations: Mapping[object, Sequence[Mapping[str, object]]] | Sequence[Mapping[str, object]],
    initial_nav_krw: float,
    cost_tax_config: Mapping[str, object] | None = None,
) -> BacktestKernelRequest:
    normalized_allocations: Mapping[object, Sequence[Mapping[str, object]]] | tuple[Mapping[str, object], ...]
    if isinstance(portfolio_allocations, Mapping):
        normalized_allocations = {
            key: tuple(_stable_record(row) for row in rows)
            for key, rows in portfolio_allocations.items()
        }
    else:
        normalized_allocations = tuple(_stable_record(row) for row in portfolio_allocations)
    return BacktestKernelRequest(
        rebalance_schedule=tuple(rebalance_schedule),
        prices=tuple(_stable_record(row) for row in prices),
        benchmark_series=tuple(_stable_record(row) for row in benchmark_series),
        fx_rates=tuple(_stable_record(row) for row in fx_rates),
        portfolio_allocations=normalized_allocations,
        initial_nav_krw=float(initial_nav_krw),
        cost_tax_config=None if cost_tax_config is None else {str(key): value for key, value in cost_tax_config.items()},
    )


def _json_ready(value: object) -> object:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def serialize_backtest_request(request: BacktestKernelRequest) -> str:
    payload: dict[str, object] = {
        "rebalance_schedule": request.rebalance_schedule,
        "prices": request.prices,
        "benchmark_series": request.benchmark_series,
        "fx_rates": request.fx_rates,
        "portfolio_allocations": request.portfolio_allocations,
        "initial_nav_krw": request.initial_nav_krw,
    }
    if request.cost_tax_config is not None:
        payload["cost_tax_config"] = request.cost_tax_config
    return json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be mapping")
    return {str(key): item for key, item in value.items()}


def _as_sequence(value: object, *, field_name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be sequence")
    return value


def _as_text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _as_date(value: object, *, field_name: str) -> date:
    if isinstance(value, date):
        return value
    text = _as_text(value, field_name=field_name)
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be ISO date") from exc


def _as_float(value: object, *, field_name: str) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite numeric")
    return parsed


def _as_optional_float(value: object, *, field_name: str) -> float | None:
    if value is None:
        return None
    return _as_float(value, field_name=field_name)


def decode_backtest_result(raw_result: object) -> BacktestResult:
    if isinstance(raw_result, BacktestResult):
        return raw_result

    payload: object = raw_result
    if isinstance(raw_result, str):
        payload = json.loads(raw_result)

    root = _as_mapping(payload, field_name="backtest_result")

    trades_rows = _as_sequence(root.get("trades", ()), field_name="backtest_result.trades")
    holdings_rows = _as_sequence(root.get("holdings", ()), field_name="backtest_result.holdings")
    period_rows = _as_sequence(root.get("returns", ()), field_name="backtest_result.returns")
    metrics_row = _as_mapping(root.get("metrics", {}), field_name="backtest_result.metrics")

    trades = tuple(
        TradeLedgerEntry(
            rebalance_date=_as_date(_as_mapping(row, field_name="trade")["rebalance_date"], field_name="trade.rebalance_date"),
            security_id=_as_text(_as_mapping(row, field_name="trade")["security_id"], field_name="trade.security_id"),
            side=_as_text(_as_mapping(row, field_name="trade")["side"], field_name="trade.side"),
            country=_as_text(_as_mapping(row, field_name="trade")["country"], field_name="trade.country"),
            currency=_as_text(_as_mapping(row, field_name="trade")["currency"], field_name="trade.currency"),
            quantity=_as_float(_as_mapping(row, field_name="trade")["quantity"], field_name="trade.quantity"),
            price_local=_as_float(_as_mapping(row, field_name="trade")["price_local"], field_name="trade.price_local"),
            price_krw=_as_float(_as_mapping(row, field_name="trade")["price_krw"], field_name="trade.price_krw"),
            notional_krw=_as_float(_as_mapping(row, field_name="trade")["notional_krw"], field_name="trade.notional_krw"),
            gross_proceeds_krw=_as_float(
                _as_mapping(row, field_name="trade")["gross_proceeds_krw"],
                field_name="trade.gross_proceeds_krw",
            ),
            gross_cost_krw=_as_float(_as_mapping(row, field_name="trade")["gross_cost_krw"], field_name="trade.gross_cost_krw"),
            commission_krw=_as_float(_as_mapping(row, field_name="trade")["commission_krw"], field_name="trade.commission_krw"),
            slippage_krw=_as_float(_as_mapping(row, field_name="trade")["slippage_krw"], field_name="trade.slippage_krw"),
            fx_fee_krw=_as_float(_as_mapping(row, field_name="trade")["fx_fee_krw"], field_name="trade.fx_fee_krw"),
            sell_tax_krw=_as_float(_as_mapping(row, field_name="trade")["sell_tax_krw"], field_name="trade.sell_tax_krw"),
            realized_gain_krw=_as_float(
                _as_mapping(row, field_name="trade")["realized_gain_krw"],
                field_name="trade.realized_gain_krw",
            ),
            realized_gains_tax_krw=_as_float(
                _as_mapping(row, field_name="trade")["realized_gains_tax_krw"],
                field_name="trade.realized_gains_tax_krw",
            ),
            total_cost_tax_krw=_as_float(
                _as_mapping(row, field_name="trade")["total_cost_tax_krw"],
                field_name="trade.total_cost_tax_krw",
            ),
        )
        for row in trades_rows
    )

    holdings = tuple(
        HoldingSnapshot(
            as_of_date=_as_date(_as_mapping(row, field_name="holding")["as_of_date"], field_name="holding.as_of_date"),
            security_id=_as_text(_as_mapping(row, field_name="holding")["security_id"], field_name="holding.security_id"),
            country=_as_text(_as_mapping(row, field_name="holding")["country"], field_name="holding.country"),
            currency=_as_text(_as_mapping(row, field_name="holding")["currency"], field_name="holding.currency"),
            quantity=_as_float(_as_mapping(row, field_name="holding")["quantity"], field_name="holding.quantity"),
            avg_cost_krw=_as_float(_as_mapping(row, field_name="holding")["avg_cost_krw"], field_name="holding.avg_cost_krw"),
            price_local=_as_float(_as_mapping(row, field_name="holding")["price_local"], field_name="holding.price_local"),
            price_krw=_as_float(_as_mapping(row, field_name="holding")["price_krw"], field_name="holding.price_krw"),
            market_value_krw=_as_float(
                _as_mapping(row, field_name="holding")["market_value_krw"],
                field_name="holding.market_value_krw",
            ),
            weight=_as_float(_as_mapping(row, field_name="holding")["weight"], field_name="holding.weight"),
        )
        for row in holdings_rows
    )

    period_returns = tuple(
        PeriodReturn(
            start_date=_as_date(_as_mapping(row, field_name="period")["start_date"], field_name="period.start_date"),
            end_date=_as_date(_as_mapping(row, field_name="period")["end_date"], field_name="period.end_date"),
            start_nav_krw=_as_float(_as_mapping(row, field_name="period")["start_nav_krw"], field_name="period.start_nav_krw"),
            end_nav_krw=_as_float(_as_mapping(row, field_name="period")["end_nav_krw"], field_name="period.end_nav_krw"),
            gross_return=_as_float(_as_mapping(row, field_name="period")["gross_return"], field_name="period.gross_return"),
            net_return=_as_float(_as_mapping(row, field_name="period")["net_return"], field_name="period.net_return"),
            benchmark_krw_return=_as_optional_float(
                _as_mapping(row, field_name="period").get("benchmark_krw_return"),
                field_name="period.benchmark_krw_return",
            ),
            total_commission_krw=_as_float(
                _as_mapping(row, field_name="period")["total_commission_krw"],
                field_name="period.total_commission_krw",
            ),
            total_slippage_krw=_as_float(
                _as_mapping(row, field_name="period")["total_slippage_krw"],
                field_name="period.total_slippage_krw",
            ),
            total_fx_fee_krw=_as_float(_as_mapping(row, field_name="period")["total_fx_fee_krw"], field_name="period.total_fx_fee_krw"),
            total_sell_tax_krw=_as_float(
                _as_mapping(row, field_name="period")["total_sell_tax_krw"],
                field_name="period.total_sell_tax_krw",
            ),
            total_realized_gains_tax_krw=_as_float(
                _as_mapping(row, field_name="period")["total_realized_gains_tax_krw"],
                field_name="period.total_realized_gains_tax_krw",
            ),
            total_cost_tax_krw=_as_float(_as_mapping(row, field_name="period")["total_cost_tax_krw"], field_name="period.total_cost_tax_krw"),
        )
        for row in period_rows
    )

    metrics = BacktestMetrics(
        periods=int(_as_float(metrics_row["periods"], field_name="metrics.periods")),
        start_nav_krw=_as_float(metrics_row["start_nav_krw"], field_name="metrics.start_nav_krw"),
        end_nav_krw=_as_float(metrics_row["end_nav_krw"], field_name="metrics.end_nav_krw"),
        cumulative_net_return=_as_float(
            metrics_row["cumulative_net_return"],
            field_name="metrics.cumulative_net_return",
        ),
        cumulative_gross_return=_as_float(
            metrics_row["cumulative_gross_return"],
            field_name="metrics.cumulative_gross_return",
        ),
        benchmark_cumulative_return=_as_optional_float(
            metrics_row.get("benchmark_cumulative_return"),
            field_name="metrics.benchmark_cumulative_return",
        ),
        total_commission_krw=_as_float(metrics_row["total_commission_krw"], field_name="metrics.total_commission_krw"),
        total_slippage_krw=_as_float(metrics_row["total_slippage_krw"], field_name="metrics.total_slippage_krw"),
        total_fx_fee_krw=_as_float(metrics_row["total_fx_fee_krw"], field_name="metrics.total_fx_fee_krw"),
        total_sell_tax_krw=_as_float(metrics_row["total_sell_tax_krw"], field_name="metrics.total_sell_tax_krw"),
        total_realized_gains_tax_krw=_as_float(
            metrics_row["total_realized_gains_tax_krw"],
            field_name="metrics.total_realized_gains_tax_krw",
        ),
        total_cost_tax_krw=_as_float(metrics_row["total_cost_tax_krw"], field_name="metrics.total_cost_tax_krw"),
    )

    return BacktestResult(
        trades=trades,
        holdings=holdings,
        returns=period_returns,
        metrics=metrics,
    )


__all__ = [
    "BacktestKernelRequest",
    "ConstructorKernelRequest",
    "RankingKernelRequest",
    "decode_ranking_result",
    "deserialize_constructor_result",
    "decode_backtest_result",
    "normalize_backtest_request",
    "normalize_constructor_request",
    "normalize_ranking_request",
    "serialize_ranking_request",
    "serialize_constructor_request",
    "serialize_backtest_request",
]
