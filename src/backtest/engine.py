"""Deterministic quarterly backtest engine with KRW-base accounting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
import math

from ..data.ingest.fx import convert_return_to_krw_base, fx_rate_on_date


@dataclass(frozen=True)
class CostTaxConfig:
    commission_rate: float
    slippage_rate: float
    fx_fee_rate: float
    kr_sell_tax_rate: float
    jp_realized_gains_tax_rate: float
    us_realized_gains_tax_rate: float


@dataclass(frozen=True)
class TradeLedgerEntry:
    rebalance_date: date
    security_id: str
    side: str
    country: str
    currency: str
    quantity: float
    price_local: float
    price_krw: float
    notional_krw: float
    gross_proceeds_krw: float
    gross_cost_krw: float
    commission_krw: float
    slippage_krw: float
    fx_fee_krw: float
    sell_tax_krw: float
    realized_gain_krw: float
    realized_gains_tax_krw: float
    total_cost_tax_krw: float


@dataclass(frozen=True)
class HoldingSnapshot:
    as_of_date: date
    security_id: str
    country: str
    currency: str
    quantity: float
    avg_cost_krw: float
    price_local: float
    price_krw: float
    market_value_krw: float
    weight: float


@dataclass(frozen=True)
class PeriodReturn:
    start_date: date
    end_date: date
    start_nav_krw: float
    end_nav_krw: float
    gross_return: float
    net_return: float
    benchmark_krw_return: float | None
    total_commission_krw: float
    total_slippage_krw: float
    total_fx_fee_krw: float
    total_sell_tax_krw: float
    total_realized_gains_tax_krw: float
    total_cost_tax_krw: float


@dataclass(frozen=True)
class BacktestMetrics:
    periods: int
    start_nav_krw: float
    end_nav_krw: float
    cumulative_net_return: float
    cumulative_gross_return: float
    benchmark_cumulative_return: float | None
    total_commission_krw: float
    total_slippage_krw: float
    total_fx_fee_krw: float
    total_sell_tax_krw: float
    total_realized_gains_tax_krw: float
    total_cost_tax_krw: float


@dataclass(frozen=True)
class BacktestResult:
    trades: tuple[TradeLedgerEntry, ...]
    holdings: tuple[HoldingSnapshot, ...]
    returns: tuple[PeriodReturn, ...]
    metrics: BacktestMetrics


@dataclass
class _Position:
    security_id: str
    country: str
    currency: str
    quantity: float
    avg_cost_krw: float


@dataclass(frozen=True)
class _Allocation:
    security_id: str
    target_weight: float


DEFAULT_COST_TAX_CONFIG = {
    "commission_rate": 0.0,
    "slippage_bps": 10.0,
    "fx_fee_rate": 0.0,
    "kr_sell_tax_rate": 0.0015,
    "jp_realized_gains_tax_rate": 0.20,
    "us_realized_gains_tax_rate": 0.22,
}


def _to_date(value: object, *, field_name: str) -> date:
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return date.fromisoformat(text)


def _to_float(value: object, *, field_name: str) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite numeric")
    return parsed


def _normalize_rate(value: object, *, field_name: str) -> float:
    parsed = _to_float(value, field_name=field_name)
    if not (0.0 <= parsed <= 1.0):
        raise ValueError(f"{field_name} must be in [0.0, 1.0]")
    return parsed


def parse_cost_tax_config(config: Mapping[str, object] | None = None) -> CostTaxConfig:
    source: dict[str, object] = dict(DEFAULT_COST_TAX_CONFIG)
    if config is not None:
        source.update(dict(config))

    if "slippage_bps" not in source:
        raise ValueError("cost_tax_config.slippage_bps is required")
    slippage_bps = _to_float(source["slippage_bps"], field_name="cost_tax_config.slippage_bps")
    if slippage_bps < 0.0:
        raise ValueError("cost_tax_config.slippage_bps must be non-negative")

    return CostTaxConfig(
        commission_rate=_normalize_rate(source["commission_rate"], field_name="cost_tax_config.commission_rate"),
        slippage_rate=slippage_bps / 10_000.0,
        fx_fee_rate=_normalize_rate(source["fx_fee_rate"], field_name="cost_tax_config.fx_fee_rate"),
        kr_sell_tax_rate=_normalize_rate(source["kr_sell_tax_rate"], field_name="cost_tax_config.kr_sell_tax_rate"),
        jp_realized_gains_tax_rate=_normalize_rate(
            source["jp_realized_gains_tax_rate"],
            field_name="cost_tax_config.jp_realized_gains_tax_rate",
        ),
        us_realized_gains_tax_rate=_normalize_rate(
            source["us_realized_gains_tax_rate"],
            field_name="cost_tax_config.us_realized_gains_tax_rate",
        ),
    )


def _realized_gains_tax_rate(country: str, config: CostTaxConfig) -> float:
    normalized = country.upper()
    if normalized == "JP":
        return config.jp_realized_gains_tax_rate
    if normalized == "US":
        return config.us_realized_gains_tax_rate
    return 0.0


def _price_date_key(row: Mapping[str, object], *, field_name: str) -> date:
    if field_name in row:
        return _to_date(row[field_name], field_name=field_name)
    return _to_date(row["as_of_date"], field_name="as_of_date")


def _build_price_lookup(
    price_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[tuple[str, date], Mapping[str, object]], dict[str, tuple[str, str]]]:
    lookup: dict[tuple[str, date], Mapping[str, object]] = {}
    metadata: dict[str, tuple[str, str]] = {}
    for index, row in enumerate(price_rows):
        security_id = str(row.get("security_id", "")).strip()
        if not security_id:
            raise ValueError(f"prices[{index}].security_id is required")
        key = (security_id, _price_date_key(row, field_name="price_date"))
        lookup[key] = row

        country = str(row.get("country", "")).strip().upper()
        currency = str(row.get("currency", "")).strip().upper()
        if country and currency:
            _ = metadata.setdefault(security_id, (country, currency))
    return lookup, metadata


def _build_benchmark_lookup(benchmark_rows: Sequence[Mapping[str, object]]) -> dict[date, Mapping[str, object]]:
    lookup: dict[date, Mapping[str, object]] = {}
    for index, row in enumerate(benchmark_rows):
        benchmark_date = _price_date_key(row, field_name="benchmark_date")
        if "close" not in row:
            raise ValueError(f"benchmark_series[{index}].close is required")
        lookup[benchmark_date] = row
    return lookup


def _normalize_allocations(
    allocations: Mapping[object, Sequence[Mapping[str, object]]] | Sequence[Mapping[str, object]],
) -> dict[date, tuple[_Allocation, ...]]:
    grouped: dict[date, list[_Allocation]] = {}
    if isinstance(allocations, Mapping):
        for raw_date, rows in allocations.items():
            rebalance_date = _to_date(raw_date, field_name="portfolio_allocations.rebalance_date")
            bucket = grouped.setdefault(rebalance_date, [])
            for row in rows:
                security_id = str(row.get("security_id", "")).strip()
                if not security_id:
                    raise ValueError("portfolio_allocations.security_id is required")
                weight_value = row["target_weight"] if "target_weight" in row else row.get("weight")
                if weight_value is None:
                    raise ValueError("portfolio_allocations.target_weight is required")
                weight = _to_float(weight_value, field_name="portfolio_allocations.target_weight")
                if weight < 0.0:
                    raise ValueError("portfolio_allocations.target_weight must be non-negative")
                bucket.append(_Allocation(security_id=security_id, target_weight=weight))
    else:
        for index, row in enumerate(allocations):
            rebalance_date = _to_date(
                row.get("rebalance_date", ""),
                field_name=f"portfolio_allocations[{index}].rebalance_date",
            )
            security_id = str(row.get("security_id", "")).strip()
            if not security_id:
                raise ValueError(f"portfolio_allocations[{index}].security_id is required")
            weight_value = row["target_weight"] if "target_weight" in row else row.get("weight")
            if weight_value is None:
                raise ValueError(f"portfolio_allocations[{index}].target_weight is required")
            weight = _to_float(weight_value, field_name=f"portfolio_allocations[{index}].target_weight")
            if weight < 0.0:
                raise ValueError("portfolio_allocations.target_weight must be non-negative")
            grouped.setdefault(rebalance_date, []).append(_Allocation(security_id=security_id, target_weight=weight))

    normalized: dict[date, tuple[_Allocation, ...]] = {}
    for rebalance_date, rows in grouped.items():
        total_weight = sum(row.target_weight for row in rows)
        if total_weight > 1.0 + 1e-12:
            raise ValueError(f"portfolio_allocations weights exceed 1.0 on {rebalance_date.isoformat()}")
        merged: dict[str, float] = {}
        for row in rows:
            merged[row.security_id] = merged.get(row.security_id, 0.0) + row.target_weight
        normalized[rebalance_date] = tuple(
            _Allocation(security_id=security_id, target_weight=merged[security_id])
            for security_id in sorted(merged)
        )
    return normalized


def _infer_currency_country(
    security_id: str,
    positions: Mapping[str, _Position],
    price_lookup: Mapping[tuple[str, date], Mapping[str, object]],
    metadata: Mapping[str, tuple[str, str]],
    as_of_date: date,
) -> tuple[str, str]:
    existing = positions.get(security_id)
    if existing is not None:
        return existing.country, existing.currency
    if security_id in metadata:
        return metadata[security_id]
    row = price_lookup.get((security_id, as_of_date))
    if row is None:
        raise ValueError(f"missing price row for security {security_id} on {as_of_date.isoformat()}")
    country = str(row.get("country", "")).strip().upper()
    currency = str(row.get("currency", "")).strip().upper()
    if not country or not currency:
        raise ValueError(f"missing country/currency metadata for {security_id}")
    return country, currency


def _to_krw_price(
    *,
    price_local: float,
    currency: str,
    as_of_date: date,
    fx_rates: Sequence[Mapping[str, object]],
) -> float:
    normalized = currency.upper()
    if normalized == "KRW":
        return float(price_local)
    usd_krw = fx_rate_on_date(fx_rates, pair="USD/KRW", fx_date=as_of_date, missing_date_policy="previous")
    if normalized == "USD":
        return float(price_local) * usd_krw
    if normalized == "JPY":
        usd_jpy = fx_rate_on_date(fx_rates, pair="USD/JPY", fx_date=as_of_date, missing_date_policy="previous")
        return float(price_local) * (usd_krw / usd_jpy)
    raise ValueError(f"unsupported currency for KRW conversion: {currency}")


def _price_krw_for_security(
    *,
    security_id: str,
    as_of_date: date,
    currency: str,
    price_lookup: Mapping[tuple[str, date], Mapping[str, object]],
    fx_rates: Sequence[Mapping[str, object]],
) -> tuple[float, float]:
    row = price_lookup.get((security_id, as_of_date))
    if row is None:
        raise ValueError(f"missing price for security {security_id} on {as_of_date.isoformat()}")
    local_price = _to_float(row["close"], field_name="prices.close")
    krw_price = _to_krw_price(price_local=local_price, currency=currency, as_of_date=as_of_date, fx_rates=fx_rates)
    return local_price, krw_price


def _benchmark_return_krw(
    *,
    start_date: date,
    end_date: date,
    benchmark_lookup: Mapping[date, Mapping[str, object]],
    fx_rates: Sequence[Mapping[str, object]],
) -> float | None:
    start = benchmark_lookup.get(start_date)
    end = benchmark_lookup.get(end_date)
    if start is None or end is None:
        return None

    start_close = _to_float(start["close"], field_name="benchmark.close")
    end_close = _to_float(end["close"], field_name="benchmark.close")
    if start_close == 0.0:
        raise ValueError("benchmark close cannot be zero")

    local_return = (end_close / start_close) - 1.0
    currency = str(start.get("currency", "")).strip().upper()
    if not currency:
        raise ValueError("benchmark currency is required")

    start_usd_krw = fx_rate_on_date(fx_rates, pair="USD/KRW", fx_date=start_date, missing_date_policy="previous")
    end_usd_krw = fx_rate_on_date(fx_rates, pair="USD/KRW", fx_date=end_date, missing_date_policy="previous")

    start_usd_jpy: float | None = None
    end_usd_jpy: float | None = None
    if currency == "JPY":
        start_usd_jpy = fx_rate_on_date(fx_rates, pair="USD/JPY", fx_date=start_date, missing_date_policy="previous")
        end_usd_jpy = fx_rate_on_date(fx_rates, pair="USD/JPY", fx_date=end_date, missing_date_policy="previous")

    return convert_return_to_krw_base(
        local_return=local_return,
        currency=currency,
        start_usd_krw=start_usd_krw,
        end_usd_krw=end_usd_krw,
        start_usd_jpy=start_usd_jpy,
        end_usd_jpy=end_usd_jpy,
    )


def run_quarterly_backtest(
    *,
    rebalance_schedule: Sequence[date | str],
    prices: Sequence[Mapping[str, object]],
    benchmark_series: Sequence[Mapping[str, object]],
    fx_rates: Sequence[Mapping[str, object]],
    portfolio_allocations: Mapping[object, Sequence[Mapping[str, object]]] | Sequence[Mapping[str, object]],
    initial_nav_krw: float = 10_000_000.0,
    cost_tax_config: Mapping[str, object] | None = None,
) -> BacktestResult:
    """Run deterministic quarterly simulation using only provided rebalance dates."""

    schedule = tuple(sorted({_to_date(item, field_name="rebalance_schedule") for item in rebalance_schedule}))
    if len(schedule) < 2:
        raise ValueError("rebalance_schedule must contain at least two dates")
    if initial_nav_krw <= 0.0:
        raise ValueError("initial_nav_krw must be positive")

    config = parse_cost_tax_config(cost_tax_config)
    price_lookup, metadata = _build_price_lookup(prices)
    benchmark_lookup = _build_benchmark_lookup(benchmark_series)
    allocation_lookup = _normalize_allocations(portfolio_allocations)

    trades: list[TradeLedgerEntry] = []
    holdings_history: list[HoldingSnapshot] = []
    period_returns: list[PeriodReturn] = []

    positions: dict[str, _Position] = {}
    cash_krw = float(initial_nav_krw)

    benchmark_growth = 1.0

    for period_index in range(len(schedule) - 1):
        start_date = schedule[period_index]
        end_date = schedule[period_index + 1]

        allocations = allocation_lookup.get(start_date, ())

        current_values: dict[str, float] = {}
        current_prices: dict[str, tuple[float, float]] = {}
        for security_id in sorted(positions):
            position = positions[security_id]
            local_price, krw_price = _price_krw_for_security(
                security_id=security_id,
                as_of_date=start_date,
                currency=position.currency,
                price_lookup=price_lookup,
                fx_rates=fx_rates,
            )
            current_prices[security_id] = (local_price, krw_price)
            current_values[security_id] = position.quantity * krw_price

        start_nav_krw = cash_krw + sum(current_values.values())

        target_weights = {row.security_id: row.target_weight for row in allocations}
        universe = sorted(set(positions) | set(target_weights))

        sell_candidates: list[tuple[str, float]] = []
        buy_candidates: list[tuple[str, float]] = []

        for security_id in universe:
            target_value = start_nav_krw * target_weights.get(security_id, 0.0)
            current_value = current_values.get(security_id, 0.0)
            delta = target_value - current_value
            if abs(delta) <= 1e-9:
                continue
            if delta < 0.0:
                sell_candidates.append((security_id, abs(delta)))
            else:
                buy_candidates.append((security_id, delta))

        period_commission = 0.0
        period_slippage = 0.0
        period_fx_fee = 0.0
        period_sell_tax = 0.0
        period_realized_gains_tax = 0.0

        for security_id, sell_value in sell_candidates:
            if security_id not in positions:
                continue

            position = positions[security_id]
            local_price, krw_price = _price_krw_for_security(
                security_id=security_id,
                as_of_date=start_date,
                currency=position.currency,
                price_lookup=price_lookup,
                fx_rates=fx_rates,
            )

            max_notional = position.quantity * krw_price
            notional_krw = min(sell_value, max_notional)
            quantity = notional_krw / krw_price if krw_price > 0.0 else 0.0
            if quantity <= 0.0:
                continue

            realized_gain = max((krw_price - position.avg_cost_krw) * quantity, 0.0)
            realized_gains_tax = realized_gain * _realized_gains_tax_rate(position.country, config)
            sell_tax = notional_krw * config.kr_sell_tax_rate if position.country == "KR" else 0.0
            commission = notional_krw * config.commission_rate
            slippage = notional_krw * config.slippage_rate
            fx_fee = notional_krw * config.fx_fee_rate if position.currency != "KRW" else 0.0
            total_cost_tax = commission + slippage + fx_fee + sell_tax + realized_gains_tax

            cash_krw += notional_krw - total_cost_tax
            position.quantity -= quantity
            if position.quantity <= 1e-12:
                _ = positions.pop(security_id)
            else:
                positions[security_id] = position

            period_commission += commission
            period_slippage += slippage
            period_fx_fee += fx_fee
            period_sell_tax += sell_tax
            period_realized_gains_tax += realized_gains_tax

            trades.append(
                TradeLedgerEntry(
                    rebalance_date=start_date,
                    security_id=security_id,
                    side="SELL",
                    country=position.country,
                    currency=position.currency,
                    quantity=quantity,
                    price_local=local_price,
                    price_krw=krw_price,
                    notional_krw=notional_krw,
                    gross_proceeds_krw=notional_krw,
                    gross_cost_krw=0.0,
                    commission_krw=commission,
                    slippage_krw=slippage,
                    fx_fee_krw=fx_fee,
                    sell_tax_krw=sell_tax,
                    realized_gain_krw=realized_gain,
                    realized_gains_tax_krw=realized_gains_tax,
                    total_cost_tax_krw=total_cost_tax,
                )
            )

        for security_id, buy_value in buy_candidates:
            country, currency = _infer_currency_country(
                security_id,
                positions,
                price_lookup,
                metadata,
                start_date,
            )

            local_price, krw_price = _price_krw_for_security(
                security_id=security_id,
                as_of_date=start_date,
                currency=currency,
                price_lookup=price_lookup,
                fx_rates=fx_rates,
            )
            if krw_price <= 0.0:
                raise ValueError(f"price must be positive for {security_id} on {start_date.isoformat()}")

            notional_krw = buy_value
            commission = notional_krw * config.commission_rate
            slippage = notional_krw * config.slippage_rate
            fx_fee = notional_krw * config.fx_fee_rate if currency != "KRW" else 0.0
            total_cost_tax = commission + slippage + fx_fee
            quantity = notional_krw / krw_price

            cash_krw -= notional_krw + total_cost_tax

            existing = positions.get(security_id)
            if existing is None:
                positions[security_id] = _Position(
                    security_id=security_id,
                    country=country,
                    currency=currency,
                    quantity=quantity,
                    avg_cost_krw=krw_price,
                )
            else:
                new_quantity = existing.quantity + quantity
                if new_quantity <= 0.0:
                    raise ValueError("resulting position quantity must be positive")
                weighted_avg_cost = ((existing.quantity * existing.avg_cost_krw) + notional_krw) / new_quantity
                positions[security_id] = _Position(
                    security_id=security_id,
                    country=existing.country,
                    currency=existing.currency,
                    quantity=new_quantity,
                    avg_cost_krw=weighted_avg_cost,
                )

            period_commission += commission
            period_slippage += slippage
            period_fx_fee += fx_fee

            trades.append(
                TradeLedgerEntry(
                    rebalance_date=start_date,
                    security_id=security_id,
                    side="BUY",
                    country=country,
                    currency=currency,
                    quantity=quantity,
                    price_local=local_price,
                    price_krw=krw_price,
                    notional_krw=notional_krw,
                    gross_proceeds_krw=0.0,
                    gross_cost_krw=notional_krw,
                    commission_krw=commission,
                    slippage_krw=slippage,
                    fx_fee_krw=fx_fee,
                    sell_tax_krw=0.0,
                    realized_gain_krw=0.0,
                    realized_gains_tax_krw=0.0,
                    total_cost_tax_krw=total_cost_tax,
                )
            )

        end_nav_krw = cash_krw
        end_values: list[tuple[str, _Position, float, float, float]] = []
        for security_id in sorted(positions):
            position = positions[security_id]
            local_price, krw_price = _price_krw_for_security(
                security_id=security_id,
                as_of_date=end_date,
                currency=position.currency,
                price_lookup=price_lookup,
                fx_rates=fx_rates,
            )
            market_value = position.quantity * krw_price
            end_nav_krw += market_value
            end_values.append((security_id, position, local_price, krw_price, market_value))

        for security_id, position, local_price, krw_price, market_value in end_values:
            weight = 0.0 if end_nav_krw == 0.0 else market_value / end_nav_krw
            holdings_history.append(
                HoldingSnapshot(
                    as_of_date=end_date,
                    security_id=security_id,
                    country=position.country,
                    currency=position.currency,
                    quantity=position.quantity,
                    avg_cost_krw=position.avg_cost_krw,
                    price_local=local_price,
                    price_krw=krw_price,
                    market_value_krw=market_value,
                    weight=weight,
                )
            )

        total_cost_tax = (
            period_commission
            + period_slippage
            + period_fx_fee
            + period_sell_tax
            + period_realized_gains_tax
        )
        gross_end_nav = end_nav_krw + total_cost_tax
        net_return = (end_nav_krw / start_nav_krw) - 1.0
        gross_return = (gross_end_nav / start_nav_krw) - 1.0

        benchmark_return = _benchmark_return_krw(
            start_date=start_date,
            end_date=end_date,
            benchmark_lookup=benchmark_lookup,
            fx_rates=fx_rates,
        )
        if benchmark_return is not None:
            benchmark_growth *= 1.0 + benchmark_return

        period_returns.append(
            PeriodReturn(
                start_date=start_date,
                end_date=end_date,
                start_nav_krw=start_nav_krw,
                end_nav_krw=end_nav_krw,
                gross_return=gross_return,
                net_return=net_return,
                benchmark_krw_return=benchmark_return,
                total_commission_krw=period_commission,
                total_slippage_krw=period_slippage,
                total_fx_fee_krw=period_fx_fee,
                total_sell_tax_krw=period_sell_tax,
                total_realized_gains_tax_krw=period_realized_gains_tax,
                total_cost_tax_krw=total_cost_tax,
            )
        )

    total_commission = sum(row.total_commission_krw for row in period_returns)
    total_slippage = sum(row.total_slippage_krw for row in period_returns)
    total_fx_fee = sum(row.total_fx_fee_krw for row in period_returns)
    total_sell_tax = sum(row.total_sell_tax_krw for row in period_returns)
    total_realized_gains_tax = sum(row.total_realized_gains_tax_krw for row in period_returns)
    total_cost_tax = sum(row.total_cost_tax_krw for row in period_returns)

    end_nav = period_returns[-1].end_nav_krw if period_returns else initial_nav_krw
    cumulative_net_return = (end_nav / initial_nav_krw) - 1.0
    cumulative_gross_return = ((end_nav + total_cost_tax) / initial_nav_krw) - 1.0

    benchmark_cumulative_return: float | None = None
    if all(row.benchmark_krw_return is not None for row in period_returns):
        benchmark_cumulative_return = benchmark_growth - 1.0

    metrics = BacktestMetrics(
        periods=len(period_returns),
        start_nav_krw=initial_nav_krw,
        end_nav_krw=end_nav,
        cumulative_net_return=cumulative_net_return,
        cumulative_gross_return=cumulative_gross_return,
        benchmark_cumulative_return=benchmark_cumulative_return,
        total_commission_krw=total_commission,
        total_slippage_krw=total_slippage,
        total_fx_fee_krw=total_fx_fee,
        total_sell_tax_krw=total_sell_tax,
        total_realized_gains_tax_krw=total_realized_gains_tax,
        total_cost_tax_krw=total_cost_tax,
    )

    return BacktestResult(
        trades=tuple(trades),
        holdings=tuple(holdings_history),
        returns=tuple(period_returns),
        metrics=metrics,
    )


__all__ = [
    "BacktestMetrics",
    "BacktestResult",
    "CostTaxConfig",
    "DEFAULT_COST_TAX_CONFIG",
    "HoldingSnapshot",
    "PeriodReturn",
    "TradeLedgerEntry",
    "parse_cost_tax_config",
    "run_quarterly_backtest",
]
