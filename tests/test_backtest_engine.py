from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Protocol, cast

import pytest


class _TradeRow(Protocol):
    commission_krw: float
    slippage_krw: float
    fx_fee_krw: float
    sell_tax_krw: float
    realized_gains_tax_krw: float
    total_cost_tax_krw: float


class _PeriodRow(Protocol):
    total_commission_krw: float
    total_slippage_krw: float
    total_fx_fee_krw: float
    total_sell_tax_krw: float
    total_realized_gains_tax_krw: float
    total_cost_tax_krw: float


class _MetricsRow(Protocol):
    periods: int
    total_slippage_krw: float
    total_sell_tax_krw: float
    total_realized_gains_tax_krw: float
    benchmark_cumulative_return: float | None


class _BacktestResult(Protocol):
    trades: tuple[_TradeRow, ...]
    returns: tuple[_PeriodRow, ...]
    metrics: _MetricsRow


class _BacktestModule(Protocol):
    def run_quarterly_backtest(
        self,
        *,
        rebalance_schedule: Sequence[date | str],
        prices: Sequence[Mapping[str, object]],
        benchmark_series: Sequence[Mapping[str, object]],
        fx_rates: Sequence[Mapping[str, object]],
        portfolio_allocations: Mapping[object, Sequence[Mapping[str, object]]] | Sequence[Mapping[str, object]],
        initial_nav_krw: float = 10_000_000.0,
        cost_tax_config: Mapping[str, object] | None = None,
    ) -> _BacktestResult: ...


_engine = cast(_BacktestModule, cast(object, importlib.import_module("src.backtest.engine")))
run_quarterly_backtest = _engine.run_quarterly_backtest


def _sample_inputs() -> tuple[
    Sequence[date],
    Sequence[Mapping[str, object]],
    Sequence[Mapping[str, object]],
    Sequence[Mapping[str, object]],
    Mapping[object, Sequence[Mapping[str, object]]],
]:
    schedule = [
        date(2025, 3, 31),
        date(2025, 6, 30),
        date(2025, 9, 30),
        date(2025, 12, 31),
    ]

    prices = [
        {"price_date": date(2025, 3, 31), "security_id": "KR_A", "country": "KR", "currency": "KRW", "close": 1000.0},
        {"price_date": date(2025, 6, 30), "security_id": "KR_A", "country": "KR", "currency": "KRW", "close": 1200.0},
        {"price_date": date(2025, 9, 30), "security_id": "KR_A", "country": "KR", "currency": "KRW", "close": 1300.0},
        {"price_date": date(2025, 12, 31), "security_id": "KR_A", "country": "KR", "currency": "KRW", "close": 1250.0},
        {"price_date": date(2025, 3, 31), "security_id": "US_A", "country": "US", "currency": "USD", "close": 10.0},
        {"price_date": date(2025, 6, 30), "security_id": "US_A", "country": "US", "currency": "USD", "close": 12.0},
        {"price_date": date(2025, 9, 30), "security_id": "US_A", "country": "US", "currency": "USD", "close": 13.0},
        {"price_date": date(2025, 12, 31), "security_id": "US_A", "country": "US", "currency": "USD", "close": 14.0},
    ]

    benchmark_series = [
        {"benchmark_date": date(2025, 3, 31), "close": 100.0, "currency": "USD"},
        {"benchmark_date": date(2025, 6, 30), "close": 105.0, "currency": "USD"},
        {"benchmark_date": date(2025, 9, 30), "close": 110.0, "currency": "USD"},
        {"benchmark_date": date(2025, 12, 31), "close": 108.0, "currency": "USD"},
    ]

    fx_rates = [
        {"pair": "USD/KRW", "fx_date": date(2025, 3, 31), "rate": 1300.0},
        {"pair": "USD/KRW", "fx_date": date(2025, 6, 30), "rate": 1310.0},
        {"pair": "USD/KRW", "fx_date": date(2025, 9, 30), "rate": 1320.0},
        {"pair": "USD/KRW", "fx_date": date(2025, 12, 31), "rate": 1330.0},
        {"pair": "USD/JPY", "fx_date": date(2025, 3, 31), "rate": 150.0},
        {"pair": "USD/JPY", "fx_date": date(2025, 6, 30), "rate": 149.0},
        {"pair": "USD/JPY", "fx_date": date(2025, 9, 30), "rate": 148.0},
        {"pair": "USD/JPY", "fx_date": date(2025, 12, 31), "rate": 147.0},
    ]

    allocations: dict[object, list[Mapping[str, object]]] = {
        date(2025, 3, 31): [
            {"security_id": "KR_A", "target_weight": 0.50},
            {"security_id": "US_A", "target_weight": 0.50},
        ],
        date(2025, 6, 30): [{"security_id": "US_A", "target_weight": 1.00}],
        date(2025, 9, 30): [{"security_id": "KR_A", "target_weight": 1.00}],
    }

    return schedule, prices, benchmark_series, fx_rates, allocations


def test_run_quarterly_backtest_happy_path_is_deterministic() -> None:
    schedule, prices, benchmark_series, fx_rates, allocations = _sample_inputs()

    first = run_quarterly_backtest(
        rebalance_schedule=schedule,
        prices=prices,
        benchmark_series=benchmark_series,
        fx_rates=fx_rates,
        portfolio_allocations=allocations,
        initial_nav_krw=1_000_000.0,
    )
    second = run_quarterly_backtest(
        rebalance_schedule=schedule,
        prices=prices,
        benchmark_series=benchmark_series,
        fx_rates=fx_rates,
        portfolio_allocations=allocations,
        initial_nav_krw=1_000_000.0,
    )

    assert first == second
    assert len(first.returns) == 3
    assert len(first.trades) >= 4
    assert first.metrics.periods == 3
    assert first.metrics.total_slippage_krw > 0.0
    assert first.metrics.total_sell_tax_krw > 0.0
    assert first.metrics.total_realized_gains_tax_krw > 0.0
    assert first.metrics.benchmark_cumulative_return is not None


def test_run_quarterly_backtest_rejects_malformed_tax_config() -> None:
    schedule, prices, benchmark_series, fx_rates, allocations = _sample_inputs()

    with pytest.raises(ValueError, match="cost_tax_config.jp_realized_gains_tax_rate"):
        _ = run_quarterly_backtest(
            rebalance_schedule=schedule,
            prices=prices,
            benchmark_series=benchmark_series,
            fx_rates=fx_rates,
            portfolio_allocations=allocations,
            cost_tax_config={"jp_realized_gains_tax_rate": "not-a-number"},
        )


def test_trade_and_period_artifacts_include_cost_tax_attribution_fields() -> None:
    schedule, prices, benchmark_series, fx_rates, allocations = _sample_inputs()
    result = run_quarterly_backtest(
        rebalance_schedule=schedule,
        prices=prices,
        benchmark_series=benchmark_series,
        fx_rates=fx_rates,
        portfolio_allocations=allocations,
        initial_nav_krw=500_000.0,
    )

    trade = result.trades[0]
    assert hasattr(trade, "commission_krw")
    assert hasattr(trade, "slippage_krw")
    assert hasattr(trade, "fx_fee_krw")
    assert hasattr(trade, "sell_tax_krw")
    assert hasattr(trade, "realized_gains_tax_krw")
    assert hasattr(trade, "total_cost_tax_krw")

    period = result.returns[0]
    assert hasattr(period, "total_commission_krw")
    assert hasattr(period, "total_slippage_krw")
    assert hasattr(period, "total_fx_fee_krw")
    assert hasattr(period, "total_sell_tax_krw")
    assert hasattr(period, "total_realized_gains_tax_krw")
    assert hasattr(period, "total_cost_tax_krw")
    assert period.total_cost_tax_krw >= (period.total_slippage_krw + period.total_sell_tax_krw)
