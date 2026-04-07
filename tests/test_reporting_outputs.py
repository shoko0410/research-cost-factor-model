from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Protocol, cast

import pytest


class _PeriodRow(Protocol):
    start_date: date
    end_date: date


class _BacktestResult(Protocol):
    returns: tuple[_PeriodRow, ...]


class _BacktestModule(Protocol):
    def run_quarterly_backtest(
        self,
        *,
        rebalance_schedule: Sequence[date | str],
        prices: Sequence[Mapping[str, object]],
        benchmark_series: Sequence[Mapping[str, object]],
        fx_rates: Sequence[Mapping[str, object]],
        portfolio_allocations: Mapping[object, Sequence[Mapping[str, object]]] | Sequence[Mapping[str, object]],
        initial_nav_krw: float = 1_000_000_000.0,
        cost_tax_config: Mapping[str, object] | None = None,
    ) -> _BacktestResult: ...


class _ReportModule(Protocol):
    def build_backtest_attribution_report(
        self,
        *,
        backtest_result: _BacktestResult,
        fx_rates: Sequence[Mapping[str, object]],
        consistency_tolerance: float = 1e-10,
    ) -> Mapping[str, object]: ...


_engine = cast(_BacktestModule, cast(object, importlib.import_module("src.backtest.engine")))
run_quarterly_backtest = _engine.run_quarterly_backtest

_reporting = cast(_ReportModule, cast(object, importlib.import_module("src.reporting.attribution")))
build_backtest_attribution_report = _reporting.build_backtest_attribution_report


def _as_float(value: object) -> float:
    return float(str(value))


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


def test_reporting_output_contains_required_tables_and_active_fields() -> None:
    schedule, prices, benchmark_series, fx_rates, allocations = _sample_inputs()
    result = run_quarterly_backtest(
        rebalance_schedule=schedule,
        prices=prices,
        benchmark_series=benchmark_series,
        fx_rates=fx_rates,
        portfolio_allocations=allocations,
        initial_nav_krw=1_000_000.0,
    )

    report = build_backtest_attribution_report(backtest_result=result, fx_rates=fx_rates)

    assert "krw_main_performance_table" in report
    assert "usd_supplemental_table" in report
    assert "fx_contribution_decomposition" in report

    krw_table = cast(Mapping[str, tuple[Mapping[str, object], ...]], report["krw_main_performance_table"])
    usd_table = cast(Mapping[str, tuple[Mapping[str, object], ...]], report["usd_supplemental_table"])
    fx_table = cast(Mapping[str, tuple[Mapping[str, object], ...]], report["fx_contribution_decomposition"])
    krw_rows = krw_table["rows"]
    usd_rows = usd_table["rows"]
    fx_rows = fx_table["rows"]

    assert len(krw_rows) == len(result.returns)
    assert len(usd_rows) == len(result.returns)
    assert len(fx_rows) == len(result.returns)

    assert "active_net_return_krw" in krw_rows[0]
    assert "active_gross_return_krw" in krw_rows[0]
    assert "active_net_return_usd" in usd_rows[0]
    assert "active_gross_return_usd" in usd_rows[0]


def test_reporting_blocks_when_benchmark_dependency_is_missing() -> None:
    schedule, prices, benchmark_series, fx_rates, allocations = _sample_inputs()
    benchmark_missing_end = benchmark_series[:-1]
    result = run_quarterly_backtest(
        rebalance_schedule=schedule,
        prices=prices,
        benchmark_series=benchmark_missing_end,
        fx_rates=fx_rates,
        portfolio_allocations=allocations,
        initial_nav_krw=1_000_000.0,
    )

    with pytest.raises(ValueError, match="benchmark dependency missing"):
        _ = build_backtest_attribution_report(backtest_result=result, fx_rates=fx_rates)


def test_fx_contribution_decomposition_is_arithmetically_consistent() -> None:
    schedule, prices, benchmark_series, fx_rates, allocations = _sample_inputs()
    result = run_quarterly_backtest(
        rebalance_schedule=schedule,
        prices=prices,
        benchmark_series=benchmark_series,
        fx_rates=fx_rates,
        portfolio_allocations=allocations,
        initial_nav_krw=1_000_000.0,
    )

    report = build_backtest_attribution_report(backtest_result=result, fx_rates=fx_rates)
    fx_table = cast(Mapping[str, tuple[Mapping[str, object], ...]], report["fx_contribution_decomposition"])
    fx_rows = fx_table["rows"]

    for row in fx_rows:
        net_lhs = _as_float(row["portfolio_net_return_krw"])
        net_rhs = _as_float(row["portfolio_net_return_usd"]) + _as_float(row["portfolio_net_fx_contribution"])
        gross_lhs = _as_float(row["portfolio_gross_return_krw"])
        gross_rhs = _as_float(row["portfolio_gross_return_usd"]) + _as_float(row["portfolio_gross_fx_contribution"])
        benchmark_lhs = _as_float(row["benchmark_return_krw"])
        benchmark_rhs = _as_float(row["benchmark_return_usd"]) + _as_float(row["benchmark_fx_contribution"])

        assert abs(net_lhs - net_rhs) <= 1e-12
        assert abs(gross_lhs - gross_rhs) <= 1e-12
        assert abs(benchmark_lhs - benchmark_rhs) <= 1e-12
        assert abs(_as_float(row["portfolio_net_consistency_gap"])) <= 1e-12
        assert abs(_as_float(row["portfolio_gross_consistency_gap"])) <= 1e-12
        assert abs(_as_float(row["benchmark_consistency_gap"])) <= 1e-12
