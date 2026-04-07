"""Parity harness skeleton for Python baseline vs Rust kernels."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date
import json
import math
from pathlib import Path
from typing import Any, cast

from src.backtest.engine import run_quarterly_backtest
from src.cli.run_pipeline import _build_qepm_ranked_rows
from src.portfolio.constructor import construct_portfolio_with_constraints
from src.rust_bridge.contracts import (
    normalize_backtest_request,
    normalize_constructor_request,
    normalize_ranking_request,
)
from src.rust_bridge.dispatch import run_backtest_kernel, run_constructor_kernel, run_ranking_kernel
from src.rust_bridge.feature_flags import RustKernelFlags


@dataclass(frozen=True)
class _FactorRow:
    security_id: str
    country: str
    factor_value: float | None
    rd_expense: float | None
    sales_ttm: float | None
    is_eligible: bool


@dataclass(frozen=True)
class _AcceptedRow:
    security_id: str
    median_traded_value_krw: float


def _serialize(value: object) -> object:
    if isinstance(value, date):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return _serialize(asdict(value))
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    return value


def _float_matches(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-6)


def _as_float(value: object) -> float:
    return float(str(value))


def _values_match(left: object, right: object) -> bool:
    if isinstance(left, bool) and isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return _float_matches(float(left), float(right))
    if isinstance(left, str) and isinstance(right, str):
        return left == right
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        return all(_values_match(lhs, rhs) for lhs, rhs in zip(left, right, strict=True))
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left.keys()) != set(right.keys()):
            return False
        return all(_values_match(left[key], right[key]) for key in sorted(left.keys()))
    return left == right


def _backtest_parity_sections(baseline: object, rust_output: object) -> dict[str, dict[str, bool | int]]:
    baseline_payload = cast(dict[str, object], _serialize(baseline))
    rust_payload = cast(dict[str, object], _serialize(rust_output))

    baseline_periods = cast(list[dict[str, object]], baseline_payload.get("returns", []))
    rust_periods = cast(list[dict[str, object]], rust_payload.get("returns", []))

    baseline_nav_path: list[float] = []
    rust_nav_path: list[float] = []
    if baseline_periods:
        baseline_nav_path.append(_as_float(baseline_periods[0]["start_nav_krw"]))
        baseline_nav_path.extend(_as_float(row["end_nav_krw"]) for row in baseline_periods)
    if rust_periods:
        rust_nav_path.append(_as_float(rust_periods[0]["start_nav_krw"]))
        rust_nav_path.extend(_as_float(row["end_nav_krw"]) for row in rust_periods)

    baseline_trades = cast(list[dict[str, object]], baseline_payload.get("trades", []))
    rust_trades = cast(list[dict[str, object]], rust_payload.get("trades", []))

    baseline_holdings = cast(list[dict[str, object]], baseline_payload.get("holdings", []))
    rust_holdings = cast(list[dict[str, object]], rust_payload.get("holdings", []))

    sections = {
        "nav_path": {
            "match": _values_match(baseline_nav_path, rust_nav_path),
            "expected_count": len(baseline_nav_path),
            "actual_count": len(rust_nav_path),
        },
        "trades": {
            "match": _values_match(baseline_trades, rust_trades),
            "expected_count": len(baseline_trades),
            "actual_count": len(rust_trades),
        },
        "holdings": {
            "match": _values_match(baseline_holdings, rust_holdings),
            "expected_count": len(baseline_holdings),
            "actual_count": len(rust_holdings),
        },
        "period_rows": {
            "match": _values_match(baseline_periods, rust_periods),
            "expected_count": len(baseline_periods),
            "actual_count": len(rust_periods),
        },
    }
    return sections


def _ranking_fixture() -> tuple[list[_FactorRow], list[_AcceptedRow], dict[str, str], dict[str, int]]:
    factor_rows = [
        _FactorRow("JP:001", "JP", 0.31, 10.0, 100.0, True),
        _FactorRow("JP:002", "JP", 0.27, 9.0, 120.0, True),
        _FactorRow("KR:001", "KR", 0.22, 8.0, 95.0, True),
        _FactorRow("US:001", "US", 0.29, 11.0, 130.0, True),
    ]
    accepted_rows = [
        _AcceptedRow("JP:001", 1_200_000_000.0),
        _AcceptedRow("JP:002", 1_100_000_000.0),
        _AcceptedRow("KR:001", 1_000_000_000.0),
        _AcceptedRow("US:001", 1_300_000_000.0),
    ]
    sectors = {
        "JP:001": "TECH",
        "JP:002": "INDUSTRIALS",
        "KR:001": "TECH",
        "US:001": "HEALTHCARE",
    }
    requested = {"JP": 1, "KR": 1, "US": 1}
    return factor_rows, accepted_rows, sectors, requested


def _ranking_selected_ids(rows: list[dict[str, object]], requested_counts: dict[str, int]) -> list[str]:
    selected: list[str] = []
    for row in rows:
        country = str(row.get("country", ""))
        target = int(requested_counts.get(country, 0))
        rank_value = row.get("rank_in_country")
        if target <= 0 or rank_value in (None, ""):
            continue
        if int(str(rank_value)) <= target:
            selected.append(str(row.get("security_id", "")))
    return sorted(selected)


def _ranking_parity_sections(
    baseline: object,
    rust_output: object,
    requested_counts: dict[str, int],
) -> dict[str, dict[str, object]]:
    baseline_rows = cast(list[dict[str, object]], _serialize(baseline))
    rust_rows = cast(list[dict[str, object]], _serialize(rust_output))

    baseline_rank_rows = [
        {
            "security_id": row.get("security_id"),
            "country": row.get("country"),
            "rank_in_country": row.get("rank_in_country"),
        }
        for row in baseline_rows
    ]
    rust_rank_rows = [
        {
            "security_id": row.get("security_id"),
            "country": row.get("country"),
            "rank_in_country": row.get("rank_in_country"),
        }
        for row in rust_rows
    ]

    baseline_eligible_rows = [
        {
            "security_id": row.get("security_id"),
            "is_eligible": row.get("is_eligible"),
        }
        for row in baseline_rows
    ]
    rust_eligible_rows = [
        {
            "security_id": row.get("security_id"),
            "is_eligible": row.get("is_eligible"),
        }
        for row in rust_rows
    ]

    baseline_selected_ids = _ranking_selected_ids(baseline_rows, requested_counts)
    rust_selected_ids = _ranking_selected_ids(rust_rows, requested_counts)

    return {
        "rank_in_country": {
            "match": _values_match(baseline_rank_rows, rust_rank_rows),
            "expected_count": len(baseline_rank_rows),
            "actual_count": len(rust_rank_rows),
        },
        "is_eligible": {
            "match": _values_match(baseline_eligible_rows, rust_eligible_rows),
            "expected_count": len(baseline_eligible_rows),
            "actual_count": len(rust_eligible_rows),
        },
        "selected_ids": {
            "match": _values_match(baseline_selected_ids, rust_selected_ids),
            "expected_count": len(baseline_selected_ids),
            "actual_count": len(rust_selected_ids),
        },
    }


def _constructor_fixture() -> list[dict[str, object]]:
    return [
        {"security_id": "JP:001", "country": "JP", "factor_value": 1.2, "rank_in_country": 1, "is_eligible": True},
        {"security_id": "KR:001", "country": "KR", "factor_value": 1.1, "rank_in_country": 1, "is_eligible": True},
        {"security_id": "US:001", "country": "US", "factor_value": 1.0, "rank_in_country": 1, "is_eligible": True},
    ]


def _constructor_infeasible_fixture() -> list[dict[str, object]]:
    return [
        {"security_id": "JP:AAA", "country": "JP", "factor_value": 0.9, "rank_in_country": 1, "is_eligible": True},
        {"security_id": "JP:AAB", "country": "JP", "factor_value": 0.9, "rank_in_country": 1, "is_eligible": True},
        {"security_id": "JP:BAA", "country": "JP", "factor_value": 0.8, "rank_in_country": 2, "is_eligible": True},
        {"security_id": "JP:C00", "country": "JP", "factor_value": 0.7, "rank_in_country": 3, "is_eligible": True},
        {"security_id": "JP:SKIP", "country": "JP", "factor_value": 0.6, "rank_in_country": 4, "is_eligible": False},
    ]


def _constructor_parity_sections(baseline: object, rust_output: object) -> dict[str, dict[str, object]]:
    baseline_payload = cast(dict[str, object], _serialize(baseline))
    rust_payload = cast(dict[str, object], _serialize(rust_output))

    baseline_holdings = cast(list[dict[str, object]], baseline_payload.get("holdings", []))
    rust_holdings = cast(list[dict[str, object]], rust_payload.get("holdings", []))

    baseline_holdings_metadata = [
        {
            "security_id": row.get("security_id"),
            "country": row.get("country"),
            "rank_in_country": row.get("rank_in_country"),
            "factor_value": row.get("factor_value"),
        }
        for row in baseline_holdings
    ]
    rust_holdings_metadata = [
        {
            "security_id": row.get("security_id"),
            "country": row.get("country"),
            "rank_in_country": row.get("rank_in_country"),
            "factor_value": row.get("factor_value"),
        }
        for row in rust_holdings
    ]

    baseline_reasons = cast(list[str], baseline_payload.get("fallback_reasons", []))
    rust_reasons = cast(list[str], rust_payload.get("fallback_reasons", []))

    sections: dict[str, dict[str, object]] = {
        "holdings_metadata": {
            "match": _values_match(baseline_holdings_metadata, rust_holdings_metadata),
            "expected_count": len(baseline_holdings_metadata),
            "actual_count": len(rust_holdings_metadata),
        },
        "fallback_triggered": {
            "match": _values_match(
                baseline_payload.get("fallback_triggered"),
                rust_payload.get("fallback_triggered"),
            ),
            "expected": baseline_payload.get("fallback_triggered"),
            "actual": rust_payload.get("fallback_triggered"),
        },
        "fallback_reasons": {
            "match": _values_match(baseline_reasons, rust_reasons),
            "expected_count": len(baseline_reasons),
            "actual_count": len(rust_reasons),
        },
    }
    return sections


def _run_constructor_case(rows: list[dict[str, object]]) -> dict[str, object]:
    request = normalize_constructor_request(
        ranked_factor_rows=rows,
        country_targets=None,
        risk_controls={
            "te_active_l2_cap": 0.08,
            "alpha_tilt_strength": 0.35,
            "max_adv_participation": 0.10,
            "portfolio_value_krw": 10_000_000.0,
            "max_turnover": 0.35,
            "previous_weights": {},
        },
    )
    python_impl = lambda: construct_portfolio_with_constraints(rows)
    baseline = python_impl()
    rust_output, backend, reason = run_constructor_kernel(
        request=request,
        flags=RustKernelFlags(constructor=True),
        python_impl=python_impl,
    )
    if backend != "rust":
        return {"status": "rust_unavailable", "reason": reason}

    parity = _constructor_parity_sections(baseline, rust_output)
    status = "pass" if all(bool(section["match"]) for section in parity.values()) else "mismatch"
    return {
        "status": status,
        "reason": None,
        "parity": parity,
    }


def _backtest_fixture() -> dict[str, object]:
    schedule = [date(2025, 3, 31), date(2025, 6, 30), date(2025, 9, 30)]
    prices = [
        {"price_date": date(2025, 3, 31), "security_id": "KR:001", "country": "KR", "currency": "KRW", "close": 1000.0},
        {"price_date": date(2025, 6, 30), "security_id": "KR:001", "country": "KR", "currency": "KRW", "close": 1100.0},
        {"price_date": date(2025, 9, 30), "security_id": "KR:001", "country": "KR", "currency": "KRW", "close": 1200.0},
    ]
    benchmark = [
        {"benchmark_date": date(2025, 3, 31), "close": 100.0, "currency": "USD"},
        {"benchmark_date": date(2025, 6, 30), "close": 102.0, "currency": "USD"},
        {"benchmark_date": date(2025, 9, 30), "close": 105.0, "currency": "USD"},
    ]
    fx = [
        {"pair": "USD/KRW", "fx_date": date(2025, 3, 31), "rate": 1300.0},
        {"pair": "USD/KRW", "fx_date": date(2025, 6, 30), "rate": 1310.0},
        {"pair": "USD/KRW", "fx_date": date(2025, 9, 30), "rate": 1320.0},
        {"pair": "USD/JPY", "fx_date": date(2025, 3, 31), "rate": 150.0},
        {"pair": "USD/JPY", "fx_date": date(2025, 6, 30), "rate": 149.0},
        {"pair": "USD/JPY", "fx_date": date(2025, 9, 30), "rate": 148.0},
    ]
    allocations = {
        date(2025, 3, 31): [{"security_id": "KR:001", "target_weight": 1.0}],
        date(2025, 6, 30): [{"security_id": "KR:001", "target_weight": 1.0}],
    }
    return {
        "schedule": schedule,
        "prices": prices,
        "benchmark": benchmark,
        "fx": fx,
        "allocations": allocations,
    }


def _run_ranking() -> dict[str, object]:
    factor_rows, accepted_rows, sectors, requested = _ranking_fixture()
    request = normalize_ranking_request(
        factor_rows=factor_rows,
        accepted_rows=accepted_rows,
        sector_by_security=sectors,
        requested_counts_by_country=requested,
        factor_model="rnd_sales_ttm",
        sector_active_band=0.10,
        use_size_stratification=True,
    )
    python_impl = lambda: _build_qepm_ranked_rows(
        factor_rows=tuple(factor_rows),
        accepted_rows=cast(list[object], accepted_rows),
        sector_by_security=sectors,
        requested_counts_by_country=requested,
        factor_model="rnd_sales_ttm",
        sector_active_band=0.10,
        use_size_stratification=True,
    )
    baseline = python_impl()
    rust_output, backend, reason = run_ranking_kernel(
        request=request,
        flags=RustKernelFlags(ranking=True),
        python_impl=python_impl,
    )
    if backend != "rust":
        return {"status": "rust_unavailable", "reason": reason}

    parity = _ranking_parity_sections(baseline, rust_output, requested)
    status = "pass" if all(bool(section["match"]) for section in parity.values()) else "mismatch"
    return {
        "status": status,
        "reason": None,
        "parity": parity,
    }


def _run_constructor() -> dict[str, object]:
    feasible = _run_constructor_case(_constructor_fixture())
    if feasible["status"] == "rust_unavailable":
        return feasible

    infeasible = _run_constructor_case(_constructor_infeasible_fixture())
    if infeasible["status"] == "rust_unavailable":
        return infeasible

    cases = {
        "feasible": feasible,
        "infeasible": infeasible,
    }
    status = "pass" if all(cast(str, case["status"]) == "pass" for case in cases.values()) else "mismatch"
    return {
        "status": status,
        "reason": None,
        "cases": cases,
    }


def _run_backtest() -> dict[str, object]:
    fixture = _backtest_fixture()
    schedule = cast(list[date], fixture["schedule"])
    prices = cast(list[dict[str, object]], fixture["prices"])
    benchmark = cast(list[dict[str, object]], fixture["benchmark"])
    fx = cast(list[dict[str, object]], fixture["fx"])
    allocations = cast(dict[object, list[dict[str, object]]], fixture["allocations"])
    request = normalize_backtest_request(
        rebalance_schedule=schedule,
        prices=prices,
        benchmark_series=benchmark,
        fx_rates=fx,
        portfolio_allocations=allocations,
        initial_nav_krw=1_000_000.0,
    )
    python_impl = lambda: run_quarterly_backtest(
        rebalance_schedule=schedule,
        prices=prices,
        benchmark_series=benchmark,
        fx_rates=fx,
        portfolio_allocations=allocations,
        initial_nav_krw=1_000_000.0,
    )
    baseline = python_impl()
    rust_output, backend, reason = run_backtest_kernel(
        request=request,
        flags=RustKernelFlags(backtest=True),
        python_impl=python_impl,
    )
    if backend != "rust":
        return {"status": "rust_unavailable", "reason": reason}

    parity = _backtest_parity_sections(baseline, rust_output)
    status = "pass" if all(bool(section["match"]) for section in parity.values()) else "mismatch"
    return {
        "status": status,
        "reason": None,
        "parity": parity,
    }


def _run(kernel: str) -> dict[str, object]:
    if kernel == "ranking":
        return {"ranking": _run_ranking()}
    if kernel == "constructor":
        return {"constructor": _run_constructor()}
    if kernel == "backtest":
        return {"backtest": _run_backtest()}
    return {
        "ranking": _run_ranking(),
        "constructor": _run_constructor(),
        "backtest": _run_backtest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Python-vs-Rust parity harness skeleton")
    _ = parser.add_argument("--kernel", choices=("all", "ranking", "constructor", "backtest"), default="all")
    _ = parser.add_argument("--fixture", default="", help="Optional fixture path. Missing file reports baseline_not_found.")
    _ = parser.add_argument("--output", default="", help="Optional output json path")
    args = parser.parse_args(argv)

    fixture_path = Path(cast(str, args.fixture)).expanduser() if cast(str, args.fixture).strip() else None
    if fixture_path is not None and not fixture_path.exists():
        payload = {
            "status": "baseline_not_found",
            "fixture": str(fixture_path),
            "results": {},
        }
    else:
        payload = {
            "status": "ok",
            "fixture": str(fixture_path) if fixture_path is not None else None,
            "results": _run(cast(str, args.kernel)),
        }

    output_path = cast(str, args.output).strip()
    if output_path:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        _ = target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
