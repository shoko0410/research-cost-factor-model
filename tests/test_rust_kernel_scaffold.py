from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date
import importlib
import json
from pathlib import Path
from typing import cast

import pytest

from src.portfolio.constructor import construct_portfolio_with_constraints
from src.backtest.engine import run_quarterly_backtest
from src.cli.run_pipeline import _build_qepm_ranked_rows
from src.rust_bridge import dispatch as rust_dispatch
from src.rust_bridge.contracts import (
    normalize_backtest_request,
    normalize_constructor_request,
    normalize_ranking_request,
)
from src.rust_bridge.dispatch import run_backtest_kernel, run_constructor_kernel, run_ranking_kernel
from src.rust_bridge.feature_flags import RustKernelFlags


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


def _backtest_sample_inputs() -> tuple[
    list[date],
    Sequence[Mapping[str, object]],
    Sequence[Mapping[str, object]],
    Sequence[Mapping[str, object]],
    Mapping[object, Sequence[Mapping[str, object]]],
]:
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
    allocations: dict[object, list[dict[str, object]]] = {
        date(2025, 3, 31): [{"security_id": "KR:001", "target_weight": 1.0}],
        date(2025, 6, 30): [{"security_id": "KR:001", "target_weight": 1.0}],
    }
    return schedule, prices, benchmark, fx, allocations


def _constructor_infeasible_rows() -> list[dict[str, object]]:
    return [
        {"security_id": "JP:AAA", "country": "JP", "factor_value": 0.9, "rank_in_country": 1, "is_eligible": True},
        {"security_id": "JP:AAB", "country": "JP", "factor_value": 0.9, "rank_in_country": 1, "is_eligible": True},
        {"security_id": "JP:BAA", "country": "JP", "factor_value": 0.8, "rank_in_country": 2, "is_eligible": True},
        {"security_id": "JP:C00", "country": "JP", "factor_value": 0.7, "rank_in_country": 3, "is_eligible": True},
        {"security_id": "JP:SKIP", "country": "JP", "factor_value": 0.6, "rank_in_country": 4, "is_eligible": False},
    ]


@dataclass(frozen=True)
class _RankingFactorRow:
    security_id: str
    country: str
    factor_value: float | None
    rd_expense: float | None
    sales_ttm: float | None
    is_eligible: bool


@dataclass(frozen=True)
class _RankingAcceptedRow:
    security_id: str
    median_traded_value_krw: float


def _ranking_fixture() -> tuple[
    Sequence[_RankingFactorRow],
    Sequence[_RankingAcceptedRow],
    dict[str, str],
    dict[str, int],
]:
    factor_rows = [
        _RankingFactorRow("JP:001", "JP", 0.31, 10.0, 100.0, True),
        _RankingFactorRow("JP:002", "JP", 0.27, 9.0, 120.0, True),
        _RankingFactorRow("KR:001", "KR", 0.22, 8.0, 95.0, True),
        _RankingFactorRow("US:001", "US", 0.29, 11.0, 130.0, True),
        _RankingFactorRow("JP:SKIP", "JP", None, 7.0, 110.0, False),
    ]
    accepted_rows = [
        _RankingAcceptedRow("JP:001", 1_200_000_000.0),
        _RankingAcceptedRow("JP:002", 1_100_000_000.0),
        _RankingAcceptedRow("KR:001", 1_000_000_000.0),
        _RankingAcceptedRow("US:001", 1_300_000_000.0),
    ]
    sectors = {
        "JP:001": "TECH",
        "JP:002": "INDUSTRIALS",
        "KR:001": "TECH",
        "US:001": "HEALTHCARE",
    }
    requested = {"JP": 1, "KR": 1, "US": 1}
    return factor_rows, accepted_rows, sectors, requested


def _ranking_selected_ids(rows: Sequence[Mapping[str, object]], requested_counts: Mapping[str, int]) -> list[str]:
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


def test_rust_dispatch_defaults_to_python_when_feature_disabled() -> None:
    request = normalize_ranking_request(
        factor_rows=[{"security_id": "JP:001"}],
        accepted_rows=[{"security_id": "JP:001", "median_traded_value_krw": 1_000_000_000.0}],
        sector_by_security={"JP:001": "TECH"},
        requested_counts_by_country={"JP": 1},
        factor_model="rnd_sales_ttm",
        sector_active_band=0.10,
        use_size_stratification=True,
    )
    called = {"value": False}

    def _python_impl() -> list[dict[str, object]]:
        called["value"] = True
        return [{"security_id": "JP:001"}]

    result, backend, reason = run_ranking_kernel(
        request=request,
        flags=RustKernelFlags(),
        python_impl=_python_impl,
    )

    assert called["value"] is True
    assert backend == "python"
    assert reason == "feature_disabled"
    assert result == [{"security_id": "JP:001"}]


def test_ranking_dispatch_decodes_rust_json_payload_with_row_level_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factor_rows, accepted_rows, sectors, requested = _ranking_fixture()
    baseline = _build_qepm_ranked_rows(
        factor_rows=tuple(factor_rows),
        accepted_rows=cast(list[object], accepted_rows),
        sector_by_security=sectors,
        requested_counts_by_country=requested,
        factor_model="rnd_sales_ttm",
        sector_active_band=0.10,
        use_size_stratification=True,
    )
    request = normalize_ranking_request(
        factor_rows=factor_rows,
        accepted_rows=accepted_rows,
        sector_by_security=sectors,
        requested_counts_by_country=requested,
        factor_model="rnd_sales_ttm",
        sector_active_band=0.10,
        use_size_stratification=True,
    )

    class _FakeRustModule:
        @staticmethod
        def run_ranking_kernel(request_json: str) -> str:
            payload = cast(dict[str, object], json.loads(request_json))
            payload_factor_rows = cast(list[dict[str, object]], payload["factor_rows"])
            payload_accepted_rows = cast(list[dict[str, object]], payload["accepted_rows"])
            payload_sectors = cast(dict[str, str], payload["sector_by_security"])
            payload_requested_raw = cast(dict[str, object], payload["requested_counts_by_country"])
            payload_requested = {
                str(country): int(str(value))
                for country, value in payload_requested_raw.items()
            }
            factor_rows_obj = [
                _RankingFactorRow(
                    security_id=str(row["security_id"]),
                    country=str(row["country"]),
                    factor_value=(float(str(row["factor_value"])) if row.get("factor_value") is not None else None),
                    rd_expense=(float(str(row["rd_expense"])) if row.get("rd_expense") is not None else None),
                    sales_ttm=(float(str(row["sales_ttm"])) if row.get("sales_ttm") is not None else None),
                    is_eligible=bool(row["is_eligible"]),
                )
                for row in payload_factor_rows
            ]
            accepted_rows_obj = [
                _RankingAcceptedRow(
                    security_id=str(row["security_id"]),
                    median_traded_value_krw=float(str(row["median_traded_value_krw"])),
                )
                for row in payload_accepted_rows
            ]
            result = _build_qepm_ranked_rows(
                factor_rows=tuple(factor_rows_obj),
                accepted_rows=cast(list[object], accepted_rows_obj),
                sector_by_security=payload_sectors,
                requested_counts_by_country=payload_requested,
                factor_model=str(payload["factor_model"]),
                sector_active_band=float(str(payload["sector_active_band"])),
                use_size_stratification=bool(payload["use_size_stratification"]),
            )
            return json.dumps(result, sort_keys=True)

    called = {"python": False}

    def _python_impl() -> object:
        called["python"] = True
        return baseline

    monkeypatch.setattr(rust_dispatch, "load_extension", lambda: (_FakeRustModule(), None))

    result, backend, reason = run_ranking_kernel(
        request=request,
        flags=RustKernelFlags(ranking=True),
        python_impl=_python_impl,
    )

    assert backend == "rust"
    assert reason is None
    assert called["python"] is False
    assert result == baseline

    result_rows = cast(list[dict[str, object]], result)
    baseline_rows = cast(list[dict[str, object]], baseline)
    assert [
        (row["security_id"], row["rank_in_country"], row["is_eligible"])
        for row in result_rows
    ] == [
        (row["security_id"], row["rank_in_country"], row["is_eligible"])
        for row in baseline_rows
    ]
    assert _ranking_selected_ids(result_rows, requested) == _ranking_selected_ids(baseline_rows, requested)


def test_ranking_dispatch_falls_back_to_python_on_decode_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factor_rows, accepted_rows, sectors, requested = _ranking_fixture()
    baseline = _build_qepm_ranked_rows(
        factor_rows=tuple(factor_rows),
        accepted_rows=cast(list[object], accepted_rows),
        sector_by_security=sectors,
        requested_counts_by_country=requested,
        factor_model="rnd_sales_ttm",
        sector_active_band=0.10,
        use_size_stratification=True,
    )
    request = normalize_ranking_request(
        factor_rows=factor_rows,
        accepted_rows=accepted_rows,
        sector_by_security=sectors,
        requested_counts_by_country=requested,
        factor_model="rnd_sales_ttm",
        sector_active_band=0.10,
        use_size_stratification=True,
    )

    class _BrokenRustModule:
        @staticmethod
        def run_ranking_kernel(request_json: str) -> str:
            _ = request_json
            return "{}"

    called = {"python": False}

    def _python_impl() -> object:
        called["python"] = True
        return baseline

    monkeypatch.setattr(rust_dispatch, "load_extension", lambda: (_BrokenRustModule(), None))

    result, backend, reason = run_ranking_kernel(
        request=request,
        flags=RustKernelFlags(ranking=True),
        python_impl=_python_impl,
    )

    assert backend == "python"
    assert called["python"] is True
    assert isinstance(reason, str)
    assert reason.startswith("rust output decode failed:")
    assert result == baseline


def test_parity_harness_reports_baseline_not_found_for_missing_fixture(tmp_path: Path) -> None:
    harness = cast(object, importlib.import_module("scripts.run_rust_parity"))
    output_file = tmp_path / "parity.json"
    exit_code = cast(int, getattr(harness, "main")([
        "--kernel",
        "all",
        "--fixture",
        str(tmp_path / "missing_fixture.json"),
        "--output",
        str(output_file),
    ]))

    assert exit_code == 0
    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert payload["status"] == "baseline_not_found"


def test_backtest_dispatch_decodes_rust_json_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    schedule, prices, benchmark, fx, allocations = _backtest_sample_inputs()
    baseline = run_quarterly_backtest(
        rebalance_schedule=schedule,
        prices=prices,
        benchmark_series=benchmark,
        fx_rates=fx,
        portfolio_allocations=allocations,
        initial_nav_krw=1_000_000.0,
    )
    request = normalize_backtest_request(
        rebalance_schedule=schedule,
        prices=prices,
        benchmark_series=benchmark,
        fx_rates=fx,
        portfolio_allocations=allocations,
        initial_nav_krw=1_000_000.0,
    )

    class _FakeRustModule:
        @staticmethod
        def run_backtest_kernel(request_json: str) -> str:
            assert isinstance(request_json, str)
            return json.dumps(_serialize(baseline), sort_keys=True)

    called = {"python": False}

    def _python_impl() -> object:
        called["python"] = True
        return baseline

    monkeypatch.setattr(rust_dispatch, "load_extension", lambda: (_FakeRustModule(), None))

    result, backend, reason = run_backtest_kernel(
        request=request,
        flags=RustKernelFlags(backtest=True),
        python_impl=_python_impl,
    )

    assert backend == "rust"
    assert reason is None
    assert called["python"] is False
    assert result == baseline


def test_constructor_dispatch_decodes_rust_json_payload_with_metadata_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _constructor_infeasible_rows()
    baseline = construct_portfolio_with_constraints(rows)
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

    class _FakeRustModule:
        @staticmethod
        def run_constructor_kernel(request_json: str) -> str:
            payload = cast(dict[str, object], json.loads(request_json))
            payload_rows = cast(list[dict[str, object]], payload["ranked_factor_rows"])
            payload_risk_controls = cast(dict[str, object], payload["risk_controls"])
            payload_targets = payload.get("country_targets")
            if payload_targets is None:
                result = construct_portfolio_with_constraints(payload_rows, risk_controls=payload_risk_controls)
            else:
                targets = tuple((str(pair[0]), float(str(pair[1]))) for pair in cast(list[list[object]], payload_targets))
                result = construct_portfolio_with_constraints(
                    payload_rows,
                    country_targets=targets,
                    risk_controls=payload_risk_controls,
                )
            return json.dumps(_serialize(result), sort_keys=True)

    called = {"python": False}

    def _python_impl() -> object:
        called["python"] = True
        return baseline

    monkeypatch.setattr(rust_dispatch, "load_extension", lambda: (_FakeRustModule(), None))

    result, backend, reason = run_constructor_kernel(
        request=request,
        flags=RustKernelFlags(constructor=True),
        python_impl=_python_impl,
    )

    assert backend == "rust"
    assert reason is None
    assert called["python"] is False
    assert result == baseline
    assert result.fallback_triggered is True
    assert result.fallback_reasons == baseline.fallback_reasons
    assert [
        (holding.security_id, holding.country, holding.rank_in_country, holding.factor_value)
        for holding in result.holdings
    ] == [
        (holding.security_id, holding.country, holding.rank_in_country, holding.factor_value)
        for holding in baseline.holdings
    ]


def test_constructor_dispatch_falls_back_to_python_on_decode_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {"security_id": "JP:001", "country": "JP", "factor_value": 1.2, "rank_in_country": 1, "is_eligible": True},
        {"security_id": "KR:001", "country": "KR", "factor_value": 1.1, "rank_in_country": 1, "is_eligible": True},
        {"security_id": "US:001", "country": "US", "factor_value": 1.0, "rank_in_country": 1, "is_eligible": True},
    ]
    baseline = construct_portfolio_with_constraints(rows)
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

    class _BrokenRustModule:
        @staticmethod
        def run_constructor_kernel(request_json: str) -> str:
            _ = request_json
            return "{}"

    called = {"python": False}

    def _python_impl() -> object:
        called["python"] = True
        return baseline

    monkeypatch.setattr(rust_dispatch, "load_extension", lambda: (_BrokenRustModule(), None))

    result, backend, reason = run_constructor_kernel(
        request=request,
        flags=RustKernelFlags(constructor=True),
        python_impl=_python_impl,
    )

    assert backend == "python"
    assert called["python"] is True
    assert isinstance(reason, str)
    assert reason.startswith("rust output decode failed:")
    assert result == baseline


def test_backtest_parity_harness_reports_nav_trades_holdings_period_sections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = cast(object, importlib.import_module("scripts.run_rust_parity"))

    def _force_rust(*, request: object, flags: object, python_impl: object) -> tuple[object, str, str | None]:
        _ = request
        _ = flags
        baseline = cast(Callable[[], object], python_impl)()
        return baseline, "rust", None

    monkeypatch.setattr(harness, "run_backtest_kernel", _force_rust)

    output_file = tmp_path / "backtest_parity.json"
    exit_code = cast(int, getattr(harness, "main")([
        "--kernel",
        "backtest",
        "--output",
        str(output_file),
    ]))

    assert exit_code == 0
    payload = cast(dict[str, object], json.loads(output_file.read_text(encoding="utf-8")))
    results = cast(dict[str, object], payload["results"])
    backtest = cast(dict[str, object], results["backtest"])
    assert backtest["status"] == "pass"
    parity = cast(dict[str, object], backtest["parity"])
    assert set(parity.keys()) == {"nav_path", "trades", "holdings", "period_rows"}
    for key in ("nav_path", "trades", "holdings", "period_rows"):
        section = cast(dict[str, object], parity[key])
        assert section["match"] is True


def test_ranking_parity_harness_reports_rank_eligibility_selected_sections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = cast(object, importlib.import_module("scripts.run_rust_parity"))

    def _force_rust(*, request: object, flags: object, python_impl: object) -> tuple[object, str, str | None]:
        _ = request
        _ = flags
        baseline = cast(Callable[[], object], python_impl)()
        return baseline, "rust", None

    monkeypatch.setattr(harness, "run_ranking_kernel", _force_rust)

    output_file = tmp_path / "ranking_parity.json"
    exit_code = cast(int, getattr(harness, "main")([
        "--kernel",
        "ranking",
        "--output",
        str(output_file),
    ]))

    assert exit_code == 0
    payload = cast(dict[str, object], json.loads(output_file.read_text(encoding="utf-8")))
    results = cast(dict[str, object], payload["results"])
    ranking = cast(dict[str, object], results["ranking"])
    assert ranking["status"] == "pass"
    parity = cast(dict[str, object], ranking["parity"])
    assert set(parity.keys()) == {"rank_in_country", "is_eligible", "selected_ids"}
    for key in ("rank_in_country", "is_eligible", "selected_ids"):
        section = cast(dict[str, object], parity[key])
        assert section["match"] is True


def test_constructor_parity_harness_reports_metadata_and_fallback_sections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = cast(object, importlib.import_module("scripts.run_rust_parity"))

    def _force_rust(*, request: object, flags: object, python_impl: object) -> tuple[object, str, str | None]:
        _ = request
        _ = flags
        baseline = cast(Callable[[], object], python_impl)()
        return baseline, "rust", None

    monkeypatch.setattr(harness, "run_constructor_kernel", _force_rust)

    output_file = tmp_path / "constructor_parity.json"
    exit_code = cast(int, getattr(harness, "main")([
        "--kernel",
        "constructor",
        "--output",
        str(output_file),
    ]))

    assert exit_code == 0
    payload = cast(dict[str, object], json.loads(output_file.read_text(encoding="utf-8")))
    results = cast(dict[str, object], payload["results"])
    constructor = cast(dict[str, object], results["constructor"])
    assert constructor["status"] == "pass"
    cases = cast(dict[str, object], constructor["cases"])
    assert set(cases.keys()) == {"feasible", "infeasible"}
    for case_name in ("feasible", "infeasible"):
        case_payload = cast(dict[str, object], cases[case_name])
        assert case_payload["status"] == "pass"
        parity = cast(dict[str, object], case_payload["parity"])
        assert set(parity.keys()) == {"holdings_metadata", "fallback_triggered", "fallback_reasons"}
        for key in ("holdings_metadata", "fallback_triggered", "fallback_reasons"):
            section = cast(dict[str, object], parity[key])
            assert section["match"] is True
