from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from typing import Protocol, cast

import pytest


class _Violation(Protocol):
    marker: str
    record_index: int | None
    field: str | None


class _ContractsModule(Protocol):
    PIT_VIOLATION: str

    def assert_fundamentals_pit(
        self,
        fundamentals: Sequence[Mapping[str, object]],
        rebalance_date: str,
    ) -> None: ...

    def canonical_benchmark(self, record: Mapping[str, object]) -> dict[str, object]: ...

    def canonical_constituent(self, record: Mapping[str, object]) -> dict[str, object]: ...

    def canonical_fundamental(self, record: Mapping[str, object]) -> dict[str, object]: ...

    def canonical_fx(self, record: Mapping[str, object]) -> dict[str, object]: ...

    def canonical_price(self, record: Mapping[str, object]) -> dict[str, object]: ...

    def validate_fundamentals_pit(
        self,
        fundamentals: Sequence[Mapping[str, object]],
        rebalance_date: str,
    ) -> list[_Violation]: ...


def _load_contracts_module() -> _ContractsModule:
    module_path = Path(__file__).resolve().parents[1] / "src" / "data" / "schema" / "contracts.py"
    spec = spec_from_file_location("schema_contracts", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec from {module_path}")

    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    typed_module = cast(_ContractsModule, cast(object, module))
    return typed_module


_contracts = _load_contracts_module()
PIT_VIOLATION = _contracts.PIT_VIOLATION
assert_fundamentals_pit = _contracts.assert_fundamentals_pit
canonical_benchmark = _contracts.canonical_benchmark
canonical_constituent = _contracts.canonical_constituent
canonical_fundamental = _contracts.canonical_fundamental
canonical_fx = _contracts.canonical_fx
canonical_price = _contracts.canonical_price
validate_fundamentals_pit = _contracts.validate_fundamentals_pit


def test_canonical_constituent_uses_scd_style_fields() -> None:
    record = canonical_constituent(
        {
            "as_of_date": "2026-01-31",
            "security_id": "US:A:NYQ",
            "issuer_id": "US:A",
            "ticker": "A:NYQ",
            "stock_code": "A",
            "exchange": "NYQ",
            "country": "US",
            "universe": "Russell3000",
            "effective_from": "2026-01-31",
            "effective_to": "",
            "is_current": "True",
        }
    )

    assert record["as_of_date"] == "2026-01-31"
    assert record["effective_from"] == "2026-01-31"
    assert record["effective_to"] is None
    assert record["is_current"] is True
    assert record["country"] == "us"
    assert record["universe"] == "russell3000"


def test_canonical_records_for_fundamentals_prices_benchmarks_fx() -> None:
    fundamental = canonical_fundamental(
        {
            "as_of_date": "2026-03-31",
            "security_id": "JP:1332:TYO",
            "country": "JP",
            "period_end": "2025-12-31",
            "filing_date": "2026-02-10",
            "rd_expense": "100.5",
            "sales_ttm": "900.0",
            "effective_from": "2026-02-10",
            "effective_to": "",
            "is_current": True,
        }
    )
    assert fundamental["as_of_date"] == "2026-03-31"
    assert fundamental["filing_date"] == "2026-02-10"
    assert fundamental["effective_to"] is None
    assert fundamental["is_current"] is True

    price = canonical_price(
        {
            "as_of_date": "2026-03-31",
            "security_id": "KRX:000100",
            "country": "KR",
            "price_date": "2026-03-31",
            "close": "105000",
            "currency": "krw",
            "effective_from": "2026-03-31",
            "effective_to": "",
            "is_current": "true",
        }
    )
    assert price["currency"] == "KRW"
    assert price["effective_to"] is None

    benchmark = canonical_benchmark(
        {
            "as_of_date": "2026-03-31",
            "benchmark_id": "US_R3000_PROXY",
            "country": "US",
            "benchmark_date": "2026-03-31",
            "close": 2500.01,
            "currency": "usd",
            "effective_from": "2026-03-31",
            "effective_to": "",
            "is_current": True,
        }
    )
    assert benchmark["currency"] == "USD"

    fx = canonical_fx(
        {
            "as_of_date": "2026-03-31",
            "pair": "usd/krw",
            "fx_date": "2026-03-31",
            "rate": "1322.40",
            "effective_from": "2026-03-31",
            "effective_to": "",
            "is_current": True,
        }
    )
    assert fx["pair"] == "USD/KRW"
    assert fx["is_current"] is True


def test_validate_fundamentals_pit_detects_future_filing_date() -> None:
    violations = validate_fundamentals_pit(
        fundamentals=[
            {
                "as_of_date": "2026-03-31",
                "security_id": "US:A:NYQ",
                "country": "US",
                "period_end": "2025-12-31",
                "filing_date": "2026-04-02",
                "rd_expense": 500.0,
                "sales_ttm": 1000.0,
                "effective_from": "2026-04-02",
                "effective_to": "",
                "is_current": True,
            }
        ],
        rebalance_date="2026-03-31",
    )

    assert len(violations) == 1
    assert violations[0].marker == PIT_VIOLATION
    assert violations[0].record_index == 0
    assert violations[0].field == "filing_date"


def test_assert_fundamentals_pit_raises_deterministic_marker() -> None:
    with pytest.raises(ValueError, match=PIT_VIOLATION):
        assert_fundamentals_pit(
            fundamentals=[
                {
                    "as_of_date": "2026-03-31",
                    "security_id": "KRX:000100",
                    "country": "KR",
                    "period_end": "2025-12-31",
                    "filing_date": "2026-04-20",
                    "rd_expense": 400.0,
                    "sales_ttm": 1000.0,
                    "effective_from": "2026-04-20",
                    "effective_to": "",
                    "is_current": True,
                }
            ],
            rebalance_date="2026-03-31",
        )


def test_assert_fundamentals_pit_allows_filing_on_or_before_rebalance() -> None:
    assert_fundamentals_pit(
        fundamentals=[
            {
                "as_of_date": "2026-03-31",
                "security_id": "JP:1332:TYO",
                "country": "JP",
                "period_end": "2025-12-31",
                "filing_date": "2026-03-31",
                "rd_expense": 200.0,
                "sales_ttm": 800.0,
                "effective_from": "2026-03-31",
                "effective_to": "",
                "is_current": True,
            }
        ],
        rebalance_date="2026-03-31",
    )


def test_canonical_contracts_reject_non_finite_numeric_values() -> None:
    with pytest.raises(ValueError, match="finite numeric"):
        _ = canonical_fundamental(
            {
                "as_of_date": "2026-03-31",
                "security_id": "US:A:NYQ",
                "country": "US",
                "period_end": "2025-12-31",
                "filing_date": "2026-03-01",
                "rd_expense": "nan",
                "sales_ttm": "1000",
                "effective_from": "2026-03-01",
                "effective_to": "",
                "is_current": True,
            }
        )

    with pytest.raises(ValueError, match="finite numeric"):
        _ = canonical_fx(
            {
                "as_of_date": "2026-03-31",
                "pair": "USD/KRW",
                "fx_date": "2026-03-31",
                "rate": "inf",
                "effective_from": "2026-03-31",
                "effective_to": "",
                "is_current": True,
            }
        )
