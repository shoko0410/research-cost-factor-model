from __future__ import annotations

import importlib
from typing import Protocol, cast

import pytest


class _AcceptedSecurity(Protocol):
    security_id: str
    median_traded_value_krw: float


class _RejectedSecurity(Protocol):
    security_id: str
    reasons: tuple[str, ...]


class _Result(Protocol):
    pit_violations: int
    accepted: tuple[_AcceptedSecurity, ...]
    rejected: tuple[_RejectedSecurity, ...]


class _PitAssemblerModule(Protocol):
    REASON_INVESTABILITY_BELOW_THRESHOLD: str
    REASON_MISSING_RD_EXPENSE: str

    class PITViolationError(Exception):
        ...

    def assemble_pit_universe(
        self,
        *,
        universe: str,
        as_of_date: str,
        constituent_rows: list[dict[str, object]],
        fundamental_rows: list[dict[str, object]],
        price_rows: list[dict[str, object]],
        fx_rates: dict[str, object],
    ) -> _Result: ...


def _load_module() -> _PitAssemblerModule:
    module = importlib.import_module("src.features.pit_assembler")
    typed_module = cast(_PitAssemblerModule, cast(object, module))
    return typed_module


_pit = _load_module()
REASON_INVESTABILITY_BELOW_THRESHOLD = _pit.REASON_INVESTABILITY_BELOW_THRESHOLD
REASON_MISSING_RD_EXPENSE = _pit.REASON_MISSING_RD_EXPENSE
PITViolationError = _pit.PITViolationError
assemble_pit_universe = _pit.assemble_pit_universe


def _constituents() -> list[dict[str, object]]:
    return [
        {
            "universe": "Russell3000",
            "security_id": "US:A:NYQ",
            "effective_from": "2026-01-01",
            "effective_to": "",
            "is_current": True,
        },
        {
            "universe": "Russell3000",
            "security_id": "KRX:000100",
            "effective_from": "2026-01-01",
            "effective_to": "",
            "is_current": True,
        },
    ]


def _fundamentals() -> list[dict[str, object]]:
    return [
        {
            "security_id": "US:A:NYQ",
            "country": "US",
            "period_end": "2025-12-31",
            "filing_date": "2026-02-15",
            "rd_expense": 100.0,
            "sales_ttm": 1000.0,
            "effective_from": "2026-02-15",
            "effective_to": "",
            "is_current": True,
        },
        {
            "security_id": "KRX:000100",
            "country": "KR",
            "period_end": "2025-12-31",
            "filing_date": "2026-02-20",
            "rd_expense": 2500.0,
            "sales_ttm": 9000.0,
            "effective_from": "2026-02-20",
            "effective_to": "",
            "is_current": True,
        },
    ]


def _prices_for_happy_path() -> list[dict[str, object]]:
    return [
        {
            "security_id": "US:A:NYQ",
            "price_date": "2026-03-27",
            "currency": "USD",
            "traded_value": 450000.0,
        },
        {
            "security_id": "US:A:NYQ",
            "price_date": "2026-03-28",
            "currency": "USD",
            "traded_value": 500000.0,
        },
        {
            "security_id": "US:A:NYQ",
            "price_date": "2026-03-31",
            "currency": "USD",
            "traded_value": 550000.0,
        },
        {
            "security_id": "KRX:000100",
            "price_date": "2026-03-27",
            "currency": "KRW",
            "traded_value": 800000000.0,
        },
        {
            "security_id": "KRX:000100",
            "price_date": "2026-03-28",
            "currency": "KRW",
            "traded_value": 820000000.0,
        },
        {
            "security_id": "KRX:000100",
            "price_date": "2026-03-31",
            "currency": "KRW",
            "traded_value": 810000000.0,
        },
    ]


def _fx_rates() -> dict[str, object]:
    return {
        "USD/KRW": {
            "2026-03-27": 1300.0,
            "2026-03-28": 1300.0,
            "2026-03-31": 1300.0,
        }
    }


def test_assemble_pit_universe_happy_path() -> None:
    result = assemble_pit_universe(
        universe="Russell3000",
        as_of_date="2026-03-31",
        constituent_rows=_constituents(),
        fundamental_rows=_fundamentals(),
        price_rows=_prices_for_happy_path(),
        fx_rates=_fx_rates(),
    )

    assert result.pit_violations == 0
    assert tuple(row.security_id for row in result.accepted) == ("KRX:000100", "US:A:NYQ")
    assert result.rejected == ()
    us_row = [row for row in result.accepted if row.security_id == "US:A:NYQ"][0]
    assert us_row.median_traded_value_krw == 650000000.0


def test_assemble_pit_universe_blocks_look_ahead_filing() -> None:
    bad_fundamentals = _fundamentals() + [
        {
            "security_id": "US:A:NYQ",
            "country": "US",
            "period_end": "2025-12-31",
            "filing_date": "2026-04-01",
            "rd_expense": 100.0,
            "sales_ttm": 1000.0,
            "effective_from": "2026-04-01",
            "effective_to": "",
            "is_current": True,
        }
    ]

    with pytest.raises(PITViolationError, match="PIT violations must be zero"):
        _ = assemble_pit_universe(
            universe="Russell3000",
            as_of_date="2026-03-31",
            constituent_rows=_constituents(),
            fundamental_rows=bad_fundamentals,
            price_rows=_prices_for_happy_path(),
            fx_rates=_fx_rates(),
        )


def test_assemble_pit_universe_excludes_missing_fundamental_fields() -> None:
    fundamentals = _fundamentals()
    fundamentals[0] = {
        **fundamentals[0],
        "rd_expense": None,
    }

    result = assemble_pit_universe(
        universe="Russell3000",
        as_of_date="2026-03-31",
        constituent_rows=_constituents(),
        fundamental_rows=fundamentals,
        price_rows=_prices_for_happy_path(),
        fx_rates=_fx_rates(),
    )

    assert tuple(row.security_id for row in result.accepted) == ("KRX:000100",)
    rejected_map = {row.security_id: row.reasons for row in result.rejected}
    assert rejected_map["US:A:NYQ"] == (REASON_MISSING_RD_EXPENSE,)


def test_assemble_pit_universe_rejects_investability_below_threshold() -> None:
    weak_prices = _prices_for_happy_path()
    weak_prices[0] = {
        "security_id": "US:A:NYQ",
        "price_date": "2026-03-27",
        "currency": "USD",
        "traded_value": 200000.0,
    }
    weak_prices[1] = {
        "security_id": "US:A:NYQ",
        "price_date": "2026-03-28",
        "currency": "USD",
        "traded_value": 220000.0,
    }
    weak_prices[2] = {
        "security_id": "US:A:NYQ",
        "price_date": "2026-03-31",
        "currency": "USD",
        "traded_value": 240000.0,
    }

    result = assemble_pit_universe(
        universe="Russell3000",
        as_of_date="2026-03-31",
        constituent_rows=_constituents(),
        fundamental_rows=_fundamentals(),
        price_rows=weak_prices,
        fx_rates=_fx_rates(),
    )

    assert tuple(row.security_id for row in result.accepted) == ("KRX:000100",)
    rejected_map = {row.security_id: row.reasons for row in result.rejected}
    assert rejected_map["US:A:NYQ"] == (REASON_INVESTABILITY_BELOW_THRESHOLD,)
