from __future__ import annotations

import importlib
from typing import Protocol, cast

import pytest


class _IngestModule(Protocol):
    UnmappedCorpCodeError: type[Exception]

    def build_corp_code_stock_code_map(
        self,
        rows: list[dict[str, object]],
        *,
        corp_code_field: str = "corp_code",
        stock_code_field: str = "stock_code",
    ) -> dict[str, str]: ...

    def transform_kr_dart_fundamentals(
        self,
        raw_rows: list[dict[str, object]],
        *,
        corp_code_to_stock_code: dict[str, str] | None = None,
        strict_unmapped_corp_code: bool = True,
        warnings: list[str] | None = None,
    ) -> list[dict[str, object]]: ...


def _load_ingest_module() -> _IngestModule:
    module = importlib.import_module("src.data.ingest.kr_dart")
    return cast(_IngestModule, cast(object, module))


_ingest = _load_ingest_module()
UnmappedCorpCodeError = _ingest.UnmappedCorpCodeError
build_corp_code_stock_code_map = _ingest.build_corp_code_stock_code_map
transform_kr_dart_fundamentals = _ingest.transform_kr_dart_fundamentals


def test_build_corp_code_stock_code_map_normalizes_rows() -> None:
    mapping = build_corp_code_stock_code_map(
        [
            {"corp_code": " 00126380 ", "stock_code": " 000100 "},
            {"corp_code": "", "stock_code": "000660"},
            {"corp_code": "00164779", "stock_code": ""},
            {"corp_code": "00164779", "stock_code": "000660"},
        ]
    )

    assert mapping == {"00126380": "000100", "00164779": "000660"}


def test_transform_kr_dart_fundamentals_returns_canonical_records() -> None:
    records = transform_kr_dart_fundamentals(
        raw_rows=[
            {
                "corp_code": "00126380",
                "stock_code": "",
                "as_of_date": "2026/04/01",
                "period_end": "20251231",
                "rcept_dt": "20260401",
                "rd_expense": "110.25",
                "sales_ttm": "9000",
            }
        ],
        corp_code_to_stock_code={"00126380": "000100"},
    )

    assert len(records) == 1
    assert records[0] == {
        "as_of_date": "2026-04-01",
        "security_id": "KRX:000100",
        "country": "kr",
        "period_end": "2025-12-31",
        "filing_date": "2026-04-01",
        "rd_expense": 110.25,
        "sales_ttm": 9000.0,
        "effective_from": "2026-04-01",
        "effective_to": None,
        "is_current": True,
    }


def test_transform_kr_dart_fundamentals_raises_for_unmapped_corp_code() -> None:
    with pytest.raises(UnmappedCorpCodeError, match="unmapped corp_code"):
        _ = transform_kr_dart_fundamentals(
            raw_rows=[
                {
                    "corp_code": "00999999",
                    "period_end": "2025-12-31",
                    "filing_date": "2026-04-01",
                    "rd_expense": "100",
                    "sales_ttm": "500",
                }
            ],
            corp_code_to_stock_code={"00126380": "000100"},
            strict_unmapped_corp_code=True,
        )


def test_transform_kr_dart_fundamentals_warns_and_skips_when_configured() -> None:
    warning_messages: list[str] = []

    records = transform_kr_dart_fundamentals(
        raw_rows=[
            {
                "corp_code": "00999999",
                "period_end": "2025-12-31",
                "filing_date": "2026-04-01",
                "rd_expense": "100",
                "sales_ttm": "500",
            }
        ],
        corp_code_to_stock_code={"00126380": "000100"},
        strict_unmapped_corp_code=False,
        warnings=warning_messages,
    )

    assert records == []
    assert warning_messages == ["unmapped corp_code: 00999999"]


def test_transform_kr_dart_fundamentals_preserves_zero_values() -> None:
    records = transform_kr_dart_fundamentals(
        raw_rows=[
            {
                "corp_code": "00126380",
                "period_end": "2025-12-31",
                "filing_date": "2026-04-01",
                "rd_expense": 0.0,
                "rnd_expense": "999.0",
                "sales_ttm": 0.0,
                "revenue_ttm": "888.0",
            }
        ],
        corp_code_to_stock_code={"00126380": "000100"},
        strict_unmapped_corp_code=False,
    )

    assert len(records) == 1
    assert records[0]["rd_expense"] == 0.0
    assert records[0]["sales_ttm"] == 0.0
