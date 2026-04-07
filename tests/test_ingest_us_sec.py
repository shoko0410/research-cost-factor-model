from __future__ import annotations

from collections.abc import Mapping, Sequence
import importlib
from typing import Protocol, cast

import pytest


class _USSecModule(Protocol):
    def handle_missing_sec_payload(
        self,
        *,
        security_id: str,
        payload_rows: Sequence[Mapping[str, object]] | None,
    ) -> tuple[list[str], list[str]]: ...

    def normalize_cik(self, value: str | int) -> str: ...

    def transform_us_sec_fundamentals(
        self,
        *,
        security_id: str,
        cik: str | int,
        payload_rows: Sequence[Mapping[str, object]] | None,
        country: str = "US",
    ) -> tuple[list[dict[str, object]], list[str], list[str]]: ...


def _load_module() -> _USSecModule:
    module = importlib.import_module("src.data.ingest.us_sec")
    typed_module = cast(_USSecModule, cast(object, module))
    return typed_module


_us_sec = _load_module()


def test_transform_us_sec_fundamentals_happy_path() -> None:
    records, skipped, warnings = _us_sec.transform_us_sec_fundamentals(
        security_id="US:AAPL:XNAS",
        cik="320193",
        payload_rows=[
            {
                "period_end": "2024-09-28",
                "filing_date": "2024-11-01",
                "rd_expense": "31370",
                "sales_ttm": "391035",
            },
            {
                "period_end": "2023-09-30",
                "filing_date": "2023-11-03",
                "rd_expense": 29915,
                "sales_ttm": 383285,
            },
        ],
    )

    assert skipped == []
    assert warnings == []
    assert [record["period_end"] for record in records] == ["2023-09-30", "2024-09-28"]
    assert records[0]["security_id"] == "US:AAPL:XNAS"
    assert records[0]["country"] == "us"
    assert records[0]["as_of_date"] == "2023-11-03"
    assert records[0]["effective_from"] == "2023-11-03"
    assert records[0]["effective_to"] is None
    assert records[0]["is_current"] is True
    assert records[1]["rd_expense"] == 31370.0
    assert records[1]["sales_ttm"] == 391035.0


def test_handle_missing_sec_payload_returns_skip_and_warning() -> None:
    missing_skipped, missing_warnings = _us_sec.handle_missing_sec_payload(
        security_id="US:MSFT:XNAS",
        payload_rows=None,
    )
    empty_skipped, empty_warnings = _us_sec.handle_missing_sec_payload(
        security_id="US:GOOGL:XNAS",
        payload_rows=[],
    )

    assert missing_skipped == ["US:MSFT:XNAS"]
    assert missing_warnings == ["US:MSFT:XNAS: SEC response missing"]
    assert empty_skipped == ["US:GOOGL:XNAS"]
    assert empty_warnings == ["US:GOOGL:XNAS: SEC payload contains no rows"]


def test_transform_us_sec_fundamentals_returns_empty_on_missing_payload() -> None:
    records, skipped, warnings = _us_sec.transform_us_sec_fundamentals(
        security_id="US:NVDA:XNAS",
        cik=1045810,
        payload_rows=None,
    )

    assert records == []
    assert skipped == ["US:NVDA:XNAS"]
    assert warnings == ["US:NVDA:XNAS: SEC response missing"]


def test_normalize_cik_zero_pads_and_validates() -> None:
    assert _us_sec.normalize_cik(320193) == "0000320193"
    assert _us_sec.normalize_cik("789019") == "0000789019"

    with pytest.raises(ValueError, match="digits only"):
        _ = _us_sec.normalize_cik("ABC123")


def test_transform_us_sec_fundamentals_missing_required_field_raises() -> None:
    with pytest.raises(ValueError, match="row 0 missing required fields"):
        _ = _us_sec.transform_us_sec_fundamentals(
            security_id="US:AMZN:XNAS",
            cik="1018724",
            payload_rows=[
                {
                    "period_end": "2024-12-31",
                    "filing_date": "2025-02-03",
                    "rd_expense": 88200,
                }
            ],
        )
