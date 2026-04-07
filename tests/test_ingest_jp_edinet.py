from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys
from typing import Protocol, cast

import pytest


class _IngestFailure(Protocol):
    row_index: int
    message: str


class _JpEdinetModule(Protocol):
    def parse_jp_edinet_fundamentals(
        self,
        payload: object,
        *,
        as_of_date: object | None = None,
        default_exchange: str = "TSE",
    ) -> tuple[list[dict[str, object]], list[_IngestFailure]]: ...


def _load_ingest_module() -> _JpEdinetModule:
    src_path = Path(__file__).resolve().parents[1] / "src"
    src_path_text = str(src_path)
    if src_path_text not in sys.path:
        sys.path.insert(0, src_path_text)
    module = import_module("data.ingest.jp_edinet")
    typed_module = cast(_JpEdinetModule, cast(object, module))
    return typed_module


parse_jp_edinet_fundamentals = _load_ingest_module().parse_jp_edinet_fundamentals


def test_parse_jp_edinet_fundamentals_normalizes_to_canonical_schema() -> None:
    records, failures = parse_jp_edinet_fundamentals(
        {
            "results": [
                {
                    "code": "1332-T",
                    "market": "xtks",
                    "periodEnd": "20251231",
                    "filingDate": "2026/02/10",
                    "research_and_development_expense": "100.5",
                    "NetSales": "1,200.0",
                    "isCurrent": "true",
                }
            ]
        }
    )

    assert failures == []
    assert len(records) == 1

    record = records[0]
    assert record["as_of_date"] == "2026-02-10"
    assert record["security_id"] == "JP:1332:TSE"
    assert record["country"] == "jp"
    assert record["period_end"] == "2025-12-31"
    assert record["filing_date"] == "2026-02-10"
    assert record["rd_expense"] == 100.5
    assert record["sales_ttm"] == 1200.0
    assert record["effective_from"] == "2026-02-10"
    assert record["effective_to"] is None
    assert record["is_current"] is True


def test_parse_jp_edinet_fundamentals_preserves_pit_fields_for_non_current_rows() -> None:
    records, failures = parse_jp_edinet_fundamentals(
        [
            {
                "jp_code": 6758,
                "period_end": "2025-12-31",
                "submitDateTime": "2026-03-31T09:00:00+09:00",
                "rd_expense": 250.0,
                "sales_ttm": 4000.0,
                "effective_to": "2026-04-30",
                "is_current": False,
            }
        ],
        as_of_date="2026-04-30",
    )

    assert failures == []
    assert len(records) == 1

    record = records[0]
    assert record["as_of_date"] == "2026-04-30"
    assert record["security_id"] == "JP:6758:TSE"
    assert record["country"] == "jp"
    assert record["filing_date"] == "2026-03-31"
    assert record["effective_from"] == "2026-03-31"
    assert record["effective_to"] == "2026-04-30"
    assert record["is_current"] is False


def test_parse_jp_edinet_fundamentals_collects_malformed_row_failures() -> None:
    records, failures = parse_jp_edinet_fundamentals(
        {
            "results": [
                "not-a-dict",
                {
                    "code": "7203",
                    "period_end": "2025-12-31",
                    "filing_date": "2026-02-14",
                    "rd_expense": "abc",
                    "sales_ttm": "1000",
                },
                {
                    "code": "9984",
                    "period_end": "2025-12-31",
                    "filing_date": "bad-date",
                    "rd_expense": "200",
                    "sales_ttm": "900",
                },
            ]
        }
    )

    assert records == []
    assert len(failures) == 3
    assert failures[0].row_index == 0
    assert failures[0].message == "row must be a mapping"
    assert failures[1].row_index == 1
    assert failures[1].message == "rd_expense must be numeric"
    assert failures[2].row_index == 2
    assert failures[2].message == "filing_date must be a valid date"


def test_parse_jp_edinet_fundamentals_rejects_invalid_payload_shape() -> None:
    with pytest.raises(ValueError, match="must include one of"):
        _ = parse_jp_edinet_fundamentals({"unexpected": []})
