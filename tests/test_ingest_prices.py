from __future__ import annotations

import importlib
from typing import Protocol, cast


class _PriceSourceSelection(Protocol):
    source: str
    fallback_used: bool
    primary_source: str
    quality_flag: str


class _PricesModule(Protocol):
    def normalize_prices_with_source_fallback(
        self,
        rows_by_source: dict[str, list[dict[str, object]]],
        source_priority: tuple[str, ...],
    ) -> tuple[list[dict[str, object]], _PriceSourceSelection]: ...


def _load_module() -> _PricesModule:
    module = importlib.import_module("src.data.ingest.prices")
    typed_module = cast(_PricesModule, cast(object, module))
    return typed_module


_prices = _load_module()


def _base_row() -> dict[str, object]:
    return {
        "as_of_date": "2026-03-31",
        "security_id": "US:AAPL:NSQ",
        "country": "US",
        "price_date": "2026-03-31",
        "close": "123.45",
        "currency": "usd",
        "effective_from": "2026-03-31",
        "effective_to": "",
        "is_current": "true",
    }


def test_normalize_prices_primary_source_success() -> None:
    rows, selection = _prices.normalize_prices_with_source_fallback(
        rows_by_source={
            "stooq": [_base_row()],
            "pykrx": [
                {
                    "as_of_date": "2026-03-31",
                    "security_id": "KRX:005930",
                    "country": "KR",
                    "price_date": "2026-03-31",
                    "close": "72000",
                    "currency": "krw",
                    "effective_from": "2026-03-31",
                    "effective_to": "",
                    "is_current": "true",
                }
            ],
        },
        source_priority=("stooq", "pykrx"),
    )

    assert selection.source == "stooq"
    assert selection.primary_source == "stooq"
    assert selection.fallback_used is False
    assert selection.quality_flag == "ok"

    assert len(rows) == 1
    row = rows[0]
    assert tuple(row.keys()) == (
        "as_of_date",
        "security_id",
        "country",
        "price_date",
        "close",
        "currency",
        "effective_from",
        "effective_to",
        "is_current",
        "source",
        "quality_flag",
    )
    assert row["country"] == "us"
    assert row["currency"] == "USD"
    assert row["close"] == 123.45
    assert row["effective_to"] is None
    assert row["source"] == "stooq"
    assert row["quality_flag"] == "ok"


def test_normalize_prices_fallback_source_sets_metadata() -> None:
    rows, selection = _prices.normalize_prices_with_source_fallback(
        rows_by_source={
            "stooq": [],
            "pykrx": [
                {
                    "as_of_date": "2026-03-31",
                    "security_id": "KRX:005930",
                    "country": "KR",
                    "price_date": "2026-03-31",
                    "close": "73000",
                    "currency": "krw",
                    "effective_from": "2026-03-31",
                    "effective_to": "",
                    "is_current": "true",
                }
            ],
        },
        source_priority=("stooq", "pykrx"),
    )

    assert selection.source == "pykrx"
    assert selection.primary_source == "stooq"
    assert selection.fallback_used is True
    assert selection.quality_flag == "fallback_source"

    assert len(rows) == 1
    row = rows[0]
    assert row["country"] == "kr"
    assert row["currency"] == "KRW"
    assert row["source"] == "pykrx"
    assert row["quality_flag"] == "fallback_source"
