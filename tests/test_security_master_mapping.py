"""Task 5 tests for cross-market security master mapping."""

from __future__ import annotations

import csv
import importlib
from pathlib import Path
from typing import Protocol, cast

import pytest


class _SecurityMasterProtocol(Protocol):
    @classmethod
    def from_rows(cls, rows: list[dict[str, str]]) -> "_SecurityMasterProtocol": ...

    def resolve(self, *, market: str, key_type: str, key_value: str, as_of: str) -> str | None: ...


_SECURITY_MASTER_MODULE = importlib.import_module("src.data.security_master")
DuplicateActiveMappingError = cast(
    type[Exception],
    getattr(_SECURITY_MASTER_MODULE, "DuplicateActiveMappingError"),
)
SecurityMaster = cast(
    type[_SecurityMasterProtocol],
    getattr(_SECURITY_MASTER_MODULE, "SecurityMaster"),
)


ROOT = Path(__file__).resolve().parents[1]


def _load_first_row(file_name: str) -> dict[str, str]:
    file_path = ROOT / file_name
    with file_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        first = next(reader)
    return dict(first)


def test_cross_market_keys_resolve_to_canonical_security_id() -> None:
    rows = [
        _load_first_row("us_sector_current.csv"),
        _load_first_row("kr_sector_current.csv"),
        _load_first_row("jp_sector_current.csv"),
    ]
    master = SecurityMaster.from_rows(rows)

    for row in rows:
        market = row["country"]
        as_of = row["effective_from"]
        for key_type in ("security_id", "issuer_id", "ticker", "stock_code"):
            resolved = master.resolve(
                market=market,
                key_type=key_type,
                key_value=row[key_type],
                as_of=as_of,
            )
            assert resolved == row["security_id"]


def test_validity_windows_switch_key_mapping_without_overlap() -> None:
    rows = [
        {
            "country": "us",
            "security_id": "US:ABC:NYQ",
            "issuer_id": "US:ABC",
            "ticker": "ABC:NYQ",
            "stock_code": "ABC",
            "effective_from": "2024-01-01",
            "effective_to": "2024-06-30",
        },
        {
            "country": "us",
            "security_id": "US:ABCV2:NYQ",
            "issuer_id": "US:ABCV2",
            "ticker": "ABC:NYQ",
            "stock_code": "ABC",
            "effective_from": "2024-07-01",
            "effective_to": "",
        },
    ]
    master = SecurityMaster.from_rows(rows)

    assert (
        master.resolve(market="US", key_type="ticker", key_value="ABC:NYQ", as_of="2024-06-30")
        == "US:ABC:NYQ"
    )
    assert (
        master.resolve(market="US", key_type="ticker", key_value="ABC:NYQ", as_of="2024-07-01")
        == "US:ABCV2:NYQ"
    )
    assert master.resolve(market="US", key_type="ticker", key_value="ABC:NYQ", as_of="2023-12-31") is None


def test_duplicate_active_key_collision_is_rejected() -> None:
    rows = [
        {
            "country": "jp",
            "security_id": "JP:1111:TYO",
            "issuer_id": "JP:1111",
            "ticker": "1111:TYO",
            "stock_code": "1111",
            "effective_from": "2024-01-01",
            "effective_to": "",
        },
        {
            "country": "jp",
            "security_id": "JP:ZZZZ:TYO",
            "issuer_id": "JP:ZZZZ",
            "ticker": "1111:TYO",
            "stock_code": "1111",
            "effective_from": "2024-06-01",
            "effective_to": "",
        },
    ]

    with pytest.raises(DuplicateActiveMappingError, match="duplicate active mapping"):
        _ = SecurityMaster.from_rows(rows)
