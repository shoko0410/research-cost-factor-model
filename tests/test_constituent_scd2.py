from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from typing import Protocol, cast

import pytest


class _ConstituentsSCD2Module(Protocol):
    class ConstituentOverlapError(Exception):
        ...

    def build_month_end_cache(
        self,
        rows: Sequence[Mapping[str, object]],
        *,
        universe: str,
        start_date: str | date | None = None,
        end_date: str | date | None = None,
    ) -> dict[date, tuple[str, ...]]: ...

    def membership_for_date(
        self,
        rows: Sequence[Mapping[str, object]],
        *,
        universe: str,
        as_of_date: str | date,
    ) -> tuple[str, ...]: ...


def _load_module() -> _ConstituentsSCD2Module:
    module_path = Path(__file__).resolve().parents[1] / "src" / "data" / "constituents_scd2.py"
    spec = spec_from_file_location("constituents_scd2", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec from {module_path}")

    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    typed_module = cast(_ConstituentsSCD2Module, cast(object, module))
    return typed_module


_scd2 = _load_module()


def _sample_rows() -> list[dict[str, object]]:
    return [
        {
            "universe": "Russell3000",
            "security_id": "US:A:NYQ",
            "effective_from": "2026-01-01",
            "effective_to": "2026-01-31",
            "is_current": False,
        },
        {
            "universe": "Russell3000",
            "security_id": "US:A:NYQ",
            "effective_from": "2026-02-01",
            "effective_to": "",
            "is_current": True,
        },
        {
            "universe": "Russell3000",
            "security_id": "US:B:NYQ",
            "effective_from": "2026-01-15",
            "effective_to": "",
            "is_current": True,
        },
        {
            "universe": "TOPIX500",
            "security_id": "JP:1332:TYO",
            "effective_from": "2026-01-01",
            "effective_to": "",
            "is_current": True,
        },
    ]


def test_membership_for_date_happy_path() -> None:
    rows = _sample_rows()

    jan_members = _scd2.membership_for_date(rows, universe="RUSSELL3000", as_of_date="2026-01-31")
    feb_members = _scd2.membership_for_date(rows, universe="Russell3000", as_of_date="2026-02-28")

    assert jan_members == ("US:A:NYQ", "US:B:NYQ")
    assert feb_members == ("US:A:NYQ", "US:B:NYQ")


def test_overlap_rejected_with_explicit_error() -> None:
    rows = [
        {
            "universe": "Russell3000",
            "security_id": "US:A:NYQ",
            "effective_from": "2026-01-01",
            "effective_to": "2026-01-31",
            "is_current": False,
        },
        {
            "universe": "Russell3000",
            "security_id": "US:A:NYQ",
            "effective_from": "2026-01-31",
            "effective_to": "",
            "is_current": True,
        },
    ]

    with pytest.raises(_scd2.ConstituentOverlapError, match="overlapping constituent windows detected"):
        _ = _scd2.membership_for_date(rows, universe="Russell3000", as_of_date="2026-01-31")


def test_month_end_cache_reproducible_and_pit_consistent() -> None:
    rows = _sample_rows()

    cache_one = _scd2.build_month_end_cache(rows, universe="Russell3000", start_date="2026-01-01", end_date="2026-03-31")
    cache_two = _scd2.build_month_end_cache(rows, universe="Russell3000", start_date="2026-01-01", end_date="2026-03-31")

    assert cache_one == cache_two
    assert cache_one[date(2026, 1, 31)] == _scd2.membership_for_date(rows, universe="Russell3000", as_of_date="2026-01-31")
    assert cache_one[date(2026, 2, 28)] == _scd2.membership_for_date(rows, universe="Russell3000", as_of_date="2026-02-28")
    assert cache_one[date(2026, 3, 31)] == _scd2.membership_for_date(rows, universe="Russell3000", as_of_date="2026-03-31")
