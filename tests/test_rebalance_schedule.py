from datetime import date
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Protocol, cast

from collections.abc import Iterable

import pytest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "core" / "calendar.py"
_SPEC = spec_from_file_location("calendar_module", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load calendar module from {_MODULE_PATH}")
_CALENDAR_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CALENDAR_MODULE)


class _GenerateQuarterlyRebalanceDates(Protocol):
    def __call__(
        self,
        market: str,
        start_date: date,
        end_date: date,
        available_trading_dates: Iterable[date],
        *,
        rebalance_anchor: str = "quarter_end",
    ) -> list[date]: ...


generate_quarterly_rebalance_dates = cast(
    _GenerateQuarterlyRebalanceDates,
    _CALENDAR_MODULE.generate_quarterly_rebalance_dates,
)


def test_generate_quarterly_rebalance_dates_us() -> None:
    trading_dates = [
        date(2024, 3, 28),
        date(2024, 3, 29),
        date(2024, 4, 1),
        date(2024, 6, 27),
        date(2024, 6, 28),
        date(2024, 7, 1),
        date(2024, 9, 27),
        date(2024, 9, 30),
        date(2024, 10, 1),
        date(2024, 12, 30),
        date(2024, 12, 31),
    ]

    schedule = generate_quarterly_rebalance_dates(
        market="US",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        available_trading_dates=trading_dates,
    )

    assert schedule == [
        date(2024, 3, 29),
        date(2024, 6, 28),
        date(2024, 9, 30),
        date(2024, 12, 31),
    ]


def test_generate_quarterly_rebalance_dates_kr() -> None:
    trading_dates = [
        date(2023, 3, 30),
        date(2023, 3, 31),
        date(2023, 6, 29),
        date(2023, 6, 30),
        date(2023, 9, 26),
        date(2023, 9, 27),
        date(2023, 10, 4),
        date(2023, 12, 27),
        date(2023, 12, 28),
        date(2023, 12, 29),
    ]

    schedule = generate_quarterly_rebalance_dates(
        market="KR",
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        available_trading_dates=trading_dates,
    )

    assert schedule == [
        date(2023, 3, 31),
        date(2023, 6, 30),
        date(2023, 9, 27),
        date(2023, 12, 29),
    ]


def test_generate_quarterly_rebalance_dates_jp() -> None:
    trading_dates = [
        date(2024, 3, 28),
        date(2024, 3, 29),
        date(2024, 4, 1),
        date(2024, 6, 28),
        date(2024, 7, 1),
        date(2024, 9, 30),
        date(2024, 10, 1),
        date(2024, 12, 27),
        date(2024, 12, 30),
    ]

    schedule = generate_quarterly_rebalance_dates(
        market="JP",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        available_trading_dates=trading_dates,
    )

    assert schedule == [
        date(2024, 3, 29),
        date(2024, 6, 28),
        date(2024, 9, 30),
        date(2024, 12, 30),
    ]


def test_generate_quarterly_rebalance_dates_rejects_unknown_market() -> None:
    with pytest.raises(ValueError, match="Unknown market"):
        _ = generate_quarterly_rebalance_dates(
            market="XX",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            available_trading_dates=[date(2024, 3, 29)],
        )


def test_generate_quarterly_rebalance_dates_rejects_invalid_window() -> None:
    with pytest.raises(ValueError, match="start_date"):
        _ = generate_quarterly_rebalance_dates(
            market="US",
            start_date=date(2024, 12, 31),
            end_date=date(2024, 1, 1),
            available_trading_dates=[date(2024, 3, 29)],
        )


def test_generate_quarterly_rebalance_dates_requires_trading_dates_in_window() -> None:
    with pytest.raises(ValueError, match="No available trading dates"):
        _ = generate_quarterly_rebalance_dates(
            market="US",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            available_trading_dates=[date(2025, 3, 31)],
        )


def test_generate_quarterly_rebalance_dates_start_date_anchor() -> None:
    trading_dates = [
        date(2024, 2, 1),
        date(2024, 5, 1),
        date(2024, 8, 1),
        date(2024, 11, 1),
    ]

    schedule = generate_quarterly_rebalance_dates(
        market="US",
        start_date=date(2024, 2, 1),
        end_date=date(2024, 12, 31),
        available_trading_dates=trading_dates,
        rebalance_anchor="start_date",
    )

    assert schedule == [
        date(2024, 2, 1),
        date(2024, 5, 1),
        date(2024, 8, 1),
        date(2024, 11, 1),
    ]


def test_generate_quarterly_rebalance_dates_start_anchor_clamps_month_end() -> None:
    trading_dates = [
        date(2024, 1, 31),
        date(2024, 4, 30),
        date(2024, 7, 31),
        date(2024, 10, 31),
    ]

    schedule = generate_quarterly_rebalance_dates(
        market="US",
        start_date=date(2024, 1, 31),
        end_date=date(2024, 12, 31),
        available_trading_dates=trading_dates,
        rebalance_anchor="start_date",
    )

    assert schedule == [
        date(2024, 1, 31),
        date(2024, 4, 30),
        date(2024, 7, 31),
        date(2024, 10, 31),
    ]
