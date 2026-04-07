"""Calendar utilities for deterministic quarterly rebalance scheduling.

Assumptions:
- Supported markets are limited to US, KR, and JP for v1.
- Trading dates are provided externally as available dates (no live exchange lookups).
- Rebalance cadence is quarterly only and each quarter target is adjusted to the
  most recent available trading date, with a forward fallback when needed.
"""

from __future__ import annotations

from calendar import monthrange
from bisect import bisect_right
from collections.abc import Iterable, Sequence
from datetime import date

SUPPORTED_MARKETS = ("US", "KR", "JP")
SUPPORTED_REBALANCE_ANCHORS = ("quarter_end", "start_date")


def _to_date(value: object) -> date:
    if not isinstance(value, date):
        raise TypeError(f"Expected datetime.date, got {type(value).__name__}")
    return value


def validate_market(market: str) -> str:
    normalized = market.upper()
    if normalized not in SUPPORTED_MARKETS:
        supported = ", ".join(SUPPORTED_MARKETS)
        raise ValueError(f"Unknown market '{market}'. Supported markets: {supported}")
    return normalized


def validate_rebalance_anchor(anchor: str) -> str:
    normalized = anchor.strip().lower()
    if normalized not in SUPPORTED_REBALANCE_ANCHORS:
        supported = ", ".join(SUPPORTED_REBALANCE_ANCHORS)
        raise ValueError(f"Unknown rebalance anchor '{anchor}'. Supported anchors: {supported}")
    return normalized


def _quarter_end_targets(start_date: date, end_date: date) -> list[date]:
    targets: list[date] = []
    year = start_date.year

    while year <= end_date.year:
        for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
            target = date(year, month, day)
            if start_date <= target <= end_date:
                targets.append(target)
        year += 1

    return targets


def _add_months(base: date, months: int, *, anchor_day: int | None = None) -> date:
    month_index = base.month - 1 + months
    year = base.year + month_index // 12
    month = month_index % 12 + 1
    target_day = base.day if anchor_day is None else anchor_day
    day = min(target_day, monthrange(year, month)[1])
    return date(year, month, day)


def _start_anchored_targets(start_date: date, end_date: date) -> list[date]:
    targets: list[date] = []
    months_offset = 0
    anchor_day = start_date.day
    while True:
        current = _add_months(start_date, months_offset, anchor_day=anchor_day)
        if current > end_date:
            break
        targets.append(current)
        months_offset += 3
    return targets


def _normalize_trading_dates(dates: Iterable[date], start_date: date, end_date: date) -> list[date]:
    normalized = sorted({_to_date(item) for item in dates})
    in_window = [item for item in normalized if start_date <= item <= end_date]

    if not in_window:
        raise ValueError("No available trading dates within the requested window")
    return in_window


def _adjust_to_available_trading_date(target: date, trading_dates: Sequence[date]) -> date:
    index = bisect_right(trading_dates, target)

    if index > 0:
        return trading_dates[index - 1]
    if trading_dates:
        return trading_dates[0]

    raise ValueError("Cannot adjust rebalance date without trading dates")


def generate_quarterly_rebalance_dates(
    market: str,
    start_date: date,
    end_date: date,
    available_trading_dates: Iterable[date],
    *,
    rebalance_anchor: str = "quarter_end",
) -> list[date]:
    """Generate quarterly rebalance dates adjusted to available market dates."""

    market = validate_market(market)
    _ = market
    anchor = validate_rebalance_anchor(rebalance_anchor)
    start = _to_date(start_date)
    end = _to_date(end_date)

    if start > end:
        raise ValueError("start_date must be on or before end_date")

    trading_dates = _normalize_trading_dates(available_trading_dates, start, end)
    quarter_targets = _quarter_end_targets(start, end) if anchor == "quarter_end" else _start_anchored_targets(start, end)

    adjusted = [
        _adjust_to_available_trading_date(target=target, trading_dates=trading_dates)
        for target in quarter_targets
    ]

    deduped: list[date] = []
    for item in adjusted:
        if not deduped or deduped[-1] != item:
            deduped.append(item)
    return deduped
