"""Deterministic walk-forward split and robustness execution helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class WalkForwardFold:
    fold_index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_dates: tuple[date, ...]
    test_dates: tuple[date, ...]


@dataclass(frozen=True)
class FoldOOSMetricsArtifact:
    fold_index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    oos_metrics: Mapping[str, object]


def _to_date(value: object, *, field_name: str) -> date:
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return date.fromisoformat(text)


def _add_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        if value.month == 2 and value.day == 29:
            return value.replace(year=value.year + years, day=28)
        raise


def _normalize_available_dates(
    available_dates: Iterable[date | str],
    *,
    start_date: date | None,
    end_date: date | None,
) -> tuple[date, ...]:
    normalized = tuple(sorted({_to_date(item, field_name="available_dates") for item in available_dates}))
    if not normalized:
        raise ValueError("available_dates cannot be empty")

    lower = normalized[0] if start_date is None else start_date
    upper = normalized[-1] if end_date is None else end_date
    if lower > upper:
        raise ValueError("start_date must be on or before end_date")

    in_window = tuple(item for item in normalized if lower <= item <= upper)
    if not in_window:
        raise ValueError("no available_dates in requested date window")
    return in_window


def assert_no_overlap_leakage(*, train_dates: Sequence[date], test_dates: Sequence[date]) -> None:
    train = tuple(sorted(train_dates))
    test = tuple(sorted(test_dates))
    if not train:
        raise ValueError("train_dates cannot be empty")
    if not test:
        raise ValueError("test_dates cannot be empty")

    overlap = sorted(set(train).intersection(test))
    if overlap:
        leaked = overlap[0].isoformat()
        raise ValueError(f"leakage detected: train/test overlap at {leaked}")

    if train[-1] >= test[0]:
        message = f"leakage detected: train_end must be strictly earlier than test_start (train_end={train[-1].isoformat()}, test_start={test[0].isoformat()})"
        raise ValueError(message)


def generate_walkforward_folds(
    available_dates: Iterable[date | str],
    *,
    train_years: int = 5,
    test_years: int = 1,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
) -> tuple[WalkForwardFold, ...]:
    """Generate deterministic rolling folds using explicit calendar boundaries."""

    if train_years <= 0:
        raise ValueError("train_years must be positive")
    if test_years <= 0:
        raise ValueError("test_years must be positive")

    start = _to_date(start_date, field_name="start_date") if start_date is not None else None
    end = _to_date(end_date, field_name="end_date") if end_date is not None else None
    dates = _normalize_available_dates(available_dates, start_date=start, end_date=end)

    fold_start = dates[0]
    last_date = dates[-1]
    folds: list[WalkForwardFold] = []

    while True:
        train_start = fold_start
        train_end = _add_years(train_start, train_years) - timedelta(days=1)
        test_start = train_end + timedelta(days=1)
        test_end = _add_years(test_start, test_years) - timedelta(days=1)

        if test_end > last_date:
            break

        train_dates = tuple(item for item in dates if train_start <= item <= train_end)
        test_dates = tuple(item for item in dates if test_start <= item <= test_end)

        if not train_dates or not test_dates:
            fold_start = _add_years(fold_start, test_years)
            continue

        assert_no_overlap_leakage(train_dates=train_dates, test_dates=test_dates)

        folds.append(
            WalkForwardFold(
                fold_index=len(folds),
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_dates=train_dates,
                test_dates=test_dates,
            )
        )

        fold_start = _add_years(fold_start, test_years)

    return tuple(folds)


def run_walkforward_robustness(
    folds: Sequence[WalkForwardFold],
    *,
    oos_metric_runner: Callable[[WalkForwardFold], Mapping[str, object]],
) -> tuple[FoldOOSMetricsArtifact, ...]:
    """Run per-fold OOS metrics and return in-memory artifacts."""

    artifacts: list[FoldOOSMetricsArtifact] = []
    for fold in folds:
        metrics = dict(oos_metric_runner(fold))
        artifacts.append(
            FoldOOSMetricsArtifact(
                fold_index=fold.fold_index,
                train_start=fold.train_start,
                train_end=fold.train_end,
                test_start=fold.test_start,
                test_end=fold.test_end,
                oos_metrics=metrics,
            )
        )
    return tuple(artifacts)


def execute_walkforward_robustness(
    available_dates: Iterable[date | str],
    *,
    oos_metric_runner: Callable[[WalkForwardFold], Mapping[str, object]],
    train_years: int = 5,
    test_years: int = 1,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
) -> tuple[FoldOOSMetricsArtifact, ...]:
    folds = generate_walkforward_folds(
        available_dates,
        train_years=train_years,
        test_years=test_years,
        start_date=start_date,
        end_date=end_date,
    )
    return run_walkforward_robustness(folds, oos_metric_runner=oos_metric_runner)


__all__ = [
    "FoldOOSMetricsArtifact",
    "WalkForwardFold",
    "assert_no_overlap_leakage",
    "execute_walkforward_robustness",
    "generate_walkforward_folds",
    "run_walkforward_robustness",
]
