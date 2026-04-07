from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from typing import Protocol, cast

import pytest


class _Fold(Protocol):
    fold_index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_dates: tuple[date, ...]
    test_dates: tuple[date, ...]


class _Artifact(Protocol):
    fold_index: int
    train_start: date
    test_end: date
    oos_metrics: Mapping[str, object]


class _WalkForwardModule(Protocol):
    def assert_no_overlap_leakage(self, *, train_dates: Sequence[date], test_dates: Sequence[date]) -> None: ...

    def generate_walkforward_folds(
        self,
        available_dates: Sequence[date],
        *,
        train_years: int = 5,
        test_years: int = 1,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
    ) -> tuple[_Fold, ...]: ...

    def run_walkforward_robustness(
        self,
        folds: Sequence[_Fold],
        *,
        oos_metric_runner: Callable[[_Fold], Mapping[str, object]],
    ) -> tuple[_Artifact, ...]: ...


_walkforward = cast(_WalkForwardModule, cast(object, importlib.import_module("src.validation.walkforward")))
assert_no_overlap_leakage = _walkforward.assert_no_overlap_leakage
generate_walkforward_folds = _walkforward.generate_walkforward_folds
run_walkforward_robustness = _walkforward.run_walkforward_robustness


def _boundary_dates(start_year: int, end_year: int) -> tuple[date, ...]:
    rows: list[date] = []
    for year in range(start_year, end_year + 1):
        rows.append(date(year, 1, 1))
        rows.append(date(year, 12, 31))
    return tuple(rows)


def test_generate_walkforward_folds_happy_path() -> None:
    available_dates = _boundary_dates(2000, 2015)

    folds = generate_walkforward_folds(
        available_dates,
        train_years=5,
        test_years=1,
    )

    assert len(folds) == 11
    first = folds[0]
    assert first.fold_index == 0
    assert first.train_start == date(2000, 1, 1)
    assert first.train_end == date(2004, 12, 31)
    assert first.test_start == date(2005, 1, 1)
    assert first.test_end == date(2005, 12, 31)
    assert first.train_dates[0] == date(2000, 1, 1)
    assert first.train_dates[-1] == date(2004, 12, 31)
    assert first.test_dates == (date(2005, 1, 1), date(2005, 12, 31))

    sensitivity_folds = generate_walkforward_folds(
        available_dates,
        train_years=5,
        test_years=1,
        start_date=date(2010, 1, 1),
        end_date=date(2015, 12, 31),
    )
    assert len(sensitivity_folds) == 1
    assert sensitivity_folds[0].test_start == date(2015, 1, 1)


def test_assert_no_overlap_leakage_raises_on_bad_split() -> None:
    with pytest.raises(ValueError, match="leakage detected"):
        assert_no_overlap_leakage(
            train_dates=(date(2024, 1, 1), date(2024, 6, 30)),
            test_dates=(date(2024, 6, 30), date(2024, 12, 31)),
        )


def test_run_walkforward_robustness_produces_fold_metrics_artifacts() -> None:
    available_dates = _boundary_dates(2010, 2018)
    folds = generate_walkforward_folds(available_dates, train_years=5, test_years=1)

    artifacts = run_walkforward_robustness(
        folds,
        oos_metric_runner=lambda fold: {
            "fold_label": f"fold-{fold.fold_index}",
            "oos_periods": len(fold.test_dates),
            "oos_return": 0.05,
        },
    )

    assert len(artifacts) == len(folds)
    first = artifacts[0]
    assert first.fold_index == 0
    assert first.train_start == date(2010, 1, 1)
    assert first.test_end == date(2015, 12, 31)
    assert "oos_return" in first.oos_metrics
    assert first.oos_metrics["oos_periods"] == 2
