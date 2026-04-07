from __future__ import annotations

import importlib
from datetime import date
from collections.abc import Callable
from typing import Protocol, cast


class _SecuritySpec(Protocol):
    security_id: str
    country: str
    rank_seed: int


_cli_module = cast(object, importlib.import_module("src.cli.run_pipeline"))


def _synthetic_rd_sales(*, security_id: str, country: str, as_of: date) -> tuple[float, float]:
    fn = cast(
        Callable[..., tuple[float, float]],
        getattr(_cli_module, "_synthetic_rd_sales"),
    )
    return fn(security_id=security_id, country=country, as_of=as_of)


def _build_security_specs() -> tuple[_SecuritySpec, ...]:
    fn = cast(Callable[[], tuple[_SecuritySpec, ...]], getattr(_cli_module, "_build_security_specs"))
    return fn()


def _build_fundamentals(*, schedule: list[date], specs: tuple[_SecuritySpec, ...]) -> list[dict[str, object]]:
    fn = cast(
        Callable[..., list[dict[str, object]]],
        getattr(_cli_module, "_build_fundamentals"),
    )
    return fn(schedule=schedule, specs=specs)


def test_synthetic_rd_sales_is_deterministic_and_date_sensitive() -> None:
    security_id = "US:EXAMPLE"
    first = _synthetic_rd_sales(security_id=security_id, country="US", as_of=date(2025, 3, 31))
    second = _synthetic_rd_sales(security_id=security_id, country="US", as_of=date(2025, 3, 31))
    later = _synthetic_rd_sales(security_id=security_id, country="US", as_of=date(2025, 6, 30))

    assert first == second
    assert later != first


def test_synthetic_fundamentals_do_not_follow_rank_seed_ordering() -> None:
    specs = _build_security_specs()
    fundamentals = _build_fundamentals(schedule=[date(2025, 3, 31)], specs=specs)

    ratios: dict[str, float] = {}
    for row in fundamentals:
        security_id = str(row["security_id"])
        rd_expense = float(str(row["rd_expense"]))
        sales_ttm = float(str(row["sales_ttm"]))
        ratios[security_id] = rd_expense / sales_ttm

    for country in ("US", "KR", "JP"):
        country_specs = sorted(
            (spec for spec in specs if spec.country == country and spec.security_id in ratios),
            key=lambda spec: spec.rank_seed,
        )
        assert len(country_specs) >= 2
        ordered_by_seed = [spec.security_id for spec in country_specs]
        ordered_by_factor = [
            spec.security_id
            for spec in sorted(country_specs, key=lambda spec: (-ratios[spec.security_id], spec.security_id))
        ]
        assert ordered_by_factor != ordered_by_seed
