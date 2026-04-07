from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from typing import Protocol, cast


class _Holding(Protocol):
    security_id: str
    country: str
    weight: float
    rank_in_country: int | None
    factor_value: float | None


class _Diagnostics(Protocol):
    requested_holdings: int
    selected_holdings: int
    country_weights: tuple[tuple[str, float], ...]
    cash_weight: float


class _Result(Protocol):
    holdings: tuple[_Holding, ...]
    fallback_triggered: bool
    fallback_reasons: tuple[str, ...]
    jp_odd_lot_enabled: bool
    diagnostics: _Diagnostics


class _PortfolioModule(Protocol):
    def construct_portfolio_with_constraints(
        self,
        ranked_factor_rows: list[dict[str, object]],
        *,
        max_holdings: int = 20,
        max_single_name_weight: float = 0.08,
    ) -> _Result: ...


def _load_module() -> _PortfolioModule:
    module_path = Path(__file__).resolve().parents[1] / "src" / "portfolio" / "constructor.py"
    spec = spec_from_file_location("portfolio_constructor", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec from {module_path}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    typed_module = cast(_PortfolioModule, cast(object, module))
    return typed_module


_portfolio = _load_module()
construct_portfolio_with_constraints = _portfolio.construct_portfolio_with_constraints


def _make_row(*, security_id: str, country: str, rank: int, factor_value: float, is_eligible: bool = True) -> dict[str, object]:
    return {
        "security_id": security_id,
        "country": country,
        "factor_value": factor_value,
        "rank_in_country": rank,
        "is_eligible": is_eligible,
    }


def test_normal_feasible_case_selects_up_to_20_and_no_fallback() -> None:
    rows: list[dict[str, object]] = []
    for country in ("JP", "KR", "US"):
        for rank in range(1, 10):
            rows.append(_make_row(security_id=f"{country}:{rank:02d}", country=country, rank=rank, factor_value=1.0 / rank))

    result = construct_portfolio_with_constraints(rows)

    assert result.fallback_triggered is False
    assert result.jp_odd_lot_enabled is True
    assert len(result.holdings) == 20
    assert result.diagnostics.requested_holdings == 20
    assert result.diagnostics.selected_holdings == 20
    assert result.diagnostics.cash_weight == 0.0
    assert all(holding.weight >= 0.0 for holding in result.holdings)
    assert all(holding.weight <= 0.08 for holding in result.holdings)


def test_max_weight_enforced_when_country_capacity_is_tight() -> None:
    rows = [
        _make_row(security_id="JP:ONLY", country="JP", rank=1, factor_value=0.9),
        _make_row(security_id="KR:ONLY", country="KR", rank=1, factor_value=0.8),
        _make_row(security_id="US:ONLY", country="US", rank=1, factor_value=0.7),
    ]

    result = construct_portfolio_with_constraints(rows, max_holdings=20)

    assert all(holding.weight <= 0.08 for holding in result.holdings)
    assert result.fallback_triggered is True
    assert "COUNTRY_CAPACITY_BIND_JP" in result.fallback_reasons
    assert "COUNTRY_CAPACITY_BIND_KR" in result.fallback_reasons
    assert "COUNTRY_CAPACITY_BIND_US" in result.fallback_reasons
    assert result.diagnostics.cash_weight > 0.0


def test_country_sleeves_are_within_target_tolerance_when_feasible() -> None:
    rows: list[dict[str, object]] = []
    for rank in range(1, 8):
        rows.append(_make_row(security_id=f"JP:{rank:02d}", country="JP", rank=rank, factor_value=0.8 - rank * 0.01))
        rows.append(_make_row(security_id=f"KR:{rank:02d}", country="KR", rank=rank, factor_value=0.7 - rank * 0.01))
        rows.append(_make_row(security_id=f"US:{rank:02d}", country="US", rank=rank, factor_value=0.6 - rank * 0.01))

    result = construct_portfolio_with_constraints(rows)

    by_country = dict(result.diagnostics.country_weights)
    assert 0.3133333333333333 <= by_country["JP"] <= 0.35333333333333333
    assert 0.3133333333333333 <= by_country["KR"] <= 0.35333333333333333
    assert 0.3133333333333333 <= by_country["US"] <= 0.35333333333333333


def test_infeasible_universe_triggers_deterministic_fallback_metadata() -> None:
    rows = [
        _make_row(security_id="JP:AAA", country="JP", rank=1, factor_value=0.9),
        _make_row(security_id="JP:AAB", country="JP", rank=1, factor_value=0.9),
        _make_row(security_id="JP:BAA", country="JP", rank=2, factor_value=0.8),
        _make_row(security_id="JP:C00", country="JP", rank=3, factor_value=0.7),
        _make_row(security_id="JP:SKIP", country="JP", rank=4, factor_value=0.6, is_eligible=False),
    ]

    first = construct_portfolio_with_constraints(rows, max_holdings=20)
    second = construct_portfolio_with_constraints(rows, max_holdings=20)

    assert first == second
    assert first.fallback_triggered is True
    assert first.fallback_reasons == (
        "COUNTRY_CAPACITY_BIND_JP",
        "INSUFFICIENT_NAMES_JP",
        "INSUFFICIENT_NAMES_KR",
        "INSUFFICIENT_NAMES_US",
        "NO_SELECTED_NAMES_KR",
        "NO_SELECTED_NAMES_US",
        "TOTAL_UNDER_MAX_HOLDINGS",
        "UNALLOCATED_CASH_DUE_TO_CAPACITY",
    )

    selected_ids = [holding.security_id for holding in first.holdings]
    assert selected_ids == ["JP:AAA", "JP:AAB", "JP:BAA", "JP:C00"]
    assert all(holding.country == "JP" for holding in first.holdings)
    assert [(holding.rank_in_country, holding.factor_value) for holding in first.holdings] == [
        (1, 0.9),
        (1, 0.9),
        (2, 0.8),
        (3, 0.7),
    ]
