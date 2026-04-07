from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from typing import Protocol, cast


class _FactorRow(Protocol):
    security_id: str
    country: str
    factor_value: float | None
    rank_in_country: int | None
    is_eligible: bool
    eligibility_reason: str


class _FactorModule(Protocol):
    REASON_ELIGIBLE: str
    REASON_MISSING_SALES_TTM: str
    REASON_NON_POSITIVE_SALES_TTM: str

    def compute_rnd_sales_ttm_factor(
        self,
        accepted_records: list[dict[str, object]],
        *,
        winsor_quantiles: tuple[float, float] | None = None,
    ) -> tuple[_FactorRow, ...]: ...


def _load_module() -> _FactorModule:
    module_path = Path(__file__).resolve().parents[1] / "src" / "factor" / "rnd_sales_ttm.py"
    spec = spec_from_file_location("rnd_sales_ttm", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec from {module_path}")

    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    typed_module = cast(_FactorModule, cast(object, module))
    return typed_module


_factor = _load_module()
REASON_ELIGIBLE = _factor.REASON_ELIGIBLE
REASON_MISSING_SALES_TTM = _factor.REASON_MISSING_SALES_TTM
REASON_NON_POSITIVE_SALES_TTM = _factor.REASON_NON_POSITIVE_SALES_TTM
compute_rnd_sales_ttm_factor = _factor.compute_rnd_sales_ttm_factor


def test_factor_values_and_deterministic_country_ranks() -> None:
    rows = compute_rnd_sales_ttm_factor(
        [
            {"security_id": "US:C", "country": "US", "rd_expense": 30.0, "sales_ttm": 200.0},
            {"security_id": "KR:Y", "country": "KR", "rd_expense": 1.0, "sales_ttm": 20.0},
            {"security_id": "US:B", "country": "US", "rd_expense": 15.0, "sales_ttm": 100.0},
            {"security_id": "US:A", "country": "US", "rd_expense": 10.0, "sales_ttm": 100.0},
            {"security_id": "KR:Z", "country": "KR", "rd_expense": 1.0, "sales_ttm": 10.0},
        ]
    )

    by_security = {row.security_id: row for row in rows}
    assert by_security["US:A"].factor_value == 0.1
    assert by_security["US:B"].factor_value == 0.15
    assert by_security["US:C"].factor_value == 0.15
    assert by_security["KR:Y"].factor_value == 0.05
    assert by_security["KR:Z"].factor_value == 0.1

    assert by_security["US:B"].rank_in_country == 1
    assert by_security["US:C"].rank_in_country == 2
    assert by_security["US:A"].rank_in_country == 3
    assert by_security["KR:Z"].rank_in_country == 1
    assert by_security["KR:Y"].rank_in_country == 2


def test_invalid_sales_values_are_flagged_and_excluded_from_ranking() -> None:
    rows = compute_rnd_sales_ttm_factor(
        [
            {"security_id": "US:VALID", "country": "US", "rd_expense": 20.0, "sales_ttm": 100.0},
            {"security_id": "US:ZERO", "country": "US", "rd_expense": 20.0, "sales_ttm": 0.0},
            {"security_id": "US:NEG", "country": "US", "rd_expense": 20.0, "sales_ttm": -50.0},
            {"security_id": "US:MISSING", "country": "US", "rd_expense": 20.0, "sales_ttm": None},
        ]
    )

    by_security = {row.security_id: row for row in rows}

    assert by_security["US:VALID"].is_eligible is True
    assert by_security["US:VALID"].eligibility_reason == REASON_ELIGIBLE
    assert by_security["US:VALID"].rank_in_country == 1

    assert by_security["US:ZERO"].is_eligible is False
    assert by_security["US:ZERO"].eligibility_reason == REASON_NON_POSITIVE_SALES_TTM
    assert by_security["US:ZERO"].factor_value is None
    assert by_security["US:ZERO"].rank_in_country is None

    assert by_security["US:NEG"].is_eligible is False
    assert by_security["US:NEG"].eligibility_reason == REASON_NON_POSITIVE_SALES_TTM
    assert by_security["US:NEG"].factor_value is None
    assert by_security["US:NEG"].rank_in_country is None

    assert by_security["US:MISSING"].is_eligible is False
    assert by_security["US:MISSING"].eligibility_reason == REASON_MISSING_SALES_TTM
    assert by_security["US:MISSING"].factor_value is None
    assert by_security["US:MISSING"].rank_in_country is None


def test_country_sleeve_ranking_is_independent() -> None:
    rows = compute_rnd_sales_ttm_factor(
        [
            {"security_id": "US:LOW", "country": "US", "rd_expense": 1.0, "sales_ttm": 100.0},
            {"security_id": "US:HIGH", "country": "US", "rd_expense": 3.0, "sales_ttm": 100.0},
            {"security_id": "KR:TOP", "country": "KR", "rd_expense": 500.0, "sales_ttm": 100.0},
            {"security_id": "KR:BOT", "country": "KR", "rd_expense": 100.0, "sales_ttm": 100.0},
        ]
    )

    by_security = {row.security_id: row for row in rows}
    assert by_security["US:HIGH"].rank_in_country == 1
    assert by_security["US:LOW"].rank_in_country == 2
    assert by_security["KR:TOP"].rank_in_country == 1
    assert by_security["KR:BOT"].rank_in_country == 2


def test_rank_and_eligibility_fields_form_consistent_row_contract() -> None:
    rows = compute_rnd_sales_ttm_factor(
        [
            {"security_id": "US:ELIGIBLE", "country": "US", "rd_expense": 30.0, "sales_ttm": 150.0},
            {"security_id": "US:INELIGIBLE", "country": "US", "rd_expense": 20.0, "sales_ttm": 0.0},
            {"security_id": "KR:ELIGIBLE", "country": "KR", "rd_expense": 10.0, "sales_ttm": 50.0},
            {"security_id": "KR:MISSING", "country": "KR", "rd_expense": 5.0, "sales_ttm": None},
        ]
    )

    for row in rows:
        if row.is_eligible:
            assert row.eligibility_reason == REASON_ELIGIBLE
            assert row.rank_in_country is not None
            assert row.factor_value is not None
        else:
            assert row.rank_in_country is None
            assert row.factor_value is None
