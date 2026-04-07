from __future__ import annotations

import importlib
import pytest
from collections.abc import Mapping, Sequence
from typing import Protocol, cast


class _FxModule(Protocol):
    DuplicateFxDateError: type[Exception]
    MissingFxRateError: type[Exception]
    UnsupportedCurrencyError: type[Exception]

    def convert_return_to_krw_base(
        self,
        *,
        local_return: float,
        currency: str,
        start_usd_krw: float,
        end_usd_krw: float,
        start_usd_jpy: float | None = None,
        end_usd_jpy: float | None = None,
    ) -> float: ...

    def decompose_return_contribution(self, *, local_return: float, fx_return: float) -> dict[str, float]: ...

    def detect_duplicate_fx_dates(self, rows: Sequence[Mapping[str, object]]) -> list[tuple[str, str]]: ...

    def fx_rate_on_date(
        self,
        rows: Sequence[Mapping[str, object]],
        *,
        pair: str,
        fx_date: str,
        missing_date_policy: str = "raise",
    ) -> float: ...

    def fx_return_to_krw(
        self,
        *,
        currency: str,
        start_usd_krw: float,
        end_usd_krw: float,
        start_usd_jpy: float | None = None,
        end_usd_jpy: float | None = None,
    ) -> float: ...

    def normalize_fx_rows(self, rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]: ...


_fx = cast(_FxModule, cast(object, importlib.import_module("src.data.ingest.fx")))
DuplicateFxDateError = _fx.DuplicateFxDateError
MissingFxRateError = _fx.MissingFxRateError
UnsupportedCurrencyError = _fx.UnsupportedCurrencyError
convert_return_to_krw_base = _fx.convert_return_to_krw_base
decompose_return_contribution = _fx.decompose_return_contribution
detect_duplicate_fx_dates = _fx.detect_duplicate_fx_dates
fx_rate_on_date = _fx.fx_rate_on_date
fx_return_to_krw = _fx.fx_return_to_krw
normalize_fx_rows = _fx.normalize_fx_rows


def test_normalize_fx_rows_canonicalizes_fields_with_temporal_defaults() -> None:
    normalized = normalize_fx_rows(
        [
            {"pair": "usd/krw", "fx_date": "2026-01-02", "rate": "1301.50"},
            {"pair": "usd/jpy", "as_of_date": "2026-01-02", "rate": 152.0},
        ]
    )

    assert [row["pair"] for row in normalized] == ["USD/JPY", "USD/KRW"]
    assert normalized[0]["fx_date"] == "2026-01-02"
    assert normalized[0]["as_of_date"] == "2026-01-02"
    assert normalized[0]["effective_from"] == "2026-01-02"
    assert normalized[0]["effective_to"] is None
    assert normalized[0]["is_current"] is True
    assert abs(float(str(normalized[1]["rate"])) - 1301.5) < 1e-12


def test_detect_duplicate_fx_dates_and_normalize_raise_on_collision() -> None:
    rows = [
        {"pair": "usd/krw", "fx_date": "2026-01-02", "rate": 1301.5},
        {"pair": "USD/KRW", "fx_date": "2026-01-02", "rate": 1302.0},
    ]

    assert detect_duplicate_fx_dates(rows) == [("USD/KRW", "2026-01-02")]
    with pytest.raises(DuplicateFxDateError, match="USD/KRW@2026-01-02"):
        _ = normalize_fx_rows(rows)


def test_fx_rate_on_date_missing_date_policy_paths() -> None:
    rows = normalize_fx_rows(
        [
            {"pair": "USD/KRW", "fx_date": "2026-01-02", "rate": 1300.0},
            {"pair": "USD/KRW", "fx_date": "2026-01-06", "rate": 1310.0},
        ]
    )

    assert abs(fx_rate_on_date(rows, pair="USD/KRW", fx_date="2026-01-06") - 1310.0) < 1e-12
    assert fx_rate_on_date(
        rows,
        pair="USD/KRW",
        fx_date="2026-01-05",
        missing_date_policy="previous",
    ) == 1300.0
    with pytest.raises(MissingFxRateError, match="missing rate"):
        _ = fx_rate_on_date(rows, pair="USD/KRW", fx_date="2026-01-05", missing_date_policy="raise")


def test_convert_return_to_krw_base_supports_usd_and_jpy_with_round_trip_check() -> None:
    local_return = 0.05
    start_usd_krw = 1300.0
    end_usd_krw = 1326.0

    fx_return_usd = fx_return_to_krw(currency="USD", start_usd_krw=start_usd_krw, end_usd_krw=end_usd_krw)
    krw_return_usd = convert_return_to_krw_base(
        local_return=local_return,
        currency="USD",
        start_usd_krw=start_usd_krw,
        end_usd_krw=end_usd_krw,
    )

    assert abs(fx_return_usd - 0.02) < 1e-12
    assert abs(krw_return_usd - ((1.05 * 1.02) - 1.0)) < 1e-12
    recovered_local = ((1.0 + krw_return_usd) / (1.0 + fx_return_usd)) - 1.0
    assert abs(recovered_local - local_return) < 1e-12

    fx_return_jpy = fx_return_to_krw(
        currency="JPY",
        start_usd_krw=1300.0,
        end_usd_krw=1300.0,
        start_usd_jpy=150.0,
        end_usd_jpy=155.0,
    )
    assert abs(fx_return_jpy - ((1300.0 / 155.0) / (1300.0 / 150.0) - 1.0)) < 1e-12


def test_decompose_return_contribution_produces_additive_components() -> None:
    result = decompose_return_contribution(local_return=0.08, fx_return=-0.02)

    assert abs(result["local"] - 0.08) < 1e-12
    assert abs(result["fx"] - (-0.02)) < 1e-12
    assert abs(result["interaction"] - (-0.0016)) < 1e-12
    assert abs(result["total"] - ((1.08 * 0.98) - 1.0)) < 1e-12


def test_convert_return_to_krw_base_rejects_unsupported_currency() -> None:
    with pytest.raises(UnsupportedCurrencyError, match="unsupported currency"):
        _ = convert_return_to_krw_base(
            local_return=0.01,
            currency="EUR",
            start_usd_krw=1300.0,
            end_usd_krw=1310.0,
        )
