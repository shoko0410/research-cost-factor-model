from copy import deepcopy
from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Protocol, TypedDict, cast

import pytest


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


_MARKET_TO_CURRENCY = {
    "US": "USD",
    "KR": "KRW",
    "JP": "JPY",
}


class MarketRecord(TypedDict):
    market: str
    security_id: str
    as_of_date: str
    currency: str
    rd_expense: int | float
    sales_ttm: int | float

_REQUIRED_FIELDS = (
    "market",
    "security_id",
    "as_of_date",
    "currency",
    "rd_expense",
    "sales_ttm",
)


def _validate_market_record(
    record: object, index: int, fixture_name: str
) -> MarketRecord:
    if not isinstance(record, dict):
        raise TypeError(
            f"{fixture_name}[{index}] must be a dict, got {type(record).__name__}."
        )

    record_map = cast(Mapping[str, object], record)

    missing_fields = [field for field in _REQUIRED_FIELDS if field not in record_map]
    if missing_fields:
        missing_text = ", ".join(missing_fields)
        raise ValueError(
            f"{fixture_name}[{index}] is missing required fields: {missing_text}."
        )

    market = cast(str, record_map["market"])
    if market not in _MARKET_TO_CURRENCY:
        allowed = ", ".join(sorted(_MARKET_TO_CURRENCY))
        raise ValueError(
            f"{fixture_name}[{index}] has unsupported market '{market}'. Allowed values: {allowed}."
        )

    expected_currency = _MARKET_TO_CURRENCY[market]
    currency = cast(str, record_map["currency"])
    if currency != expected_currency:
        raise ValueError(
            f"{fixture_name}[{index}] has invalid currency '{currency}' for market '{market}'. Expected '{expected_currency}'."
        )

    if record_map["rd_expense"] is None:
        raise ValueError(
            f"{fixture_name}[{index}] has invalid rd_expense: None is not allowed."
        )

    sales_ttm = cast(int | float, record_map["sales_ttm"])
    if sales_ttm <= 0:
        raise ValueError(
            f"{fixture_name}[{index}] has invalid sales_ttm '{sales_ttm}': must be greater than 0."
        )

    return {
        "market": market,
        "security_id": cast(str, record_map["security_id"]),
        "as_of_date": cast(str, record_map["as_of_date"]),
        "currency": currency,
        "rd_expense": cast(int | float, record_map["rd_expense"]),
        "sales_ttm": sales_ttm,
    }


class FixtureLoader(Protocol):
    def __call__(
        self, records: object, fixture_name: str = "market_records"
    ) -> list[MarketRecord]: ...


@pytest.fixture
def market_fixture_loader() -> FixtureLoader:
    def _load(
        records: object, fixture_name: str = "market_records"
    ) -> list[MarketRecord]:
        if not isinstance(records, list):
            raise TypeError(
                f"{fixture_name} must be a list of dict records, got {type(records).__name__}."
            )

        records_list = cast(list[object], records)
        validated_records: list[MarketRecord] = []
        for index, record in enumerate(records_list):
            validated_records.append(
                _validate_market_record(record=record, index=index, fixture_name=fixture_name)
            )
        return validated_records

    return _load


@pytest.fixture
def market_sample_records(market_fixture_loader: FixtureLoader) -> list[MarketRecord]:
    records: list[object] = [
        {
            "market": "US",
            "security_id": "US_AAPL_001",
            "as_of_date": "2024-03-31",
            "currency": "USD",
            "rd_expense": 30000000000,
            "sales_ttm": 380000000000,
        },
        {
            "market": "KR",
            "security_id": "KR_005930_001",
            "as_of_date": "2024-03-31",
            "currency": "KRW",
            "rd_expense": 28900000000000,
            "sales_ttm": 258900000000000,
        },
        {
            "market": "JP",
            "security_id": "JP_7203_001",
            "as_of_date": "2024-03-31",
            "currency": "JPY",
            "rd_expense": 1240000000000,
            "sales_ttm": 45270000000000,
        },
    ]
    return market_fixture_loader(deepcopy(records), fixture_name="market_sample_records")


@pytest.fixture
def market_records_by_code(
    market_sample_records: list[MarketRecord],
) -> dict[str, MarketRecord]:
    return {record["market"]: record for record in market_sample_records}
