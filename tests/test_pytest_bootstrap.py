import pytest
from typing import Callable


FixtureLoader = Callable[..., list[dict[str, object]]]


def test_pytest_bootstrap_discovery():
    assert True


def test_market_fixtures_cover_us_kr_jp(
    market_records_by_code: dict[str, dict[str, object]],
):
    assert set(market_records_by_code) == {"US", "KR", "JP"}


def test_market_fixture_loader_rejects_missing_fields(
    market_fixture_loader: FixtureLoader,
):
    malformed_records = [
        {
            "market": "US",
            "security_id": "US_BAD_001",
            "currency": "USD",
            "rd_expense": 1,
        }
    ]

    with pytest.raises(
        ValueError,
        match=r"broken_market_fixture\[0\] is missing required fields:",
    ):
        _ = market_fixture_loader(malformed_records, fixture_name="broken_market_fixture")


def test_market_fixture_loader_rejects_unsupported_market(
    market_fixture_loader: FixtureLoader,
):
    malformed_records = [
        {
            "market": "CN",
            "security_id": "CN_BAD_001",
            "as_of_date": "2024-03-31",
            "currency": "CNY",
            "rd_expense": 100,
            "sales_ttm": 1000,
        }
    ]

    with pytest.raises(
        ValueError,
        match=r"broken_market_fixture\[0\] has unsupported market 'CN'",
    ):
        _ = market_fixture_loader(malformed_records, fixture_name="broken_market_fixture")
