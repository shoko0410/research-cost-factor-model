from __future__ import annotations

import importlib
import io
import json
import os
import sys
import types
import zipfile
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol, cast

import pytest


class _ConfigProtocol(Protocol):
    sec_user_agent: str
    dart_api_key: str
    edinet_api_key: str
    benchmark_us_ticker: str
    benchmark_kr_ticker: str
    benchmark_jp_ticker: str


class _ModuleProtocol(Protocol):
    LiveDataConfigError: type[Exception]

    def load_live_data_config_from_env(self) -> _ConfigProtocol: ...


_module = cast(_ModuleProtocol, cast(object, importlib.import_module("src.data.ingest.live_sources")))


def test_load_live_data_config_from_env_requires_required_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.delenv("DART_API_KEY", raising=False)
    monkeypatch.delenv("EDINET_API_KEY", raising=False)

    with pytest.raises(_module.LiveDataConfigError, match="missing required environment variables"):
        _ = _module.load_live_data_config_from_env()


def test_load_live_data_config_from_env_parses_values_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", '"Jungjae ao-buta@shoko.moe"')
    monkeypatch.setenv("DART_API_KEY", "'dart-key'")
    monkeypatch.setenv("EDINET_API_KEY", "'edinet-key'")
    monkeypatch.delenv("BENCH_US", raising=False)
    monkeypatch.delenv("BENCH_KR", raising=False)
    monkeypatch.delenv("BENCH_JP", raising=False)

    config = _module.load_live_data_config_from_env()

    assert config.sec_user_agent == "Jungjae ao-buta@shoko.moe"
    assert config.dart_api_key == "dart-key"
    assert config.edinet_api_key == "edinet-key"
    assert config.benchmark_us_ticker == "IWB"
    assert config.benchmark_kr_ticker == "069500.KS"
    assert config.benchmark_jp_ticker == "1306.T"


def test_to_float_rejects_non_finite_values() -> None:
    to_float = cast(Callable[[object], float | None], getattr(cast(object, _module), "_to_float"))

    assert to_float("nan") is None
    assert to_float("inf") is None
    assert to_float("-inf") is None
    assert to_float("123.4") == 123.4


@dataclass(frozen=True)
class _DummySpec:
    security_id: str = "US:BFB:NYQ"
    universe: str = "russell3000"
    country: str = "US"
    currency: str = "USD"
    ticker: str = "BF.B:NYQ"
    stock_code: str = "BF.B"
    stock_name: str = "Brown-Forman"


def test_security_price_ticker_normalizes_us_dot_class_shares() -> None:
    security_price_ticker = cast(
        Callable[[object], str],
        getattr(cast(object, _module), "_security_price_ticker"),
    )
    assert security_price_ticker(_DummySpec()) == "BF-B"


def test_normalized_stock_code_preserves_jp_alphanumeric_suffix() -> None:
    normalized_stock_code = cast(
        Callable[[str], str],
        getattr(cast(object, _module), "_normalized_stock_code"),
    )

    assert normalized_stock_code("285A") == "285A"
    assert normalized_stock_code("417A") == "417A"
    assert normalized_stock_code("285") == "0285"


def test_security_price_ticker_preserves_jp_alphanumeric_code() -> None:
    security_price_ticker = cast(
        Callable[[object], str],
        getattr(cast(object, _module), "_security_price_ticker"),
    )

    jp_spec = _DummySpec(
        security_id="JP:285A:TYO",
        universe="topix500",
        country="JP",
        currency="JPY",
        ticker="285A:TYO",
        stock_code="285A",
        stock_name="Kioxia Holdings Corp",
    )
    jp_numeric_spec = _DummySpec(
        security_id="JP:0285:TYO",
        universe="topix500",
        country="JP",
        currency="JPY",
        ticker="0285:TYO",
        stock_code="285",
        stock_name="Dummy JP",
    )

    assert security_price_ticker(jp_spec) == "285A.T"
    assert security_price_ticker(jp_numeric_spec) == "0285.T"


def test_extract_dart_rd_sales_supports_broader_rd_account_tokens() -> None:
    extract_dart_rd_sales = cast(
        Callable[[object], tuple[float | None, float | None]],
        getattr(cast(object, _module), "_extract_dart_rd_sales"),
    )

    rows = [
        {
            "account_id": "ifrs-full_CostOfSales",
            "account_nm": "매출원가",
            "thstrm_amount": "1800",
        },
        {
            "account_id": "dart_TotalResearchAndDevelopmentExpenses",
            "account_nm": "연구개발비",
            "thstrm_amount": "320",
        },
        {
            "account_id": "ifrs-full_Revenue",
            "account_nm": "매출액",
            "thstrm_amount": "5000",
        },
    ]

    rd_value, sales_value = extract_dart_rd_sales(rows)
    assert rd_value == 320.0
    assert sales_value == 5000.0


def test_extract_jp_rd_sales_reads_unmapped_fields_for_rd() -> None:
    extract_jp_rd_sales = cast(
        Callable[[object], tuple[float | None, float | None]],
        getattr(cast(object, _module), "_extract_jp_rd_sales"),
    )

    class _DummyReport:
        net_sales: float | None = None
        raw_fields: dict[str, str] = {
            "ifrs-full:CostOfSales": "1000",
            "ifrs-full:Revenue": "7000",
        }
        unmapped_fields: dict[str, str] = {
            "jpcrp_cor:ResearchAndDevelopmentExpenses": "420",
            "jpcrp_cor:ResearchAndDevelopmentActivitiesTextBlock": "ignore",
        }

    rd_value, sales_value = extract_jp_rd_sales(_DummyReport())
    assert rd_value == 420.0
    assert sales_value == 7000.0


def test_load_or_download_edinet_zip_bytes_prefers_cached_payload(tmp_path: Path) -> None:
    load_or_download_edinet_zip_bytes = cast(
        Callable[..., bytes | None],
        getattr(cast(object, _module), "_load_or_download_edinet_zip_bytes"),
    )
    edinet_download_cache_file = cast(
        Callable[..., Path],
        getattr(cast(object, _module), "_edinet_download_cache_file"),
    )

    cache_file = edinet_download_cache_file(cache_root=tmp_path, doc_id="S100TTJ3")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    expected = b"cached-zip"
    _ = cache_file.write_bytes(expected)

    class _ClientShouldNotBeCalled:
        def download_filing_raw(self, _: str) -> bytes:
            raise AssertionError("download_filing_raw should not be called when cache exists")

    payload = load_or_download_edinet_zip_bytes(
        client=_ClientShouldNotBeCalled(),
        cache_root=tmp_path,
        doc_id="S100TTJ3",
    )
    assert payload == expected


def test_extract_dart_rd_from_xbrl_bytes_prefers_current_year_consolidated() -> None:
    extract_dart_rd_from_xbrl_bytes = cast(
        Callable[..., float | None],
        getattr(cast(object, _module), "_extract_dart_rd_from_xbrl_bytes"),
    )

    xbrl_text = """<?xml version='1.0' encoding='UTF-8'?>
<xbrl>
  <OrdinaryDevelopmentExpenseSellingGeneralAdministrativeExpenses contextRef='PFY2023dFY_ifrs-full_ConsolidatedMember'>28339724000000</OrdinaryDevelopmentExpenseSellingGeneralAdministrativeExpenses>
  <OrdinaryDevelopmentExpenseSellingGeneralAdministrativeExpenses contextRef='CFY2024dFY_ifrs-full_SeparateMember'>30158017000000</OrdinaryDevelopmentExpenseSellingGeneralAdministrativeExpenses>
  <OrdinaryDevelopmentExpenseSellingGeneralAdministrativeExpenses contextRef='CFY2024dFY_ifrs-full_ConsolidatedMember'>34998142000000</OrdinaryDevelopmentExpenseSellingGeneralAdministrativeExpenses>
  <SellingGeneralAndAdministrativeExpenseExceptForResearchAndDevelopment contextRef='CFY2024dFY_ifrs-full_ConsolidatedMember'>46584532000000</SellingGeneralAndAdministrativeExpenseExceptForResearchAndDevelopment>
</xbrl>
"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("entity.xbrl", xbrl_text)

    rd_value = extract_dart_rd_from_xbrl_bytes(xbrl_zip_bytes=buffer.getvalue(), year=2024)
    assert rd_value == 34998142000000.0


def test_extract_dart_rd_from_xbrl_bytes_excludes_capitalised_rd_tags() -> None:
    extract_dart_rd_from_xbrl_bytes = cast(
        Callable[..., float | None],
        getattr(cast(object, _module), "_extract_dart_rd_from_xbrl_bytes"),
    )

    xbrl_text = """<?xml version='1.0' encoding='UTF-8'?>
<xbrl>
  <CapitalisedResearchAndDevelopmentExpenseForBioindustry contextRef='CFY2024dFY_ifrs-full_ConsolidatedMember'>4475076000000</CapitalisedResearchAndDevelopmentExpenseForBioindustry>
  <OrdinaryDevelopmentExpenseSellingGeneralAdministrativeExpenses contextRef='CFY2024dFY_ifrs-full_ConsolidatedMember'>310000000000</OrdinaryDevelopmentExpenseSellingGeneralAdministrativeExpenses>
</xbrl>
"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("entity.xbrl", xbrl_text)

    rd_value = extract_dart_rd_from_xbrl_bytes(xbrl_zip_bytes=buffer.getvalue(), year=2024)
    assert rd_value == 310000000000.0


def test_extract_dart_rd_from_xbrl_bytes_returns_none_when_only_capitalised_rd_exists() -> None:
    extract_dart_rd_from_xbrl_bytes = cast(
        Callable[..., float | None],
        getattr(cast(object, _module), "_extract_dart_rd_from_xbrl_bytes"),
    )

    xbrl_text = """<?xml version='1.0' encoding='UTF-8'?>
<xbrl>
  <CapitalisedResearchAndDevelopmentExpenseForBioindustry contextRef='CFY2024dFY_ifrs-full_ConsolidatedMember'>4475076000000</CapitalisedResearchAndDevelopmentExpenseForBioindustry>
</xbrl>
"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("entity.xbrl", xbrl_text)

    rd_value = extract_dart_rd_from_xbrl_bytes(xbrl_zip_bytes=buffer.getvalue(), year=2024)
    assert rd_value is None


def test_dart_xbrl_zip_bytes_returns_none_for_invalid_receipt_number(tmp_path: Path) -> None:
    dart_xbrl_zip_bytes = cast(
        Callable[..., bytes | None],
        getattr(cast(object, _module), "_dart_xbrl_zip_bytes"),
    )
    live_options_ctor = cast(Callable[[], object], getattr(cast(object, _module), "LiveIngestOptions"))

    class _DummyConfig:
        dart_api_key = "dummy"

    payload = dart_xbrl_zip_bytes(
        config=_DummyConfig(),
        cache_root=tmp_path,
        rcept_no="abc",
        options=live_options_ctor(),
    )
    assert payload is None


def test_edinet_daily_docs_merges_120_and_130_and_deduplicates(tmp_path: Path) -> None:
    edinet_daily_docs = cast(
        Callable[..., list[dict[str, object]]],
        getattr(cast(object, _module), "_edinet_daily_docs"),
    )
    live_options_ctor = cast(Callable[[], object], getattr(cast(object, _module), "LiveIngestOptions"))

    class _Client:
        def get_documents_by_date(self, day: date, doc_type: str) -> list[dict[str, object]]:
            assert day.isoformat() == "2025-06-25"
            if doc_type == "120":
                return [{"docID": "S100W4E8", "secCode": "68570"}]
            if doc_type == "130":
                return [
                    {"docID": "S100W4E8", "secCode": "68570"},
                    {"docID": "S100W4E9", "secCode": "68570"},
                ]
            return []

    rows = edinet_daily_docs(
        client=_Client(),
        cache_root=tmp_path,
        day=date(2025, 6, 25),
        options=live_options_ctor(),
    )
    doc_ids = sorted(str(row.get("docID", "")) for row in rows)
    assert doc_ids == ["S100W4E8", "S100W4E9"]


def test_jp_document_index_for_window_reuses_overlapping_days_and_preserves_order(tmp_path: Path) -> None:
    jp_document_index_for_window = cast(
        Callable[..., list[dict[str, object]]],
        getattr(cast(object, _module), "_jp_document_index_for_window"),
    )
    live_options_ctor = cast(Callable[[], object], getattr(cast(object, _module), "LiveIngestOptions"))

    calls: list[tuple[str, str]] = []

    class _Client:
        def get_documents_by_date(self, day: date, doc_type: str) -> list[dict[str, object]]:
            day_text = day.isoformat()
            calls.append((day_text, doc_type))
            doc_prefix = day.strftime("%Y%m%d")
            if doc_type == "120":
                return [{"docID": f"{doc_prefix}-A", "secCode": "68570"}]
            if doc_type == "130":
                return [
                    {"docID": f"{doc_prefix}-A", "secCode": "68570"},
                    {"docID": f"{doc_prefix}-B", "secCode": "68570"},
                ]
            return []

    first_rows = jp_document_index_for_window(
        client=_Client(),
        cache_root=tmp_path,
        start_day=date(2025, 6, 1),
        end_day=date(2025, 6, 3),
        options=live_options_ctor(),
    )
    second_rows = jp_document_index_for_window(
        client=_Client(),
        cache_root=tmp_path,
        start_day=date(2025, 6, 2),
        end_day=date(2025, 6, 4),
        options=live_options_ctor(),
    )

    first_doc_ids = [str(row.get("docID", "")) for row in first_rows]
    second_doc_ids = [str(row.get("docID", "")) for row in second_rows]
    assert first_doc_ids == [
        "20250601-A",
        "20250601-B",
        "20250602-A",
        "20250602-B",
        "20250603-A",
        "20250603-B",
    ]
    assert second_doc_ids == [
        "20250602-A",
        "20250602-B",
        "20250603-A",
        "20250603-B",
        "20250604-A",
        "20250604-B",
    ]

    day_call_counts = Counter(day_text for day_text, _ in calls)
    assert day_call_counts == Counter(
        {
            "2025-06-01": 2,
            "2025-06-02": 2,
            "2025-06-03": 2,
            "2025-06-04": 2,
        }
    )


def test_jp_documents_for_window_falls_back_to_day_scan_on_corrupted_index(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    jp_documents_for_window = cast(
        Callable[..., list[dict[str, object]]],
        getattr(cast(object, _module), "_jp_documents_for_window"),
    )
    live_options_ctor = cast(Callable[[], object], getattr(cast(object, _module), "LiveIngestOptions"))

    def _corrupted_index(**_: object) -> list[object]:
        return [123]

    day_calls: list[date] = []

    def _fake_daily_docs(**kwargs: object) -> list[dict[str, object]]:
        day = cast(date, kwargs["day"])
        day_calls.append(day)
        return [{"docID": day.strftime("%Y%m%d") + "-A", "secCode": "68570"}]

    monkeypatch.setattr(cast(object, _module), "_jp_document_index_for_window", _corrupted_index)
    monkeypatch.setattr(cast(object, _module), "_edinet_daily_docs", _fake_daily_docs)

    rows = jp_documents_for_window(
        client=object(),
        cache_root=tmp_path,
        start_day=date(2025, 7, 1),
        end_day=date(2025, 7, 3),
        options=live_options_ctor(),
    )

    assert [str(row.get("docID", "")) for row in rows] == ["20250701-A", "20250702-A", "20250703-A"]
    assert day_calls == [date(2025, 7, 1), date(2025, 7, 2), date(2025, 7, 3)]


def test_build_jp_fundamentals_uses_index_docs_with_key_field_parity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    build_jp_fundamentals = cast(
        Callable[..., list[dict[str, object]]],
        getattr(cast(object, _module), "_build_jp_fundamentals"),
    )
    live_options_ctor = cast(Callable[[], object], getattr(cast(object, _module), "LiveIngestOptions"))

    index_rows = [
        {
            "docID": "DOC-2",
            "secCode": "68570",
            "submitDateTime": "2025-08-02T09:00:00+09:00",
            "periodEnd": "2025-06-30",
        },
        {
            "docID": "DOC-1",
            "secCode": "68570",
            "submitDateTime": "2025-07-01T09:00:00+09:00",
            "periodEnd": "2025-03-31",
        },
        {
            "docID": "DOC-1",
            "secCode": "68570",
            "submitDateTime": "2025-07-01T09:01:00+09:00",
            "periodEnd": "2025-03-31",
        },
        {
            "docID": "DOC-X",
            "secCode": "99990",
            "submitDateTime": "2025-09-01T09:00:00+09:00",
            "periodEnd": "2025-06-30",
        },
    ]

    class _Report:
        def __init__(self, doc_id: str) -> None:
            if doc_id == "DOC-1":
                self.net_sales = 1000.0
                self.unmapped_fields = {"jpcrp_cor:ResearchAndDevelopmentExpenses": "110"}
            else:
                self.net_sales = 1200.0
                self.unmapped_fields = {"jpcrp_cor:ResearchAndDevelopmentExpenses": "140"}
            self.raw_fields = {}

    class _Document:
        def __init__(self, row: Mapping[str, object], client: object) -> None:
            _ = client
            self._doc_id = str(row.get("docID", ""))

        def parse(self) -> _Report:
            return _Report(self._doc_id)

    monkeypatch.setattr(cast(object, _module), "_edinet_client", lambda **_: object())
    monkeypatch.setattr(cast(object, _module), "_jp_documents_for_window", lambda **_: list(index_rows))
    monkeypatch.setattr(cast(object, _module), "_load_or_download_edinet_zip_bytes", lambda **_: None)
    monkeypatch.setattr(cast(object, _module), "parse_jp_edinet_fundamentals", lambda payload: (list(payload), []))

    monkeypatch.setitem(sys.modules, "edinet_tools", types.ModuleType("edinet_tools"))
    monkeypatch.setitem(sys.modules, "edinet_tools.document", types.SimpleNamespace(Document=_Document))
    monkeypatch.setitem(sys.modules, "edinet_tools.parsers", types.SimpleNamespace(parse=lambda document: _Report(document.doc_id)))

    class _Config:
        edinet_api_key = "dummy"

    records = build_jp_fundamentals(
        start=date(2025, 8, 1),
        end=date(2025, 8, 2),
        specs=[
            _DummySpec(
                security_id="JP:6857:TSE",
                universe="topix500",
                country="JP",
                currency="JPY",
                ticker="6857:TYO",
                stock_code="6857",
                stock_name="Dummy JP",
            )
        ],
        config=_Config(),
        cache_root=tmp_path,
        options=live_options_ctor(),
    )

    key_fields = [
        {
            "security_id": str(row.get("security_id", "")),
            "filing_date": str(row.get("filing_date", "")),
            "period_end": str(row.get("period_end", "")),
            "rd_expense": float(cast(float, row.get("rd_expense"))),
            "sales_ttm": float(cast(float, row.get("sales_ttm"))),
        }
        for row in records
    ]
    assert key_fields == [
        {
            "security_id": "JP:6857:TSE",
            "filing_date": "2025-07-01",
            "period_end": "2025-03-31",
            "rd_expense": 110.0,
            "sales_ttm": 1000.0,
        },
        {
            "security_id": "JP:6857:TSE",
            "filing_date": "2025-08-02",
            "period_end": "2025-06-30",
            "rd_expense": 140.0,
            "sales_ttm": 1200.0,
        },
    ]


def test_build_price_rows_filters_symbols_by_fundamental_security_ids(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    build_price_rows = cast(
        Callable[..., list[dict[str, object]]],
        getattr(cast(object, _module), "_build_price_rows"),
    )
    live_options_ctor = cast(Callable[[], object], getattr(cast(object, _module), "LiveIngestOptions"))

    captured: dict[str, list[str]] = {}

    def _fake_bulk_series(**kwargs: object) -> dict[str, dict[date, tuple[float, float]]]:
        tickers = cast(list[str], kwargs["tickers"])
        captured["tickers"] = list(tickers)
        return {ticker: {date(2020, 1, 1): (100.0, 1000.0)} for ticker in tickers}

    def _unexpected_single_series(*args: object, **kwargs: object) -> dict[date, tuple[float, float]]:
        raise AssertionError("single ticker fallback should not run in this test")

    monkeypatch.setattr(cast(object, _module), "_download_yfinance_bulk_series", _fake_bulk_series)
    monkeypatch.setattr(cast(object, _module), "_download_yfinance_series", _unexpected_single_series)

    specs = [
        _DummySpec(security_id="US:AAA:SP500", stock_code="AAA", ticker="AAA:SP500", stock_name="AAA"),
        _DummySpec(security_id="US:BBB:SP500", stock_code="BBB", ticker="BBB:SP500", stock_name="BBB"),
    ]

    rows = build_price_rows(
        schedule=[date(2025, 3, 31), date(2025, 6, 30)],
        specs=specs,
        cache_root=tmp_path,
        options=live_options_ctor(),
        included_security_ids={"US:AAA:SP500"},
    )

    assert captured["tickers"] == ["AAA"]
    assert rows
    assert {str(row["security_id"]) for row in rows} == {"US:AAA:SP500"}


def test_build_price_rows_uses_v1_fallback_then_v2_hit_without_schema_change(tmp_path: Path) -> None:
    build_price_rows = cast(
        Callable[..., list[dict[str, object]]],
        getattr(cast(object, _module), "_build_price_rows"),
    )
    yfinance_cache_file = cast(Callable[..., Path], getattr(cast(object, _module), "_yfinance_cache_file"))
    save_cached_yfinance_series = cast(
        Callable[..., None],
        getattr(cast(object, _module), "_save_cached_yfinance_series"),
    )
    get_stats = cast(Callable[[], dict[str, int]], getattr(cast(object, _module), "_get_yfinance_migration_stats"))
    reset_stats = cast(Callable[[], None], getattr(cast(object, _module), "_reset_yfinance_migration_stats"))
    live_options_ctor = cast(Callable[[], object], getattr(cast(object, _module), "LiveIngestOptions"))

    target_day = date(2025, 3, 31)
    required_days = [target_day - timedelta(days=14), target_day - timedelta(days=7), target_day]
    fetch_start = min(required_days) - timedelta(days=40)
    fetch_end = target_day
    cached_series = {day: (125.0 + idx, 1000.0 + idx) for idx, day in enumerate(required_days)}
    cache_file = yfinance_cache_file(
        cache_root=tmp_path,
        ticker="AAA",
        start=fetch_start,
        end=fetch_end,
    )
    save_cached_yfinance_series(
        cache_file=cache_file,
        ticker="AAA",
        start=fetch_start,
        end=fetch_end,
        series=cached_series,
    )

    specs = [_DummySpec(security_id="US:AAA:SP500", stock_code="AAA", ticker="AAA:SP500", stock_name="AAA")]

    reset_stats()
    first_rows = build_price_rows(
        schedule=[target_day],
        specs=specs,
        cache_root=tmp_path,
        options=live_options_ctor(),
    )
    first_stats = get_stats()

    second_rows = build_price_rows(
        schedule=[target_day],
        specs=specs,
        cache_root=tmp_path,
        options=live_options_ctor(),
    )
    second_stats = get_stats()

    assert first_rows
    assert all(set(row.keys()) == {"security_id", "country", "currency", "price_date", "close", "traded_value"} for row in first_rows)
    assert first_stats["v1_fallback"] == 1
    assert first_stats["v2_hit"] == 0
    assert first_stats["network_fetch"] == 0

    assert second_rows
    assert all(set(row.keys()) == {"security_id", "country", "currency", "price_date", "close", "traded_value"} for row in second_rows)
    assert second_stats["v1_fallback"] == 1
    assert second_stats["v2_hit"] == 1
    assert second_stats["network_fetch"] == 0


def test_build_fx_rows_uses_v2_cache_hits_and_preserves_schema(tmp_path: Path) -> None:
    build_fx_rows = cast(
        Callable[..., tuple[list[dict[str, object]], dict[str, dict[date, float]]]],
        getattr(cast(object, _module), "_build_fx_rows"),
    )
    save_cached_yfinance_series_v2 = cast(
        Callable[..., None],
        getattr(cast(object, _module), "_save_cached_yfinance_series_v2"),
    )
    get_stats = cast(Callable[[], dict[str, int]], getattr(cast(object, _module), "_get_yfinance_migration_stats"))
    reset_stats = cast(Callable[[], None], getattr(cast(object, _module), "_reset_yfinance_migration_stats"))
    live_options_ctor = cast(Callable[[], object], getattr(cast(object, _module), "LiveIngestOptions"))

    schedule = [date(2025, 5, 1), date(2025, 5, 2)]
    fetch_start = min(schedule) - timedelta(days=15)
    fetch_end = max(schedule)
    required_days = [fetch_start, schedule[0], schedule[1]]

    usd_krw_series = {required_days[0]: (1350.0, 0.0), required_days[1]: (1355.0, 0.0), required_days[2]: (1360.0, 0.0)}
    usd_jpy_series = {required_days[0]: (149.0, 0.0), required_days[1]: (150.0, 0.0), required_days[2]: (151.0, 0.0)}
    save_cached_yfinance_series_v2(
        cache_root=tmp_path,
        ticker="USDKRW=X",
        start=fetch_start,
        end=fetch_end,
        series=usd_krw_series,
    )
    save_cached_yfinance_series_v2(
        cache_root=tmp_path,
        ticker="USDJPY=X",
        start=fetch_start,
        end=fetch_end,
        series=usd_jpy_series,
    )

    reset_stats()
    rows, fx_map = build_fx_rows(schedule=schedule, cache_root=tmp_path, options=live_options_ctor())
    stats = get_stats()

    assert rows
    assert all(set(row.keys()) == {"pair", "fx_date", "rate"} for row in rows)
    assert fx_map.keys() == {"USD/KRW", "USD/JPY"}
    expected_fx_dates = {
        schedule[0] - timedelta(days=14),
        schedule[0] - timedelta(days=7),
        schedule[0],
        schedule[1] - timedelta(days=14),
        schedule[1] - timedelta(days=7),
        schedule[1],
    }
    assert set(fx_map["USD/KRW"].keys()) == expected_fx_dates
    assert set(fx_map["USD/JPY"].keys()) == expected_fx_dates
    assert stats["v2_hit"] == 2
    assert stats["v1_fallback"] == 0
    assert stats["network_fetch"] == 0


def test_build_benchmark_rows_uses_v1_fallback_and_repairs_v2_without_schema_change(tmp_path: Path) -> None:
    build_benchmark_rows = cast(
        Callable[..., list[dict[str, object]]],
        getattr(cast(object, _module), "_build_benchmark_rows"),
    )
    yfinance_cache_file = cast(Callable[..., Path], getattr(cast(object, _module), "_yfinance_cache_file"))
    save_cached_yfinance_series = cast(
        Callable[..., None],
        getattr(cast(object, _module), "_save_cached_yfinance_series"),
    )
    load_cached_yfinance_series_v2 = cast(
        Callable[..., dict[date, tuple[float, float]] | None],
        getattr(cast(object, _module), "_load_cached_yfinance_series_v2"),
    )
    get_stats = cast(Callable[[], dict[str, int]], getattr(cast(object, _module), "_get_yfinance_migration_stats"))
    reset_stats = cast(Callable[[], None], getattr(cast(object, _module), "_reset_yfinance_migration_stats"))
    live_options_ctor = cast(Callable[[], object], getattr(cast(object, _module), "LiveIngestOptions"))

    class _Config:
        benchmark_us_ticker = "IWB"

    schedule = [date(2025, 6, 2), date(2025, 6, 3)]
    fetch_start = min(schedule) - timedelta(days=15)
    fetch_end = max(schedule)
    v1_series = {fetch_start: (295.0, 0.0), schedule[0]: (297.0, 0.0), schedule[1]: (299.0, 0.0)}
    cache_file = yfinance_cache_file(cache_root=tmp_path, ticker="IWB", start=fetch_start, end=fetch_end)
    save_cached_yfinance_series(
        cache_file=cache_file,
        ticker="IWB",
        start=fetch_start,
        end=fetch_end,
        series=v1_series,
    )

    reset_stats()
    rows = build_benchmark_rows(
        schedule=schedule,
        config=_Config(),
        cache_root=tmp_path,
        options=live_options_ctor(),
    )
    repaired_v2 = load_cached_yfinance_series_v2(
        cache_root=tmp_path,
        ticker="IWB",
        start=fetch_start,
        end=fetch_end,
        cache_ttl_days=3,
    )
    stats = get_stats()

    assert rows
    assert all(set(row.keys()) == {"benchmark_date", "close", "currency"} for row in rows)
    assert all(str(row["currency"]) == "USD" for row in rows)
    assert repaired_v2 is not None
    assert stats["v2_hit"] == 0
    assert stats["v1_fallback"] == 1
    assert stats["network_fetch"] == 0
    assert stats["repair_success"] == 1


def test_yfinance_v2_partition_write_read_stitch_happy_path(tmp_path: Path) -> None:
    save_cached_yfinance_series_v2 = cast(
        Callable[..., None],
        getattr(cast(object, _module), "_save_cached_yfinance_series_v2"),
    )
    load_cached_yfinance_series_v2 = cast(
        Callable[..., dict[date, tuple[float, float]] | None],
        getattr(cast(object, _module), "_load_cached_yfinance_series_v2"),
    )

    series = {
        date(2025, 1, 3): (102.0, 1002.0),
        date(2024, 12, 31): (100.0, 1000.0),
        date(2025, 1, 2): (101.0, 1001.0),
    }
    save_cached_yfinance_series_v2(
        cache_root=tmp_path,
        ticker="AAPL",
        start=date(2024, 12, 30),
        end=date(2025, 1, 3),
        series=series,
    )

    loaded = load_cached_yfinance_series_v2(
        cache_root=tmp_path,
        ticker="AAPL",
        start=date(2024, 12, 31),
        end=date(2025, 1, 3),
        cache_ttl_days=3,
    )

    assert loaded == {
        date(2024, 12, 31): (100.0, 1000.0),
        date(2025, 1, 2): (101.0, 1001.0),
        date(2025, 1, 3): (102.0, 1002.0),
    }


def test_yfinance_v2_stitch_slice_across_year_boundary(tmp_path: Path) -> None:
    save_cached_yfinance_series_v2 = cast(
        Callable[..., None],
        getattr(cast(object, _module), "_save_cached_yfinance_series_v2"),
    )
    load_cached_yfinance_series_v2 = cast(
        Callable[..., dict[date, tuple[float, float]] | None],
        getattr(cast(object, _module), "_load_cached_yfinance_series_v2"),
    )

    save_cached_yfinance_series_v2(
        cache_root=tmp_path,
        ticker="AAPL",
        start=date(2024, 12, 29),
        end=date(2025, 1, 4),
        series={
            date(2024, 12, 29): (98.0, 980.0),
            date(2024, 12, 31): (100.0, 1000.0),
            date(2025, 1, 1): (101.0, 1100.0),
            date(2025, 1, 4): (104.0, 1400.0),
        },
    )

    loaded = load_cached_yfinance_series_v2(
        cache_root=tmp_path,
        ticker="AAPL",
        start=date(2024, 12, 31),
        end=date(2025, 1, 1),
        cache_ttl_days=3,
    )

    assert loaded == {
        date(2024, 12, 31): (100.0, 1000.0),
        date(2025, 1, 1): (101.0, 1100.0),
    }


def test_yfinance_v2_partition_ttl_expiry_returns_none_with_stale_mtime(tmp_path: Path) -> None:
    save_cached_yfinance_series_v2 = cast(
        Callable[..., None],
        getattr(cast(object, _module), "_save_cached_yfinance_series_v2"),
    )
    yfinance_cache_v2_partition_file = cast(
        Callable[..., Path],
        getattr(cast(object, _module), "_yfinance_cache_v2_partition_file"),
    )
    load_cached_yfinance_partition_series_v2 = cast(
        Callable[..., dict[date, tuple[float, float]] | None],
        getattr(cast(object, _module), "_load_cached_yfinance_partition_series_v2"),
    )

    save_cached_yfinance_series_v2(
        cache_root=tmp_path,
        ticker="IBM",
        start=date(2025, 1, 2),
        end=date(2025, 1, 3),
        series={
            date(2025, 1, 2): (220.0, 3000.0),
            date(2025, 1, 3): (221.0, 3100.0),
        },
    )

    partition_file = yfinance_cache_v2_partition_file(cache_root=tmp_path, ticker="IBM", year=2025)
    os.utime(partition_file, (1, 1))

    loaded = load_cached_yfinance_partition_series_v2(
        cache_file=partition_file,
        ticker="IBM",
        year=2025,
        cache_ttl_days=3,
    )
    assert loaded is None


def test_yfinance_v2_stitched_series_order_is_stable_across_repeated_runs(tmp_path: Path) -> None:
    save_cached_yfinance_series_v2 = cast(
        Callable[..., None],
        getattr(cast(object, _module), "_save_cached_yfinance_series_v2"),
    )
    load_cached_yfinance_series_v2 = cast(
        Callable[..., dict[date, tuple[float, float]] | None],
        getattr(cast(object, _module), "_load_cached_yfinance_series_v2"),
    )

    first_write = {
        date(2025, 1, 7): (107.0, 1700.0),
        date(2024, 12, 31): (99.0, 990.0),
        date(2025, 1, 3): (103.0, 1300.0),
    }
    second_write = {
        date(2025, 1, 3): (103.0, 1300.0),
        date(2025, 1, 7): (107.0, 1700.0),
        date(2024, 12, 31): (99.0, 990.0),
    }

    save_cached_yfinance_series_v2(
        cache_root=tmp_path,
        ticker="QQQ",
        start=date(2024, 12, 31),
        end=date(2025, 1, 7),
        series=first_write,
    )
    first_loaded = load_cached_yfinance_series_v2(
        cache_root=tmp_path,
        ticker="QQQ",
        start=date(2024, 12, 31),
        end=date(2025, 1, 7),
        cache_ttl_days=3,
    )

    save_cached_yfinance_series_v2(
        cache_root=tmp_path,
        ticker="QQQ",
        start=date(2024, 12, 31),
        end=date(2025, 1, 7),
        series=second_write,
    )
    second_loaded = load_cached_yfinance_series_v2(
        cache_root=tmp_path,
        ticker="QQQ",
        start=date(2024, 12, 31),
        end=date(2025, 1, 7),
        cache_ttl_days=3,
    )

    expected_order = [date(2024, 12, 31), date(2025, 1, 3), date(2025, 1, 7)]
    assert first_loaded is not None
    assert second_loaded is not None
    assert list(first_loaded.keys()) == expected_order
    assert list(second_loaded.keys()) == expected_order
    assert second_loaded == first_loaded


def test_yfinance_v2_partition_rows_and_stitched_series_are_deterministically_ordered(tmp_path: Path) -> None:
    save_cached_yfinance_series_v2 = cast(
        Callable[..., None],
        getattr(cast(object, _module), "_save_cached_yfinance_series_v2"),
    )
    load_cached_yfinance_series_v2 = cast(
        Callable[..., dict[date, tuple[float, float]] | None],
        getattr(cast(object, _module), "_load_cached_yfinance_series_v2"),
    )
    yfinance_cache_v2_partition_file = cast(
        Callable[..., Path],
        getattr(cast(object, _module), "_yfinance_cache_v2_partition_file"),
    )

    save_cached_yfinance_series_v2(
        cache_root=tmp_path,
        ticker="BRK.B",
        start=date(2025, 1, 1),
        end=date(2025, 1, 31),
        series={
            date(2025, 1, 10): (310.0, 1200.0),
            date(2025, 1, 3): (300.0, 1000.0),
            date(2025, 1, 7): (305.0, 1100.0),
        },
    )

    partition_file = yfinance_cache_v2_partition_file(cache_root=tmp_path, ticker="BRK.B", year=2025)
    payload = json.loads(partition_file.read_text(encoding="utf-8"))
    assert [row["date"] for row in payload["rows"]] == ["2025-01-03", "2025-01-07", "2025-01-10"]

    loaded = load_cached_yfinance_series_v2(
        cache_root=tmp_path,
        ticker="BRK.B",
        start=date(2025, 1, 1),
        end=date(2025, 1, 31),
        cache_ttl_days=3,
    )
    assert loaded is not None
    assert list(loaded.keys()) == [date(2025, 1, 3), date(2025, 1, 7), date(2025, 1, 10)]


def test_load_cached_yfinance_series_with_migration_dual_read_fallback_repairs_then_hits_v2(tmp_path: Path) -> None:
    yfinance_cache_file = cast(Callable[..., Path], getattr(cast(object, _module), "_yfinance_cache_file"))
    save_cached_yfinance_series = cast(
        Callable[..., None],
        getattr(cast(object, _module), "_save_cached_yfinance_series"),
    )
    load_cached_yfinance_series_v2 = cast(
        Callable[..., dict[date, tuple[float, float]] | None],
        getattr(cast(object, _module), "_load_cached_yfinance_series_v2"),
    )
    load_cached_yfinance_series_with_migration = cast(
        Callable[..., dict[date, tuple[float, float]] | None],
        getattr(cast(object, _module), "_load_cached_yfinance_series_with_migration"),
    )
    get_stats = cast(Callable[[], dict[str, int]], getattr(cast(object, _module), "_get_yfinance_migration_stats"))
    reset_stats = cast(Callable[[], None], getattr(cast(object, _module), "_reset_yfinance_migration_stats"))

    start = date(2025, 2, 3)
    end = date(2025, 2, 4)
    expected = {
        date(2025, 2, 3): (111.0, 2100.0),
        date(2025, 2, 4): (112.0, 2200.0),
    }
    cache_file = yfinance_cache_file(cache_root=tmp_path, ticker="MSFT", start=start, end=end)
    save_cached_yfinance_series(
        cache_file=cache_file,
        ticker="MSFT",
        start=start,
        end=end,
        series=expected,
    )

    reset_stats()
    first_loaded = load_cached_yfinance_series_with_migration(
        cache_root=tmp_path,
        ticker="MSFT",
        start=start,
        end=end,
        cache_ttl_days=3,
    )
    first_stats = get_stats()
    repaired_v2 = load_cached_yfinance_series_v2(
        cache_root=tmp_path,
        ticker="MSFT",
        start=start,
        end=end,
        cache_ttl_days=3,
    )

    second_loaded = load_cached_yfinance_series_with_migration(
        cache_root=tmp_path,
        ticker="MSFT",
        start=start,
        end=end,
        cache_ttl_days=3,
    )
    second_stats = get_stats()

    assert first_loaded == expected
    assert repaired_v2 == expected
    assert first_stats["v2_hit"] == 0
    assert first_stats["v1_fallback"] == 1
    assert first_stats["network_fetch"] == 0
    assert first_stats["repair_success"] == 1
    assert first_stats["repair_failure"] == 0

    assert second_loaded == expected
    assert second_stats["v2_hit"] == 1
    assert second_stats["v1_fallback"] == 1
    assert second_stats["network_fetch"] == 0
    assert second_stats["repair_success"] == 1
    assert second_stats["repair_failure"] == 0


def test_yfinance_v2_partition_year_mismatch_is_rejected(tmp_path: Path) -> None:
    yfinance_cache_v2_partition_file = cast(
        Callable[..., Path],
        getattr(cast(object, _module), "_yfinance_cache_v2_partition_file"),
    )
    load_cached_yfinance_partition_series_v2 = cast(
        Callable[..., dict[date, tuple[float, float]] | None],
        getattr(cast(object, _module), "_load_cached_yfinance_partition_series_v2"),
    )

    partition_file = yfinance_cache_v2_partition_file(cache_root=tmp_path, ticker="AAPL", year=2025)
    partition_file.parent.mkdir(parents=True, exist_ok=True)
    _ = partition_file.write_text(
        json.dumps(
            {
                "version": "v2",
                "provider": "yfinance",
                "ticker": "AAPL",
                "ticker_norm": "AAPL",
                "year": 2025,
                "generated_at_utc": "2026-02-19T08:00:00Z",
                "source_window": {"start": "2025-01-01", "end": "2025-12-31"},
                "rows": [{"date": "2024-12-31", "close": 100.0, "volume": 1000.0}],
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    loaded = load_cached_yfinance_partition_series_v2(
        cache_file=partition_file,
        ticker="AAPL",
        year=2025,
        cache_ttl_days=3,
    )
    assert loaded is None


def test_download_yfinance_series_prefers_v2_cache_hit(tmp_path: Path) -> None:
    save_cached_yfinance_series_v2 = cast(
        Callable[..., None],
        getattr(cast(object, _module), "_save_cached_yfinance_series_v2"),
    )
    download_yfinance_series = cast(
        Callable[..., dict[date, tuple[float, float]]],
        getattr(cast(object, _module), "_download_yfinance_series"),
    )
    get_stats = cast(Callable[[], dict[str, int]], getattr(cast(object, _module), "_get_yfinance_migration_stats"))
    reset_stats = cast(Callable[[], None], getattr(cast(object, _module), "_reset_yfinance_migration_stats"))

    expected = {
        date(2025, 1, 2): (100.0, 1000.0),
        date(2025, 1, 3): (101.0, 1100.0),
    }
    save_cached_yfinance_series_v2(
        cache_root=tmp_path,
        ticker="AAPL",
        start=date(2025, 1, 2),
        end=date(2025, 1, 3),
        series=expected,
    )

    reset_stats()
    loaded = download_yfinance_series(
        "AAPL",
        start=date(2025, 1, 2),
        end=date(2025, 1, 3),
        cache_root=tmp_path,
        cache_ttl_days=3,
    )

    stats = get_stats()
    assert loaded == expected
    assert stats["v2_hit"] == 1
    assert stats["v1_fallback"] == 0
    assert stats["network_fetch"] == 0
    assert stats["repair_success"] == 0
    assert stats["repair_failure"] == 0


def test_download_yfinance_series_uses_v1_fallback_and_repairs_v2(tmp_path: Path) -> None:
    yfinance_cache_file = cast(Callable[..., Path], getattr(cast(object, _module), "_yfinance_cache_file"))
    save_cached_yfinance_series = cast(
        Callable[..., None],
        getattr(cast(object, _module), "_save_cached_yfinance_series"),
    )
    load_cached_yfinance_series_v2 = cast(
        Callable[..., dict[date, tuple[float, float]] | None],
        getattr(cast(object, _module), "_load_cached_yfinance_series_v2"),
    )
    download_yfinance_series = cast(
        Callable[..., dict[date, tuple[float, float]]],
        getattr(cast(object, _module), "_download_yfinance_series"),
    )
    get_stats = cast(Callable[[], dict[str, int]], getattr(cast(object, _module), "_get_yfinance_migration_stats"))
    reset_stats = cast(Callable[[], None], getattr(cast(object, _module), "_reset_yfinance_migration_stats"))

    expected = {
        date(2025, 2, 3): (111.0, 2100.0),
        date(2025, 2, 4): (112.0, 2200.0),
    }
    cache_file = yfinance_cache_file(cache_root=tmp_path, ticker="MSFT", start=date(2025, 2, 3), end=date(2025, 2, 4))
    save_cached_yfinance_series(
        cache_file=cache_file,
        ticker="MSFT",
        start=date(2025, 2, 3),
        end=date(2025, 2, 4),
        series=expected,
    )

    reset_stats()
    loaded = download_yfinance_series(
        "MSFT",
        start=date(2025, 2, 3),
        end=date(2025, 2, 4),
        cache_root=tmp_path,
        cache_ttl_days=3,
    )
    repaired_v2 = load_cached_yfinance_series_v2(
        cache_root=tmp_path,
        ticker="MSFT",
        start=date(2025, 2, 3),
        end=date(2025, 2, 4),
        cache_ttl_days=3,
    )

    stats = get_stats()
    assert loaded == expected
    assert repaired_v2 == expected
    assert stats["v2_hit"] == 0
    assert stats["v1_fallback"] == 1
    assert stats["network_fetch"] == 0
    assert stats["repair_success"] == 1
    assert stats["repair_failure"] == 0


def test_download_yfinance_series_returns_v1_when_repair_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    yfinance_cache_file = cast(Callable[..., Path], getattr(cast(object, _module), "_yfinance_cache_file"))
    save_cached_yfinance_series = cast(
        Callable[..., None],
        getattr(cast(object, _module), "_save_cached_yfinance_series"),
    )
    download_yfinance_series = cast(
        Callable[..., dict[date, tuple[float, float]]],
        getattr(cast(object, _module), "_download_yfinance_series"),
    )
    get_stats = cast(Callable[[], dict[str, int]], getattr(cast(object, _module), "_get_yfinance_migration_stats"))
    reset_stats = cast(Callable[[], None], getattr(cast(object, _module), "_reset_yfinance_migration_stats"))

    expected = {date(2025, 3, 3): (99.0, 1500.0)}
    cache_file = yfinance_cache_file(cache_root=tmp_path, ticker="TSLA", start=date(2025, 3, 3), end=date(2025, 3, 3))
    save_cached_yfinance_series(
        cache_file=cache_file,
        ticker="TSLA",
        start=date(2025, 3, 3),
        end=date(2025, 3, 3),
        series=expected,
    )

    def _boom(**_: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(cast(object, _module), "_save_cached_yfinance_series_v2", _boom)

    reset_stats()
    loaded = download_yfinance_series(
        "TSLA",
        start=date(2025, 3, 3),
        end=date(2025, 3, 3),
        cache_root=tmp_path,
        cache_ttl_days=3,
    )

    stats = get_stats()
    assert loaded == expected
    assert stats["v2_hit"] == 0
    assert stats["v1_fallback"] == 1
    assert stats["network_fetch"] == 0
    assert stats["repair_success"] == 0
    assert stats["repair_failure"] == 1


def test_download_yfinance_bulk_series_network_fetch_writes_both_caches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    download_yfinance_bulk_series = cast(
        Callable[..., dict[str, dict[date, tuple[float, float]]]],
        getattr(cast(object, _module), "_download_yfinance_bulk_series"),
    )
    yfinance_cache_file = cast(Callable[..., Path], getattr(cast(object, _module), "_yfinance_cache_file"))
    load_cached_yfinance_series_v2 = cast(
        Callable[..., dict[date, tuple[float, float]] | None],
        getattr(cast(object, _module), "_load_cached_yfinance_series_v2"),
    )
    live_options_ctor = cast(Callable[[], object], getattr(cast(object, _module), "LiveIngestOptions"))
    get_stats = cast(Callable[[], dict[str, int]], getattr(cast(object, _module), "_get_yfinance_migration_stats"))
    reset_stats = cast(Callable[[], None], getattr(cast(object, _module), "_reset_yfinance_migration_stats"))

    expected = {
        date(2025, 4, 1): (201.0, 3100.0),
        date(2025, 4, 2): (202.0, 3200.0),
    }

    class _Frame:
        empty = False

    def _fake_download(**_: object) -> _Frame:
        return _Frame()

    def _fake_extract_series_from_frame(*, frame: object, ticker: str) -> dict[date, tuple[float, float]]:
        _ = frame
        if ticker == "NVDA":
            return expected
        return {}

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=_fake_download))
    monkeypatch.setattr(cast(object, _module), "_extract_series_from_frame", _fake_extract_series_from_frame)

    reset_stats()
    loaded = download_yfinance_bulk_series(
        tickers=["NVDA"],
        start=date(2025, 4, 1),
        end=date(2025, 4, 2),
        cache_root=tmp_path,
        options=live_options_ctor(),
    )

    v1_cache_file = yfinance_cache_file(cache_root=tmp_path, ticker="NVDA", start=date(2025, 4, 1), end=date(2025, 4, 2))
    v2_loaded = load_cached_yfinance_series_v2(
        cache_root=tmp_path,
        ticker="NVDA",
        start=date(2025, 4, 1),
        end=date(2025, 4, 2),
        cache_ttl_days=3,
    )
    stats = get_stats()

    assert loaded == {"NVDA": expected}
    assert v1_cache_file.exists()
    assert v2_loaded == expected
    assert stats["v2_hit"] == 0
    assert stats["v1_fallback"] == 0
    assert stats["network_fetch"] == 1
    assert stats["repair_success"] == 0
    assert stats["repair_failure"] == 0
