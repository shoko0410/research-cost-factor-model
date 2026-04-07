"""CLI orchestration for deterministic end-to-end research pipeline."""

from __future__ import annotations

import argparse
from calendar import monthrange
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import cast

from ..backtest.engine import run_quarterly_backtest
from ..core.calendar import generate_quarterly_rebalance_dates
from ..data.ingest.benchmark import load_benchmark_series
from ..data.ingest.fx import normalize_fx_rows
from ..data.ingest.jp_edinet import parse_jp_edinet_fundamentals
from ..data.ingest.kr_dart import transform_kr_dart_fundamentals
from ..data.ingest import live_sources as live_sources_module
from ..data.ingest.live_sources import LiveIngestOptions, build_live_inputs, load_live_data_config_from_env
from ..data.ingest.us_sec import transform_us_sec_fundamentals
from ..factor.rnd_sales_ttm import compute_rnd_sales_ttm_factor
from ..features.pit_assembler import assemble_pit_universe
from ..portfolio.constructor import DEFAULT_COUNTRY_TARGETS, DEFAULT_MAX_HOLDINGS, construct_portfolio_with_constraints
from ..reporting.attribution import build_backtest_attribution_report, build_walkforward_attribution_report
from ..rust_bridge.contracts import (
    normalize_backtest_request,
    normalize_constructor_request,
    normalize_ranking_request,
)
from ..rust_bridge.dispatch import run_backtest_kernel, run_constructor_kernel, run_ranking_kernel
from ..rust_bridge.feature_flags import RustKernelFlags
from ..validation.walkforward import execute_walkforward_robustness

ROLLOUT_SUMMARY_ARTIFACT = "rollout_summary.json"

REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "holdings.csv",
    "trades.csv",
    "metrics.json",
    "data_quality_report.json",
    "factor_integrity_report.json",
    "qepm_alignment_report.json",
    "perf_telemetry.json",
    "perf_comparison_report.json",
    ROLLOUT_SUMMARY_ARTIFACT,
    "manifest.json",
)

PERF_TELEMETRY_SCHEMA_VERSION = "v1"
DEFAULT_PERF_BASELINE_SUMMARY = Path("outputs") / "perf_baseline" / "summary.csv"


@dataclass(frozen=True)
class SecuritySpec:
    security_id: str
    universe: str
    country: str
    currency: str
    ticker: str
    stock_code: str
    stock_name: str
    sector: str
    rank_seed: int


@dataclass(frozen=True)
class _FactorSeedRow:
    security_id: str
    country: str
    rd_expense: float | None
    sales_ttm: float | None
    factor_value: float | None
    rank_in_country: int | None
    is_eligible: bool
    eligibility_reason: str


def _parse_iso_date(value: str, *, field_name: str) -> date:
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return date.fromisoformat(text)


def _parse_env_line(raw_line: str) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[7:].strip()
    if "=" not in line:
        return None

    key_text, value_text = line.split("=", 1)
    key = key_text.strip()
    if not key:
        return None

    value = value_text.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    elif " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return key, value


def _load_env_file(env_file: Path) -> set[str]:
    if not env_file.exists() or not env_file.is_file():
        return set()

    loaded_keys: set[str] = set()
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        current = os.getenv(key)
        if current is not None and current.strip():
            continue
        os.environ[key] = value
        loaded_keys.add(key)
    return loaded_keys


def _serialize(value: object) -> object:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, tuple):
        items = cast(tuple[object, ...], value)
        return [_serialize(item) for item in items]
    if isinstance(value, list):
        items = cast(list[object], value)
        return [_serialize(item) for item in items]
    if isinstance(value, dict):
        items = cast(dict[object, object], value)
        return {str(key): _serialize(item) for key, item in items.items()}
    return value


def _count_non_finite_numbers(value: object) -> int:
    if isinstance(value, float):
        return 0 if math.isfinite(value) else 1
    if isinstance(value, int):
        return 0
    if isinstance(value, list):
        return sum(_count_non_finite_numbers(item) for item in value)
    if isinstance(value, tuple):
        return sum(_count_non_finite_numbers(item) for item in value)
    if isinstance(value, dict):
        return sum(_count_non_finite_numbers(item) for item in value.values())
    return 0


def _get_record_field(record: object, field_name: str, default: object = None) -> object:
    if isinstance(record, Mapping):
        return record.get(field_name, default)
    return getattr(record, field_name, default)


def make_run_id(start: date, end: date) -> str:
    return f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"


def _quarter_targets(start: date, end: date) -> list[date]:
    return _quarter_targets_with_anchor(start=start, end=end, rebalance_anchor="quarter_end")


def _add_months(base: date, months: int, *, anchor_day: int | None = None) -> date:
    month_index = base.month - 1 + months
    year = base.year + month_index // 12
    month = month_index % 12 + 1
    target_day = base.day if anchor_day is None else anchor_day
    day = min(target_day, monthrange(year, month)[1])
    return date(year, month, day)


def _quarter_targets_with_anchor(*, start: date, end: date, rebalance_anchor: str) -> list[date]:
    anchor = rebalance_anchor.strip().lower()
    if anchor == "quarter_end":
        values: list[date] = []
        for year in range(start.year, end.year + 1):
            for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
                target = date(year, month, day)
                if start <= target <= end:
                    values.append(target)
        return values
    if anchor == "start_date":
        values: list[date] = []
        months_offset = 0
        anchor_day = start.day
        while True:
            current = _add_months(start, months_offset, anchor_day=anchor_day)
            if current > end:
                break
            values.append(current)
            months_offset += 3
        return values
    raise ValueError("--rebalance-anchor must be one of: quarter_end, start_date")


def _month_end_targets(*, start: date, end: date) -> list[date]:
    if start > end:
        return []
    values: list[date] = []
    current = date(start.year, start.month, 1)
    while current <= end:
        day = monthrange(current.year, current.month)[1]
        month_end = date(current.year, current.month, day)
        if start <= month_end <= end:
            values.append(month_end)
        current = _add_months(current, 1, anchor_day=1)
    return values


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _stable_hash_int(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _synthetic_rd_sales(
    *,
    security_id: str,
    country: str,
    as_of: date,
) -> tuple[float, float]:
    seed = _stable_hash_int(f"{security_id}|{country}|{as_of.isoformat()}|rnd-sales")
    country_adjust = {"US": 0.0, "KR": 12.5, "JP": 7.5}.get(country.upper(), 0.0)
    rd_expense = 60.0 + country_adjust + float(seed % 180)
    sales_ttm = 700.0 + float((seed // 97) % 2200)
    return rd_expense, sales_ttm


def _load_market_specs(csv_file: Path, *, expected_country: str, currency: str) -> tuple[SecuritySpec, ...]:
    rows: list[SecuritySpec] = []
    with csv_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            country = str(raw.get("country", "")).strip().upper()
            if country != expected_country:
                continue
            if str(raw.get("is_current", "")).strip().lower() not in {"true", "1", "yes"}:
                continue
            security_id = str(raw.get("security_id", "")).strip()
            universe = str(raw.get("universe", "")).strip().lower()
            ticker = str(raw.get("ticker", "")).strip()
            stock_code = str(raw.get("stock_code", "")).strip()
            stock_name = str(raw.get("stock_name", "")).strip()
            sector = str(raw.get("icb_proxy_level1", "")).strip() or str(raw.get("native_taxonomy_label", "")).strip() or "UNKNOWN"
            if not security_id or not universe:
                continue
            rows.append(
                SecuritySpec(
                    security_id=security_id,
                    universe=universe,
                    country=expected_country,
                    currency=currency,
                    ticker=ticker,
                    stock_code=stock_code,
                    stock_name=stock_name,
                    sector=sector,
                    rank_seed=len(rows) + 1,
                )
            )
    if not rows:
        raise ValueError(f"no current symbols found in {csv_file.name} for {expected_country}")
    return tuple(rows)


def _load_sp500_symbols(snapshot_file: Path) -> set[str]:
    if not snapshot_file.exists():
        raise ValueError(f"missing S&P500 snapshot: {snapshot_file}")
    with snapshot_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        symbols = {
            str(row.get("symbol", "")).strip().upper().replace("-", ".")
            for row in reader
            if str(row.get("symbol", "")).strip()
        }
    if not symbols:
        raise ValueError("S&P500 snapshot has no symbols")
    return symbols


def _load_nikkei225_codes(snapshot_file: Path) -> set[str]:
    if not snapshot_file.exists():
        raise ValueError(f"missing Nikkei225 snapshot: {snapshot_file}")
    with snapshot_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        codes = {
            str(row.get("stock_code", "")).strip().upper()
            for row in reader
            if str(row.get("stock_code", "")).strip()
        }
    if not codes:
        raise ValueError("Nikkei225 snapshot has no stock codes")
    return codes


def _remap_universe(specs: tuple[SecuritySpec, ...], *, new_universe: str) -> tuple[SecuritySpec, ...]:
    return tuple(
        SecuritySpec(
            security_id=spec.security_id,
            universe=new_universe,
            country=spec.country,
            currency=spec.currency,
            ticker=spec.ticker,
            stock_code=spec.stock_code,
            stock_name=spec.stock_name,
            sector=spec.sector,
            rank_seed=spec.rank_seed,
        )
        for spec in specs
    )


def _filter_to_core_indices(
    *,
    us_specs: tuple[SecuritySpec, ...],
    kr_specs: tuple[SecuritySpec, ...],
    jp_specs: tuple[SecuritySpec, ...],
    root: Path,
) -> tuple[SecuritySpec, ...]:
    sp500_symbols = _load_sp500_symbols(root / "data" / "universe" / "sp500_symbols.csv")
    nikkei_codes = _load_nikkei225_codes(root / "data" / "universe" / "nikkei225_codes.csv")

    us_filtered = [
        spec
        for spec in us_specs
        if spec.stock_code.strip().upper().replace("-", ".") in sp500_symbols
    ]
    kr_filtered = [spec for spec in kr_specs if spec.universe == "kospi200"]
    jp_filtered = [
        spec
        for spec in jp_specs
        if spec.stock_code.strip().upper() in nikkei_codes
    ]

    if not us_filtered:
        raise ValueError("core_indices profile produced zero US names")
    if not kr_filtered:
        raise ValueError("core_indices profile produced zero KR names")
    if not jp_filtered:
        raise ValueError("core_indices profile produced zero JP names")

    return (
        *_remap_universe(tuple(us_filtered), new_universe="sp500"),
        *_remap_universe(tuple(kr_filtered), new_universe="kospi200"),
        *_remap_universe(tuple(jp_filtered), new_universe="nikkei225"),
    )


def _build_security_specs(*, universe_profile: str = "broad") -> tuple[SecuritySpec, ...]:
    root = _project_root()
    us_specs = _load_market_specs(
        root / "us_sector_current.csv",
        expected_country="US",
        currency="USD",
    )
    kr_specs = _load_market_specs(
        root / "kr_sector_current.csv",
        expected_country="KR",
        currency="KRW",
    )
    jp_specs = _load_market_specs(
        root / "jp_sector_current.csv",
        expected_country="JP",
        currency="JPY",
    )
    if universe_profile == "core_indices":
        return _filter_to_core_indices(us_specs=us_specs, kr_specs=kr_specs, jp_specs=jp_specs, root=root)
    return (*us_specs, *kr_specs, *jp_specs)


def _security_metadata(specs: tuple[SecuritySpec, ...]) -> dict[str, dict[str, str]]:
    return {
        spec.security_id: {
            "ticker": spec.ticker,
            "stock_code": spec.stock_code,
            "stock_name": spec.stock_name,
        }
        for spec in specs
    }


def _build_constituent_rows(specs: tuple[SecuritySpec, ...], start: date) -> list[dict[str, object]]:
    effective_from = (start - timedelta(days=365)).isoformat()
    return [
        {
            "universe": spec.universe,
            "security_id": spec.security_id,
            "effective_from": effective_from,
            "effective_to": "",
            "is_current": True,
        }
        for spec in specs
    ]


def _build_fundamentals(
    *,
    schedule: list[date],
    specs: tuple[SecuritySpec, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    kr_rows: list[dict[str, object]] = []
    jp_payload: list[dict[str, object]] = []
    corp_map: dict[str, str] = {}

    for as_of in schedule:
        for spec in specs:
            if spec.country == "US":
                filing_date = as_of - timedelta(days=60)
                period_end = as_of - timedelta(days=150)
                rd, sales = _synthetic_rd_sales(
                    security_id=spec.security_id,
                    country=spec.country,
                    as_of=as_of,
                )
                transformed, _, _ = transform_us_sec_fundamentals(
                    security_id=spec.security_id,
                    cik=f"{spec.rank_seed}",
                    payload_rows=[
                        {
                            "period_end": period_end.isoformat(),
                            "filing_date": filing_date.isoformat(),
                            "rd_expense": rd,
                            "sales_ttm": sales,
                        }
                    ],
                    country="US",
                )
                rows.extend(transformed)
            elif spec.country == "KR":
                filing_date = as_of - timedelta(days=75)
                period_end = as_of - timedelta(days=170)
                corp_code = f"C{spec.rank_seed:04d}"
                stock_code = spec.stock_code
                corp_map[corp_code] = stock_code
                rd, sales = _synthetic_rd_sales(
                    security_id=spec.security_id,
                    country=spec.country,
                    as_of=as_of,
                )
                kr_rows.append(
                    {
                        "corp_code": corp_code,
                        "stock_code": stock_code,
                        "filing_date": filing_date.isoformat(),
                        "period_end": period_end.isoformat(),
                        "rd_expense": rd,
                        "sales_ttm": sales,
                    }
                )
            else:
                filing_date = as_of - timedelta(days=75)
                period_end = as_of - timedelta(days=170)
                rd, sales = _synthetic_rd_sales(
                    security_id=spec.security_id,
                    country=spec.country,
                    as_of=as_of,
                )
                jp_payload.append(
                    {
                        "security_id": spec.security_id,
                        "jp_code": spec.stock_code,
                        "filing_date": filing_date.isoformat(),
                        "period_end": period_end.isoformat(),
                        "rd_expense": rd,
                        "sales_ttm": sales,
                    }
                )

    rows.extend(transform_kr_dart_fundamentals(kr_rows, corp_code_to_stock_code=corp_map, strict_unmapped_corp_code=False))
    jp_records, failures = parse_jp_edinet_fundamentals(jp_payload)
    if failures:
        raise ValueError(f"jp fundamental parse failures: {len(failures)}")
    rows.extend(jp_records)
    return rows


def _build_price_rows(
    *,
    schedule: list[date],
    specs: tuple[SecuritySpec, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    all_dates = sorted({*(schedule), *(day - timedelta(days=7) for day in schedule), *(day - timedelta(days=14) for day in schedule)})
    base_by_country = {"US": 25.0, "KR": 20_000.0, "JP": 2_500.0}

    for spec in specs:
        base = base_by_country[spec.country] + float(spec.rank_seed)
        for day in all_dates:
            quarter_index = max(0, schedule.index(min((item for item in schedule if item >= day), default=schedule[-1])))
            growth = 1.0 + (0.015 * quarter_index)
            discount = 0.99 if day not in schedule else 1.0
            close = round(base * growth * discount, 6)
            traded_value = 800_000_000.0 + float(spec.rank_seed * 25_000_000.0)
            rows.append(
                {
                    "security_id": spec.security_id,
                    "country": spec.country,
                    "currency": spec.currency,
                    "price_date": day,
                    "close": close,
                    "traded_value": traded_value,
                }
            )
    return rows


def _build_fx_rates(schedule: list[date]) -> tuple[list[dict[str, str | date | float]], dict[str, dict[date, float]]]:
    all_dates = sorted({*(schedule), *(day - timedelta(days=7) for day in schedule), *(day - timedelta(days=14) for day in schedule)})
    raw_rows: list[dict[str, object]] = []
    mapping: dict[str, dict[date, float]] = {"USD/KRW": {}, "USD/JPY": {}}

    for index, day in enumerate(all_dates):
        usd_krw = 1300.0 + float(index)
        usd_jpy = 150.0 - float(index) * 0.1
        mapping["USD/KRW"][day] = usd_krw
        mapping["USD/JPY"][day] = usd_jpy
        raw_rows.append({"pair": "USD/KRW", "fx_date": day.isoformat(), "rate": usd_krw})
        raw_rows.append({"pair": "USD/JPY", "fx_date": day.isoformat(), "rate": usd_jpy})

    normalized = normalize_fx_rows(raw_rows)
    normalized_rows = [
        {
            "pair": str(row["pair"]),
            "fx_date": date.fromisoformat(str(row["fx_date"])),
            "rate": float(str(row["rate"])),
        }
        for row in normalized
    ]
    return normalized_rows, mapping


def _build_benchmark_series(schedule: list[date]) -> list[dict[str, object]]:
    official = [
        {"as_of_date": day.isoformat(), "level": 1000.0 + float(index) * 22.5}
        for index, day in enumerate(schedule)
    ]
    selected = load_benchmark_series(official_series=official, proxy_series=None)
    return [
        {"benchmark_date": point.as_of_date, "close": point.level, "currency": "USD"}
        for point in selected.series
    ]


def _build_allocations(
    *,
    schedule: list[date],
    constituent_rows: list[dict[str, object]],
    fundamental_rows: list[dict[str, object]],
    price_rows: list[dict[str, object]],
    fx_rate_map: dict[str, dict[date, float]],
    sector_by_security: dict[str, str],
    factor_model: str,
    sector_active_band: float,
    use_size_stratification: bool,
    qepm_te_active_l2_cap: float,
    qepm_alpha_tilt_strength: float,
    qepm_max_adv_participation: float,
    qepm_max_turnover: float,
    portfolio_value_krw: float,
    qepm_staggered_sleeves: bool,
    qepm_sleeve_count: int,
    country_targets: tuple[tuple[str, float], ...] | None = None,
    rust_kernel_flags: RustKernelFlags | None = None,
) -> tuple[dict[date, list[dict[str, object]]], int, int, list[dict[str, object]]]:
    allocations: dict[date, list[dict[str, object]]] = {}
    total_rejected = 0
    fallback_count = 0
    factor_integrity_rows: list[dict[str, object]] = []
    universes = tuple(sorted({str(row.get("universe", "")).strip().lower() for row in constituent_rows if row.get("universe")}))
    if not universes:
        raise ValueError("constituent_rows must contain at least one universe")

    if country_targets is None:
        effective_country_targets = tuple(DEFAULT_COUNTRY_TARGETS)
    else:
        effective_country_targets = country_targets
    target_weights = {country: float(weight) for country, weight in effective_country_targets}
    requested_counts_by_country = _requested_counts_by_country(DEFAULT_MAX_HOLDINGS, target_weights)
    effective_rust_flags = rust_kernel_flags if rust_kernel_flags is not None else RustKernelFlags()
    sales_required_models = {"rnd_sales_ttm", "rnd_sales_size_proxy"}
    previous_weights: dict[str, float] = {}
    sleeve_history: list[dict[str, float]] = []

    def _mean(values: list[float]) -> float | None:
        if not values:
            return None
        return sum(values) / len(values)

    def _median(values: list[float]) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2 == 1:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2.0

    for as_of in schedule[:-1]:
        accepted: list[object] = []
        require_positive_sales = factor_model in sales_required_models
        fundamentals_as_of = [
            row
            for row in fundamental_rows
            if date.fromisoformat(str(row["filing_date"])) <= as_of
        ]
        for universe in universes:
            result = assemble_pit_universe(
                universe=universe,
                as_of_date=as_of,
                constituent_rows=constituent_rows,
                fundamental_rows=fundamentals_as_of,
                price_rows=price_rows,
                fx_rates=fx_rate_map,
                require_positive_sales_ttm=require_positive_sales,
            )
            total_rejected += len(result.rejected)
            accepted.extend(result.accepted)

        if require_positive_sales:
            factor_rows: tuple[object, ...] = tuple(compute_rnd_sales_ttm_factor(accepted, winsor_quantiles=(0.01, 0.99)))
        else:
            factor_rows = tuple(
                _FactorSeedRow(
                    security_id=str(getattr(row, "security_id")),
                    country=str(getattr(row, "country")),
                    rd_expense=cast(float | None, getattr(row, "rd_expense", None)),
                    sales_ttm=cast(float | None, getattr(row, "sales_ttm", None)),
                    factor_value=0.0,
                    rank_in_country=None,
                    is_eligible=bool(getattr(row, "rd_expense", None) is not None),
                    eligibility_reason="ELIGIBLE" if getattr(row, "rd_expense", None) is not None else "MISSING_RD_EXPENSE",
                )
                for row in accepted
            )
        ranking_request = normalize_ranking_request(
            factor_rows=factor_rows,
            accepted_rows=accepted,
            sector_by_security=sector_by_security,
            requested_counts_by_country=requested_counts_by_country,
            factor_model=factor_model,
            sector_active_band=sector_active_band,
            use_size_stratification=use_size_stratification,
        )
        qepm_ranked_rows_raw, ranking_backend, ranking_reason = run_ranking_kernel(
            request=ranking_request,
            flags=effective_rust_flags,
            python_impl=lambda: _build_qepm_ranked_rows(
                factor_rows=factor_rows,
                accepted_rows=accepted,
                sector_by_security=sector_by_security,
                requested_counts_by_country=requested_counts_by_country,
                factor_model=factor_model,
                sector_active_band=sector_active_band,
                use_size_stratification=use_size_stratification,
            ),
        )
        qepm_ranked_rows = cast(list[dict[str, object]], qepm_ranked_rows_raw)
        ranking_rows: tuple[object, ...] = tuple(qepm_ranked_rows) if qepm_ranked_rows else tuple(factor_rows)
        risk_controls = {
            "te_active_l2_cap": qepm_te_active_l2_cap,
            "alpha_tilt_strength": qepm_alpha_tilt_strength,
            "max_adv_participation": qepm_max_adv_participation,
            "portfolio_value_krw": portfolio_value_krw,
            "max_turnover": qepm_max_turnover,
            "previous_weights": previous_weights,
        }
        constructor_request = normalize_constructor_request(
            ranked_factor_rows=ranking_rows,
            country_targets=country_targets,
            risk_controls=risk_controls,
        )

        portfolio_raw, constructor_backend, constructor_reason = run_constructor_kernel(
            request=constructor_request,
            flags=effective_rust_flags,
            python_impl=lambda: construct_portfolio_with_constraints(
                ranking_rows,
                risk_controls=risk_controls,
            )
            if country_targets is None
            else construct_portfolio_with_constraints(
                ranking_rows,
                country_targets=country_targets,
                risk_controls=risk_controls,
            ),
        )
        portfolio = portfolio_raw
        if portfolio.fallback_triggered:
            fallback_count += 1
        current_weights = {holding.security_id: holding.weight for holding in portfolio.holdings}
        blended_weights = current_weights
        if qepm_staggered_sleeves and qepm_sleeve_count > 1 and current_weights:
            sleeve_window = [*sleeve_history, current_weights][-qepm_sleeve_count:]
            ids = {security_id for sleeve in sleeve_window for security_id in sleeve}
            blended_weights = {
                security_id: sum(float(sleeve.get(security_id, 0.0)) for sleeve in sleeve_window) / len(sleeve_window)
                for security_id in ids
            }
            current_total = sum(current_weights.values())
            blended_total = sum(blended_weights.values())
            if current_total > 0.0 and blended_total > 0.0:
                scale = current_total / blended_total
                blended_weights = {security_id: weight * scale for security_id, weight in blended_weights.items()}
        allocations[as_of] = [
            {"security_id": security_id, "target_weight": weight}
            for security_id, weight in sorted(blended_weights.items(), key=lambda item: item[0])
            if weight > 0.0
        ]
        previous_weights = dict(blended_weights)
        sleeve_history.append(dict(current_weights))
        if len(sleeve_history) > qepm_sleeve_count:
            sleeve_history = sleeve_history[-qepm_sleeve_count:]

        selected_ids = set(blended_weights)
        eligible_rows = [
            row
            for row in ranking_rows
            if bool(_get_record_field(row, "is_eligible", False)) and _get_record_field(row, "factor_value", None) is not None
        ]
        selected_rows = [row for row in eligible_rows if str(_get_record_field(row, "security_id")) in selected_ids]
        non_selected_rows = [row for row in eligible_rows if str(_get_record_field(row, "security_id")) not in selected_ids]

        eligible_factors = [float(str(_get_record_field(row, "factor_value"))) for row in eligible_rows]
        selected_factors = [float(str(_get_record_field(row, "factor_value"))) for row in selected_rows]
        non_selected_factors = [float(str(_get_record_field(row, "factor_value"))) for row in non_selected_rows]
        eligible_median = _median(eligible_factors)

        selected_at_or_above_median = None
        if selected_rows and eligible_median is not None:
            selected_at_or_above_median = sum(
                1
                for row in selected_rows
                if float(str(_get_record_field(row, "factor_value"))) >= eligible_median
            ) / len(selected_rows)

        eligible_count_by_country: dict[str, int] = defaultdict(int)
        for row in eligible_rows:
            eligible_count_by_country[str(_get_record_field(row, "country"))] += 1

        top_quartile_selected = 0
        for row in selected_rows:
            country = str(_get_record_field(row, "country"))
            rank_value = _get_record_field(row, "rank_in_country", None)
            rank = int(str(rank_value)) if rank_value not in (None, "") else None
            if rank is None:
                continue
            cutoff = max(1, (eligible_count_by_country[country] + 3) // 4)
            if rank <= cutoff:
                top_quartile_selected += 1
        selected_top_quartile_ratio = (top_quartile_selected / len(selected_rows)) if selected_rows else None

        country_breakdown: dict[str, dict[str, object]] = {}
        for country in sorted(eligible_count_by_country):
            country_eligible = [row for row in eligible_rows if str(_get_record_field(row, "country")) == country]
            country_selected = [row for row in selected_rows if str(_get_record_field(row, "country")) == country]
            country_breakdown[country] = {
                "eligible_count": len(country_eligible),
                "selected_count": len(country_selected),
                "selected_factor_mean": _mean([float(str(_get_record_field(row, "factor_value"))) for row in country_selected]),
                "eligible_factor_mean": _mean([float(str(_get_record_field(row, "factor_value"))) for row in country_eligible]),
            }

        selected_mean = _mean(selected_factors)
        non_selected_mean = _mean(non_selected_factors)
        spread = None
        if selected_mean is not None and non_selected_mean is not None:
            spread = selected_mean - non_selected_mean

        factor_integrity_rows.append(
            {
                "as_of_date": as_of,
                "eligible_count": len(eligible_rows),
                "selected_count": len(selected_rows),
                "eligible_factor_median": eligible_median,
                "selected_factor_mean": selected_mean,
                "non_selected_factor_mean": non_selected_mean,
                "factor_spread_selected_minus_non_selected": spread,
                "selected_at_or_above_eligible_median_ratio": selected_at_or_above_median,
                "selected_top_quartile_rank_ratio": selected_top_quartile_ratio,
                "fallback_triggered": portfolio.fallback_triggered,
                "ranking_backend": ranking_backend,
                "ranking_reason": ranking_reason,
                "constructor_backend": constructor_backend,
                "constructor_reason": constructor_reason,
                "country_breakdown": country_breakdown,
            }
        )

    return allocations, total_rejected, fallback_count, factor_integrity_rows


def _artifact_checksum(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def enforce_quality_gate(output_dir: Path, required_artifacts: tuple[str, ...] = REQUIRED_ARTIFACTS) -> None:
    missing = [name for name in required_artifacts if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"quality gate failed: missing required artifacts: {', '.join(missing)}")

    quality_path = output_dir / "data_quality_report.json"
    if "data_quality_report.json" in required_artifacts and quality_path.exists():
        payload = cast(dict[str, object], json.loads(quality_path.read_text(encoding="utf-8")))
        status = str(payload.get("status", "")).strip().lower()
        if status != "pass":
            raise ValueError("quality gate failed: data_quality_report status is not pass")


def _write_csv(rows: list[dict[str, object]], output_file: Path) -> None:
    if not rows:
        _ = output_file.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _serialize(row.get(key)) for key in columns})


def _round_seconds(value: float) -> float:
    return round(float(value), 6)


def _reset_live_source_cache_counters() -> None:
    reset_fn = getattr(live_sources_module, "_reset_yfinance_migration_stats", None)
    if callable(reset_fn):
        reset_fn()


def _read_live_source_cache_counters() -> dict[str, int]:
    read_fn = getattr(live_sources_module, "_get_yfinance_migration_stats", None)
    if not callable(read_fn):
        return {
            "v2_hit": 0,
            "v1_fallback": 0,
            "network_fetch": 0,
            "repair_success": 0,
            "repair_failure": 0,
        }
    raw = cast(Mapping[str, object], read_fn())
    return {
        "v2_hit": int(str(raw.get("v2_hit", 0))),
        "v1_fallback": int(str(raw.get("v1_fallback", 0))),
        "network_fetch": int(str(raw.get("network_fetch", 0))),
        "repair_success": int(str(raw.get("repair_success", 0))),
        "repair_failure": int(str(raw.get("repair_failure", 0))),
    }


def _build_baseline_comparison_report(
    *,
    baseline_summary_path: Path,
    run_id: str,
    scenario_start: date,
    market_scope: str,
    runtime_total_seconds: float,
    fallback_count: int,
    rejected_count: int,
) -> dict[str, object]:
    report: dict[str, object] = {
        "schema_version": PERF_TELEMETRY_SCHEMA_VERSION,
        "baseline_path": str(baseline_summary_path),
        "baseline_available": False,
        "status": "baseline_missing",
        "match_key": {
            "scenario_start": scenario_start.isoformat(),
            "run_id": run_id,
            "mode": "combined" if market_scope == "combined" else "separate_markets",
            "market": "n/a" if market_scope == "combined" else market_scope,
        },
        "post_refactor": {
            "runtime_seconds": _round_seconds(runtime_total_seconds),
            "portfolio_fallback_count": int(fallback_count),
            "rejected_candidates": int(rejected_count),
        },
    }
    if not baseline_summary_path.exists():
        return report

    report["baseline_available"] = True
    target_mode = str(cast(dict[str, object], report["match_key"])["mode"])
    target_market = str(cast(dict[str, object], report["match_key"])["market"])
    target_start = scenario_start.isoformat()
    target_run_id = run_id

    with baseline_summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = cast(list[dict[str, str]], list(csv.DictReader(handle)))

    matched_row = next(
        (
            row
            for row in rows
            if row.get("scenario_start", "") == target_start
            and row.get("run_id", "") == target_run_id
            and row.get("mode", "") == target_mode
            and row.get("market", "") == target_market
        ),
        None,
    )
    if matched_row is None:
        report["status"] = "baseline_not_found"
        return report

    baseline_runtime = float(str(matched_row.get("runtime_seconds", "0") or "0"))
    baseline_fallback = int(str(matched_row.get("portfolio_fallback_count", "0") or "0"))
    baseline_rejected = int(str(matched_row.get("rejected_candidates", "0") or "0"))
    report.update(
        {
            "status": "baseline_matched",
            "baseline": {
                "runtime_seconds": baseline_runtime,
                "portfolio_fallback_count": baseline_fallback,
                "rejected_candidates": baseline_rejected,
            },
            "delta": {
                "runtime_seconds": _round_seconds(runtime_total_seconds - baseline_runtime),
                "portfolio_fallback_count": int(fallback_count - baseline_fallback),
                "rejected_candidates": int(rejected_count - baseline_rejected),
            },
        }
    )
    return report


def _build_rollout_summary_artifact(
    *,
    run_id: str,
    start: date,
    end: date,
    metrics_payload: dict[str, object],
    quality_report: dict[str, object],
    qepm_alignment_payload: dict[str, object],
    perf_comparison_payload: dict[str, object],
) -> dict[str, object]:
    quality_status = str(quality_report.get("status", "fail"))
    qepm_status = str(qepm_alignment_payload.get("status", "fail"))
    comparison_status = str(perf_comparison_payload.get("status", "baseline_missing"))
    baseline_available = bool(perf_comparison_payload.get("baseline_available", False))

    benchmark_status = "warning"
    if comparison_status == "baseline_matched":
        benchmark_status = "pass"
    elif comparison_status not in {"baseline_missing", "baseline_not_found"}:
        benchmark_status = "fail"

    signal_statuses = {
        "benchmark": benchmark_status,
        "parity": quality_status,
        "anchor": qepm_status,
    }

    overall_status = "pass"
    if any(status == "fail" for status in signal_statuses.values()):
        overall_status = "fail"
    elif any(status == "warning" for status in signal_statuses.values()):
        overall_status = "warning"

    config_payload = cast(dict[str, object], metrics_payload.get("config", {}))
    checks_payload = cast(dict[str, object], quality_report.get("checks", {}))
    qepm_checks = cast(dict[str, object], qepm_alignment_payload.get("checks", {}))
    baseline_payload = cast(dict[str, object], perf_comparison_payload.get("baseline", {}))
    post_refactor_payload = cast(dict[str, object], perf_comparison_payload.get("post_refactor", {}))

    speedup_ratio: float | None = None
    speedup_pct: float | None = None
    if comparison_status == "baseline_matched":
        baseline_runtime = float(str(baseline_payload.get("runtime_seconds", 0.0) or 0.0))
        post_refactor_runtime = float(str(post_refactor_payload.get("runtime_seconds", 0.0) or 0.0))
        if baseline_runtime > 0.0:
            speedup_ratio = _round_seconds((baseline_runtime - post_refactor_runtime) / baseline_runtime)
            speedup_pct = _round_seconds(speedup_ratio * 100.0)

    return {
        "schema_version": PERF_TELEMETRY_SCHEMA_VERSION,
        "run_id": run_id,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "status": overall_status,
        "signals": {
            "benchmark": {
                "status": benchmark_status,
                "comparison_status": comparison_status,
                "baseline_available": baseline_available,
                "speedup_ratio": speedup_ratio,
                "speedup_pct": speedup_pct,
            },
            "parity": {
                "status": quality_status,
                "non_finite_output_values": int(str(checks_payload.get("non_finite_output_values", 0) or 0)),
            },
            "anchor": {
                "status": qepm_status,
                "requested": str(config_payload.get("rebalance_anchor", "")),
                "effective": str(config_payload.get("rebalance_anchor_effective", "")),
                "phase_lock_workaround_used": bool(qepm_checks.get("phase_lock_workaround_used", False)),
            },
        },
        "rust_backend": {
            "flags": cast(dict[str, object], config_payload.get("rust_kernel_flags", {})),
            "backtest_backend": str(config_payload.get("rust_backtest_backend", "python")),
            "backtest_reason": str(config_payload.get("rust_backtest_reason", "")),
        },
        "evidence_files": {
            "metrics": "metrics.json",
            "data_quality": "data_quality_report.json",
            "qepm_alignment": "qepm_alignment_report.json",
            "perf_telemetry": "perf_telemetry.json",
            "perf_comparison": "perf_comparison_report.json",
        },
    }


def _compound_returns(values: list[float]) -> float:
    growth = 1.0
    for value in values:
        growth *= 1.0 + float(value)
    return growth - 1.0


def _zscore_by_key(values: list[tuple[str, float]]) -> dict[str, float]:
    if not values:
        return {}
    mean_value = sum(value for _, value in values) / len(values)
    variance = sum((value - mean_value) ** 2 for _, value in values) / len(values)
    std_value = math.sqrt(variance)
    if std_value <= 1e-12:
        return {key: 0.0 for key, _ in values}
    return {key: (value - mean_value) / std_value for key, value in values}


def _requested_counts_by_country(max_holdings: int, target_weights: dict[str, float]) -> dict[str, int]:
    countries = tuple(sorted(target_weights))
    raw = {country: max_holdings * target_weights[country] for country in countries}
    floored = {country: int(raw[country]) for country in countries}
    remaining = max_holdings - sum(floored.values())
    remainders = sorted(countries, key=lambda country: (-((raw[country]) - floored[country]), country))
    for country in remainders[:remaining]:
        floored[country] += 1
    return floored


def _compute_sector_quotas(*, total: int, sector_counts: dict[str, int]) -> dict[str, int]:
    if total <= 0 or not sector_counts:
        return {}
    total_count = sum(sector_counts.values())
    if total_count <= 0:
        return {sector: 0 for sector in sector_counts}
    raw = {sector: total * (count / total_count) for sector, count in sector_counts.items()}
    quotas = {sector: int(raw[sector]) for sector in sector_counts}
    if total >= len(sector_counts):
        for sector in sector_counts:
            if quotas[sector] == 0:
                quotas[sector] = 1
    used = sum(quotas.values())
    if used > total:
        overflow = used - total
        for sector, _ in sorted(quotas.items(), key=lambda pair: (-pair[1], pair[0])):
            if overflow <= 0:
                break
            reducible = max(0, quotas[sector] - 1)
            if reducible <= 0:
                continue
            step = min(reducible, overflow)
            quotas[sector] -= step
            overflow -= step
    elif used < total:
        shortfall = total - used
        for sector in sorted(sector_counts, key=lambda item: (-((raw[item]) - quotas[item]), item)):
            if shortfall <= 0:
                break
            quotas[sector] += 1
            shortfall -= 1
    return quotas


def _benchmark_sector_weights(rows: list[dict[str, object]]) -> dict[str, float]:
    sector_liquidity: dict[str, float] = defaultdict(float)
    for row in rows:
        sector = str(row["sector"])
        liquidity = max(float(str(row["median_traded_value_krw"])), 1.0)
        sector_liquidity[sector] += liquidity
    total = sum(sector_liquidity.values())
    if total <= 0.0:
        return {}
    return {sector: liquidity / total for sector, liquidity in sector_liquidity.items()}


def _compute_sector_quota_bounds(
    *,
    total: int,
    sector_counts: dict[str, int],
    sector_weights: dict[str, float],
    active_band: float,
) -> tuple[dict[str, int], dict[str, int]]:
    if total <= 0 or not sector_counts:
        return {}, {}

    normalized_band = max(0.0, min(active_band, 1.0))
    min_quota: dict[str, int] = {}
    max_quota: dict[str, int] = {}
    for sector, available in sector_counts.items():
        weight = float(sector_weights.get(sector, 0.0))
        lower_weight = max(0.0, weight - normalized_band)
        upper_weight = min(1.0, weight + normalized_band)
        minimum = int(math.floor(total * lower_weight + 1e-12))
        maximum = int(math.ceil(total * upper_weight - 1e-12))
        minimum = min(max(0, minimum), available)
        maximum = min(max(minimum, maximum), available)
        min_quota[sector] = minimum
        max_quota[sector] = maximum

    minimum_total = sum(min_quota.values())
    if minimum_total > total:
        overflow = minimum_total - total
        for sector in sorted(min_quota, key=lambda item: (min_quota[item], item), reverse=True):
            if overflow <= 0:
                break
            reducible = min_quota[sector]
            if reducible <= 0:
                continue
            step = min(reducible, overflow)
            min_quota[sector] -= step
            overflow -= step

    return min_quota, max_quota


def _select_sector_rows(
    *,
    rows: list[dict[str, object]],
    pick_count: int,
    use_size_stratification: bool,
) -> list[dict[str, object]]:
    if pick_count <= 0 or not rows:
        return []
    ordered_rows = sorted(rows, key=lambda item: (-float(str(item["factor_value"])), str(item["security_id"])))
    if not use_size_stratification or pick_count >= len(ordered_rows):
        return ordered_rows[:pick_count]

    size_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in ordered_rows:
        size_groups[str(row.get("size_bucket", "mid"))].append(row)
    for bucket_rows in size_groups.values():
        bucket_rows.sort(key=lambda item: (-float(str(item["factor_value"])), str(item["security_id"])))

    bucket_counts = {bucket: len(bucket_rows) for bucket, bucket_rows in size_groups.items()}
    bucket_quotas = _compute_sector_quotas(total=pick_count, sector_counts=bucket_counts)

    selected: list[dict[str, object]] = []
    for bucket, quota in bucket_quotas.items():
        selected.extend(size_groups[bucket][:quota])

    if len(selected) < pick_count:
        selected_ids = {str(item["security_id"]) for item in selected}
        leftovers = [item for item in ordered_rows if str(item["security_id"]) not in selected_ids]
        selected.extend(leftovers[: pick_count - len(selected)])
    return selected[:pick_count]


def _compute_raw_metric_for_model(
    *,
    model: str,
    rd_expense: float,
    sales_ttm: float | None,
    liquidity_krw: float,
) -> float:
    safe_sales = max(float(sales_ttm), 1e-9) if sales_ttm is not None else None
    safe_liquidity = max(liquidity_krw, 1.0)
    liquidity_scale = max(math.log1p(safe_liquidity / 1_000_000_000.0), 1e-6)
    sales_ratio = (rd_expense / safe_sales) if safe_sales is not None else None
    sales_size_ratio = (sales_ratio / liquidity_scale) if sales_ratio is not None else None
    mktcap_proxy_ratio = rd_expense / safe_liquidity
    ev_like_ratio = rd_expense / (safe_liquidity * (1.0 + liquidity_scale))

    if model == "rnd_sales_ttm":
        if sales_ratio is None:
            raise ValueError("sales_ttm is required for rnd_sales_ttm")
        return sales_ratio
    if model == "rnd_sales_size_proxy":
        if sales_size_ratio is None:
            raise ValueError("sales_ttm is required for rnd_sales_size_proxy")
        return sales_size_ratio
    if model == "rnd_mktcap_proxy":
        return mktcap_proxy_ratio
    if model == "rnd_ev_proxy":
        return ev_like_ratio
    if model == "rnd_robust_composite":
        components = [mktcap_proxy_ratio, ev_like_ratio]
        if sales_ratio is not None:
            components.append(sales_ratio)
        if sales_size_ratio is not None:
            components.append(sales_size_ratio)
        return sum(components) / len(components)
    raise ValueError("unsupported factor model")


def _factor_metadata_for_model(model: str) -> tuple[str, str]:
    mapping = {
        "rnd_sales_ttm": ("sales_ttm", "R&D/Sales(TTM)"),
        "rnd_sales_size_proxy": ("sales_size_proxy", "R&D/Sales(Size-Adjusted Proxy)"),
        "rnd_mktcap_proxy": ("mktcap_proxy", "R&D/MktCap(Proxy)"),
        "rnd_ev_proxy": ("ev_proxy", "R&D/EV(Proxy)"),
        "rnd_robust_composite": ("robust_composite", "R&D/Composite(Robust)"),
    }
    return mapping.get(model, ("unknown", "R&D/Unknown"))


def _build_qepm_alignment_report(
    *,
    run_id: str,
    start: date,
    end: date,
    factor_model: str,
    denominator_model: str,
    rebalance_anchor_requested: str,
    rebalance_anchor_effective: str,
    qepm_phase_lock_quarter_end: bool,
    qepm_sector_active_band: float,
    qepm_size_stratification: bool,
    qepm_te_active_l2_cap: float,
    qepm_max_adv_participation: float,
    qepm_max_turnover: float,
    qepm_staggered_sleeves: bool,
    qepm_sleeve_count: int,
) -> dict[str, object]:
    checks = {
        "factor_axis_rd": True,
        "denominator_model": denominator_model,
        "model_in_rd_family": factor_model in {
            "rnd_sales_ttm",
            "rnd_sales_size_proxy",
            "rnd_mktcap_proxy",
            "rnd_ev_proxy",
            "rnd_robust_composite",
        },
        "benchmark_relative_sector_band_enabled": qepm_sector_active_band > 0.0,
        "size_stratification_enabled": qepm_size_stratification,
        "te_constraint_enabled": qepm_te_active_l2_cap > 0.0,
        "adv_participation_constraint_enabled": qepm_max_adv_participation > 0.0,
        "turnover_constraint_enabled": qepm_max_turnover > 0.0,
        "staggered_sleeves_enabled": qepm_staggered_sleeves and qepm_sleeve_count >= 2,
        "phase_lock_workaround_used": rebalance_anchor_requested == "start_date"
        and rebalance_anchor_effective == "quarter_end"
        and qepm_phase_lock_quarter_end,
    }

    failed = [
        key
        for key in (
            "model_in_rd_family",
            "benchmark_relative_sector_band_enabled",
            "size_stratification_enabled",
            "te_constraint_enabled",
            "adv_participation_constraint_enabled",
            "turnover_constraint_enabled",
            "staggered_sleeves_enabled",
        )
        if not bool(checks[key])
    ]
    warnings = []
    if bool(checks["phase_lock_workaround_used"]):
        warnings.append("phase_lock_workaround_used")

    status = "pass"
    if failed:
        status = "fail"
    elif warnings:
        status = "warning"

    return {
        "run_id": run_id,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "status": status,
        "checks": checks,
        "failed_checks": failed,
        "warnings": warnings,
        "notes": {
            "qepm_reference": "QEPM_PURE_CHUNK_02.md, QEPM_PURE_CHUNK_03.md",
            "operator_action_if_warning": "Run start-date sensitivity and avoid phase lock for production decisions.",
        },
    }


def _build_qepm_ranked_rows(
    *,
    factor_rows: tuple[object, ...],
    accepted_rows: list[object],
    sector_by_security: dict[str, str],
    requested_counts_by_country: dict[str, int],
    factor_model: str,
    sector_active_band: float,
    use_size_stratification: bool,
) -> list[dict[str, object]]:
    accepted_by_security = {
        str(getattr(row, "security_id")): row
        for row in accepted_rows
    }

    eligible_rows: list[dict[str, object]] = []
    for row in factor_rows:
        security_id = str(getattr(row, "security_id"))
        factor_value = cast(float | None, getattr(row, "factor_value"))
        rd_expense = cast(float | None, getattr(row, "rd_expense"))
        sales_ttm = cast(float | None, getattr(row, "sales_ttm"))
        if not bool(getattr(row, "is_eligible")) or factor_value is None:
            continue
        accepted = accepted_by_security.get(security_id)
        if accepted is None or rd_expense is None:
            continue
        liquidity = float(getattr(accepted, "median_traded_value_krw"))
        sector = sector_by_security.get(security_id, "UNKNOWN")
        try:
            raw_metric = _compute_raw_metric_for_model(
                model=factor_model,
                rd_expense=float(rd_expense),
                sales_ttm=float(sales_ttm) if sales_ttm is not None else None,
                liquidity_krw=liquidity,
            )
        except ValueError:
            continue
        eligible_rows.append(
            {
                "security_id": security_id,
                "country": str(getattr(row, "country")),
                "factor_value": raw_metric,
                "base_factor_value": raw_metric,
                "sector": sector,
                "median_traded_value_krw": liquidity,
                "rd_expense": float(rd_expense),
                "sales_ttm": (float(sales_ttm) if sales_ttm is not None else None),
            }
        )

    sector_zscores: dict[str, float] = {}
    size_zscores: dict[str, float] = {}

    groups_by_country_sector: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    groups_by_country: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in eligible_rows:
        country = str(row["country"])
        sector = str(row["sector"])
        groups_by_country_sector[(country, sector)].append(row)
        groups_by_country[country].append(row)

    for rows in groups_by_country_sector.values():
        zmap = _zscore_by_key([(str(row["security_id"]), float(str(row["factor_value"]))) for row in rows])
        sector_zscores.update(zmap)

    for country, rows in groups_by_country.items():
        size_adjusted_values: list[tuple[str, float]] = []
        sorted_liquidity = sorted(max(float(str(row["median_traded_value_krw"])), 1.0) for row in rows)
        if sorted_liquidity:
            low_cut = sorted_liquidity[max(0, len(sorted_liquidity) // 3 - 1)]
            high_cut = sorted_liquidity[max(0, (2 * len(sorted_liquidity)) // 3 - 1)]
        else:
            low_cut = 0.0
            high_cut = 0.0
        for row in rows:
            liquidity = max(float(str(row["median_traded_value_krw"])), 1.0)
            size_scale = math.log1p(liquidity / 1_000_000_000.0)
            adjusted = float(str(row["factor_value"])) / max(size_scale, 1e-6)
            size_adjusted_values.append((str(row["security_id"]), adjusted))
            if liquidity <= low_cut:
                row["size_bucket"] = "small"
            elif liquidity <= high_cut:
                row["size_bucket"] = "mid"
            else:
                row["size_bucket"] = "large"
        zmap = _zscore_by_key(size_adjusted_values)
        for key, value in zmap.items():
            size_zscores[f"{country}:{key}"] = value

    for row in eligible_rows:
        security_id = str(row["security_id"])
        country = str(row["country"])
        sector_z = sector_zscores.get(security_id, 0.0)
        size_z = size_zscores.get(f"{country}:{security_id}", 0.0)
        score_weights = {
            "rnd_sales_ttm": (0.7, 0.3),
            "rnd_sales_size_proxy": (0.4, 0.6),
            "rnd_mktcap_proxy": (0.3, 0.7),
            "rnd_ev_proxy": (0.6, 0.4),
            "rnd_robust_composite": (0.5, 0.5),
        }
        sector_weight, size_weight = score_weights.get(factor_model, (0.5, 0.5))
        score = (sector_weight * sector_z) + (size_weight * size_z)
        row["factor_value"] = score

    ranked_rows: list[dict[str, object]] = []
    for country, rows in groups_by_country.items():
        target_count = requested_counts_by_country.get(country, 0)
        ordered_rows = sorted(rows, key=lambda item: (-float(str(item["factor_value"])), str(item["security_id"])))
        prioritized_rows = ordered_rows

        if target_count > 0 and len(rows) > target_count:
            sector_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
            for row in rows:
                sector_groups[str(row["sector"])].append(row)
            for sector_rows in sector_groups.values():
                sector_rows.sort(key=lambda item: (-float(str(item["factor_value"])), str(item["security_id"])))

            sector_counts = {sector: len(sector_rows) for sector, sector_rows in sector_groups.items()}
            benchmark_sector_weights = _benchmark_sector_weights(rows)
            min_quota, max_quota = _compute_sector_quota_bounds(
                total=target_count,
                sector_counts=sector_counts,
                sector_weights=benchmark_sector_weights,
                active_band=sector_active_band,
            )
            selected: list[dict[str, object]] = []
            selected_ids: set[str] = set()
            selected_by_sector: dict[str, int] = {sector: 0 for sector in sector_groups}

            for sector in sorted(sector_groups, key=lambda item: (-benchmark_sector_weights.get(item, 0.0), item)):
                quota = min_quota.get(sector, 0)
                picks = _select_sector_rows(
                    rows=sector_groups[sector],
                    pick_count=quota,
                    use_size_stratification=use_size_stratification,
                )
                for item in picks:
                    security_id = str(item["security_id"])
                    if security_id in selected_ids:
                        continue
                    selected.append(item)
                    selected_ids.add(security_id)
                    selected_by_sector[sector] += 1

            capacity_candidates: list[dict[str, object]] = []
            for sector in sorted(sector_groups):
                cap = max_quota.get(sector, sector_counts[sector])
                if selected_by_sector[sector] >= cap:
                    continue
                for item in sector_groups[sector]:
                    if str(item["security_id"]) in selected_ids:
                        continue
                    candidate = dict(item)
                    candidate["_sector_cap"] = cap
                    capacity_candidates.append(candidate)

            capacity_candidates.sort(key=lambda item: (-float(str(item["factor_value"])), str(item["security_id"])))
            for item in capacity_candidates:
                if len(selected) >= target_count:
                    break
                sector = str(item["sector"])
                cap = int(str(item.get("_sector_cap", sector_counts.get(sector, 0))))
                if selected_by_sector.get(sector, 0) >= cap:
                    continue
                security_id = str(item["security_id"])
                if security_id in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(security_id)
                selected_by_sector[sector] = selected_by_sector.get(sector, 0) + 1

            if len(selected) < target_count:
                leftovers = [
                    item
                    for item in ordered_rows
                    if str(item["security_id"]) not in selected_ids
                ]
                selected.extend(leftovers[: target_count - len(selected)])

            selected_ids = {str(item["security_id"]) for item in selected[:target_count]}
            prioritized_rows = [item for item in ordered_rows if str(item["security_id"]) in selected_ids] + [
                item for item in ordered_rows if str(item["security_id"]) not in selected_ids
            ]

        for rank, row in enumerate(prioritized_rows, start=1):
            row["rank_in_country"] = rank
            row["is_eligible"] = True
            ranked_rows.append(row)

    ranked = sorted(
        ranked_rows,
        key=lambda item: (
            str(item["country"]),
            int(str(item.get("rank_in_country") or 10**9)),
            str(item["security_id"]),
        ),
    )
    return ranked


def _normalize_market_scope(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"combined", "us", "kr", "jp"}:
        raise ValueError("--market must be one of: combined, us, kr, jp")
    return normalized


def _fundamental_rd_coverage_by_country(
    *,
    specs: Sequence[SecuritySpec],
    fundamental_rows: Sequence[Mapping[str, object]],
) -> dict[str, tuple[int, int, float]]:
    expected_by_country: dict[str, set[str]] = defaultdict(set)
    for spec in specs:
        expected_by_country[spec.country.lower()].add(spec.security_id)

    observed_by_country: dict[str, set[str]] = defaultdict(set)
    for row in fundamental_rows:
        security_id = str(row.get("security_id", "")).strip()
        country = str(row.get("country", "")).strip().lower()
        rd_expense = row.get("rd_expense")
        if not security_id or not country:
            continue
        rd_value = None
        if isinstance(rd_expense, (int, float)):
            rd_value = float(rd_expense)
        elif isinstance(rd_expense, str):
            try:
                rd_value = float(rd_expense.replace(",", "").strip())
            except ValueError:
                rd_value = None
        if rd_value is None or not math.isfinite(rd_value):
            continue
        observed_by_country[country].add(security_id)

    coverage: dict[str, tuple[int, int, float]] = {}
    for country, expected in expected_by_country.items():
        expected_count = len(expected)
        observed_count = len(observed_by_country.get(country, set()) & expected)
        ratio = (observed_count / expected_count) if expected_count > 0 else 1.0
        coverage[country] = (observed_count, expected_count, ratio)
    return coverage


def _enforce_rd_coverage_gate(
    *,
    specs: Sequence[SecuritySpec],
    fundamental_rows: Sequence[Mapping[str, object]],
    market_scope: str,
    min_rd_coverage_kr: float,
    min_rd_coverage_jp: float,
) -> None:
    coverage = _fundamental_rd_coverage_by_country(specs=specs, fundamental_rows=fundamental_rows)

    required: dict[str, float] = {}
    if market_scope in {"combined", "kr"}:
        required["kr"] = min_rd_coverage_kr
    if market_scope in {"combined", "jp"}:
        required["jp"] = min_rd_coverage_jp

    failures: list[str] = []
    for country, threshold in required.items():
        observed, expected, ratio = coverage.get(country, (0, 0, 0.0))
        if ratio + 1e-12 < threshold:
            failures.append(
                f"{country.upper()} rd_coverage={ratio:.3f} (observed={observed}, expected={expected}, min={threshold:.3f})"
            )

    expected_by_country: dict[str, set[str]] = defaultdict(set)
    for spec in specs:
        expected_by_country[spec.country.lower()].add(spec.security_id)

    anchor_targets = {
        "kr": "KRX:005930",
        "jp": "JP:6857:TYO",
    }
    observed_security_ids = {
        str(row.get("security_id", "")).strip()
        for row in fundamental_rows
        if row.get("rd_expense") not in (None, "")
    }
    for country, anchor_security_id in anchor_targets.items():
        if country not in required:
            continue
        if anchor_security_id not in expected_by_country.get(country, set()):
            continue
        if anchor_security_id not in observed_security_ids:
            failures.append(f"{country.upper()} anchor missing rd_expense: {anchor_security_id}")

    if failures:
        joined = "; ".join(failures)
        raise ValueError(f"real-data R&D coverage gate failed: {joined}")


def run_pipeline(
    *,
    start: date,
    end: date,
    output_root: Path | None = None,
    data_source: str = "synthetic",
    market: str = "combined",
    rebalance_anchor: str = "quarter_end",
    universe_profile: str = "broad",
    factor_model: str = "rnd_sales_ttm",
    qepm_phase_lock_quarter_end: bool = True,
    qepm_sector_active_band: float = 0.10,
    qepm_size_stratification: bool = True,
    qepm_te_active_l2_cap: float = 0.08,
    qepm_alpha_tilt_strength: float = 0.35,
    qepm_max_adv_participation: float = 0.10,
    qepm_max_turnover: float = 0.35,
    qepm_staggered_sleeves: bool = True,
    qepm_sleeve_count: int = 3,
    seed_krw: float = 10_000_000.0,
    sec_stage_budget_sec: int | None = 600,
    sec_timeout_sec: int = 20,
    sec_max_retries: int = 2,
    sec_backoff_sec: float = 1.0,
    sec_max_rps: float = 8.0,
    sec_max_workers: int = 4,
    sec_allow_stale_cache: bool = True,
    yfinance_cache_ttl_days: int = 3,
    yfinance_chunk_size: int = 80,
    yfinance_max_workers: int = 4,
    yfinance_max_retries: int = 2,
    yfinance_backoff_sec: float = 1.0,
    dart_timeout_sec: int = 20,
    dart_max_retries: int = 2,
    dart_backoff_sec: float = 1.0,
    dart_max_rps: float = 8.0,
    dart_max_workers: int = 6,
    edinet_max_workers: int = 8,
    edinet_max_rps: float = 5.0,
    edinet_max_retries: int = 2,
    edinet_backoff_sec: float = 1.0,
    min_rd_coverage_kr: float = 0.60,
    min_rd_coverage_jp: float = 0.60,
    perf_baseline_summary_path: Path | None = None,
    rust_kernels: bool = False,
    rust_kernel_ranking: bool = False,
    rust_kernel_constructor: bool = False,
    rust_kernel_backtest: bool = False,
    rust_kernel_strict: bool = False,
) -> Path:
    if start > end:
        raise ValueError("--start must be on or before --end")
    market_scope = _normalize_market_scope(market)
    rebalance_anchor_key = rebalance_anchor.strip().lower()
    if rebalance_anchor_key not in {"quarter_end", "start_date"}:
        raise ValueError("--rebalance-anchor must be one of: quarter_end, start_date")
    factor_model_key = factor_model.strip().lower()
    universe_profile_key = universe_profile.strip().lower()
    if universe_profile_key not in {"broad", "core_indices"}:
        raise ValueError("--universe-profile must be one of: broad, core_indices")
    supported_factor_models = {
        "rnd_sales_ttm",
        "rnd_sales_size_proxy",
        "rnd_mktcap_proxy",
        "rnd_ev_proxy",
        "rnd_robust_composite",
    }
    if factor_model_key not in supported_factor_models:
        raise ValueError(
            "--factor-model must be one of: rnd_sales_ttm, rnd_sales_size_proxy, rnd_mktcap_proxy, rnd_ev_proxy, rnd_robust_composite"
        )
    if seed_krw <= 0.0:
        raise ValueError("--seed-krw must be positive")
    if not (0.0 <= qepm_sector_active_band <= 1.0):
        raise ValueError("--qepm-sector-active-band must be in [0, 1]")
    if qepm_te_active_l2_cap < 0.0:
        raise ValueError("--qepm-te-active-l2-cap must be non-negative")
    if qepm_alpha_tilt_strength < 0.0:
        raise ValueError("--qepm-alpha-tilt-strength must be non-negative")
    if qepm_max_adv_participation <= 0.0:
        raise ValueError("--qepm-max-adv-participation must be positive")
    if qepm_max_turnover < 0.0:
        raise ValueError("--qepm-max-turnover must be non-negative")
    if qepm_sleeve_count <= 0:
        raise ValueError("--qepm-sleeve-count must be positive")
    if sec_timeout_sec <= 0:
        raise ValueError("--sec-timeout-sec must be positive")
    if sec_max_retries < 0:
        raise ValueError("--sec-max-retries must be non-negative")
    if sec_backoff_sec < 0.0:
        raise ValueError("--sec-backoff-sec must be non-negative")
    if sec_max_rps <= 0.0:
        raise ValueError("--sec-max-rps must be positive")
    if sec_max_workers <= 0:
        raise ValueError("--sec-max-workers must be positive")
    if yfinance_cache_ttl_days < 0:
        raise ValueError("--yfinance-cache-ttl-days must be non-negative")
    if yfinance_chunk_size <= 0:
        raise ValueError("--yfinance-chunk-size must be positive")
    if yfinance_max_workers <= 0:
        raise ValueError("--yfinance-max-workers must be positive")
    if yfinance_max_retries < 0:
        raise ValueError("--yfinance-max-retries must be non-negative")
    if yfinance_backoff_sec < 0.0:
        raise ValueError("--yfinance-backoff-sec must be non-negative")
    if dart_timeout_sec <= 0:
        raise ValueError("--dart-timeout-sec must be positive")
    if dart_max_retries < 0:
        raise ValueError("--dart-max-retries must be non-negative")
    if dart_backoff_sec < 0.0:
        raise ValueError("--dart-backoff-sec must be non-negative")
    if dart_max_rps <= 0.0:
        raise ValueError("--dart-max-rps must be positive")
    if dart_max_workers <= 0:
        raise ValueError("--dart-max-workers must be positive")
    if edinet_max_workers <= 0:
        raise ValueError("--edinet-max-workers must be positive")
    if edinet_max_rps <= 0.0:
        raise ValueError("--edinet-max-rps must be positive")
    if edinet_max_retries < 0:
        raise ValueError("--edinet-max-retries must be non-negative")
    if edinet_backoff_sec < 0.0:
        raise ValueError("--edinet-backoff-sec must be non-negative")
    if not (0.0 <= min_rd_coverage_kr <= 1.0):
        raise ValueError("--min-rd-coverage-kr must be in [0, 1]")
    if not (0.0 <= min_rd_coverage_jp <= 1.0):
        raise ValueError("--min-rd-coverage-jp must be in [0, 1]")
    if sec_stage_budget_sec is not None and sec_stage_budget_sec <= 0:
        sec_stage_budget_sec = None

    rust_flags = RustKernelFlags.from_inputs(
        enable_all=rust_kernels,
        enable_ranking=rust_kernel_ranking,
        enable_constructor=rust_kernel_constructor,
        enable_backtest=rust_kernel_backtest,
        strict=rust_kernel_strict,
    )

    pipeline_t0 = time.perf_counter()
    ingestion_stage_seconds = 0.0
    compute_stage_seconds = 0.0
    cache_counters = {
        "yfinance_v2_hit": 0,
        "yfinance_v1_fallback": 0,
        "yfinance_network_fetch": 0,
        "yfinance_repair_success": 0,
        "yfinance_repair_failure": 0,
    }

    denominator_model, factor_label = _factor_metadata_for_model(factor_model_key)

    effective_anchor = "quarter_end" if (rebalance_anchor_key == "start_date" and qepm_phase_lock_quarter_end) else rebalance_anchor_key

    available_trading_dates = _quarter_targets_with_anchor(start=start, end=end, rebalance_anchor=effective_anchor)
    schedule_country = market_scope.upper() if market_scope in {"us", "kr", "jp"} else "KR"
    schedule = list(
        generate_quarterly_rebalance_dates(
            schedule_country,
            start,
            end,
            available_trading_dates,
            rebalance_anchor=effective_anchor,
        )
    )
    if len(schedule) < 2:
        raise ValueError("date window must include at least two rebalance dates")

    allocation_schedule = list(schedule)
    if rebalance_anchor_key == "start_date" and qepm_staggered_sleeves and qepm_sleeve_count > 1:
        warmup_start = date(start.year - 1, 1, 1)
        month_targets = _month_end_targets(start=warmup_start, end=end)
        allocation_schedule = sorted(set(schedule) | set(month_targets))

    all_specs = _build_security_specs(universe_profile=universe_profile_key)
    specs = all_specs if market_scope == "combined" else tuple(spec for spec in all_specs if spec.country == market_scope.upper())
    if not specs:
        raise ValueError(f"no securities available for market scope: {market_scope}")
    security_metadata = _security_metadata(specs)
    sector_by_security = {spec.security_id: spec.sector for spec in specs}
    constituent_rows = _build_constituent_rows(specs, start)
    source_key = data_source.strip().lower()
    ingestion_t0 = time.perf_counter()
    if source_key == "synthetic":
        fundamental_rows = _build_fundamentals(schedule=allocation_schedule, specs=specs)
        price_rows = _build_price_rows(schedule=allocation_schedule, specs=specs)
        fx_rows, fx_rate_map = _build_fx_rates(allocation_schedule)
        benchmark_rows = _build_benchmark_series(schedule)
    elif source_key == "real":
        _reset_live_source_cache_counters()
        live_config = load_live_data_config_from_env()
        cache_root = _project_root() / ".cache" / "live_sources"
        benchmark_ticker: str | None = None
        benchmark_currency: str | None = None
        if market_scope == "us":
            benchmark_ticker = live_config.benchmark_us_ticker
            benchmark_currency = "USD"
        elif market_scope == "kr":
            benchmark_ticker = live_config.benchmark_kr_ticker
            benchmark_currency = "KRW"
        elif market_scope == "jp":
            benchmark_ticker = live_config.benchmark_jp_ticker
            benchmark_currency = "JPY"
        live_options = LiveIngestOptions(
            sec_timeout_sec=sec_timeout_sec,
            sec_max_retries=sec_max_retries,
            sec_backoff_sec=sec_backoff_sec,
            sec_max_rps=sec_max_rps,
            sec_max_workers=sec_max_workers,
            sec_stage_budget_sec=sec_stage_budget_sec,
            sec_allow_stale_cache=sec_allow_stale_cache,
            yfinance_cache_ttl_days=yfinance_cache_ttl_days,
            yfinance_chunk_size=yfinance_chunk_size,
            yfinance_max_workers=yfinance_max_workers,
            yfinance_max_retries=yfinance_max_retries,
            yfinance_backoff_sec=yfinance_backoff_sec,
            dart_timeout_sec=dart_timeout_sec,
            dart_max_retries=dart_max_retries,
            dart_backoff_sec=dart_backoff_sec,
            dart_max_rps=dart_max_rps,
            dart_max_workers=dart_max_workers,
            edinet_max_workers=edinet_max_workers,
            edinet_max_rps=edinet_max_rps,
            edinet_max_retries=edinet_max_retries,
            edinet_backoff_sec=edinet_backoff_sec,
        )
        fundamental_rows, price_rows, fx_rows, fx_rate_map, benchmark_rows = build_live_inputs(
            start=start,
            end=end,
            schedule=allocation_schedule,
            specs=specs,
            config=live_config,
            cache_root=cache_root,
            options=live_options,
            benchmark_ticker=benchmark_ticker,
            benchmark_currency=benchmark_currency,
        )
        _enforce_rd_coverage_gate(
            specs=specs,
            fundamental_rows=fundamental_rows,
            market_scope=market_scope,
            min_rd_coverage_kr=min_rd_coverage_kr,
            min_rd_coverage_jp=min_rd_coverage_jp,
        )
        migration_counters = _read_live_source_cache_counters()
        cache_counters = {
            "yfinance_v2_hit": migration_counters["v2_hit"],
            "yfinance_v1_fallback": migration_counters["v1_fallback"],
            "yfinance_network_fetch": migration_counters["network_fetch"],
            "yfinance_repair_success": migration_counters["repair_success"],
            "yfinance_repair_failure": migration_counters["repair_failure"],
        }
    else:
        raise ValueError("--data-source must be either 'synthetic' or 'real'")
    ingestion_stage_seconds = _round_seconds(time.perf_counter() - ingestion_t0)

    country_targets = None if market_scope == "combined" else ((market_scope.upper(), 1.0),)

    compute_t0 = time.perf_counter()
    allocations, rejected_count, fallback_count, factor_integrity_rows = _build_allocations(
        schedule=allocation_schedule,
        constituent_rows=constituent_rows,
        fundamental_rows=fundamental_rows,
        price_rows=price_rows,
        fx_rate_map=fx_rate_map,
        sector_by_security=sector_by_security,
        factor_model=factor_model_key,
        sector_active_band=qepm_sector_active_band,
        use_size_stratification=qepm_size_stratification,
        qepm_te_active_l2_cap=qepm_te_active_l2_cap,
        qepm_alpha_tilt_strength=qepm_alpha_tilt_strength,
        qepm_max_adv_participation=qepm_max_adv_participation,
        qepm_max_turnover=qepm_max_turnover,
        portfolio_value_krw=seed_krw,
        qepm_staggered_sleeves=qepm_staggered_sleeves,
        qepm_sleeve_count=qepm_sleeve_count,
        country_targets=country_targets,
        rust_kernel_flags=rust_flags,
    )

    backtest_prices = [row for row in price_rows if row["price_date"] in set(schedule)]
    backtest_request = normalize_backtest_request(
        rebalance_schedule=schedule,
        prices=backtest_prices,
        benchmark_series=benchmark_rows,
        fx_rates=fx_rows,
        portfolio_allocations=cast(dict[object, list[dict[str, object]]], allocations),
        initial_nav_krw=seed_krw,
    )
    backtest_result_raw, backtest_backend, backtest_reason = run_backtest_kernel(
        request=backtest_request,
        flags=rust_flags,
        python_impl=lambda: run_quarterly_backtest(
            rebalance_schedule=schedule,
            prices=backtest_prices,
            benchmark_series=benchmark_rows,
            fx_rates=fx_rows,
            portfolio_allocations=cast(dict[object, list[dict[str, object]]], allocations),
            initial_nav_krw=seed_krw,
        ),
    )
    backtest_result = backtest_result_raw

    def _oos_runner(fold: object) -> dict[str, object]:
        test_start = cast(date, getattr(fold, "test_start"))
        test_end = cast(date, getattr(fold, "test_end"))
        period_rows = [
            row
            for row in backtest_result.returns
            if row.start_date >= test_start and row.end_date <= test_end
        ]
        if not period_rows:
            return {"portfolio_net_return": 0.0, "benchmark_krw_return": 0.0}

        portfolio_net = _compound_returns([row.net_return for row in period_rows])
        benchmark_krw = _compound_returns([
            float(row.benchmark_krw_return or 0.0) for row in period_rows
        ])
        return {
            "portfolio_net_return": portfolio_net,
            "benchmark_krw_return": benchmark_krw,
        }

    walkforward_artifacts = execute_walkforward_robustness(
        schedule,
        oos_metric_runner=_oos_runner,
        train_years=5,
        test_years=1,
        start_date=start,
        end_date=end,
    )

    backtest_report = build_backtest_attribution_report(backtest_result=backtest_result, fx_rates=fx_rows)
    walkforward_report = build_walkforward_attribution_report(artifacts=walkforward_artifacts)
    compute_stage_seconds = _round_seconds(time.perf_counter() - compute_t0)

    run_id = make_run_id(start, end)
    root = Path("outputs") if output_root is None else output_root
    output_dir = root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    holdings_rows = [cast(dict[str, object], asdict(item)) for item in backtest_result.holdings]
    trades_rows = [cast(dict[str, object], asdict(item)) for item in backtest_result.trades]

    for row in holdings_rows:
        security_id = str(row.get("security_id", ""))
        meta = security_metadata.get(security_id, {"ticker": "", "stock_code": "", "stock_name": ""})
        row.update(meta)

    for row in trades_rows:
        security_id = str(row.get("security_id", ""))
        meta = security_metadata.get(security_id, {"ticker": "", "stock_code": "", "stock_name": ""})
        row.update(meta)

    _write_csv(holdings_rows, output_dir / "holdings.csv")
    _write_csv(trades_rows, output_dir / "trades.csv")

    holdings_non_finite = _count_non_finite_numbers(holdings_rows)
    trades_non_finite = _count_non_finite_numbers(trades_rows)

    metrics_payload = {
        "config": {
            "base_currency": "KRW",
            "supplemental_currency": "USD",
            "seed_krw": seed_krw,
            "factor": factor_label,
            "factor_axis": "R&D",
            "denominator_model": denominator_model,
            "factor_model": factor_model_key,
            "universe_profile": universe_profile_key,
            "qepm_phase_lock_quarter_end": qepm_phase_lock_quarter_end,
            "qepm_sector_active_band": qepm_sector_active_band,
            "qepm_size_stratification": qepm_size_stratification,
            "qepm_te_active_l2_cap": qepm_te_active_l2_cap,
            "qepm_alpha_tilt_strength": qepm_alpha_tilt_strength,
            "qepm_max_adv_participation": qepm_max_adv_participation,
            "qepm_max_turnover": qepm_max_turnover,
            "qepm_staggered_sleeves": qepm_staggered_sleeves,
            "qepm_sleeve_count": qepm_sleeve_count,
            "rebalance_frequency": "quarterly",
            "rebalance_anchor": rebalance_anchor_key,
            "rebalance_anchor_effective": effective_anchor,
            "data_source": source_key,
            "market_scope": market_scope,
            "sec_stage_budget_sec": sec_stage_budget_sec,
            "sec_timeout_sec": sec_timeout_sec,
            "sec_max_retries": sec_max_retries,
            "sec_backoff_sec": sec_backoff_sec,
            "sec_max_rps": sec_max_rps,
            "sec_max_workers": sec_max_workers,
            "sec_allow_stale_cache": sec_allow_stale_cache,
            "yfinance_cache_ttl_days": yfinance_cache_ttl_days,
            "yfinance_chunk_size": yfinance_chunk_size,
            "yfinance_max_workers": yfinance_max_workers,
            "yfinance_max_retries": yfinance_max_retries,
            "yfinance_backoff_sec": yfinance_backoff_sec,
            "dart_timeout_sec": dart_timeout_sec,
            "dart_max_retries": dart_max_retries,
            "dart_backoff_sec": dart_backoff_sec,
            "dart_max_rps": dart_max_rps,
            "dart_max_workers": dart_max_workers,
            "edinet_max_workers": edinet_max_workers,
            "edinet_max_rps": edinet_max_rps,
            "edinet_max_retries": edinet_max_retries,
            "edinet_backoff_sec": edinet_backoff_sec,
            "min_rd_coverage_kr": min_rd_coverage_kr,
            "min_rd_coverage_jp": min_rd_coverage_jp,
            "rust_kernel_flags": rust_flags.as_dict(),
            "rust_backtest_backend": backtest_backend,
            "rust_backtest_reason": backtest_reason,
        },
        "backtest_metrics": _serialize(asdict(backtest_result.metrics)),
        "backtest_report": _serialize(backtest_report),
        "walkforward_report": _serialize(walkforward_report),
        "walkforward_folds": len(walkforward_artifacts),
    }
    metrics_non_finite = _count_non_finite_numbers(metrics_payload)
    _ = (output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )

    non_finite_output_values = holdings_non_finite + trades_non_finite + metrics_non_finite

    quality_report = {
        "run_id": run_id,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "status": "pass" if non_finite_output_values == 0 else "fail",
        "checks": {
            "pit_violations": 0,
            "rejected_candidates": rejected_count,
            "portfolio_fallback_count": fallback_count,
            "holdings_rows": len(holdings_rows),
            "trades_rows": len(trades_rows),
            "periods": int(backtest_result.metrics.periods),
            "non_finite_output_values": non_finite_output_values,
        },
    }
    _ = (output_dir / "data_quality_report.json").write_text(
        json.dumps(quality_report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )

    spreads = [
        float(value)
        for row in factor_integrity_rows
        for value in [row.get("factor_spread_selected_minus_non_selected")]
        if isinstance(value, (int, float))
    ]
    median_ratios = [
        float(value)
        for row in factor_integrity_rows
        for value in [row.get("selected_at_or_above_eligible_median_ratio")]
        if isinstance(value, (int, float))
    ]
    top_quartile_ratios = [
        float(value)
        for row in factor_integrity_rows
        for value in [row.get("selected_top_quartile_rank_ratio")]
        if isinstance(value, (int, float))
    ]
    integrity_payload = {
        "run_id": run_id,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "summary": {
            "periods_evaluated": len(factor_integrity_rows),
            "periods_with_positive_spread": sum(1 for value in spreads if value > 0.0),
            "mean_factor_spread_selected_minus_non_selected": (sum(spreads) / len(spreads)) if spreads else None,
            "mean_selected_at_or_above_eligible_median_ratio": (sum(median_ratios) / len(median_ratios)) if median_ratios else None,
            "mean_selected_top_quartile_rank_ratio": (sum(top_quartile_ratios) / len(top_quartile_ratios)) if top_quartile_ratios else None,
        },
        "periods": _serialize(factor_integrity_rows),
    }
    _ = (output_dir / "factor_integrity_report.json").write_text(
        json.dumps(integrity_payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )

    qepm_alignment_payload = _build_qepm_alignment_report(
        run_id=run_id,
        start=start,
        end=end,
        factor_model=factor_model_key,
        denominator_model=denominator_model,
        rebalance_anchor_requested=rebalance_anchor_key,
        rebalance_anchor_effective=effective_anchor,
        qepm_phase_lock_quarter_end=qepm_phase_lock_quarter_end,
        qepm_sector_active_band=qepm_sector_active_band,
        qepm_size_stratification=qepm_size_stratification,
        qepm_te_active_l2_cap=qepm_te_active_l2_cap,
        qepm_max_adv_participation=qepm_max_adv_participation,
        qepm_max_turnover=qepm_max_turnover,
        qepm_staggered_sleeves=qepm_staggered_sleeves,
        qepm_sleeve_count=qepm_sleeve_count,
    )
    _ = (output_dir / "qepm_alignment_report.json").write_text(
        json.dumps(qepm_alignment_payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )

    runtime_total_seconds = _round_seconds(time.perf_counter() - pipeline_t0)
    baseline_summary = DEFAULT_PERF_BASELINE_SUMMARY if perf_baseline_summary_path is None else perf_baseline_summary_path
    perf_comparison_payload = _build_baseline_comparison_report(
        baseline_summary_path=baseline_summary,
        run_id=run_id,
        scenario_start=start,
        market_scope=market_scope,
        runtime_total_seconds=runtime_total_seconds,
        fallback_count=fallback_count,
        rejected_count=rejected_count,
    )

    perf_telemetry_payload = {
        "schema_version": PERF_TELEMETRY_SCHEMA_VERSION,
        "run": {
            "run_id": run_id,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "market": market_scope,
            "data_source": source_key,
        },
        "runtime_seconds": {
            "total": runtime_total_seconds,
            "ingestion": ingestion_stage_seconds,
            "compute": compute_stage_seconds,
            "artifact_write": _round_seconds(max(0.0, runtime_total_seconds - ingestion_stage_seconds - compute_stage_seconds)),
        },
        "cache_counters": cache_counters,
        "fallback_counters": {
            "portfolio_fallback_count": int(fallback_count),
            "rejected_candidates": int(rejected_count),
        },
        "baseline_comparison_status": str(perf_comparison_payload.get("status", "baseline_missing")),
    }
    _ = (output_dir / "perf_telemetry.json").write_text(
        json.dumps(perf_telemetry_payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    _ = (output_dir / "perf_comparison_report.json").write_text(
        json.dumps(perf_comparison_payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )

    rollout_summary_payload = _build_rollout_summary_artifact(
        run_id=run_id,
        start=start,
        end=end,
        metrics_payload=metrics_payload,
        quality_report=cast(dict[str, object], quality_report),
        qepm_alignment_payload=qepm_alignment_payload,
        perf_comparison_payload=perf_comparison_payload,
    )
    _ = (output_dir / ROLLOUT_SUMMARY_ARTIFACT).write_text(
        json.dumps(rollout_summary_payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )

    pre_manifest_required = tuple(name for name in REQUIRED_ARTIFACTS if name != "manifest.json")
    enforce_quality_gate(output_dir, required_artifacts=pre_manifest_required)

    manifest_checksums: dict[str, dict[str, object]] = {}
    for artifact_name in pre_manifest_required:
        artifact_path = output_dir / artifact_name
        manifest_checksums[artifact_name] = {
            "sha256": _artifact_checksum(artifact_path),
            "size_bytes": artifact_path.stat().st_size,
        }

    manifest_payload = {
        "run_id": run_id,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "checksums": manifest_checksums,
    }
    _ = (output_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )

    enforce_quality_gate(output_dir)
    return output_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic end-to-end artifact pipeline")
    _ = parser.add_argument("--start", required=True, help="ISO date inclusive window start (YYYY-MM-DD)")
    _ = parser.add_argument("--end", required=True, help="ISO date inclusive window end (YYYY-MM-DD)")
    _ = parser.add_argument(
        "--env-file",
        default=".env",
        help="Optional dotenv-style file path for loading environment variables",
    )
    _ = parser.add_argument(
        "--data-source",
        default="synthetic",
        choices=("synthetic", "real"),
        help="Use synthetic fixtures or live external data",
    )
    _ = parser.add_argument(
        "--market",
        default="combined",
        choices=("combined", "us", "kr", "jp"),
        help="Run combined portfolio or a single market portfolio",
    )
    _ = parser.add_argument(
        "--universe-profile",
        default="broad",
        choices=("broad", "core_indices"),
        help="Universe profile: broad (Russell3000+KOSPI200/KOSDAQ150+TOPIX500) or core_indices (S&P500+KOSPI200+Nikkei225)",
    )
    _ = parser.add_argument(
        "--rebalance-anchor",
        default="quarter_end",
        choices=("quarter_end", "start_date"),
        help="Quarterly schedule anchor: quarter_end or start_date",
    )
    _ = parser.add_argument(
        "--factor-model",
        default="rnd_sales_ttm",
        choices=(
            "rnd_sales_ttm",
            "rnd_sales_size_proxy",
            "rnd_mktcap_proxy",
            "rnd_ev_proxy",
            "rnd_robust_composite",
        ),
        help="Factor scoring model variant",
    )
    _ = parser.add_argument(
        "--qepm-phase-lock-quarter-end",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If start_date anchor is requested, optionally lock to quarter_end schedule (diagnostic workaround)",
    )
    _ = parser.add_argument(
        "--qepm-sector-active-band",
        default=0.10,
        type=float,
        help="Benchmark-relative sector active weight band used for QEPM stratification",
    )
    _ = parser.add_argument(
        "--qepm-size-stratification",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable within-sector size-bucket stratification",
    )
    _ = parser.add_argument(
        "--qepm-te-active-l2-cap",
        default=0.08,
        type=float,
        help="Active-weight L2 cap proxy for tracking-error style risk control",
    )
    _ = parser.add_argument(
        "--qepm-alpha-tilt-strength",
        default=0.35,
        type=float,
        help="Strength of factor tilt around benchmark proxy weights",
    )
    _ = parser.add_argument(
        "--qepm-max-adv-participation",
        default=0.10,
        type=float,
        help="Maximum participation cap as a fraction of ADV proxy",
    )
    _ = parser.add_argument(
        "--qepm-max-turnover",
        default=0.35,
        type=float,
        help="Maximum one-way turnover cap per rebalance",
    )
    _ = parser.add_argument(
        "--qepm-staggered-sleeves",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Blend current and prior sleeve targets to reduce phase sensitivity",
    )
    _ = parser.add_argument(
        "--qepm-sleeve-count",
        default=3,
        type=int,
        help="Number of overlapping sleeves for staggered blending",
    )
    _ = parser.add_argument(
        "--separate-markets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run US/KR/JP as separate portfolios in one command",
    )
    _ = parser.add_argument(
        "--separate-markets-max-workers",
        default=3,
        type=int,
        help="Maximum worker processes for separate-market orchestration (1-3)",
    )
    _ = parser.add_argument(
        "--separate-markets-sequential",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force sequential orchestration for separate-market runs",
    )
    _ = parser.add_argument(
        "--rust-kernels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Rust kernels for ranking/constructor/backtest (opt-in)",
    )
    _ = parser.add_argument(
        "--rust-kernel-ranking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Rust ranking kernel only (opt-in)",
    )
    _ = parser.add_argument(
        "--rust-kernel-constructor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Rust constructor kernel only (opt-in)",
    )
    _ = parser.add_argument(
        "--rust-kernel-backtest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Rust backtest kernel only (opt-in)",
    )
    _ = parser.add_argument(
        "--rust-kernel-strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail instead of falling back when Rust kernel is enabled but unavailable",
    )
    _ = parser.add_argument("--seed-krw", default=10_000_000.0, type=float, help="Initial portfolio NAV in KRW")
    _ = parser.add_argument("--sec-stage-budget-sec", default=600, type=int, help="SEC stage time budget in seconds (real mode)")
    _ = parser.add_argument("--sec-timeout-sec", default=20, type=int, help="SEC request timeout seconds")
    _ = parser.add_argument("--sec-max-retries", default=2, type=int, help="SEC request retry count")
    _ = parser.add_argument("--sec-backoff-sec", default=1.0, type=float, help="SEC retry exponential backoff base seconds")
    _ = parser.add_argument("--sec-max-rps", default=8.0, type=float, help="SEC maximum requests per second")
    _ = parser.add_argument("--sec-max-workers", default=4, type=int, help="SEC parallel worker count")
    _ = parser.add_argument(
        "--sec-allow-stale-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow stale SEC cache fallback on network failures",
    )
    _ = parser.add_argument("--yfinance-cache-ttl-days", default=3, type=int, help="yfinance cache TTL in days")
    _ = parser.add_argument("--yfinance-chunk-size", default=80, type=int, help="yfinance multi-ticker chunk size")
    _ = parser.add_argument("--yfinance-max-workers", default=4, type=int, help="yfinance chunk parallel worker count")
    _ = parser.add_argument("--yfinance-max-retries", default=2, type=int, help="yfinance chunk retry count")
    _ = parser.add_argument("--yfinance-backoff-sec", default=1.0, type=float, help="yfinance retry exponential backoff base seconds")
    _ = parser.add_argument("--dart-timeout-sec", default=20, type=int, help="DART request timeout seconds")
    _ = parser.add_argument("--dart-max-retries", default=2, type=int, help="DART request retry count")
    _ = parser.add_argument("--dart-backoff-sec", default=1.0, type=float, help="DART retry exponential backoff base seconds")
    _ = parser.add_argument("--dart-max-rps", default=8.0, type=float, help="DART maximum requests per second")
    _ = parser.add_argument("--dart-max-workers", default=6, type=int, help="DART parallel worker count")
    _ = parser.add_argument("--edinet-max-workers", default=8, type=int, help="EDINET parallel worker count")
    _ = parser.add_argument("--edinet-max-rps", default=5.0, type=float, help="EDINET maximum requests per second")
    _ = parser.add_argument("--edinet-max-retries", default=2, type=int, help="EDINET request retry count")
    _ = parser.add_argument("--edinet-backoff-sec", default=1.0, type=float, help="EDINET retry exponential backoff base seconds")
    _ = parser.add_argument(
        "--min-rd-coverage-kr",
        default=0.60,
        type=float,
        help="Minimum KR security coverage ratio with numeric rd_expense in real mode",
    )
    _ = parser.add_argument(
        "--min-rd-coverage-jp",
        default=0.60,
        type=float,
        help="Minimum JP security coverage ratio with numeric rd_expense in real mode",
    )
    return parser


@dataclass(frozen=True)
class _PipelineSharedConfig:
    data_source: str
    rebalance_anchor: str
    universe_profile: str
    factor_model: str
    qepm_phase_lock_quarter_end: bool
    qepm_sector_active_band: float
    qepm_size_stratification: bool
    qepm_te_active_l2_cap: float
    qepm_alpha_tilt_strength: float
    qepm_max_adv_participation: float
    qepm_max_turnover: float
    qepm_staggered_sleeves: bool
    qepm_sleeve_count: int
    rust_kernels: bool
    rust_kernel_ranking: bool
    rust_kernel_constructor: bool
    rust_kernel_backtest: bool
    rust_kernel_strict: bool
    seed_krw: float
    sec_stage_budget_sec: int
    sec_timeout_sec: int
    sec_max_retries: int
    sec_backoff_sec: float
    sec_max_rps: float
    sec_max_workers: int
    sec_allow_stale_cache: bool
    yfinance_cache_ttl_days: int
    yfinance_chunk_size: int
    yfinance_max_workers: int
    yfinance_max_retries: int
    yfinance_backoff_sec: float
    dart_timeout_sec: int
    dart_max_retries: int
    dart_backoff_sec: float
    dart_max_rps: float
    dart_max_workers: int
    edinet_max_workers: int
    edinet_max_rps: float
    edinet_max_retries: int
    edinet_backoff_sec: float
    min_rd_coverage_kr: float
    min_rd_coverage_jp: float


def _invoke_pipeline_for_market(
    *,
    start: date,
    end: date,
    market_scope: str,
    shared_config: _PipelineSharedConfig,
    output_root: Path | None = None,
) -> Path:
    return run_pipeline(
        start=start,
        end=end,
        output_root=output_root,
        market=market_scope,
        data_source=shared_config.data_source,
        rebalance_anchor=shared_config.rebalance_anchor,
        universe_profile=shared_config.universe_profile,
        factor_model=shared_config.factor_model,
        qepm_phase_lock_quarter_end=shared_config.qepm_phase_lock_quarter_end,
        qepm_sector_active_band=shared_config.qepm_sector_active_band,
        qepm_size_stratification=shared_config.qepm_size_stratification,
        qepm_te_active_l2_cap=shared_config.qepm_te_active_l2_cap,
        qepm_alpha_tilt_strength=shared_config.qepm_alpha_tilt_strength,
        qepm_max_adv_participation=shared_config.qepm_max_adv_participation,
        qepm_max_turnover=shared_config.qepm_max_turnover,
        qepm_staggered_sleeves=shared_config.qepm_staggered_sleeves,
        qepm_sleeve_count=shared_config.qepm_sleeve_count,
        rust_kernels=shared_config.rust_kernels,
        rust_kernel_ranking=shared_config.rust_kernel_ranking,
        rust_kernel_constructor=shared_config.rust_kernel_constructor,
        rust_kernel_backtest=shared_config.rust_kernel_backtest,
        rust_kernel_strict=shared_config.rust_kernel_strict,
        seed_krw=shared_config.seed_krw,
        sec_stage_budget_sec=shared_config.sec_stage_budget_sec,
        sec_timeout_sec=shared_config.sec_timeout_sec,
        sec_max_retries=shared_config.sec_max_retries,
        sec_backoff_sec=shared_config.sec_backoff_sec,
        sec_max_rps=shared_config.sec_max_rps,
        sec_max_workers=shared_config.sec_max_workers,
        sec_allow_stale_cache=shared_config.sec_allow_stale_cache,
        yfinance_cache_ttl_days=shared_config.yfinance_cache_ttl_days,
        yfinance_chunk_size=shared_config.yfinance_chunk_size,
        yfinance_max_workers=shared_config.yfinance_max_workers,
        yfinance_max_retries=shared_config.yfinance_max_retries,
        yfinance_backoff_sec=shared_config.yfinance_backoff_sec,
        dart_timeout_sec=shared_config.dart_timeout_sec,
        dart_max_retries=shared_config.dart_max_retries,
        dart_backoff_sec=shared_config.dart_backoff_sec,
        dart_max_rps=shared_config.dart_max_rps,
        dart_max_workers=shared_config.dart_max_workers,
        edinet_max_workers=shared_config.edinet_max_workers,
        edinet_max_rps=shared_config.edinet_max_rps,
        edinet_max_retries=shared_config.edinet_max_retries,
        edinet_backoff_sec=shared_config.edinet_backoff_sec,
        min_rd_coverage_kr=shared_config.min_rd_coverage_kr,
        min_rd_coverage_jp=shared_config.min_rd_coverage_jp,
    )


def _run_market_pipeline_task(
    start: date,
    end: date,
    market_scope: str,
    output_root: Path,
    shared_config: _PipelineSharedConfig,
) -> tuple[str, Path]:
    output = _invoke_pipeline_for_market(
        start=start,
        end=end,
        market_scope=market_scope,
        shared_config=shared_config,
        output_root=output_root,
    )
    return market_scope, output


def _run_separate_market_pipelines(
    *,
    start: date,
    end: date,
    shared_config: _PipelineSharedConfig,
    max_workers: int,
    force_sequential: bool,
) -> list[Path]:
    market_scopes = ("us", "kr", "jp")
    max_market_workers = len(market_scopes)
    if max_workers <= 0:
        raise ValueError("--separate-markets-max-workers must be positive")
    if max_workers > max_market_workers:
        raise ValueError(f"--separate-markets-max-workers must be <= {max_market_workers}")

    effective_workers = min(max_workers, max_market_workers)
    outputs_by_market: dict[str, Path] = {}
    if force_sequential or effective_workers <= 1:
        for market_scope in market_scopes:
            _, market_output = _run_market_pipeline_task(
                start,
                end,
                market_scope,
                Path("outputs") / "markets" / market_scope,
                shared_config,
            )
            outputs_by_market[market_scope] = market_output
        return [outputs_by_market[scope] for scope in market_scopes]

    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        futures = {
            executor.submit(
                _run_market_pipeline_task,
                start,
                end,
                market_scope,
                Path("outputs") / "markets" / market_scope,
                shared_config,
            ): market_scope
            for market_scope in market_scopes
        }
        for future in as_completed(futures):
            market_scope, market_output = future.result()
            outputs_by_market[market_scope] = market_output
    return [outputs_by_market[scope] for scope in market_scopes]


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _ = _load_env_file(Path(cast(str, args.env_file)))

    try:
        start = _parse_iso_date(cast(str, args.start), field_name="--start")
        end = _parse_iso_date(cast(str, args.end), field_name="--end")
        data_source = cast(str, args.data_source)
        rebalance_anchor = cast(str, args.rebalance_anchor)
        universe_profile = cast(str, args.universe_profile)
        factor_model = cast(str, args.factor_model)
        qepm_phase_lock_quarter_end = cast(bool, args.qepm_phase_lock_quarter_end)
        qepm_sector_active_band = cast(float, args.qepm_sector_active_band)
        qepm_size_stratification = cast(bool, args.qepm_size_stratification)
        qepm_te_active_l2_cap = cast(float, args.qepm_te_active_l2_cap)
        qepm_alpha_tilt_strength = cast(float, args.qepm_alpha_tilt_strength)
        qepm_max_adv_participation = cast(float, args.qepm_max_adv_participation)
        qepm_max_turnover = cast(float, args.qepm_max_turnover)
        qepm_staggered_sleeves = cast(bool, args.qepm_staggered_sleeves)
        qepm_sleeve_count = cast(int, args.qepm_sleeve_count)
        rust_kernels = cast(bool, args.rust_kernels)
        rust_kernel_ranking = cast(bool, args.rust_kernel_ranking)
        rust_kernel_constructor = cast(bool, args.rust_kernel_constructor)
        rust_kernel_backtest = cast(bool, args.rust_kernel_backtest)
        rust_kernel_strict = cast(bool, args.rust_kernel_strict)
        seed_krw = cast(float, args.seed_krw)
        sec_stage_budget_sec = cast(int, args.sec_stage_budget_sec)
        sec_timeout_sec = cast(int, args.sec_timeout_sec)
        sec_max_retries = cast(int, args.sec_max_retries)
        sec_backoff_sec = cast(float, args.sec_backoff_sec)
        sec_max_rps = cast(float, args.sec_max_rps)
        sec_max_workers = cast(int, args.sec_max_workers)
        sec_allow_stale_cache = cast(bool, args.sec_allow_stale_cache)
        yfinance_cache_ttl_days = cast(int, args.yfinance_cache_ttl_days)
        yfinance_chunk_size = cast(int, args.yfinance_chunk_size)
        yfinance_max_workers = cast(int, args.yfinance_max_workers)
        yfinance_max_retries = cast(int, args.yfinance_max_retries)
        yfinance_backoff_sec = cast(float, args.yfinance_backoff_sec)
        dart_timeout_sec = cast(int, args.dart_timeout_sec)
        dart_max_retries = cast(int, args.dart_max_retries)
        dart_backoff_sec = cast(float, args.dart_backoff_sec)
        dart_max_rps = cast(float, args.dart_max_rps)
        dart_max_workers = cast(int, args.dart_max_workers)
        edinet_max_workers = cast(int, args.edinet_max_workers)
        edinet_max_rps = cast(float, args.edinet_max_rps)
        edinet_max_retries = cast(int, args.edinet_max_retries)
        edinet_backoff_sec = cast(float, args.edinet_backoff_sec)
        min_rd_coverage_kr = cast(float, args.min_rd_coverage_kr)
        min_rd_coverage_jp = cast(float, args.min_rd_coverage_jp)
        separate_markets_max_workers = cast(int, args.separate_markets_max_workers)
        separate_markets_sequential = cast(bool, args.separate_markets_sequential)

        shared_pipeline_config = _PipelineSharedConfig(
            data_source=data_source,
            rebalance_anchor=rebalance_anchor,
            universe_profile=universe_profile,
            factor_model=factor_model,
            qepm_phase_lock_quarter_end=qepm_phase_lock_quarter_end,
            qepm_sector_active_band=qepm_sector_active_band,
            qepm_size_stratification=qepm_size_stratification,
            qepm_te_active_l2_cap=qepm_te_active_l2_cap,
            qepm_alpha_tilt_strength=qepm_alpha_tilt_strength,
            qepm_max_adv_participation=qepm_max_adv_participation,
            qepm_max_turnover=qepm_max_turnover,
            qepm_staggered_sleeves=qepm_staggered_sleeves,
            qepm_sleeve_count=qepm_sleeve_count,
            rust_kernels=rust_kernels,
            rust_kernel_ranking=rust_kernel_ranking,
            rust_kernel_constructor=rust_kernel_constructor,
            rust_kernel_backtest=rust_kernel_backtest,
            rust_kernel_strict=rust_kernel_strict,
            seed_krw=seed_krw,
            sec_stage_budget_sec=sec_stage_budget_sec,
            sec_timeout_sec=sec_timeout_sec,
            sec_max_retries=sec_max_retries,
            sec_backoff_sec=sec_backoff_sec,
            sec_max_rps=sec_max_rps,
            sec_max_workers=sec_max_workers,
            sec_allow_stale_cache=sec_allow_stale_cache,
            yfinance_cache_ttl_days=yfinance_cache_ttl_days,
            yfinance_chunk_size=yfinance_chunk_size,
            yfinance_max_workers=yfinance_max_workers,
            yfinance_max_retries=yfinance_max_retries,
            yfinance_backoff_sec=yfinance_backoff_sec,
            dart_timeout_sec=dart_timeout_sec,
            dart_max_retries=dart_max_retries,
            dart_backoff_sec=dart_backoff_sec,
            dart_max_rps=dart_max_rps,
            dart_max_workers=dart_max_workers,
            edinet_max_workers=edinet_max_workers,
            edinet_max_rps=edinet_max_rps,
            edinet_max_retries=edinet_max_retries,
            edinet_backoff_sec=edinet_backoff_sec,
            min_rd_coverage_kr=min_rd_coverage_kr,
            min_rd_coverage_jp=min_rd_coverage_jp,
        )

        if cast(bool, args.separate_markets):
            if cast(str, args.market) != "combined":
                raise ValueError("--separate-markets cannot be combined with --market other than 'combined'")
            outputs = _run_separate_market_pipelines(
                start=start,
                end=end,
                shared_config=shared_pipeline_config,
                max_workers=separate_markets_max_workers,
                force_sequential=separate_markets_sequential,
            )
            for market_scope, market_output in zip(("us", "kr", "jp"), outputs, strict=True):
                print(f"pipeline complete ({market_scope}): {market_output}")
            print(f"separate market pipelines complete: {len(outputs)} runs")
            return 0

        output_dir = _invoke_pipeline_for_market(
            start=start,
            end=end,
            market_scope=cast(str, args.market),
            shared_config=shared_pipeline_config,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"pipeline complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REQUIRED_ARTIFACTS",
    "enforce_quality_gate",
    "main",
    "make_run_id",
    "run_pipeline",
]
