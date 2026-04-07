from __future__ import annotations

import io
import json
import math
import os
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Protocol, cast
from urllib import parse, request
from urllib.error import HTTPError, URLError
from zipfile import BadZipFile, ZipFile

from .benchmark import load_benchmark_series
from .fx import normalize_fx_rows
from .jp_edinet import parse_jp_edinet_fundamentals
from .kr_dart import transform_kr_dart_fundamentals
from .us_sec import transform_us_sec_fundamentals


class LiveDataConfigError(ValueError):
    pass


@dataclass(frozen=True)
class LiveDataConfig:
    sec_user_agent: str
    dart_api_key: str
    edinet_api_key: str
    benchmark_us_ticker: str
    benchmark_kr_ticker: str
    benchmark_jp_ticker: str


@dataclass(frozen=True)
class LiveIngestOptions:
    sec_timeout_sec: int = 20
    sec_max_retries: int = 2
    sec_backoff_sec: float = 1.0
    sec_max_rps: float = 8.0
    sec_max_workers: int = 4
    sec_stage_budget_sec: int | None = 600
    sec_allow_stale_cache: bool = True
    yfinance_cache_ttl_days: int = 3
    yfinance_chunk_size: int = 80
    yfinance_max_workers: int = 4
    yfinance_max_retries: int = 2
    yfinance_backoff_sec: float = 1.0
    dart_timeout_sec: int = 20
    dart_max_retries: int = 2
    dart_backoff_sec: float = 1.0
    dart_max_rps: float = 8.0
    dart_max_workers: int = 6
    edinet_max_workers: int = 8
    edinet_max_rps: float = 5.0
    edinet_max_retries: int = 2
    edinet_backoff_sec: float = 1.0


_SEC_RATE_LOCK = threading.Lock()
_sec_next_request_at = 0.0
_EDINET_RATE_LOCK = threading.Lock()
_edinet_next_request_at = 0.0
_JP_DOC_INDEX_LOCK = threading.Lock()
_jp_doc_index_cache: dict[str, dict[date, tuple[dict[str, object], ...]]] = {}
_YFINANCE_MIGRATION_EVENTS = (
    "v2_hit",
    "v1_fallback",
    "network_fetch",
    "repair_success",
    "repair_failure",
)
_YFINANCE_MIGRATION_STATS_LOCK = threading.Lock()
_yfinance_migration_stats: dict[str, int] = {event: 0 for event in _YFINANCE_MIGRATION_EVENTS}


class SecuritySpecLike(Protocol):
    @property
    def security_id(self) -> str: ...

    @property
    def universe(self) -> str: ...

    @property
    def country(self) -> str: ...

    @property
    def currency(self) -> str: ...

    @property
    def ticker(self) -> str: ...

    @property
    def stock_code(self) -> str: ...

    @property
    def stock_name(self) -> str: ...


def _clean_secret(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1].strip()
    return text


def _required_env(name: str) -> str:
    raw = os.getenv(name, "")
    cleaned = _clean_secret(raw)
    if not cleaned:
        raise LiveDataConfigError(f"missing required environment variable: {name}")
    return cleaned


def _optional_env(name: str, default: str) -> str:
    raw = os.getenv(name, default)
    cleaned = _clean_secret(raw)
    return cleaned or default


def load_live_data_config_from_env() -> LiveDataConfig:
    missing: list[str] = []
    for key in ("SEC_USER_AGENT", "DART_API_KEY", "EDINET_API_KEY"):
        if not _clean_secret(os.getenv(key, "")):
            missing.append(key)
    if missing:
        raise LiveDataConfigError(f"missing required environment variables: {', '.join(missing)}")

    return LiveDataConfig(
        sec_user_agent=_required_env("SEC_USER_AGENT"),
        dart_api_key=_required_env("DART_API_KEY"),
        edinet_api_key=_required_env("EDINET_API_KEY"),
        benchmark_us_ticker=_optional_env("BENCH_US", "IWB"),
        benchmark_kr_ticker=_optional_env("BENCH_KR", "069500.KS"),
        benchmark_jp_ticker=_optional_env("BENCH_JP", "1306.T"),
    )


def _cache_path(cache_root: Path, *parts: str) -> Path:
    path = cache_root.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_cached_json(path: Path) -> object | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, JSONDecodeError):
        return None


def _save_cached_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    _ = temp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    _ = temp_path.replace(path)


def _sanitize_cache_key(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text)


def _sec_rate_limit_wait(max_rps: float) -> None:
    global _sec_next_request_at
    if max_rps <= 0:
        return
    min_interval = 1.0 / max_rps
    while True:
        with _SEC_RATE_LOCK:
            now = time.monotonic()
            if now >= _sec_next_request_at:
                _sec_next_request_at = now + min_interval
                return
            wait_for = _sec_next_request_at - now
        if wait_for > 0:
            time.sleep(wait_for)


def _edinet_rate_limit_wait(max_rps: float) -> None:
    global _edinet_next_request_at
    if max_rps <= 0:
        return
    min_interval = 1.0 / max_rps
    while True:
        with _EDINET_RATE_LOCK:
            now = time.monotonic()
            if now >= _edinet_next_request_at:
                _edinet_next_request_at = now + min_interval
                return
            wait_for = _edinet_next_request_at - now
        if wait_for > 0:
            time.sleep(wait_for)


def _http_get_json(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout_sec: int = 60,
    max_retries: int = 0,
    backoff_sec: float = 0.0,
    sec_max_rps: float | None = None,
) -> object:
    req = request.Request(url=url, headers=dict(headers or {}))
    attempts = max(1, max_retries + 1)
    last_error: Exception | None = None
    for attempt in range(attempts):
        if sec_max_rps is not None:
            _sec_rate_limit_wait(sec_max_rps)
        try:
            with request.urlopen(req, timeout=timeout_sec) as response:
                body = response.read()
            return json.loads(body.decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, JSONDecodeError) as exc:
            last_error = exc
            if attempt >= attempts - 1:
                break
            if backoff_sec > 0.0:
                time.sleep(backoff_sec * (2**attempt))
    raise LiveDataConfigError(f"http request failed for {url}: {last_error}") from last_error


def _http_get_bytes(url: str, *, headers: Mapping[str, str] | None = None) -> bytes:
    req = request.Request(url=url, headers=dict(headers or {}))
    try:
        with request.urlopen(req, timeout=60) as response:
            return response.read()
    except (HTTPError, URLError) as exc:
        raise LiveDataConfigError(f"http request failed for {url}: {exc}") from exc


def _as_date(text: str) -> date:
    return date.fromisoformat(text[:10])


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(str(value))
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if not cleaned:
            return None
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = f"-{cleaned[1:-1]}"
        try:
            parsed = float(cleaned)
            return parsed if math.isfinite(parsed) else None
        except ValueError:
            return None
    try:
        parsed = float(str(value).strip())
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def _normalized_label(value: object) -> str:
    text = str(value or "").strip().lower()
    return "".join(ch for ch in text if ch.isalnum())


def _normalized_stock_code(value: str) -> str:
    compact = "".join(ch for ch in value.upper() if ch.isalnum())
    if not compact:
        return ""
    if compact.isdigit():
        return compact.zfill(4) if len(compact) < 4 else compact[:4]
    return compact[:4]


def _required_dates(schedule: Sequence[date]) -> list[date]:
    return sorted({*schedule, *(day - timedelta(days=7) for day in schedule), *(day - timedelta(days=14) for day in schedule)})


def _value_on_or_before(series: Mapping[date, tuple[float, float]], target: date) -> tuple[float, float]:
    candidates = [day for day in series if day <= target]
    if not candidates:
        raise LiveDataConfigError(f"no time-series value available on or before {target.isoformat()}")
    latest = max(candidates)
    return series[latest]


def _yfinance_cache_file(*, cache_root: Path, ticker: str, start: date, end: date) -> Path:
    return _cache_path(
        cache_root,
        "yfinance",
        f"{_sanitize_cache_key(ticker)}_{start.isoformat()}_{end.isoformat()}.json",
    )


def _load_cached_yfinance_series(*, cache_file: Path, cache_ttl_days: int) -> dict[date, tuple[float, float]] | None:
    cached = _load_cached_json(cache_file)
    if not isinstance(cached, Mapping):
        return None
    try:
        age_days = (time.time() - cache_file.stat().st_mtime) / 86400.0
    except OSError:
        return None
    if cache_ttl_days > 0 and age_days > float(cache_ttl_days):
        return None

    raw_rows = cached.get("rows")
    if not isinstance(raw_rows, Sequence):
        return None
    parsed_rows: dict[date, tuple[float, float]] = {}
    for row in raw_rows:
        if not isinstance(row, Mapping):
            continue
        row_date_text = str(row.get("date", "")).strip()
        close = _to_float(row.get("close"))
        volume = _to_float(row.get("volume"))
        if not row_date_text or close is None:
            continue
        try:
            row_date = date.fromisoformat(row_date_text)
        except ValueError:
            continue
        parsed_rows[row_date] = (close, max(volume or 0.0, 0.0))
    return parsed_rows or None


def _save_cached_yfinance_series(
    *,
    cache_file: Path,
    ticker: str,
    start: date,
    end: date,
    series: Mapping[date, tuple[float, float]],
) -> None:
    _save_cached_json(
        cache_file,
        {
            "ticker": ticker,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "rows": [
                {"date": row_date.isoformat(), "close": values[0], "volume": values[1]}
                for row_date, values in sorted(series.items(), key=lambda pair: pair[0])
            ],
        },
    )


def _yfinance_cache_v2_partition_file(*, cache_root: Path, ticker: str, year: int) -> Path:
    return _cache_path(
        cache_root,
        "yfinance",
        "v2",
        _sanitize_cache_key(ticker),
        f"{year:04d}.json",
    )


def _load_cached_yfinance_partition_series_v2(
    *,
    cache_file: Path,
    ticker: str,
    year: int,
    cache_ttl_days: int,
) -> dict[date, tuple[float, float]] | None:
    cached = _load_cached_json(cache_file)
    if not isinstance(cached, Mapping):
        return None
    try:
        age_days = (time.time() - cache_file.stat().st_mtime) / 86400.0
    except OSError:
        return None
    if cache_ttl_days > 0 and age_days > float(cache_ttl_days):
        return None

    expected_ticker_norm = _sanitize_cache_key(ticker)
    if str(cached.get("version", "")).strip() != "v2":
        return None
    if str(cached.get("provider", "")).strip() != "yfinance":
        return None
    if str(cached.get("ticker", "")).strip() != ticker:
        return None
    if str(cached.get("ticker_norm", "")).strip() != expected_ticker_norm:
        return None
    payload_year = cached.get("year")
    if not isinstance(payload_year, int) or isinstance(payload_year, bool) or payload_year != year:
        return None

    source_window = cached.get("source_window")
    if not isinstance(source_window, Mapping):
        return None
    source_start = str(source_window.get("start", "")).strip()
    source_end = str(source_window.get("end", "")).strip()
    if not source_start or not source_end:
        return None
    try:
        _ = date.fromisoformat(source_start)
        _ = date.fromisoformat(source_end)
    except ValueError:
        return None

    raw_rows = cached.get("rows")
    if not isinstance(raw_rows, Sequence):
        return None
    parsed_rows: dict[date, tuple[float, float]] = {}
    for row in raw_rows:
        if not isinstance(row, Mapping):
            continue
        row_date_text = str(row.get("date", "")).strip()
        close = _to_float(row.get("close"))
        volume = _to_float(row.get("volume"))
        if not row_date_text or close is None:
            continue
        try:
            row_date = date.fromisoformat(row_date_text)
        except ValueError:
            continue
        if row_date.year != year:
            return None
        parsed_rows[row_date] = (close, max(volume or 0.0, 0.0))

    if not parsed_rows:
        return None
    return {row_date: parsed_rows[row_date] for row_date in sorted(parsed_rows)}


def _save_cached_yfinance_partition_series_v2(
    *,
    cache_file: Path,
    ticker: str,
    year: int,
    start: date,
    end: date,
    series: Mapping[date, tuple[float, float]],
) -> None:
    for row_date in series:
        if row_date.year != year:
            raise LiveDataConfigError(
                f"yfinance v2 partition-year mismatch for {ticker}: expected {year}, got {row_date.isoformat()}"
            )

    _save_cached_json(
        cache_file,
        {
            "version": "v2",
            "provider": "yfinance",
            "ticker": ticker,
            "ticker_norm": _sanitize_cache_key(ticker),
            "year": year,
            "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "source_window": {"start": start.isoformat(), "end": end.isoformat()},
            "rows": [
                {"date": row_date.isoformat(), "close": values[0], "volume": values[1]}
                for row_date, values in sorted(series.items(), key=lambda pair: pair[0])
            ],
        },
    )


def _save_cached_yfinance_series_v2(
    *,
    cache_root: Path,
    ticker: str,
    start: date,
    end: date,
    series: Mapping[date, tuple[float, float]],
) -> None:
    per_year: dict[int, dict[date, tuple[float, float]]] = {}
    for row_date, values in sorted(series.items(), key=lambda pair: pair[0]):
        bucket = per_year.setdefault(row_date.year, {})
        bucket[row_date] = values

    for partition_year, partition_rows in sorted(per_year.items(), key=lambda pair: pair[0]):
        cache_file = _yfinance_cache_v2_partition_file(cache_root=cache_root, ticker=ticker, year=partition_year)
        _save_cached_yfinance_partition_series_v2(
            cache_file=cache_file,
            ticker=ticker,
            year=partition_year,
            start=start,
            end=end,
            series=partition_rows,
        )


def _stitch_yfinance_partition_series_v2(
    *,
    partitions: Mapping[int, Mapping[date, tuple[float, float]]],
    start: date,
    end: date,
) -> dict[date, tuple[float, float]]:
    stitched_rows: list[tuple[date, tuple[float, float]]] = []
    for _, partition_rows in sorted(partitions.items(), key=lambda pair: pair[0]):
        for row_date, values in partition_rows.items():
            if start <= row_date <= end:
                stitched_rows.append((row_date, values))

    stitched_rows.sort(key=lambda pair: pair[0])
    return {row_date: values for row_date, values in stitched_rows}


def _record_yfinance_migration_stat(event: str) -> None:
    if event not in _yfinance_migration_stats:
        return
    with _YFINANCE_MIGRATION_STATS_LOCK:
        _yfinance_migration_stats[event] = _yfinance_migration_stats.get(event, 0) + 1


def _reset_yfinance_migration_stats() -> None:
    with _YFINANCE_MIGRATION_STATS_LOCK:
        for event in _YFINANCE_MIGRATION_EVENTS:
            _yfinance_migration_stats[event] = 0


def _get_yfinance_migration_stats() -> dict[str, int]:
    with _YFINANCE_MIGRATION_STATS_LOCK:
        return {event: int(_yfinance_migration_stats.get(event, 0)) for event in _YFINANCE_MIGRATION_EVENTS}


def _load_cached_yfinance_series_v2(
    *,
    cache_root: Path,
    ticker: str,
    start: date,
    end: date,
    cache_ttl_days: int,
) -> dict[date, tuple[float, float]] | None:
    partitions: dict[int, dict[date, tuple[float, float]]] = {}
    for partition_year in range(start.year, end.year + 1):
        cache_file = _yfinance_cache_v2_partition_file(cache_root=cache_root, ticker=ticker, year=partition_year)
        partition_rows = _load_cached_yfinance_partition_series_v2(
            cache_file=cache_file,
            ticker=ticker,
            year=partition_year,
            cache_ttl_days=cache_ttl_days,
        )
        if partition_rows is None:
            return None
        partitions[partition_year] = partition_rows

    stitched = _stitch_yfinance_partition_series_v2(partitions=partitions, start=start, end=end)
    return stitched or None


def _repair_yfinance_series_v2_from_v1(
    *,
    cache_root: Path,
    ticker: str,
    start: date,
    end: date,
    series: Mapping[date, tuple[float, float]],
) -> None:
    try:
        _save_cached_yfinance_series_v2(
            cache_root=cache_root,
            ticker=ticker,
            start=start,
            end=end,
            series=series,
        )
    except Exception:
        _record_yfinance_migration_stat("repair_failure")
    else:
        _record_yfinance_migration_stat("repair_success")


def _load_cached_yfinance_series_with_migration(
    *,
    cache_root: Path,
    ticker: str,
    start: date,
    end: date,
    cache_ttl_days: int,
) -> dict[date, tuple[float, float]] | None:
    cached_v2 = _load_cached_yfinance_series_v2(
        cache_root=cache_root,
        ticker=ticker,
        start=start,
        end=end,
        cache_ttl_days=cache_ttl_days,
    )
    if cached_v2 is not None:
        _record_yfinance_migration_stat("v2_hit")
        return cached_v2

    cache_file = _yfinance_cache_file(cache_root=cache_root, ticker=ticker, start=start, end=end)
    cached_v1 = _load_cached_yfinance_series(cache_file=cache_file, cache_ttl_days=cache_ttl_days)
    if cached_v1 is None:
        return None

    _record_yfinance_migration_stat("v1_fallback")
    _repair_yfinance_series_v2_from_v1(
        cache_root=cache_root,
        ticker=ticker,
        start=start,
        end=end,
        series=cached_v1,
    )
    return cached_v1


def _extract_series_from_frame(*, frame: object, ticker: str) -> dict[date, tuple[float, float]]:
    frame_any = cast(Any, frame)
    close_series = None
    volume_series = None
    try:
        if ticker in frame_any:
            sub = frame_any[ticker]
            if hasattr(sub, "__contains__") and "Close" in sub:
                close_series = sub["Close"]
            if hasattr(sub, "__contains__") and "Volume" in sub:
                volume_series = sub["Volume"]
    except Exception:
        pass

    if close_series is None:
        candidate = ("Close", ticker)
        if candidate in frame_any:
            close_series = frame_any[candidate]
    if volume_series is None:
        candidate = ("Volume", ticker)
        if candidate in frame_any:
            volume_series = frame_any[candidate]
    if close_series is None and "Close" in frame_any:
        close_series = frame_any["Close"]
    if volume_series is None and "Volume" in frame_any:
        volume_series = frame_any["Volume"]

    if close_series is None:
        return {}

    if hasattr(close_series, "ndim") and cast(int, getattr(close_series, "ndim")) == 2:
        close_series = close_series.iloc[:, 0]
    if volume_series is not None and hasattr(volume_series, "ndim") and cast(int, getattr(volume_series, "ndim")) == 2:
        volume_series = volume_series.iloc[:, 0]

    series: dict[date, tuple[float, float]] = {}
    for timestamp, close_value in close_series.items():
        close = _to_float(close_value)
        if close is None:
            continue
        volume = 0.0
        if volume_series is not None:
            raw_volume = volume_series.get(timestamp)
            parsed_volume = _to_float(raw_volume)
            if parsed_volume is not None:
                volume = max(parsed_volume, 0.0)
        row_date = cast(datetime, timestamp).date() if hasattr(timestamp, "date") else date.fromisoformat(str(timestamp)[:10])
        series[row_date] = (close, volume)
    return series


def _download_yfinance_bulk_series(
    *,
    tickers: Sequence[str],
    start: date,
    end: date,
    cache_root: Path,
    options: LiveIngestOptions,
) -> dict[str, dict[date, tuple[float, float]]]:
    by_ticker: dict[str, dict[date, tuple[float, float]]] = {}
    missing: list[str] = []
    for ticker in tickers:
        cached = _load_cached_yfinance_series_with_migration(
            cache_root=cache_root,
            ticker=ticker,
            start=start,
            end=end,
            cache_ttl_days=options.yfinance_cache_ttl_days,
        )
        if cached is not None:
            by_ticker[ticker] = cached
        else:
            missing.append(ticker)

    if not missing:
        return by_ticker

    try:
        import yfinance as yf
    except ImportError as exc:
        raise LiveDataConfigError("yfinance package is required for real data mode") from exc

    def _download_chunk(chunk: list[str]) -> dict[str, dict[date, tuple[float, float]]]:
        attempts = max(1, options.yfinance_max_retries + 1)
        for attempt in range(attempts):
            frame = yf.download(
                tickers=list(chunk),
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                progress=False,
                auto_adjust=False,
                interval="1d",
                group_by="column",
                threads=False,
            )
            if frame is not None and not getattr(frame, "empty", True):
                mapped: dict[str, dict[date, tuple[float, float]]] = {}
                for ticker in chunk:
                    series = _extract_series_from_frame(frame=frame, ticker=ticker)
                    if series:
                        mapped[ticker] = series
                if mapped:
                    return mapped
            if attempt < attempts - 1 and options.yfinance_backoff_sec > 0.0:
                time.sleep(options.yfinance_backoff_sec * (2**attempt))
        return {}

    size = max(1, options.yfinance_chunk_size)
    chunks = [missing[start_index : start_index + size] for start_index in range(0, len(missing), size)]
    workers = max(1, options.yfinance_max_workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_chunk, chunk): chunk for chunk in chunks}
        for future in as_completed(futures):
            chunk = futures[future]
            chunk_result = future.result()
            for ticker, series in chunk_result.items():
                by_ticker[ticker] = series
                _record_yfinance_migration_stat("network_fetch")
                cache_file = _yfinance_cache_file(cache_root=cache_root, ticker=ticker, start=start, end=end)
                _save_cached_yfinance_series(
                    cache_file=cache_file,
                    ticker=ticker,
                    start=start,
                    end=end,
                    series=series,
                )
                _save_cached_yfinance_series_v2(
                    cache_root=cache_root,
                    ticker=ticker,
                    start=start,
                    end=end,
                    series=series,
                )
            unresolved = [ticker for ticker in chunk if ticker not in chunk_result]
            for ticker in unresolved:
                try:
                    series = _download_yfinance_series(
                        ticker,
                        start=start,
                        end=end,
                        cache_root=cache_root,
                        cache_ttl_days=options.yfinance_cache_ttl_days,
                    )
                except LiveDataConfigError:
                    continue
                by_ticker[ticker] = series
    return by_ticker


def _download_yfinance_series(
    ticker: str,
    *,
    start: date,
    end: date,
    cache_root: Path,
    cache_ttl_days: int,
) -> dict[date, tuple[float, float]]:
    cached = _load_cached_yfinance_series_with_migration(
        cache_root=cache_root,
        ticker=ticker,
        start=start,
        end=end,
        cache_ttl_days=cache_ttl_days,
    )
    if cached is not None:
        return cached

    try:
        import yfinance as yf
    except ImportError as exc:
        raise LiveDataConfigError("yfinance package is required for real data mode") from exc

    frame = yf.download(
        tickers=ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=False,
        interval="1d",
        group_by="column",
        threads=False,
    )
    if frame is None or getattr(frame, "empty", True):
        raise LiveDataConfigError(f"yfinance returned no rows for ticker: {ticker}")

    close_series = None
    volume_series = None
    if "Close" in frame:
        close_series = frame["Close"]
    if "Volume" in frame:
        volume_series = frame["Volume"]

    if close_series is None:
        candidate = ("Close", ticker)
        if candidate in frame:
            close_series = frame[candidate]
    if volume_series is None:
        candidate = ("Volume", ticker)
        if candidate in frame:
            volume_series = frame[candidate]

    if close_series is None:
        raise LiveDataConfigError(f"close prices missing for ticker: {ticker}")

    if hasattr(close_series, "ndim") and cast(int, getattr(close_series, "ndim")) == 2:
        close_series = close_series.iloc[:, 0]
    if volume_series is not None and hasattr(volume_series, "ndim") and cast(int, getattr(volume_series, "ndim")) == 2:
        volume_series = volume_series.iloc[:, 0]

    series: dict[date, tuple[float, float]] = {}
    for timestamp, close_value in close_series.items():
        close = _to_float(close_value)
        if close is None:
            continue
        volume = 0.0
        if volume_series is not None:
            raw_volume = volume_series.get(timestamp)
            parsed_volume = _to_float(raw_volume)
            if parsed_volume is not None:
                volume = max(parsed_volume, 0.0)
        row_date = cast(datetime, timestamp).date() if hasattr(timestamp, "date") else date.fromisoformat(str(timestamp)[:10])
        series[row_date] = (close, volume)

    if not series:
        raise LiveDataConfigError(f"no valid close rows for ticker: {ticker}")

    _record_yfinance_migration_stat("network_fetch")
    cache_file = _yfinance_cache_file(cache_root=cache_root, ticker=ticker, start=start, end=end)
    _save_cached_yfinance_series(
        cache_file=cache_file,
        ticker=ticker,
        start=start,
        end=end,
        series=series,
    )
    _save_cached_yfinance_series_v2(
        cache_root=cache_root,
        ticker=ticker,
        start=start,
        end=end,
        series=series,
    )
    return series


def _read_yfinance_series_for_loader(
    ticker: str,
    *,
    start: date,
    end: date,
    cache_root: Path,
    options: LiveIngestOptions,
) -> dict[date, tuple[float, float]]:
    return _download_yfinance_series(
        ticker,
        start=start,
        end=end,
        cache_root=cache_root,
        cache_ttl_days=options.yfinance_cache_ttl_days,
    )


def _read_yfinance_bulk_series_for_loader(
    *,
    tickers: Sequence[str],
    start: date,
    end: date,
    cache_root: Path,
    options: LiveIngestOptions,
) -> dict[str, dict[date, tuple[float, float]]]:
    return _download_yfinance_bulk_series(
        tickers=tickers,
        start=start,
        end=end,
        cache_root=cache_root,
        options=options,
    )


def _sec_ticker_to_cik(*, config: LiveDataConfig, cache_root: Path) -> dict[str, str]:
    cache_file = _cache_path(cache_root, "sec", "company_tickers.json")
    cached = _load_cached_json(cache_file)
    payload = cached
    if payload is None:
        payload = _http_get_json(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": config.sec_user_agent, "Accept": "application/json"},
        )
        _save_cached_json(cache_file, payload)

    if not isinstance(payload, Mapping):
        raise LiveDataConfigError("SEC ticker payload is malformed")

    mapping: dict[str, str] = {}
    for row in payload.values():
        if not isinstance(row, Mapping):
            continue
        ticker = str(row.get("ticker", "")).strip().upper()
        cik_raw = str(row.get("cik_str", "")).strip()
        if not ticker or not cik_raw.isdigit():
            continue
        mapping[ticker] = cik_raw.zfill(10)
    return mapping


def _sec_company_facts(
    cik: str,
    *,
    config: LiveDataConfig,
    cache_root: Path,
    options: LiveIngestOptions,
    deadline_monotonic: float | None = None,
) -> Mapping[str, object]:
    cache_file = _cache_path(cache_root, "sec", "companyfacts", f"{cik}.json")
    cached = _load_cached_json(cache_file)
    payload = cached if isinstance(cached, Mapping) else None
    if payload is None:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise LiveDataConfigError(f"SEC stage budget exhausted before fetching CIK {cik}")
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        try:
            payload = _http_get_json(
                url,
                headers={"User-Agent": config.sec_user_agent, "Accept": "application/json"},
                timeout_sec=options.sec_timeout_sec,
                max_retries=options.sec_max_retries,
                backoff_sec=options.sec_backoff_sec,
                sec_max_rps=options.sec_max_rps,
            )
            _save_cached_json(cache_file, payload)
        except LiveDataConfigError:
            if options.sec_allow_stale_cache and isinstance(cached, Mapping):
                payload = cached
            else:
                raise
    if not isinstance(payload, Mapping):
        raise LiveDataConfigError(f"SEC companyfacts malformed for CIK {cik}")
    return cast(Mapping[str, object], payload)


def _prefetch_sec_companyfacts(
    *,
    ciks: Sequence[str],
    config: LiveDataConfig,
    cache_root: Path,
    options: LiveIngestOptions,
    deadline_monotonic: float | None,
) -> None:
    if not ciks:
        return

    def _load_single(cik: str) -> None:
        _ = _sec_company_facts(
            cik,
            config=config,
            cache_root=cache_root,
            options=options,
            deadline_monotonic=deadline_monotonic,
        )

    workers = max(1, options.sec_max_workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_load_single, cik): cik for cik in ciks}
        for future in as_completed(futures):
            _ = futures[future]
            try:
                future.result()
            except LiveDataConfigError:
                continue


def _sec_points(facts: Mapping[str, object], tags: Sequence[str]) -> dict[date, tuple[date, float]]:
    us_gaap = cast(Mapping[str, object], cast(Mapping[str, object], facts.get("facts", {})).get("us-gaap", {}))
    collected: dict[date, tuple[date, float]] = {}
    allowed_forms = {"10-Q", "10-K", "20-F", "40-F"}
    for tag in tags:
        tag_payload = us_gaap.get(tag)
        if not isinstance(tag_payload, Mapping):
            continue
        units = cast(Mapping[str, object], tag_payload.get("units", {}))
        for rows in units.values():
            if not isinstance(rows, Sequence):
                continue
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                form = str(row.get("form", "")).strip().upper()
                if form and form not in allowed_forms:
                    continue
                raw_end = str(row.get("end", "")).strip()
                raw_filed = str(row.get("filed", "")).strip()
                value = _to_float(row.get("val"))
                if not raw_end or not raw_filed or value is None:
                    continue
                try:
                    period_end = _as_date(raw_end)
                    filing_date = _as_date(raw_filed)
                except ValueError:
                    continue
                previous = collected.get(period_end)
                if previous is None or filing_date >= previous[0]:
                    collected[period_end] = (filing_date, value)
    return collected


def _dart_corp_code_map(*, config: LiveDataConfig, cache_root: Path) -> dict[str, str]:
    cache_file = _cache_path(cache_root, "dart", "corp_code_map.json")
    cached = _load_cached_json(cache_file)
    if isinstance(cached, Mapping):
        return {str(key): str(value) for key, value in cached.items()}

    url = f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={parse.quote(config.dart_api_key)}"
    payload = _http_get_bytes(url)
    mapping: dict[str, str] = {}
    with ZipFile(io.BytesIO(payload)) as archive:
        if not archive.namelist():
            raise LiveDataConfigError("DART corpCode response archive is empty")
        xml_bytes = archive.read(archive.namelist()[0])
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_bytes)
    for row in root.findall(".//list"):
        corp_code = (row.findtext("corp_code") or "").strip()
        stock_code = (row.findtext("stock_code") or "").strip()
        if corp_code and stock_code:
            mapping[stock_code] = corp_code
    _save_cached_json(cache_file, mapping)
    return mapping


def _dart_statement_rows(
    *,
    config: LiveDataConfig,
    cache_root: Path,
    corp_code: str,
    year: int,
    options: LiveIngestOptions,
) -> list[dict[str, object]]:
    def _rows_for_fs(fs_div: str) -> list[dict[str, object]]:
        cache_file = _cache_path(cache_root, "dart", "financials", f"{corp_code}_{year}_11011_{fs_div}.json")
        cached = _load_cached_json(cache_file)
        payload = cached
        if payload is None and fs_div == "CFS":
            legacy_cache_file = _cache_path(cache_root, "dart", "financials", f"{corp_code}_{year}_11011.json")
            payload = _load_cached_json(legacy_cache_file)
            if payload is not None:
                _save_cached_json(cache_file, payload)
        if payload is None:
            query = parse.urlencode(
                {
                    "crtfc_key": config.dart_api_key,
                    "corp_code": corp_code,
                    "bsns_year": str(year),
                    "reprt_code": "11011",
                    "fs_div": fs_div,
                }
            )
            url = f"https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json?{query}"
            payload = _http_get_json(
                url,
                timeout_sec=options.dart_timeout_sec,
                max_retries=options.dart_max_retries,
                backoff_sec=options.dart_backoff_sec,
                sec_max_rps=options.dart_max_rps,
            )
            _save_cached_json(cache_file, payload)

        if not isinstance(payload, Mapping):
            return []
        status = str(payload.get("status", "")).strip()
        if status != "000":
            return []
        rows = payload.get("list", [])
        if not isinstance(rows, Sequence):
            return []
        normalized: list[dict[str, object]] = []
        for row in rows:
            if isinstance(row, Mapping):
                normalized.append(dict(cast(Mapping[str, object], row)))
        return normalized

    cfs_rows = _rows_for_fs("CFS")
    if cfs_rows:
        return cfs_rows
    return _rows_for_fs("OFS")


def _dart_xbrl_zip_bytes(
    *,
    config: LiveDataConfig,
    cache_root: Path,
    rcept_no: str,
    options: LiveIngestOptions,
) -> bytes | None:
    normalized_rcept_no = "".join(ch for ch in rcept_no if ch.isdigit())
    if len(normalized_rcept_no) != 14:
        return None

    cache_file = _cache_path(cache_root, "dart", "xbrl", f"{normalized_rcept_no}.zip")
    if cache_file.exists():
        try:
            cached = cache_file.read_bytes()
        except OSError:
            cached = b""
        if cached:
            return cached

    query = parse.urlencode(
        {
            "crtfc_key": config.dart_api_key,
            "rcept_no": normalized_rcept_no,
        }
    )
    url = f"https://opendart.fss.or.kr/api/fnlttXbrl.xml?{query}"

    attempts = max(1, options.dart_max_retries + 1)
    last_error: Exception | None = None
    payload: bytes | None = None
    for attempt in range(attempts):
        _sec_rate_limit_wait(options.dart_max_rps)
        try:
            req = request.Request(url=url)
            with request.urlopen(req, timeout=options.dart_timeout_sec) as response:
                payload = response.read()
            break
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt >= attempts - 1:
                break
            if options.dart_backoff_sec > 0.0:
                time.sleep(options.dart_backoff_sec * (2**attempt))

    if payload is None or not payload:
        _ = last_error
        return None

    try:
        with ZipFile(io.BytesIO(payload)) as archive:
            if not archive.namelist():
                return None
    except BadZipFile:
        return None

    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        _ = cache_file.write_bytes(payload)
    except OSError:
        pass
    return payload


def _extract_dart_rd_from_xbrl_bytes(*, xbrl_zip_bytes: bytes, year: int) -> float | None:
    import xml.etree.ElementTree as ET

    try:
        with ZipFile(io.BytesIO(xbrl_zip_bytes)) as archive:
            xbrl_names = [name for name in archive.namelist() if name.lower().endswith(".xbrl")]
            if not xbrl_names:
                return None
            xbrl_payload = archive.read(xbrl_names[0])
    except (BadZipFile, KeyError, OSError):
        return None

    try:
        root = ET.fromstring(xbrl_payload)
    except ET.ParseError:
        return None

    rd_positive_tokens = (
        "researchanddevelopment",
        "researchdevelopment",
        "ordinarydevelopmentexpense",
        "developmentexpense",
        "developmentcost",
        "randd",
    )
    rd_negative_tokens = (
        "exceptforresearchanddevelopment",
        "appropriationofresearchanddevelopmentreserves",
        "capitalised",
        "capitalized",
        "capitalisation",
        "capitalization",
        "accumulated",
        "amortisation",
        "amortization",
        "impairment",
        "gross",
        "remainingamortisationperiod",
        "remainingamortizationperiod",
        "preclinical",
        "approvalfornewdrug",
        "forbioindustry",
        "tableofitems",
        "tableofmember",
        "lineitems",
        "memberof",
        "reserve",
        "textblock",
    )
    context_exclusion_tokens = (
        "significantinvestmentsinsubsidiariesaxis",
        "entitystotalforsubsidiariesmember",
        "researchanddevelopmentexpenseforbioindustrytable",
        "memberofresearchanddevelopmentexpense",
    )

    candidates: list[tuple[int, float, float]] = []
    for elem in root.iter():
        context_ref = str(elem.attrib.get("contextRef", "")).strip()
        if not context_ref:
            continue
        parsed_value = _to_float((elem.text or "").strip())
        if parsed_value is None or parsed_value <= 0:
            continue

        raw_tag = str(elem.tag)
        local_tag = raw_tag.split("}", 1)[1] if "}" in raw_tag else raw_tag
        tag_key = _normalized_label(local_tag)
        if not tag_key:
            continue
        if any(token in tag_key for token in rd_negative_tokens):
            continue
        if not any(token in tag_key for token in rd_positive_tokens):
            continue

        context_key = _normalized_label(context_ref)
        if any(token in context_key for token in context_exclusion_tokens):
            continue

        score = 0
        if "ordinarydevelopmentexpensesellinggeneraladministrativeexpenses" in tag_key:
            score += 5
        if "researchanddevelopmentexpense" in tag_key:
            score += 3
        if f"cfy{year}" in context_key:
            score += 8
        if "consolidatedmember" in context_key:
            score += 3
        if "separatemember" in context_key:
            score -= 1
        if "dfy" in context_key or "duration" in context_key:
            score += 1
        candidates.append((score, abs(parsed_value), parsed_value))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _extract_dart_rd_sales(rows: Sequence[Mapping[str, object]]) -> tuple[float | None, float | None]:
    rd_candidates: list[float] = []
    sales_candidates: list[float] = []

    rd_id_tokens = (
        "researchanddevelopment",
        "researchdevelopment",
        "randd",
        "developmentcost",
        "developmentexpense",
        "researchcost",
        "researchexpense",
    )
    rd_name_tokens = (
        "연구개발",
        "연구개발비",
        "연구비",
        "개발비",
    )
    rd_negative_tokens = (
        "capitalised",
        "capitalized",
        "accumulated",
        "amortisation",
        "amortization",
        "impairment",
        "자본화",
        "누계",
        "상각",
        "손상",
    )

    for row in rows:
        account_id = _normalized_label(row.get("account_id", ""))
        account_nm = _normalized_label(row.get("account_nm", ""))
        value = _to_float(row.get("thstrm_amount"))
        if value is None:
            value = _to_float(row.get("thstrm_add_amount"))
        if value is None:
            continue

        is_rd = any(token in account_id for token in rd_id_tokens) or any(token in account_nm for token in rd_name_tokens)
        if is_rd and (any(token in account_id for token in rd_negative_tokens) or any(token in account_nm for token in rd_negative_tokens)):
            is_rd = False
        is_sales = (
            "revenue" in account_id
            or "sales" in account_id
            or "netsales" in account_id
            or "매출" in account_nm
            or "수익" in account_nm
        )
        if is_sales and ("costofsales" in account_id or "costsales" in account_id):
            is_sales = False
        if is_rd:
            rd_candidates.append(value)
        if is_sales:
            sales_candidates.append(value)

    rd_value = max(rd_candidates, key=abs) if rd_candidates else None
    sales_value = max(sales_candidates, key=abs) if sales_candidates else None
    return rd_value, sales_value


def _edinet_client(config: LiveDataConfig, cache_root: Path) -> object:
    try:
        from edinet_tools import EdinetClient
    except ImportError as exc:
        raise LiveDataConfigError("edinet-tools package is required for real data mode") from exc
    download_dir = _cache_path(cache_root, "edinet", "downloads")
    return EdinetClient(api_key=config.edinet_api_key, download_dir=str(download_dir))


def _edinet_daily_docs(
    *,
    client: object,
    cache_root: Path,
    day: date,
    options: LiveIngestOptions,
) -> list[dict[str, object]]:
    def _load_doc_type(doc_type: str) -> list[dict[str, object]]:
        cache_file = _cache_path(cache_root, "edinet", "documents", f"{day.isoformat()}_{doc_type}.json")
        cached = _load_cached_json(cache_file)
        payload = cached
        if payload is None:
            attempts = max(1, options.edinet_max_retries + 1)
            rows: list[dict[str, object]] | None = None
            for attempt in range(attempts):
                _edinet_rate_limit_wait(options.edinet_max_rps)
                try:
                    rows = cast(list[dict[str, object]], getattr(client, "get_documents_by_date")(day, doc_type=doc_type))
                    break
                except Exception:
                    if attempt >= attempts - 1:
                        rows = []
                        break
                    if options.edinet_backoff_sec > 0.0:
                        time.sleep(options.edinet_backoff_sec * (2**attempt))

            payload = rows if rows is not None else []
            _save_cached_json(cache_file, payload)

        if not isinstance(payload, Sequence):
            return []
        normalized_rows: list[dict[str, object]] = []
        for row in payload:
            if isinstance(row, Mapping):
                normalized_rows.append(dict(cast(Mapping[str, object], row)))
        return normalized_rows

    merged: list[dict[str, object]] = []
    merged.extend(_load_doc_type("120"))
    merged.extend(_load_doc_type("130"))

    deduped: list[dict[str, object]] = []
    seen_doc_ids: set[str] = set()
    for row in merged:
        doc_id = str(row.get("docID", "")).strip()
        if doc_id and doc_id in seen_doc_ids:
            continue
        if doc_id:
            seen_doc_ids.add(doc_id)
        deduped.append(row)
    return deduped


def _jp_doc_index_cache_namespace(*, cache_root: Path) -> str:
    try:
        return str(cache_root.resolve())
    except OSError:
        return str(cache_root)


def _jp_document_index_for_window(
    *,
    client: object,
    cache_root: Path,
    start_day: date,
    end_day: date,
    options: LiveIngestOptions,
) -> list[dict[str, object]]:
    all_days: list[date] = []
    current = start_day
    while current <= end_day:
        all_days.append(current)
        current += timedelta(days=1)

    namespace = _jp_doc_index_cache_namespace(cache_root=cache_root)
    with _JP_DOC_INDEX_LOCK:
        day_cache = _jp_doc_index_cache.setdefault(namespace, {})
        missing_days = [day for day in all_days if day not in day_cache]

    if missing_days:
        fetched_by_day: dict[date, tuple[dict[str, object], ...]] = {}
        workers = max(1, options.edinet_max_workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_edinet_daily_docs, client=client, cache_root=cache_root, day=day, options=options): day
                for day in missing_days
            }
            for future in as_completed(futures):
                day = futures[future]
                rows = future.result()
                fetched_by_day[day] = tuple(dict(row) for row in rows)

        with _JP_DOC_INDEX_LOCK:
            day_cache = _jp_doc_index_cache.setdefault(namespace, {})
            for day in missing_days:
                if day in day_cache:
                    continue
                day_cache[day] = fetched_by_day.get(day, ())

    indexed: list[dict[str, object]] = []
    with _JP_DOC_INDEX_LOCK:
        day_cache = _jp_doc_index_cache.get(namespace, {})
        for day in all_days:
            for row in day_cache.get(day, ()):
                indexed.append(dict(row))
    return indexed


def _jp_documents_for_window(
    *,
    client: object,
    cache_root: Path,
    start_day: date,
    end_day: date,
    options: LiveIngestOptions,
) -> list[dict[str, object]]:
    try:
        indexed_docs = _jp_document_index_for_window(
            client=client,
            cache_root=cache_root,
            start_day=start_day,
            end_day=end_day,
            options=options,
        )
        normalized_indexed: list[dict[str, object]] = []
        for row in indexed_docs:
            if not isinstance(row, Mapping):
                raise TypeError("indexed JP document row must be a mapping")
            normalized_indexed.append(dict(cast(Mapping[str, object], row)))
        return normalized_indexed
    except Exception:
        fallback_rows: list[dict[str, object]] = []
        current = start_day
        while current <= end_day:
            for row in _edinet_daily_docs(
                client=client,
                cache_root=cache_root,
                day=current,
                options=options,
            ):
                if not isinstance(row, Mapping):
                    continue
                fallback_rows.append(dict(cast(Mapping[str, object], row)))
            current += timedelta(days=1)
        return fallback_rows


def _edinet_download_cache_file(*, cache_root: Path, doc_id: str) -> Path:
    return _cache_path(cache_root, "edinet", "downloads", f"{_sanitize_cache_key(doc_id)}.zip")


def _load_or_download_edinet_zip_bytes(*, client: object, cache_root: Path, doc_id: str) -> bytes | None:
    doc_id_clean = doc_id.strip()
    if not doc_id_clean:
        return None
    cache_file = _edinet_download_cache_file(cache_root=cache_root, doc_id=doc_id_clean)
    if cache_file.exists():
        try:
            cached = cache_file.read_bytes()
        except OSError:
            cached = b""
        if cached:
            return cached

    try:
        raw_bytes = cast(bytes, getattr(client, "download_filing_raw")(doc_id_clean))
    except Exception:
        return None
    if not isinstance(raw_bytes, (bytes, bytearray)):
        return None
    payload = bytes(raw_bytes)
    if not payload:
        return None
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        _ = cache_file.write_bytes(payload)
    except OSError:
        pass
    return payload


def _extract_jp_rd_sales(report: object) -> tuple[float | None, float | None]:
    sales = _to_float(getattr(report, "net_sales", None))

    rd_candidates: list[float] = []
    sales_candidates: list[float] = []

    direct_rd_fields = (
        "research_and_development_expense",
        "research_and_development_expenses",
        "research_and_development_cost",
        "research_and_development_costs",
        "rd_expense",
        "r_and_d_expense",
    )
    for attr in direct_rd_fields:
        parsed = _to_float(getattr(report, attr, None))
        if parsed is not None:
            rd_candidates.append(parsed)

    if sales is not None:
        sales_candidates.append(sales)

    rd_key_tokens = (
        "researchanddevelopmentexpense",
        "researchanddevelopmentexpenses",
        "researchanddevelopmentcost",
        "researchanddevelopmentcosts",
        "researchanddevelopmentexpenditure",
        "researchanddevelopmentexpenditures",
        "researchanddevelopmentamount",
        "randdexpense",
        "randdcost",
    )

    def _scan_mapping(mapping: object) -> None:
        if not isinstance(mapping, Mapping):
            return
        for key, raw in mapping.items():
            key_text = _normalized_label(key)
            if not key_text or "textblock" in key_text:
                continue
            parsed = _to_float(raw)
            if parsed is None:
                continue
            if any(token in key_text for token in rd_key_tokens):
                rd_candidates.append(parsed)
            is_sales_key = (
                "netsales" in key_text
                or "revenue" in key_text
                or "sales" in key_text
                or "operatingrevenue" in key_text
            )
            if is_sales_key and "costofsales" not in key_text and "costsales" not in key_text:
                sales_candidates.append(parsed)

    _scan_mapping(getattr(report, "raw_fields", None))
    _scan_mapping(getattr(report, "unmapped_fields", None))

    rd_value = max(rd_candidates, key=abs) if rd_candidates else None
    sales_value = max(sales_candidates, key=abs) if sales_candidates else None
    if sales_value is None:
        sales_value = sales
    return rd_value, sales_value


def _security_price_ticker(spec: SecuritySpecLike) -> str:
    country = spec.country.upper()
    if country == "US":
        return spec.stock_code.replace(".", "-")
    if country == "KR":
        suffix = ".KQ" if "kosdaq" in spec.universe.lower() else ".KS"
        return f"{spec.stock_code}{suffix}"
    if country == "JP":
        return f"{_normalized_stock_code(spec.stock_code)}.T"
    raise LiveDataConfigError(f"unsupported country for price ticker: {spec.country}")


def build_live_inputs(
    *,
    start: date,
    end: date,
    schedule: Sequence[date],
    specs: Sequence[SecuritySpecLike],
    config: LiveDataConfig,
    cache_root: Path,
    options: LiveIngestOptions | None = None,
    benchmark_ticker: str | None = None,
    benchmark_currency: str | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, dict[date, float]], list[dict[str, object]]]:
    scoped_specs = tuple(specs)
    runtime_options = options or LiveIngestOptions()
    fundamentals: list[dict[str, object]] = []
    fundamentals.extend(
        _build_us_fundamentals(
            start=start,
            end=end,
            specs=scoped_specs,
            config=config,
            cache_root=cache_root,
            options=runtime_options,
        )
    )
    fundamentals.extend(
        _build_kr_fundamentals(
            start=start,
            end=end,
            specs=scoped_specs,
            config=config,
            cache_root=cache_root,
            options=runtime_options,
        )
    )
    fundamentals.extend(
        _build_jp_fundamentals(
            start=start,
            end=end,
            specs=scoped_specs,
            config=config,
            cache_root=cache_root,
            options=runtime_options,
        )
    )

    fundamental_security_ids = {
        str(row.get("security_id", "")).strip()
        for row in fundamentals
        if str(row.get("security_id", "")).strip()
    }
    price_security_ids: set[str] | None = fundamental_security_ids if fundamental_security_ids else None

    price_rows = _build_price_rows(
        schedule=schedule,
        specs=scoped_specs,
        cache_root=cache_root,
        options=runtime_options,
        included_security_ids=price_security_ids,
    )
    fx_rows, fx_rate_map = _build_fx_rows(schedule=schedule, cache_root=cache_root, options=runtime_options)
    benchmark_rows = _build_benchmark_rows(
        schedule=schedule,
        config=config,
        cache_root=cache_root,
        options=runtime_options,
        benchmark_ticker=benchmark_ticker,
        benchmark_currency=benchmark_currency,
    )
    return fundamentals, price_rows, fx_rows, fx_rate_map, benchmark_rows


def _build_us_fundamentals(
    *,
    start: date,
    end: date,
    specs: Sequence[SecuritySpecLike],
    config: LiveDataConfig,
    cache_root: Path,
    options: LiveIngestOptions,
) -> list[dict[str, object]]:
    us_specs = [spec for spec in specs if spec.country.upper() == "US"]
    if not us_specs:
        return []

    ticker_to_cik = _sec_ticker_to_cik(config=config, cache_root=cache_root)
    records: list[dict[str, object]] = []
    rd_tags = ("ResearchAndDevelopmentExpense",)
    sales_tags = (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
    )
    min_period = start - timedelta(days=500)

    mapped_pairs: list[tuple[SecuritySpecLike, str]] = []
    for spec in us_specs:
        cik = ticker_to_cik.get(spec.stock_code.upper())
        if cik is None:
            continue
        mapped_pairs.append((spec, cik))

    required_ciks = sorted({cik for _, cik in mapped_pairs})
    deadline_monotonic: float | None = None
    if options.sec_stage_budget_sec is not None and options.sec_stage_budget_sec > 0:
        deadline_monotonic = time.monotonic() + float(options.sec_stage_budget_sec)
    _prefetch_sec_companyfacts(
        ciks=required_ciks,
        config=config,
        cache_root=cache_root,
        options=options,
        deadline_monotonic=deadline_monotonic,
    )

    for spec, cik in mapped_pairs:
        try:
            facts = _sec_company_facts(
                cik,
                config=config,
                cache_root=cache_root,
                options=options,
                deadline_monotonic=deadline_monotonic,
            )
        except LiveDataConfigError:
            continue
        rd_points = _sec_points(facts, rd_tags)
        sales_points = _sec_points(facts, sales_tags)
        period_ends = sorted(set(rd_points).intersection(sales_points))
        payload_rows: list[dict[str, object]] = []
        for period_end in period_ends:
            rd_filed, rd_value = rd_points[period_end]
            sales_filed, sales_value = sales_points[period_end]
            filing_date = max(rd_filed, sales_filed)
            if filing_date < min_period or filing_date > end + timedelta(days=365):
                continue
            payload_rows.append(
                {
                    "period_end": period_end.isoformat(),
                    "filing_date": filing_date.isoformat(),
                    "rd_expense": rd_value,
                    "sales_ttm": sales_value,
                }
            )
        transformed, _, _ = transform_us_sec_fundamentals(
            security_id=spec.security_id,
            cik=cik,
            payload_rows=payload_rows,
            country="US",
        )
        records.extend(transformed)
    return records


def _build_kr_fundamentals(
    *,
    start: date,
    end: date,
    specs: Sequence[SecuritySpecLike],
    config: LiveDataConfig,
    cache_root: Path,
    options: LiveIngestOptions,
) -> list[dict[str, object]]:
    kr_specs = [spec for spec in specs if spec.country.upper() == "KR"]
    if not kr_specs:
        return []

    stock_to_corp = _dart_corp_code_map(config=config, cache_root=cache_root)
    raw_rows: list[dict[str, object]] = []
    def _collect_rows_for_spec(spec: SecuritySpecLike) -> list[dict[str, object]]:
        corp_code = stock_to_corp.get(spec.stock_code)
        if not corp_code:
            return []
        collected: list[dict[str, object]] = []
        for year in range(start.year - 1, end.year + 1):
            statement_rows = _dart_statement_rows(
                config=config,
                cache_root=cache_root,
                corp_code=corp_code,
                year=year,
                options=options,
            )
            if not statement_rows:
                continue
            rd_value, sales_value = _extract_dart_rd_sales(statement_rows)
            if rd_value is None:
                receipt_number = ""
                for row in statement_rows:
                    candidate = str(row.get("rcept_no", "")).strip()
                    digits = "".join(ch for ch in candidate if ch.isdigit())
                    if len(digits) == 14:
                        receipt_number = digits
                        break
                if receipt_number:
                    xbrl_zip_bytes = _dart_xbrl_zip_bytes(
                        config=config,
                        cache_root=cache_root,
                        rcept_no=receipt_number,
                        options=options,
                    )
                    if xbrl_zip_bytes is not None:
                        rd_from_xbrl = _extract_dart_rd_from_xbrl_bytes(xbrl_zip_bytes=xbrl_zip_bytes, year=year)
                        if rd_from_xbrl is not None:
                            rd_value = rd_from_xbrl
            if rd_value is None or sales_value is None:
                continue

            filing_date = None
            period_end = None
            for row in statement_rows:
                rcept_no = str(row.get("rcept_no", "")).strip()
                if len(rcept_no) >= 8 and rcept_no[:8].isdigit():
                    filing_date = f"{rcept_no[0:4]}-{rcept_no[4:6]}-{rcept_no[6:8]}"
                    break
            for row in statement_rows:
                candidate = str(row.get("thstrm_dt", "")).strip()
                digits = "".join(ch for ch in candidate if ch.isdigit())
                if len(digits) >= 8:
                    period_end = f"{digits[0:4]}-{digits[4:6]}-{digits[6:8]}"
                    break
            if filing_date is None:
                filing_date = f"{year + 1}-03-31"
            if period_end is None:
                period_end = f"{year}-12-31"

            collected.append(
                {
                    "corp_code": corp_code,
                    "stock_code": spec.stock_code,
                    "filing_date": filing_date,
                    "period_end": period_end,
                    "rd_expense": rd_value,
                    "sales_ttm": sales_value,
                }
            )
        return collected

    workers = max(1, options.dart_max_workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_collect_rows_for_spec, spec) for spec in kr_specs]
        for future in as_completed(futures):
            try:
                raw_rows.extend(future.result())
            except LiveDataConfigError:
                continue

    raw_rows.sort(key=lambda row: (str(row.get("stock_code", "")), str(row.get("period_end", "")), str(row.get("filing_date", ""))))

    return transform_kr_dart_fundamentals(raw_rows, strict_unmapped_corp_code=False)


def _build_jp_fundamentals(
    *,
    start: date,
    end: date,
    specs: Sequence[SecuritySpecLike],
    config: LiveDataConfig,
    cache_root: Path,
    options: LiveIngestOptions,
) -> list[dict[str, object]]:
    jp_specs = [spec for spec in specs if spec.country.upper() == "JP"]
    if not jp_specs:
        return []

    client = _edinet_client(config=config, cache_root=cache_root)
    target_codes = {_normalized_stock_code(spec.stock_code): spec for spec in jp_specs}
    scan_start = start - timedelta(days=420)
    scan_end = end
    matched_docs: list[dict[str, object]] = []

    indexed_docs = _jp_documents_for_window(
        client=client,
        cache_root=cache_root,
        start_day=scan_start,
        end_day=scan_end,
        options=options,
    )
    for row in indexed_docs:
        sec_code = _normalized_stock_code(str(row.get("secCode", "")))
        if not sec_code or sec_code not in target_codes:
            continue
        row_copy = dict(row)
        row_copy["_normalized_sec_code"] = sec_code
        matched_docs.append(row_copy)

    matched_docs.sort(
        key=lambda row: (
            str(row.get("submitDateTime", "")),
            str(row.get("docID", "")),
            str(row.get("_normalized_sec_code", "")),
        )
    )

    try:
        from edinet_tools.document import Document
        from edinet_tools.parsers import parse as parse_edinet_document
    except ImportError as exc:
        raise LiveDataConfigError("edinet-tools package is required for JP real-data ingestion") from exc

    deduped_docs: list[dict[str, object]] = []
    seen_doc_ids: set[str] = set()
    for row in matched_docs:
        doc_id = str(row.get("docID", "")).strip()
        if doc_id and doc_id in seen_doc_ids:
            continue
        if doc_id:
            seen_doc_ids.add(doc_id)
        deduped_docs.append(row)

    def _parse_document_row(row: dict[str, object]) -> dict[str, object] | None:
        sec_code = str(row.get("_normalized_sec_code", ""))
        spec = target_codes.get(sec_code)
        if spec is None:
            return None

        doc_id = str(row.get("docID", "")).strip()
        cached_zip = _load_or_download_edinet_zip_bytes(client=client, cache_root=cache_root, doc_id=doc_id)

        report = None
        try:
            document = Document(row, client=client)
            report = document.parse()
        except Exception:
            report = None

        if report is None and cached_zip is not None:
            class _CachedDocument:
                def __init__(self, *, document_id: str, document_row: Mapping[str, object], zip_bytes: bytes) -> None:
                    self.doc_id = document_id
                    self.doc_type_code = str(document_row.get("docTypeCode", "")).strip()
                    self.filer_name = str(document_row.get("filerName", "")).strip() or None
                    self.filer_edinet_code = str(document_row.get("edinetCode", "")).strip() or None
                    self._zip_bytes = zip_bytes

                def fetch(self) -> bytes:
                    return self._zip_bytes

            try:
                report = parse_edinet_document(_CachedDocument(document_id=doc_id, document_row=row, zip_bytes=cached_zip))
            except Exception:
                report = None

        if report is None:
            return None

        rd_value, sales_value = _extract_jp_rd_sales(report)
        if rd_value is None or sales_value is None:
            return None

        submit = str(row.get("submitDateTime", "")).strip()
        filing_date = submit[:10] if len(submit) >= 10 else ""
        if not filing_date:
            filing_attr = getattr(report, "filing_date", None)
            if isinstance(filing_attr, (date, datetime)):
                filing_date = filing_attr.date().isoformat() if isinstance(filing_attr, datetime) else filing_attr.isoformat()
            elif isinstance(filing_attr, str) and len(filing_attr) >= 10:
                filing_date = filing_attr[:10]
        period_end = str(row.get("periodEnd", "")).strip()
        if not period_end:
            fiscal_end = getattr(report, "fiscal_year_end", None)
            if isinstance(fiscal_end, (date, datetime)):
                period_end = fiscal_end.date().isoformat() if isinstance(fiscal_end, datetime) else fiscal_end.isoformat()
            elif isinstance(fiscal_end, str) and len(fiscal_end) >= 10:
                period_end = fiscal_end[:10]
        if not filing_date or not period_end:
            return None

        return {
            "security_id": spec.security_id,
            "jp_code": spec.stock_code,
            "filing_date": filing_date,
            "period_end": period_end,
            "rd_expense": rd_value,
            "sales_ttm": sales_value,
        }

    payload_rows: list[dict[str, object]] = []
    parse_workers = max(1, min(options.edinet_max_workers, 8))
    with ThreadPoolExecutor(max_workers=parse_workers) as pool:
        futures = [pool.submit(_parse_document_row, row) for row in deduped_docs]
        for future in as_completed(futures):
            parsed = future.result()
            if parsed is not None:
                payload_rows.append(parsed)

    payload_rows.sort(
        key=lambda row: (
            str(row.get("security_id", "")),
            str(row.get("period_end", "")),
            str(row.get("filing_date", "")),
        )
    )

    records, failures = parse_jp_edinet_fundamentals(payload_rows)
    _ = failures
    return records


def _build_price_rows(
    *,
    schedule: Sequence[date],
    specs: Sequence[SecuritySpecLike],
    cache_root: Path,
    options: LiveIngestOptions,
    included_security_ids: set[str] | None = None,
) -> list[dict[str, object]]:
    required_dates = _required_dates(schedule)
    fetch_start = min(required_dates) - timedelta(days=40)
    fetch_end = max(required_dates)
    rows: list[dict[str, object]] = []
    filtered_specs = [spec for spec in specs if included_security_ids is None or spec.security_id in included_security_ids]
    ticker_by_security = {spec.security_id: _security_price_ticker(spec) for spec in filtered_specs}
    batch = _read_yfinance_bulk_series_for_loader(
        tickers=sorted(set(ticker_by_security.values())),
        start=fetch_start,
        end=fetch_end,
        cache_root=cache_root,
        options=options,
    )

    for spec in filtered_specs:
        ticker = ticker_by_security[spec.security_id]
        series = batch.get(ticker)
        if not series:
            try:
                series = _read_yfinance_series_for_loader(
                    ticker,
                    start=fetch_start,
                    end=fetch_end,
                    cache_root=cache_root,
                    options=options,
                )
            except LiveDataConfigError:
                continue

        spec_rows: list[dict[str, object]] = []
        has_all_required_dates = True
        for day in required_dates:
            try:
                close, volume = _value_on_or_before(series, day)
            except LiveDataConfigError:
                has_all_required_dates = False
                break
            traded_value = max(close * max(volume, 1.0), close)
            spec_rows.append(
                {
                    "security_id": spec.security_id,
                    "country": spec.country,
                    "currency": spec.currency,
                    "price_date": day,
                    "close": round(close, 6),
                    "traded_value": float(traded_value),
                }
            )
        if has_all_required_dates:
            rows.extend(spec_rows)
    return rows


def _build_fx_rows(
    schedule: Sequence[date],
    *,
    cache_root: Path,
    options: LiveIngestOptions,
) -> tuple[list[dict[str, object]], dict[str, dict[date, float]]]:
    required_dates = _required_dates(schedule)
    fetch_start = min(required_dates) - timedelta(days=15)
    fetch_end = max(required_dates)
    usd_krw = _read_yfinance_series_for_loader(
        "USDKRW=X",
        start=fetch_start,
        end=fetch_end,
        cache_root=cache_root,
        options=options,
    )
    usd_jpy = _read_yfinance_series_for_loader(
        "USDJPY=X",
        start=fetch_start,
        end=fetch_end,
        cache_root=cache_root,
        options=options,
    )

    raw_rows: list[dict[str, object]] = []
    fx_map: dict[str, dict[date, float]] = {"USD/KRW": {}, "USD/JPY": {}}
    for day in required_dates:
        usd_krw_rate, _ = _value_on_or_before(usd_krw, day)
        usd_jpy_rate, _ = _value_on_or_before(usd_jpy, day)
        fx_map["USD/KRW"][day] = float(usd_krw_rate)
        fx_map["USD/JPY"][day] = float(usd_jpy_rate)
        raw_rows.append({"pair": "USD/KRW", "fx_date": day.isoformat(), "rate": usd_krw_rate})
        raw_rows.append({"pair": "USD/JPY", "fx_date": day.isoformat(), "rate": usd_jpy_rate})

    normalized = normalize_fx_rows(raw_rows)
    rows: list[dict[str, object]] = [
        {
            "pair": str(row["pair"]),
            "fx_date": date.fromisoformat(str(row["fx_date"])),
            "rate": float(str(row["rate"])),
        }
        for row in normalized
    ]
    return rows, fx_map


def _build_benchmark_rows(
    *,
    schedule: Sequence[date],
    config: LiveDataConfig,
    cache_root: Path,
    options: LiveIngestOptions,
    benchmark_ticker: str | None = None,
    benchmark_currency: str | None = None,
) -> list[dict[str, object]]:
    fetch_start = min(schedule) - timedelta(days=15)
    fetch_end = max(schedule)
    us_series = _read_yfinance_series_for_loader(
        benchmark_ticker or config.benchmark_us_ticker,
        start=fetch_start,
        end=fetch_end,
        cache_root=cache_root,
        options=options,
    )

    official_rows = []
    for day in schedule:
        close, _ = _value_on_or_before(us_series, day)
        official_rows.append({"as_of_date": day.isoformat(), "level": close})

    selected = load_benchmark_series(official_series=official_rows, proxy_series=None)
    currency = (benchmark_currency or "USD").strip().upper() or "USD"
    return [
        {
            "benchmark_date": point.as_of_date,
            "close": point.level,
            "currency": currency,
        }
        for point in selected.series
    ]


__all__ = [
    "LiveDataConfig",
    "LiveDataConfigError",
    "build_live_inputs",
    "load_live_data_config_from_env",
]
