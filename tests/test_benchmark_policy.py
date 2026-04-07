from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Protocol, cast


class _ProxyQualityMetricsProtocol(Protocol):
    correlation: float
    tracking_error: float
    passed: bool


class _BenchmarkStatusProtocol(Protocol):
    level: str
    code: str


class _BenchmarkLoadResultProtocol(Protocol):
    source_tag: str
    fallback_reason: str | None
    proxy_quality: _ProxyQualityMetricsProtocol | None
    status: _BenchmarkStatusProtocol


class _BenchmarkModuleProtocol(Protocol):
    def load_benchmark_series(
        self,
        *,
        official_series: Sequence[Mapping[str, object]] | None,
        proxy_series: Sequence[Mapping[str, object]] | None,
        official_overlap_returns: Sequence[float] | None = None,
        proxy_overlap_returns: Sequence[float] | None = None,
        fallback_reason: str = "official_unavailable",
        correlation_threshold: float = 0.95,
        tracking_error_threshold: float = 0.05,
    ) -> _BenchmarkLoadResultProtocol: ...


_BENCHMARK_MODULE = importlib.import_module("src.data.ingest.benchmark")
_benchmark = cast(_BenchmarkModuleProtocol, cast(object, _BENCHMARK_MODULE))
load_benchmark_series = _benchmark.load_benchmark_series


def _official_series() -> list[dict[str, object]]:
    return [
        {"as_of_date": "2026-01-31", "level": 100.0},
        {"as_of_date": "2026-02-28", "level": 101.2},
    ]


def _proxy_series() -> list[dict[str, object]]:
    return [
        {"as_of_date": "2026-01-31", "level": 99.5},
        {"as_of_date": "2026-02-28", "level": 101.0},
    ]


def test_official_is_selected_when_available() -> None:
    result = load_benchmark_series(
        official_series=_official_series(),
        proxy_series=_proxy_series(),
        official_overlap_returns=[0.01, 0.02, -0.01, 0.015],
        proxy_overlap_returns=[0.0105, 0.0195, -0.009, 0.014],
    )

    assert result.source_tag == "official"
    assert result.fallback_reason is None
    assert result.proxy_quality is None
    assert result.status.level == "ok"
    assert result.status.code == "official_selected"


def test_proxy_fallback_used_with_quality_guardrail_pass() -> None:
    result = load_benchmark_series(
        official_series=None,
        proxy_series=_proxy_series(),
        official_overlap_returns=[0.01, 0.02, -0.01, 0.015],
        proxy_overlap_returns=[0.0105, 0.0195, -0.009, 0.014],
        fallback_reason="official_feed_unavailable",
    )

    assert result.source_tag == "proxy"
    assert result.fallback_reason == "official_feed_unavailable"
    assert result.proxy_quality is not None
    assert result.proxy_quality.passed is True
    assert result.proxy_quality.correlation >= 0.95
    assert result.proxy_quality.tracking_error <= 0.05
    assert result.status.level == "warning"
    assert result.status.code == "proxy_fallback_used"


def test_proxy_fallback_guardrail_breach_emits_warning_status() -> None:
    result = load_benchmark_series(
        official_series=None,
        proxy_series=_proxy_series(),
        official_overlap_returns=[0.01, 0.02, -0.01, 0.015],
        proxy_overlap_returns=[0.08, -0.06, 0.09, -0.07],
        fallback_reason="official_feed_unavailable",
    )

    assert result.source_tag == "proxy"
    assert result.fallback_reason == "official_feed_unavailable"
    assert result.proxy_quality is not None
    assert result.proxy_quality.passed is False
    assert result.status.level == "warning"
    assert result.status.code == "proxy_quality_guardrail_breached"
