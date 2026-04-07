"""Benchmark loader policy: official-first with proxy fallback guardrails."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from math import sqrt


class BenchmarkPolicyError(ValueError):
    """Raised when benchmark loader inputs are invalid."""


def _to_date(value: object, field_name: str) -> date:
    if isinstance(value, date):
        return value
    text = str(value).strip() if value is not None else ""
    if not text:
        raise BenchmarkPolicyError(f"{field_name} is required")
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise BenchmarkPolicyError(f"{field_name} must be ISO date (YYYY-MM-DD)") from exc


def _to_float(value: object, field_name: str) -> float:
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise BenchmarkPolicyError(f"{field_name} must be numeric")
        try:
            return float(text)
        except ValueError as exc:
            raise BenchmarkPolicyError(f"{field_name} must be numeric") from exc
    raise BenchmarkPolicyError(f"{field_name} must be numeric")


@dataclass(frozen=True)
class BenchmarkPoint:
    """One benchmark observation."""

    as_of_date: date
    level: float


@dataclass(frozen=True)
class ProxyQualityMetrics:
    """Proxy quality metrics against official overlap window."""

    correlation: float
    tracking_error: float
    correlation_threshold: float
    tracking_error_threshold: float
    passed: bool


@dataclass(frozen=True)
class BenchmarkStatus:
    """Structured status object for benchmark selection and quality."""

    level: str
    code: str
    message: str


@dataclass(frozen=True)
class BenchmarkLoadResult:
    """Result of benchmark policy selection."""

    series: tuple[BenchmarkPoint, ...]
    source_tag: str
    fallback_reason: str | None
    proxy_quality: ProxyQualityMetrics | None
    status: BenchmarkStatus


def _canonicalize_series(rows: Sequence[Mapping[str, object]], *, field_name: str) -> tuple[BenchmarkPoint, ...]:
    points: list[BenchmarkPoint] = []
    for index, row in enumerate(rows):
        if "as_of_date" not in row or "level" not in row:
            raise BenchmarkPolicyError(f"{field_name}[{index}] requires as_of_date and level")
        points.append(
            BenchmarkPoint(
                as_of_date=_to_date(row["as_of_date"], f"{field_name}[{index}].as_of_date"),
                level=_to_float(row["level"], f"{field_name}[{index}].level"),
            )
        )
    return tuple(sorted(points, key=lambda point: point.as_of_date))


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return sqrt(variance)


def _pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise BenchmarkPolicyError("correlation inputs must have equal length")
    if len(left) < 2:
        return 0.0

    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)

    centered_left = [value - mean_left for value in left]
    centered_right = [value - mean_right for value in right]

    sum_squares_left = sum(value * value for value in centered_left)
    sum_squares_right = sum(value * value for value in centered_right)

    if sum_squares_left == 0.0 or sum_squares_right == 0.0:
        return 1.0 if centered_left == centered_right else 0.0

    covariance = sum(x * y for x, y in zip(centered_left, centered_right, strict=True))
    return covariance / sqrt(sum_squares_left * sum_squares_right)


def evaluate_proxy_quality(
    *,
    official_returns: Sequence[float],
    proxy_returns: Sequence[float],
    correlation_threshold: float = 0.95,
    tracking_error_threshold: float = 0.05,
    annualization_factor: int = 252,
) -> ProxyQualityMetrics:
    """Evaluate proxy guardrails using overlap returns."""

    if len(official_returns) != len(proxy_returns):
        raise BenchmarkPolicyError("official_returns and proxy_returns must have equal length")
    if len(official_returns) < 2:
        raise BenchmarkPolicyError("at least two overlap returns are required")
    if annualization_factor <= 0:
        raise BenchmarkPolicyError("annualization_factor must be positive")

    official = [float(value) for value in official_returns]
    proxy = [float(value) for value in proxy_returns]

    correlation = _pearson_correlation(official, proxy)
    active_returns = [p - o for o, p in zip(official, proxy, strict=True)]
    tracking_error = _sample_std(active_returns) * sqrt(annualization_factor)

    passed = correlation >= correlation_threshold and tracking_error <= tracking_error_threshold
    return ProxyQualityMetrics(
        correlation=correlation,
        tracking_error=tracking_error,
        correlation_threshold=correlation_threshold,
        tracking_error_threshold=tracking_error_threshold,
        passed=passed,
    )


def load_benchmark_series(
    *,
    official_series: Sequence[Mapping[str, object]] | None,
    proxy_series: Sequence[Mapping[str, object]] | None,
    official_overlap_returns: Sequence[float] | None = None,
    proxy_overlap_returns: Sequence[float] | None = None,
    fallback_reason: str = "official_unavailable",
    correlation_threshold: float = 0.95,
    tracking_error_threshold: float = 0.05,
) -> BenchmarkLoadResult:
    """Apply official-first benchmark policy with proxy quality guardrails."""

    if official_series:
        series = _canonicalize_series(official_series, field_name="official_series")
        status = BenchmarkStatus(
            level="ok",
            code="official_selected",
            message="Official benchmark series selected.",
        )
        return BenchmarkLoadResult(
            series=series,
            source_tag="official",
            fallback_reason=None,
            proxy_quality=None,
            status=status,
        )

    if not proxy_series:
        raise BenchmarkPolicyError("proxy_series is required when official series is unavailable")

    series = _canonicalize_series(proxy_series, field_name="proxy_series")

    if official_overlap_returns is None or proxy_overlap_returns is None:
        status = BenchmarkStatus(
            level="warning",
            code="proxy_fallback_missing_quality_inputs",
            message="Proxy fallback used but overlap returns are missing for guardrail check.",
        )
        return BenchmarkLoadResult(
            series=series,
            source_tag="proxy",
            fallback_reason=fallback_reason,
            proxy_quality=None,
            status=status,
        )

    quality = evaluate_proxy_quality(
        official_returns=official_overlap_returns,
        proxy_returns=proxy_overlap_returns,
        correlation_threshold=correlation_threshold,
        tracking_error_threshold=tracking_error_threshold,
    )

    if quality.passed:
        status = BenchmarkStatus(
            level="warning",
            code="proxy_fallback_used",
            message="Proxy fallback used and quality guardrails passed.",
        )
    else:
        status = BenchmarkStatus(
            level="warning",
            code="proxy_quality_guardrail_breached",
            message=(
                "Proxy fallback used but quality guardrail breached: "
                f"corr={quality.correlation:.6f} (< {quality.correlation_threshold:.2f}) or "
                f"te={quality.tracking_error:.6f} (> {quality.tracking_error_threshold:.2f})."
            ),
        )

    return BenchmarkLoadResult(
        series=series,
        source_tag="proxy",
        fallback_reason=fallback_reason,
        proxy_quality=quality,
        status=status,
    )


__all__ = [
    "BenchmarkLoadResult",
    "BenchmarkPoint",
    "BenchmarkPolicyError",
    "BenchmarkStatus",
    "ProxyQualityMetrics",
    "evaluate_proxy_quality",
    "load_benchmark_series",
]
