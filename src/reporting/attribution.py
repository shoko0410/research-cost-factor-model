"""Deterministic reporting builders for attribution outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from ..backtest.engine import BacktestResult, PeriodReturn
from ..validation.walkforward import FoldOOSMetricsArtifact


def _to_date(value: object, *, field_name: str) -> date:
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return date.fromisoformat(text)


def _to_float(value: object, *, field_name: str) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc


def _build_usd_krw_lookup(fx_rates: Sequence[Mapping[str, object]]) -> dict[date, float]:
    lookup: dict[date, float] = {}
    for index, row in enumerate(fx_rates):
        pair = str(row.get("pair", "")).strip().upper()
        if pair != "USD/KRW":
            continue
        fx_date = _to_date(row.get("fx_date", ""), field_name=f"fx_rates[{index}].fx_date")
        rate = _to_float(row.get("rate"), field_name=f"fx_rates[{index}].rate")
        if rate <= 0.0:
            raise ValueError(f"fx_rates[{index}].rate must be positive")
        lookup[fx_date] = rate
    return lookup


def _usd_return_from_krw(*, krw_return: float, fx_return: float) -> float:
    return ((1.0 + krw_return) / (1.0 + fx_return)) - 1.0


def _validate_benchmark_dependency(returns: Sequence[PeriodReturn]) -> None:
    for index, period in enumerate(returns):
        if period.benchmark_krw_return is None:
            raise ValueError(
                f"benchmark dependency missing: backtest_result.returns[{index}].benchmark_krw_return is required"
            )


def build_backtest_attribution_report(
    *,
    backtest_result: BacktestResult,
    fx_rates: Sequence[Mapping[str, object]],
    consistency_tolerance: float = 1e-10,
) -> dict[str, object]:
    """Build KRW/USD performance tables and FX decomposition from backtest outputs."""

    if consistency_tolerance < 0.0:
        raise ValueError("consistency_tolerance must be non-negative")

    periods = tuple(backtest_result.returns)
    _validate_benchmark_dependency(periods)

    usd_krw_lookup = _build_usd_krw_lookup(fx_rates)

    krw_rows: list[dict[str, object]] = []
    usd_rows: list[dict[str, object]] = []
    fx_rows: list[dict[str, object]] = []

    for index, period in enumerate(periods):
        start_rate = usd_krw_lookup.get(period.start_date)
        end_rate = usd_krw_lookup.get(period.end_date)
        if start_rate is None:
            raise ValueError(
                f"fx dependency missing: USD/KRW rate for period start {period.start_date.isoformat()}"
            )
        if end_rate is None:
            raise ValueError(
                f"fx dependency missing: USD/KRW rate for period end {period.end_date.isoformat()}"
            )

        fx_return = (end_rate / start_rate) - 1.0
        benchmark_krw = period.benchmark_krw_return
        assert benchmark_krw is not None

        portfolio_net_usd = _usd_return_from_krw(krw_return=period.net_return, fx_return=fx_return)
        portfolio_gross_usd = _usd_return_from_krw(krw_return=period.gross_return, fx_return=fx_return)
        benchmark_usd = _usd_return_from_krw(krw_return=benchmark_krw, fx_return=fx_return)

        net_fx_contribution = (1.0 + portfolio_net_usd) * fx_return
        gross_fx_contribution = (1.0 + portfolio_gross_usd) * fx_return
        benchmark_fx_contribution = (1.0 + benchmark_usd) * fx_return

        net_gap = period.net_return - (portfolio_net_usd + net_fx_contribution)
        gross_gap = period.gross_return - (portfolio_gross_usd + gross_fx_contribution)
        benchmark_gap = benchmark_krw - (benchmark_usd + benchmark_fx_contribution)

        if abs(net_gap) > consistency_tolerance:
            raise ValueError(f"fx decomposition inconsistency on period[{index}] net return")
        if abs(gross_gap) > consistency_tolerance:
            raise ValueError(f"fx decomposition inconsistency on period[{index}] gross return")
        if abs(benchmark_gap) > consistency_tolerance:
            raise ValueError(f"fx decomposition inconsistency on period[{index}] benchmark return")

        active_net_krw = period.net_return - benchmark_krw
        active_gross_krw = period.gross_return - benchmark_krw
        active_net_usd = portfolio_net_usd - benchmark_usd
        active_gross_usd = portfolio_gross_usd - benchmark_usd

        krw_rows.append(
            {
                "start_date": period.start_date,
                "end_date": period.end_date,
                "portfolio_net_return_krw": period.net_return,
                "portfolio_gross_return_krw": period.gross_return,
                "benchmark_return_krw": benchmark_krw,
                "active_net_return_krw": active_net_krw,
                "active_gross_return_krw": active_gross_krw,
                "total_cost_tax_krw": period.total_cost_tax_krw,
            }
        )

        usd_rows.append(
            {
                "start_date": period.start_date,
                "end_date": period.end_date,
                "portfolio_net_return_usd": portfolio_net_usd,
                "portfolio_gross_return_usd": portfolio_gross_usd,
                "benchmark_return_usd": benchmark_usd,
                "active_net_return_usd": active_net_usd,
                "active_gross_return_usd": active_gross_usd,
                "fx_return_usdkrw": fx_return,
            }
        )

        fx_rows.append(
            {
                "start_date": period.start_date,
                "end_date": period.end_date,
                "fx_return_usdkrw": fx_return,
                "portfolio_net_return_krw": period.net_return,
                "portfolio_net_return_usd": portfolio_net_usd,
                "portfolio_net_fx_contribution": net_fx_contribution,
                "portfolio_gross_return_krw": period.gross_return,
                "portfolio_gross_return_usd": portfolio_gross_usd,
                "portfolio_gross_fx_contribution": gross_fx_contribution,
                "benchmark_return_krw": benchmark_krw,
                "benchmark_return_usd": benchmark_usd,
                "benchmark_fx_contribution": benchmark_fx_contribution,
                "portfolio_net_consistency_gap": net_gap,
                "portfolio_gross_consistency_gap": gross_gap,
                "benchmark_consistency_gap": benchmark_gap,
            }
        )

    return {
        "krw_main_performance_table": {"rows": tuple(krw_rows)},
        "usd_supplemental_table": {"rows": tuple(usd_rows)},
        "fx_contribution_decomposition": {"rows": tuple(fx_rows)},
    }


def build_walkforward_attribution_report(
    *,
    artifacts: Sequence[FoldOOSMetricsArtifact],
) -> dict[str, object]:
    """Build benchmark-relative fold metrics from walk-forward artifacts."""

    rows: list[dict[str, object]] = []
    for index, artifact in enumerate(artifacts):
        benchmark = artifact.oos_metrics.get("benchmark_krw_return")
        if benchmark is None:
            raise ValueError(
                f"benchmark dependency missing: artifacts[{index}].oos_metrics['benchmark_krw_return'] is required"
            )
        portfolio = _to_float(
            artifact.oos_metrics.get("portfolio_net_return", 0.0),
            field_name=f"artifacts[{index}].oos_metrics['portfolio_net_return']",
        )
        benchmark_krw = _to_float(
            benchmark,
            field_name=f"artifacts[{index}].oos_metrics['benchmark_krw_return']",
        )

        rows.append(
            {
                "fold_index": artifact.fold_index,
                "train_start": artifact.train_start,
                "train_end": artifact.train_end,
                "test_start": artifact.test_start,
                "test_end": artifact.test_end,
                "portfolio_net_return_krw": portfolio,
                "benchmark_return_krw": benchmark_krw,
                "active_net_return_krw": portfolio - benchmark_krw,
            }
        )

    return {"walkforward_benchmark_relative_table": {"rows": tuple(rows)}}


__all__ = [
    "build_backtest_attribution_report",
    "build_walkforward_attribution_report",
]
