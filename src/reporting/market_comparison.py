from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt

from .analysis_artifacts import generate_analysis_artifacts


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _annualized(cumulative: float, years: float) -> float:
    if years <= 0.0:
        return 0.0
    return (1.0 + cumulative) ** (1.0 / years) - 1.0


def _growth_and_drawdown(rows: Sequence[Mapping[str, object]]) -> tuple[list[float], list[float]]:
    growth: list[float] = []
    drawdown: list[float] = []
    value = 1.0
    peak = 1.0
    for row in rows:
        value *= 1.0 + float(str(row["portfolio_net_return_krw"]))
        peak = max(peak, value)
        growth.append(value)
        drawdown.append((value / peak) - 1.0)
    return growth, drawdown


def generate_market_comparison_artifacts(*, run_id: str, markets_root: Path | None = None) -> Path:
    root = Path("outputs") / "markets" if markets_root is None else markets_root
    markets = ("us", "kr", "jp")
    market_dirs = {market: root / market / run_id for market in markets}

    for market, run_dir in market_dirs.items():
        if not run_dir.exists():
            raise FileNotFoundError(f"missing market run directory: {market} -> {run_dir}")
        _ = generate_analysis_artifacts(run_dir)

    comparison_dir = root / f"comparison_{run_id}"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    quarterly_by_market: dict[str, list[Mapping[str, object]]] = {}
    spread_by_market: dict[str, dict[str, float]] = {}

    for market in markets:
        run_dir = market_dirs[market]
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        quality = json.loads((run_dir / "data_quality_report.json").read_text(encoding="utf-8"))
        factor = json.loads((run_dir / "factor_integrity_report.json").read_text(encoding="utf-8"))

        backtest_metrics = metrics["backtest_metrics"]
        periods = int(backtest_metrics["periods"])
        years = periods / 4.0 if periods else 0.0
        cumulative_net = float(backtest_metrics["cumulative_net_return"])
        cumulative_benchmark = float(backtest_metrics["benchmark_cumulative_return"])
        annualized_net = _annualized(cumulative_net, years)
        annualized_benchmark = _annualized(cumulative_benchmark, years)

        with (run_dir / "analysis" / "quarterly_performance_krw.csv").open("r", encoding="utf-8", newline="") as handle:
            qrows = cast(list[Mapping[str, object]], list(csv.DictReader(handle)))
        quarterly_by_market[market] = qrows

        _, drawdown = _growth_and_drawdown(qrows)
        mdd = min(drawdown) if drawdown else 0.0
        active_positive = sum(1 for row in qrows if float(str(row["active_net_return_krw"])) > 0.0)

        spread_map: dict[str, float] = {}
        for row in factor["periods"]:
            value = row.get("factor_spread_selected_minus_non_selected")
            if value is None:
                continue
            spread_map[str(row["as_of_date"])] = float(str(value))
        spread_by_market[market] = spread_map

        summary_rows.append(
            {
                "market": market.upper(),
                "periods": periods,
                "years": years,
                "start_nav_krw": float(backtest_metrics["start_nav_krw"]),
                "end_nav_krw": float(backtest_metrics["end_nav_krw"]),
                "cumulative_net_return": cumulative_net,
                "cumulative_gross_return": float(backtest_metrics["cumulative_gross_return"]),
                "benchmark_cumulative_return": cumulative_benchmark,
                "active_cumulative_return": cumulative_net - cumulative_benchmark,
                "annualized_net_return": annualized_net,
                "annualized_benchmark_return": annualized_benchmark,
                "annualized_alpha": annualized_net - annualized_benchmark,
                "max_drawdown_krw_net": mdd,
                "active_positive_quarters": active_positive,
                "active_total_quarters": len(qrows),
                "active_positive_ratio": (active_positive / len(qrows)) if qrows else 0.0,
                "total_cost_tax_krw": float(backtest_metrics["total_cost_tax_krw"]),
                "total_slippage_krw": float(backtest_metrics["total_slippage_krw"]),
                "total_sell_tax_krw": float(backtest_metrics["total_sell_tax_krw"]),
                "total_realized_gains_tax_krw": float(backtest_metrics["total_realized_gains_tax_krw"]),
                "fallback_count": int(quality["checks"]["portfolio_fallback_count"]),
                "rejected_candidates": int(quality["checks"]["rejected_candidates"]),
                "quality_status": str(quality["status"]),
                "factor_positive_spread_periods": int(factor["summary"]["periods_with_positive_spread"]),
                "factor_periods_evaluated": int(factor["summary"]["periods_evaluated"]),
                "factor_mean_spread": float(factor["summary"]["mean_factor_spread_selected_minus_non_selected"]),
            }
        )

    summary_columns = list(summary_rows[0].keys())
    _write_csv(comparison_dir / "market_comparison_summary.csv", summary_columns, summary_rows)

    all_end_dates = sorted({str(row["end_date"]) for rows in quarterly_by_market.values() for row in rows})
    quarter_maps = {
        market: {str(row["end_date"]): row for row in rows}
        for market, rows in quarterly_by_market.items()
    }
    quarter_rows: list[dict[str, object]] = []
    for end_date in all_end_dates:
        record: dict[str, object] = {"end_date": end_date}
        for market in markets:
            row = quarter_maps[market].get(end_date)
            record[f"{market}_portfolio_net_return_krw"] = "" if row is None else row["portfolio_net_return_krw"]
            record[f"{market}_benchmark_return_krw"] = "" if row is None else row["benchmark_return_krw"]
            record[f"{market}_active_net_return_krw"] = "" if row is None else row["active_net_return_krw"]
        quarter_rows.append(record)
    _write_csv(
        comparison_dir / "quarterly_return_comparison.csv",
        [
            "end_date",
            "us_portfolio_net_return_krw",
            "us_benchmark_return_krw",
            "us_active_net_return_krw",
            "kr_portfolio_net_return_krw",
            "kr_benchmark_return_krw",
            "kr_active_net_return_krw",
            "jp_portfolio_net_return_krw",
            "jp_benchmark_return_krw",
            "jp_active_net_return_krw",
        ],
        quarter_rows,
    )

    spread_dates = sorted({day for market in markets for day in spread_by_market[market]})
    spread_rows: list[dict[str, object]] = []
    for spread_date in spread_dates:
        spread_rows.append(
            {
                "as_of_date": spread_date,
                "us_factor_spread": spread_by_market["us"].get(spread_date, ""),
                "kr_factor_spread": spread_by_market["kr"].get(spread_date, ""),
                "jp_factor_spread": spread_by_market["jp"].get(spread_date, ""),
            }
        )
    _write_csv(
        comparison_dir / "factor_spread_comparison.csv",
        ["as_of_date", "us_factor_spread", "kr_factor_spread", "jp_factor_spread"],
        spread_rows,
    )

    palette = {"us": "#1f77b4", "kr": "#2ca02c", "jp": "#ff7f0e"}

    plt.figure(figsize=(10, 5))
    for market in markets:
        growth, _ = _growth_and_drawdown(quarterly_by_market[market])
        x = list(range(len(growth)))
        _ = plt.plot(x, growth, label=market.upper(), color=palette[market], linewidth=2)
    _ = plt.title("Cumulative Growth by Market (KRW net)")
    _ = plt.xlabel("Quarter index")
    _ = plt.ylabel("Growth Multiple")
    _ = plt.grid(alpha=0.3)
    _ = plt.legend()
    plt.tight_layout()
    plt.savefig(comparison_dir / "market_cumulative_growth.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    for market in markets:
        _, drawdown = _growth_and_drawdown(quarterly_by_market[market])
        x = list(range(len(drawdown)))
        _ = plt.plot(x, drawdown, label=market.upper(), color=palette[market], linewidth=2)
    _ = plt.title("Drawdown by Market (KRW net)")
    _ = plt.xlabel("Quarter index")
    _ = plt.ylabel("Drawdown")
    _ = plt.grid(alpha=0.3)
    _ = plt.legend()
    plt.tight_layout()
    plt.savefig(comparison_dir / "market_drawdown.png", dpi=150)
    plt.close()

    labels = [str(row["market"]) for row in summary_rows]
    ann_port = [float(str(row["annualized_net_return"])) for row in summary_rows]
    ann_bench = [float(str(row["annualized_benchmark_return"])) for row in summary_rows]
    x = list(range(len(labels)))
    width = 0.35

    plt.figure(figsize=(8, 5))
    _ = plt.bar([value - width / 2.0 for value in x], ann_port, width=width, label="Portfolio", color="#34495e")
    _ = plt.bar([value + width / 2.0 for value in x], ann_bench, width=width, label="Benchmark", color="#95a5a6")
    _ = plt.xticks(x, labels)
    _ = plt.title("Annualized Return Comparison")
    _ = plt.ylabel("Return")
    _ = plt.grid(axis="y", alpha=0.3)
    _ = plt.legend()
    plt.tight_layout()
    plt.savefig(comparison_dir / "annualized_return_comparison.png", dpi=150)
    plt.close()

    active_ratio = [float(str(row["active_positive_ratio"])) for row in summary_rows]
    plt.figure(figsize=(8, 5))
    _ = plt.bar(labels, active_ratio, color=[palette["us"], palette["kr"], palette["jp"]])
    _ = plt.ylim(0.0, 1.0)
    _ = plt.title("Active Positive Quarter Ratio")
    _ = plt.ylabel("Ratio")
    _ = plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(comparison_dir / "active_positive_ratio.png", dpi=150)
    plt.close()

    slippage = [float(str(row["total_slippage_krw"])) for row in summary_rows]
    sell_tax = [float(str(row["total_sell_tax_krw"])) for row in summary_rows]
    realized_tax = [float(str(row["total_realized_gains_tax_krw"])) for row in summary_rows]

    plt.figure(figsize=(8, 5))
    _ = plt.bar(labels, slippage, label="Slippage", color="#3498db")
    _ = plt.bar(labels, sell_tax, bottom=slippage, label="Sell Tax", color="#9b59b6")
    _ = plt.bar(
        labels,
        realized_tax,
        bottom=[left + right for left, right in zip(slippage, sell_tax)],
        label="Realized Gains Tax",
        color="#e74c3c",
    )
    _ = plt.title("Total Cost/Tax Decomposition (KRW)")
    _ = plt.ylabel("KRW")
    _ = plt.grid(axis="y", alpha=0.3)
    _ = plt.legend()
    plt.tight_layout()
    plt.savefig(comparison_dir / "cost_decomposition_comparison.png", dpi=150)
    plt.close()

    return comparison_dir


__all__ = ["generate_market_comparison_artifacts"]
