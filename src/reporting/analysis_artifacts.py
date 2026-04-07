from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import matplotlib.pyplot as plt


def _to_date(value: str) -> date:
    return datetime.fromisoformat(value).date()


def _write_rows(path: Path, columns: list[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def generate_analysis_artifacts(run_dir: Path) -> Path:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    factor_integrity = json.loads((run_dir / "factor_integrity_report.json").read_text(encoding="utf-8"))

    output_dir = run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    backtest_metrics = metrics["backtest_metrics"]
    krw_rows = metrics["backtest_report"]["krw_main_performance_table"]["rows"]
    usd_rows = metrics["backtest_report"]["usd_supplemental_table"]["rows"]
    fx_rows = metrics["backtest_report"]["fx_contribution_decomposition"]["rows"]

    periods = int(backtest_metrics["periods"])
    years = periods / 4.0 if periods else 0.0
    cumulative_net = float(backtest_metrics["cumulative_net_return"])
    cumulative_gross = float(backtest_metrics["cumulative_gross_return"])
    benchmark_cumulative = float(backtest_metrics["benchmark_cumulative_return"])
    annualized_net = ((1.0 + cumulative_net) ** (1.0 / years) - 1.0) if years > 0 else 0.0
    annualized_benchmark = ((1.0 + benchmark_cumulative) ** (1.0 / years) - 1.0) if years > 0 else 0.0

    summary_rows = [
        {"metric": "run_id", "value": run_dir.name},
        {"metric": "periods_quarterly", "value": periods},
        {"metric": "years", "value": round(years, 2)},
        {"metric": "start_nav_krw", "value": float(backtest_metrics["start_nav_krw"])},
        {"metric": "end_nav_krw", "value": float(backtest_metrics["end_nav_krw"])},
        {"metric": "cumulative_net_return", "value": cumulative_net},
        {"metric": "cumulative_gross_return", "value": cumulative_gross},
        {"metric": "benchmark_cumulative_return", "value": benchmark_cumulative},
        {"metric": "active_cumulative_return", "value": cumulative_net - benchmark_cumulative},
        {"metric": "annualized_net_return", "value": annualized_net},
        {"metric": "annualized_benchmark_return", "value": annualized_benchmark},
        {"metric": "annualized_alpha", "value": annualized_net - annualized_benchmark},
        {"metric": "total_cost_tax_krw", "value": float(backtest_metrics["total_cost_tax_krw"])},
        {"metric": "total_slippage_krw", "value": float(backtest_metrics["total_slippage_krw"])},
        {"metric": "total_sell_tax_krw", "value": float(backtest_metrics["total_sell_tax_krw"])},
        {"metric": "total_realized_gains_tax_krw", "value": float(backtest_metrics["total_realized_gains_tax_krw"])},
    ]
    _write_rows(output_dir / "summary_metrics.csv", ["metric", "value"], summary_rows)

    _write_rows(
        output_dir / "quarterly_performance_krw.csv",
        [
            "start_date",
            "end_date",
            "portfolio_net_return_krw",
            "portfolio_gross_return_krw",
            "benchmark_return_krw",
            "active_net_return_krw",
            "active_gross_return_krw",
            "total_cost_tax_krw",
        ],
        krw_rows,
    )

    _write_rows(
        output_dir / "quarterly_performance_usd.csv",
        [
            "start_date",
            "end_date",
            "portfolio_net_return_usd",
            "portfolio_gross_return_usd",
            "benchmark_return_usd",
            "active_net_return_usd",
            "active_gross_return_usd",
            "fx_return_usdkrw",
        ],
        usd_rows,
    )

    _write_rows(
        output_dir / "fx_contribution_table.csv",
        [
            "start_date",
            "end_date",
            "portfolio_net_return_krw",
            "portfolio_net_return_usd",
            "portfolio_net_fx_contribution",
            "benchmark_return_krw",
            "benchmark_return_usd",
            "benchmark_fx_contribution",
            "fx_return_usdkrw",
        ],
        fx_rows,
    )

    _write_rows(
        output_dir / "factor_integrity_by_period.csv",
        [
            "as_of_date",
            "eligible_count",
            "selected_count",
            "factor_spread_selected_minus_non_selected",
            "selected_at_or_above_eligible_median_ratio",
            "selected_top_quartile_rank_ratio",
        ],
        factor_integrity["periods"],
    )

    with (run_dir / "holdings.csv").open("r", encoding="utf-8", newline="") as handle:
        holdings = list(csv.DictReader(handle))

    latest_date = max(row["as_of_date"] for row in holdings)
    latest_holdings = [row for row in holdings if row["as_of_date"] == latest_date]
    latest_holdings_sorted = sorted(latest_holdings, key=lambda row: float(row["weight"]), reverse=True)

    _write_rows(
        output_dir / "latest_top_holdings.csv",
        ["as_of_date", "security_id", "country", "ticker", "stock_code", "stock_name", "weight", "market_value_krw"],
        latest_holdings_sorted,
    )

    country_map: dict[str, dict[str, float]] = defaultdict(lambda: {"market_value_krw": 0.0, "weight": 0.0, "holding_count": 0.0})
    for row in latest_holdings:
        country = row["country"]
        country_map[country]["market_value_krw"] += float(row["market_value_krw"])
        country_map[country]["weight"] += float(row["weight"])
        country_map[country]["holding_count"] += 1.0

    country_rows: list[dict[str, object]] = []
    for country in sorted(country_map):
        value = country_map[country]
        country_rows.append(
            {
                "as_of_date": latest_date,
                "country": country,
                "holding_count": int(value["holding_count"]),
                "total_weight": value["weight"],
                "total_market_value_krw": value["market_value_krw"],
            }
        )
    _write_rows(
        output_dir / "latest_country_allocation.csv",
        ["as_of_date", "country", "holding_count", "total_weight", "total_market_value_krw"],
        country_rows,
    )

    end_dates = [_to_date(str(row["end_date"])) for row in krw_rows]
    x_index = list(range(len(end_dates)))
    tick_step = max(1, len(end_dates) // 10)
    tick_positions = x_index[::tick_step]
    tick_labels = [end_dates[index].isoformat() for index in tick_positions]
    growth_portfolio: list[float] = []
    growth_benchmark: list[float] = []
    active_net: list[float] = []
    drawdown: list[float] = []

    portfolio = 1.0
    benchmark = 1.0
    peak = 1.0
    for row in krw_rows:
        net = float(row["portfolio_net_return_krw"])
        bench = float(row["benchmark_return_krw"])
        active = float(row["active_net_return_krw"])
        portfolio *= 1.0 + net
        benchmark *= 1.0 + bench
        peak = max(peak, portfolio)
        growth_portfolio.append(portfolio)
        growth_benchmark.append(benchmark)
        active_net.append(active)
        drawdown.append((portfolio / peak) - 1.0)

    plt.figure(figsize=(10, 5))
    plt.plot(x_index, growth_portfolio, label="Portfolio KRW (net)", linewidth=2)
    plt.plot(x_index, growth_benchmark, label="Benchmark KRW", linewidth=2)
    plt.title("Cumulative Growth (2016-2025)")
    plt.xlabel("Date")
    plt.ylabel("Growth Multiple")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.xticks(tick_positions, tick_labels, rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "cumulative_growth_krw.png", dpi=140)
    plt.close()

    plt.figure(figsize=(10, 5))
    colors = ["#2E8B57" if value >= 0 else "#B22222" for value in active_net]
    plt.bar(x_index, active_net, color=colors)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.title("Quarterly Active Net Return (KRW)")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(tick_positions, tick_labels, rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "quarterly_active_return_krw.png", dpi=140)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.fill_between(x_index, drawdown, 0.0, color="#C0392B", alpha=0.35)
    plt.plot(x_index, drawdown, color="#922B21", linewidth=1.5)
    plt.title("Portfolio Drawdown (KRW net)")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(alpha=0.3)
    plt.xticks(tick_positions, tick_labels, rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "portfolio_drawdown_krw.png", dpi=140)
    plt.close()

    plt.figure(figsize=(7, 5))
    country_labels = [str(row["country"]) for row in country_rows]
    country_weights = [float(str(row["total_weight"])) for row in country_rows]
    plt.bar(country_labels, country_weights, color=["#1F77B4", "#2CA02C", "#FF7F0E"])
    plt.title(f"Latest Country Allocation ({latest_date})")
    plt.xlabel("Country")
    plt.ylabel("Weight")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "latest_country_allocation.png", dpi=140)
    plt.close()

    spread_dates = [_to_date(str(row["as_of_date"])) for row in factor_integrity["periods"]]
    spread_index = list(range(len(spread_dates)))
    spread_step = max(1, len(spread_dates) // 10)
    spread_positions = spread_index[::spread_step]
    spread_labels = [spread_dates[index].isoformat() for index in spread_positions]
    spread_values = [
        float(str(value)) if value is not None else float("nan")
        for value in [row.get("factor_spread_selected_minus_non_selected") for row in factor_integrity["periods"]]
    ]

    plt.figure(figsize=(10, 5))
    plt.plot(spread_index, spread_values, color="#6A1B9A", linewidth=2)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.title("Factor Spread by Rebalance")
    plt.xlabel("Date")
    plt.ylabel("R&D/Sales Spread")
    plt.grid(alpha=0.3)
    plt.xticks(spread_positions, spread_labels, rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "factor_spread_trend.png", dpi=140)
    plt.close()

    return output_dir


__all__ = ["generate_analysis_artifacts"]
