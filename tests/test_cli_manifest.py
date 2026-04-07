from __future__ import annotations

import concurrent.futures
import hashlib
import importlib
import json
import os
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Protocol, cast

import pytest


class _CliModule(Protocol):
    REQUIRED_ARTIFACTS: tuple[str, ...]

    def main(self, argv: list[str] | None = None) -> int: ...

    def make_run_id(self, start: date, end: date) -> str: ...


cli = cast(_CliModule, cast(object, importlib.import_module("src.cli.run_pipeline")))


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _as_int(value: object) -> int:
    return int(str(value))


def _as_float(value: object) -> float:
    return float(str(value))


def test_cli_generates_required_artifacts_and_manifest_checksums(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    start = date(2025, 3, 31)
    end = date(2025, 12, 31)
    exit_code = cli.main(["--start", start.isoformat(), "--end", end.isoformat()])

    assert exit_code == 0

    run_id = cli.make_run_id(start, end)
    output_dir = tmp_path / "outputs" / run_id
    rollout_summary_path = output_dir / "rollout_summary.json"

    for artifact in cli.REQUIRED_ARTIFACTS:
        assert (output_dir / artifact).exists()
    assert rollout_summary_path.exists()

    manifest = cast(dict[str, object], json.loads((output_dir / "manifest.json").read_text(encoding="utf-8")))
    required_artifacts = cast(list[str], manifest["required_artifacts"])
    assert required_artifacts == list(cli.REQUIRED_ARTIFACTS)
    assert "rollout_summary.json" in required_artifacts

    rollout_summary = cast(dict[str, object], json.loads(rollout_summary_path.read_text(encoding="utf-8")))
    signals = cast(dict[str, object], rollout_summary["signals"])
    assert set(signals) == {"anchor", "benchmark", "parity"}

    checksums = cast(dict[str, dict[str, object]], manifest["checksums"])
    for artifact_name in (
        "holdings.csv",
        "trades.csv",
        "metrics.json",
        "data_quality_report.json",
        "factor_integrity_report.json",
        "perf_telemetry.json",
        "perf_comparison_report.json",
        "rollout_summary.json",
    ):
        artifact_path = output_dir / artifact_name
        assert checksums[artifact_name]["sha256"] == _sha256(artifact_path)
        assert checksums[artifact_name]["size_bytes"] == artifact_path.stat().st_size


def test_cli_default_mode_preserves_python_backend_for_rollback_safety(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(["--start", "2025-03-31", "--end", "2025-12-31"])

    assert exit_code == 0
    output_dir = tmp_path / "outputs" / cli.make_run_id(date(2025, 3, 31), date(2025, 12, 31))
    payload = cast(dict[str, object], json.loads((output_dir / "metrics.json").read_text(encoding="utf-8")))
    config = cast(dict[str, object], payload["config"])
    rollout_summary = cast(
        dict[str, object],
        json.loads((output_dir / "rollout_summary.json").read_text(encoding="utf-8")),
    )
    rust_backend = cast(dict[str, object], rollout_summary["rust_backend"])

    assert cast(dict[str, bool], config["rust_kernel_flags"]) == {
        "ranking": False,
        "constructor": False,
        "backtest": False,
        "strict": False,
    }
    assert config["rust_backtest_backend"] == "python"
    assert config["rust_backtest_reason"] == "feature_disabled"
    assert cast(dict[str, bool], rust_backend["flags"]) == {
        "ranking": False,
        "constructor": False,
        "backtest": False,
        "strict": False,
    }
    assert rust_backend["backtest_backend"] == "python"
    assert rust_backend["backtest_reason"] == "feature_disabled"


def test_cli_generates_factor_integrity_report_with_expected_structure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    start = date(2025, 3, 31)
    end = date(2025, 12, 31)
    exit_code = cli.main(["--start", start.isoformat(), "--end", end.isoformat()])

    assert exit_code == 0

    run_id = cli.make_run_id(start, end)
    output_dir = tmp_path / "outputs" / run_id
    report_path = output_dir / "factor_integrity_report.json"

    payload = cast(dict[str, object], json.loads(report_path.read_text(encoding="utf-8")))
    summary = cast(dict[str, object], payload["summary"])
    periods = cast(list[dict[str, object]], payload["periods"])

    assert _as_int(summary["periods_evaluated"]) == len(periods)
    assert len(periods) > 0
    assert 0 <= _as_int(summary["periods_with_positive_spread"]) <= len(periods)

    for row in periods:
        eligible_count = _as_int(row["eligible_count"])
        selected_count = _as_int(row["selected_count"])
        assert selected_count <= eligible_count
        ratio = row["selected_top_quartile_rank_ratio"]
        if ratio is not None:
            parsed_ratio = _as_float(ratio)
            assert 0.0 <= parsed_ratio <= 1.0


def test_cli_emits_perf_telemetry_schema_and_missing_baseline_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    start = date(2025, 3, 31)
    end = date(2025, 12, 31)
    exit_code = cli.main(["--start", start.isoformat(), "--end", end.isoformat()])

    assert exit_code == 0

    run_id = cli.make_run_id(start, end)
    output_dir = tmp_path / "outputs" / run_id
    telemetry_payload = cast(dict[str, object], json.loads((output_dir / "perf_telemetry.json").read_text(encoding="utf-8")))
    comparison_payload = cast(dict[str, object], json.loads((output_dir / "perf_comparison_report.json").read_text(encoding="utf-8")))

    assert telemetry_payload["schema_version"] == "v1"
    run_meta = cast(dict[str, object], telemetry_payload["run"])
    assert run_meta["run_id"] == run_id
    assert run_meta["market"] == "combined"

    runtime_seconds = cast(dict[str, object], telemetry_payload["runtime_seconds"])
    assert float(str(runtime_seconds["ingestion"])) >= 0.0
    assert float(str(runtime_seconds["compute"])) >= 0.0
    assert float(str(runtime_seconds["total"])) >= float(str(runtime_seconds["ingestion"]))

    cache_counters = cast(dict[str, object], telemetry_payload["cache_counters"])
    assert set(cache_counters) == {
        "yfinance_v2_hit",
        "yfinance_v1_fallback",
        "yfinance_network_fetch",
        "yfinance_repair_success",
        "yfinance_repair_failure",
    }

    fallback_counters = cast(dict[str, object], telemetry_payload["fallback_counters"])
    assert int(str(fallback_counters["portfolio_fallback_count"])) >= 0
    assert int(str(fallback_counters["rejected_candidates"])) >= 0

    assert comparison_payload["status"] == "baseline_missing"
    assert comparison_payload["baseline_available"] is False
    assert telemetry_payload["baseline_comparison_status"] == "baseline_missing"


def test_cli_returns_nonzero_when_quality_gate_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli,
        "REQUIRED_ARTIFACTS",
        cli.REQUIRED_ARTIFACTS + ("missing_artifact.csv",),
    )

    exit_code = cli.main(["--start", "2025-03-31", "--end", "2025-12-31"])

    assert exit_code != 0


def test_cli_real_mode_returns_nonzero_when_required_env_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.delenv("DART_API_KEY", raising=False)
    monkeypatch.delenv("EDINET_API_KEY", raising=False)

    exit_code = cli.main([
        "--start",
        "2025-03-31",
        "--end",
        "2025-12-31",
        "--data-source",
        "real",
    ])

    assert exit_code != 0


def test_cli_real_mode_loads_required_env_from_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.delenv("DART_API_KEY", raising=False)
    monkeypatch.delenv("EDINET_API_KEY", raising=False)

    env_file = tmp_path / ".env"
    _ = env_file.write_text(
        "\n".join(
            [
                'SEC_USER_AGENT="Env User env@example.com"',
                "DART_API_KEY='env-dart'",
                "EDINET_API_KEY='env-edinet'",
            ]
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def _fake_run_pipeline(**_: object) -> Path:
        captured["sec_user_agent"] = os.getenv("SEC_USER_AGENT")
        captured["dart_api_key"] = os.getenv("DART_API_KEY")
        captured["edinet_api_key"] = os.getenv("EDINET_API_KEY")
        return tmp_path / "outputs" / "fake"

    monkeypatch.setattr(cli, "run_pipeline", _fake_run_pipeline)

    exit_code = cli.main([
        "--start",
        "2025-03-31",
        "--end",
        "2025-12-31",
        "--data-source",
        "real",
    ])

    assert exit_code == 0
    assert captured["sec_user_agent"] == "Env User env@example.com"
    assert captured["dart_api_key"] == "env-dart"
    assert captured["edinet_api_key"] == "env-edinet"


def test_cli_accepts_ingestion_tuning_flags_in_synthetic_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main([
        "--start",
        "2025-03-31",
        "--end",
        "2025-12-31",
        "--sec-stage-budget-sec",
        "300",
        "--sec-timeout-sec",
        "15",
        "--sec-max-retries",
        "1",
        "--sec-backoff-sec",
        "0.5",
        "--sec-max-rps",
        "6",
        "--sec-max-workers",
        "2",
        "--no-sec-allow-stale-cache",
        "--yfinance-cache-ttl-days",
        "2",
        "--yfinance-chunk-size",
        "40",
        "--yfinance-max-workers",
        "2",
        "--yfinance-max-retries",
        "1",
        "--yfinance-backoff-sec",
        "0.5",
        "--dart-timeout-sec",
        "15",
        "--dart-max-retries",
        "1",
        "--dart-backoff-sec",
        "0.5",
        "--dart-max-rps",
        "6",
        "--dart-max-workers",
        "2",
        "--edinet-max-workers",
        "2",
        "--edinet-max-rps",
        "4",
        "--edinet-max-retries",
        "1",
        "--edinet-backoff-sec",
        "0.5",
    ])

    assert exit_code == 0


def test_cli_returns_nonzero_for_invalid_chunk_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main([
        "--start",
        "2025-03-31",
        "--end",
        "2025-12-31",
        "--yfinance-chunk-size",
        "0",
    ])

    assert exit_code != 0


def test_cli_returns_nonzero_for_invalid_min_rd_coverage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main([
        "--start",
        "2025-03-31",
        "--end",
        "2025-12-31",
        "--min-rd-coverage-kr",
        "1.1",
    ])

    assert exit_code != 0


def test_cli_accepts_start_date_rebalance_anchor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    start = date(2025, 1, 1)
    end = date(2025, 12, 31)
    exit_code = cli.main([
        "--start",
        start.isoformat(),
        "--end",
        end.isoformat(),
        "--rebalance-anchor",
        "start_date",
    ])

    assert exit_code == 0
    run_id = cli.make_run_id(start, end)
    output_dir = tmp_path / "outputs" / run_id
    payload = cast(dict[str, object], json.loads((output_dir / "metrics.json").read_text(encoding="utf-8")))
    config = cast(dict[str, object], payload["config"])
    assert config["rebalance_anchor"] == "start_date"


def test_cli_start_date_anchor_is_not_phase_locked_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main([
        "--start",
        "2025-01-01",
        "--end",
        "2025-12-31",
        "--rebalance-anchor",
        "start_date",
    ])

    assert exit_code == 0
    output_dir = tmp_path / "outputs" / cli.make_run_id(date(2025, 1, 1), date(2025, 12, 31))
    payload = cast(dict[str, object], json.loads((output_dir / "metrics.json").read_text(encoding="utf-8")))
    config = cast(dict[str, object], payload["config"])
    assert config["rebalance_anchor"] == "start_date"
    assert config["rebalance_anchor_effective"] == "start_date"


def test_cli_can_enable_phase_lock_explicitly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main([
        "--start",
        "2025-01-01",
        "--end",
        "2025-12-31",
        "--rebalance-anchor",
        "start_date",
        "--qepm-phase-lock-quarter-end",
    ])

    assert exit_code == 0
    output_dir = tmp_path / "outputs" / cli.make_run_id(date(2025, 1, 1), date(2025, 12, 31))
    payload = cast(dict[str, object], json.loads((output_dir / "metrics.json").read_text(encoding="utf-8")))
    config = cast(dict[str, object], payload["config"])
    assert config["rebalance_anchor"] == "start_date"
    assert config["rebalance_anchor_effective"] == "quarter_end"


def test_cli_accepts_new_rnd_denominator_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main([
        "--start",
        "2025-03-31",
        "--end",
        "2025-12-31",
        "--factor-model",
        "rnd_mktcap_proxy",
    ])

    assert exit_code == 0
    output_dir = tmp_path / "outputs" / cli.make_run_id(date(2025, 3, 31), date(2025, 12, 31))
    payload = cast(dict[str, object], json.loads((output_dir / "metrics.json").read_text(encoding="utf-8")))
    config = cast(dict[str, object], payload["config"])
    assert config["factor_model"] == "rnd_mktcap_proxy"
    assert config["factor_axis"] == "R&D"
    assert config["denominator_model"] == "mktcap_proxy"

    alignment = cast(
        dict[str, object],
        json.loads((output_dir / "qepm_alignment_report.json").read_text(encoding="utf-8")),
    )
    assert alignment["status"] in {"pass", "warning"}


def test_cli_accepts_rust_kernel_flags_with_python_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main([
        "--start",
        "2025-03-31",
        "--end",
        "2025-12-31",
        "--rust-kernels",
    ])

    assert exit_code == 0
    output_dir = tmp_path / "outputs" / cli.make_run_id(date(2025, 3, 31), date(2025, 12, 31))
    payload = cast(dict[str, object], json.loads((output_dir / "metrics.json").read_text(encoding="utf-8")))
    config = cast(dict[str, object], payload["config"])
    rust_flags = cast(dict[str, bool], config["rust_kernel_flags"])
    assert rust_flags == {
        "ranking": True,
        "constructor": True,
        "backtest": True,
        "strict": False,
    }


def test_cli_accepts_core_indices_universe_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main([
        "--start",
        "2025-03-31",
        "--end",
        "2025-12-31",
        "--market",
        "us",
        "--universe-profile",
        "core_indices",
    ])

    assert exit_code == 0
    output_dir = tmp_path / "outputs" / cli.make_run_id(date(2025, 3, 31), date(2025, 12, 31))
    payload = cast(dict[str, object], json.loads((output_dir / "metrics.json").read_text(encoding="utf-8")))
    config = cast(dict[str, object], payload["config"])
    assert config["universe_profile"] == "core_indices"


def test_cli_separate_markets_parallel_uses_bounded_workers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    start = date(2025, 3, 31)
    end = date(2025, 12, 31)
    run_id = cli.make_run_id(start, end)
    captured_markets: list[str] = []
    executor_workers: dict[str, int] = {}

    class _InlineProcessPoolExecutor:
        def __init__(self, max_workers: int | None = None, **_: object) -> None:
            executor_workers["value"] = int(str(max_workers)) if max_workers is not None else 0

        def __enter__(self) -> _InlineProcessPoolExecutor:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

        def submit(self, fn: object, *args: object, **kwargs: object) -> concurrent.futures.Future[object]:
            future: concurrent.futures.Future[object] = concurrent.futures.Future()
            runner = cast(Callable[..., object], fn)
            try:
                result = runner(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - parity with executor behavior
                future.set_exception(exc)
            else:
                future.set_result(result)
            return future

    def _fake_run_pipeline(**kwargs: object) -> Path:
        market = str(kwargs["market"])
        output_root = cast(Path, kwargs["output_root"])
        output_dir = output_root / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        captured_markets.append(market)
        return output_dir

    monkeypatch.setattr(cli, "ProcessPoolExecutor", _InlineProcessPoolExecutor)
    monkeypatch.setattr(cli, "run_pipeline", _fake_run_pipeline)

    exit_code = cli.main([
        "--start",
        start.isoformat(),
        "--end",
        end.isoformat(),
        "--separate-markets",
        "--separate-markets-max-workers",
        "2",
    ])

    assert exit_code == 0
    assert executor_workers["value"] == 2
    assert sorted(captured_markets) == ["jp", "kr", "us"]
    assert (tmp_path / "outputs" / "markets" / "us" / run_id).exists()
    assert (tmp_path / "outputs" / "markets" / "kr" / run_id).exists()
    assert (tmp_path / "outputs" / "markets" / "jp" / run_id).exists()


def test_cli_separate_markets_sequential_mode_skips_process_pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    start = date(2025, 3, 31)
    end = date(2025, 12, 31)
    run_id = cli.make_run_id(start, end)
    call_order: list[str] = []

    def _pool_not_expected(*_: object, **_kwargs: object) -> None:
        raise AssertionError("ProcessPoolExecutor should not be used in sequential mode")

    def _fake_run_pipeline(**kwargs: object) -> Path:
        market = str(kwargs["market"])
        output_root = cast(Path, kwargs["output_root"])
        output_dir = output_root / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        call_order.append(market)
        return output_dir

    monkeypatch.setattr(cli, "ProcessPoolExecutor", _pool_not_expected)
    monkeypatch.setattr(cli, "run_pipeline", _fake_run_pipeline)

    exit_code = cli.main([
        "--start",
        start.isoformat(),
        "--end",
        end.isoformat(),
        "--separate-markets",
        "--separate-markets-sequential",
        "--separate-markets-max-workers",
        "3",
    ])

    assert exit_code == 0
    assert call_order == ["us", "kr", "jp"]


def test_cli_separate_markets_rejects_worker_count_above_market_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    def _unexpected_run_pipeline(**_: object) -> Path:
        raise AssertionError("run_pipeline should not execute when worker guard fails")

    monkeypatch.setattr(cli, "run_pipeline", _unexpected_run_pipeline)

    exit_code = cli.main([
        "--start",
        "2025-03-31",
        "--end",
        "2025-12-31",
        "--separate-markets",
        "--separate-markets-max-workers",
        "4",
    ])

    assert exit_code != 0
