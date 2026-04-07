# Research and Development Cost Factor Pipeline

This repository contains a deterministic research pipeline for building and evaluating an R&D-based equity factor strategy across the US, Korea, and Japan.

The pipeline covers:

- universe assembly for US/KR/JP equities
- synthetic or live-data ingestion
- point-in-time feature assembly
- factor scoring based on R&D intensity variants
- QEPM-style portfolio construction with country and risk controls
- quarterly backtesting
- validation and reporting artifact generation
- optional Rust kernel parity and fallback testing

## What the project does

The main workflow builds a long-only portfolio from R&D-related factor signals and produces a full research artifact set for each run. By default, it runs in a deterministic synthetic mode so the entire pipeline can be exercised without external APIs.

Supported markets:

- `combined`
- `us`
- `kr`
- `jp`

Supported data modes:

- `synthetic`
- `real`

Supported universe profiles:

- `broad`
- `core_indices`

Supported factor models:

- `rnd_sales_ttm`
- `rnd_sales_size_proxy`
- `rnd_mktcap_proxy`
- `rnd_ev_proxy`
- `rnd_robust_composite`

## Repository layout

```text
src/
  backtest/       Quarterly backtest engine
  cli/            End-to-end pipeline CLI
  config/         Strategy constants
  core/           Calendar and scheduling utilities
  data/           Ingestion, schema, security master, constituent history
  factor/         Factor computation logic
  features/       Point-in-time feature assembly
  portfolio/      Portfolio construction and constraints
  reporting/      Output reports and attribution
  rust_bridge/    Rust kernel dispatch, contracts, feature flags
  validation/     Walk-forward robustness checks

scripts/
  run_rust_parity.py

data/universe/
  sp500_symbols.csv
  nikkei225_codes.csv
```

## Quick start

Python 3.11 or newer is required.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .
python -m pip install pytest
```

## Main pipeline

The main pipeline entry point is the module CLI:

```bash
python -m src.cli.run_pipeline --start 2025-03-31 --end 2025-12-31
```

This default command:

- runs in `synthetic` mode
- uses `combined` market scope
- uses the `broad` universe profile
- rebalances on `quarter_end`
- uses the `rnd_sales_ttm` factor model
- writes outputs under `outputs/<run_id>/`

Example: single-market run

```bash
python -m src.cli.run_pipeline --start 2025-03-31 --end 2025-12-31 --market us
```

Example: core index universe

```bash
python -m src.cli.run_pipeline --start 2025-03-31 --end 2025-12-31 --market us --universe-profile core_indices
```

Example: alternate factor model

```bash
python -m src.cli.run_pipeline --start 2025-03-31 --end 2025-12-31 --factor-model rnd_mktcap_proxy
```

Example: separate US/KR/JP runs from one command

```bash
python -m src.cli.run_pipeline --start 2025-03-31 --end 2025-12-31 --separate-markets
```

Example: enable Rust kernels with Python fallback

```bash
python -m src.cli.run_pipeline --start 2025-03-31 --end 2025-12-31 --rust-kernels
```

## Live data mode

To run with live external sources, create a `.env` file or copy `.env.example`:

```env
SEC_USER_AGENT="Your Name your-email@example.com"
DART_API_KEY="your_dart_api_key"
EDINET_API_KEY="your_edinet_api_key"

BENCH_US="IWB"
BENCH_KR="069500.KS"
BENCH_JP="1306.T"
```

Then run:

```bash
python -m src.cli.run_pipeline --start 2025-03-31 --end 2025-12-31 --data-source real
```

The CLI loads environment variables from `.env` by default. You can override that path with `--env-file`.

## Output artifacts

Each successful run writes a directory like `outputs/20250331_20251231/` containing:

- `holdings.csv`
- `trades.csv`
- `metrics.json`
- `data_quality_report.json`
- `factor_integrity_report.json`
- `qepm_alignment_report.json`
- `perf_telemetry.json`
- `perf_comparison_report.json`
- `rollout_summary.json`
- `manifest.json`

The manifest includes required artifacts, checksums, and size metadata.

## Testing

Run the full test suite:

```bash
pytest
```

Run the smoke test for the installed package entry point:

```bash
pytest tests/smoke/test_cli_smoke.py
```

Note: the packaged `qsf` CLI currently provides a minimal scaffold interface, while the research pipeline itself is executed through `python -m src.cli.run_pipeline`.

## Rust parity harness

The repository includes a helper script for Python-vs-Rust parity checks:

```bash
python scripts/run_rust_parity.py --kernel all
```

Available kernels:

- `all`
- `ranking`
- `constructor`
- `backtest`

## Notes

- Default portfolio construction targets 20 holdings.
- Default country target weights are approximately one-third each for JP, KR, and US.
- Default single-name cap is `0.08`.
- Rust execution is opt-in and defaults to safe Python fallback when strict mode is not enabled.
