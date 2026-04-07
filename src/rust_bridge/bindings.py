"""Runtime binding loader for optional Rust extension module."""

from __future__ import annotations

import importlib
from pathlib import Path

RUST_EXTENSION_MODULE = "qsf_rust_kernels"
RUST_CRATE_DIR = Path(__file__).resolve().parents[2] / "rust" / "qsf_rust_kernels"


def load_extension() -> tuple[object | None, str | None]:
    try:
        module = importlib.import_module(RUST_EXTENSION_MODULE)
    except Exception as exc:
        return None, f"rust extension import failed: {exc}"
    return module, None


__all__ = ["RUST_CRATE_DIR", "RUST_EXTENSION_MODULE", "load_extension"]
