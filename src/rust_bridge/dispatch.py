"""Kernel dispatch helpers that keep Python as default-safe path."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .bindings import load_extension
from .contracts import (
    BacktestKernelRequest,
    ConstructorKernelRequest,
    RankingKernelRequest,
    decode_ranking_result,
    deserialize_constructor_result,
    decode_backtest_result,
    serialize_ranking_request,
    serialize_constructor_request,
    serialize_backtest_request,
)
from .feature_flags import RustKernelFlags


def _call_rust_or_fallback(
    *,
    enabled: bool,
    strict: bool,
    function_name: str,
    request: object,
    python_impl: Callable[[], Any],
) -> tuple[Any, str, str | None]:
    if not enabled:
        return python_impl(), "python", "feature_disabled"

    module, load_error = load_extension()
    if module is None:
        if strict:
            raise RuntimeError(load_error or "rust extension unavailable")
        return python_impl(), "python", load_error

    rust_fn = getattr(module, function_name, None)
    if not callable(rust_fn):
        message = f"rust extension missing callable: {function_name}"
        if strict:
            raise RuntimeError(message)
        return python_impl(), "python", message

    try:
        rust_result = rust_fn(request)
    except Exception as exc:
        if strict:
            raise
        return python_impl(), "python", f"rust call failed: {exc}"
    return rust_result, "rust", None


def run_ranking_kernel(
    *,
    request: RankingKernelRequest,
    flags: RustKernelFlags,
    python_impl: Callable[[], Any],
) -> tuple[Any, str, str | None]:
    rust_result, backend, reason = _call_rust_or_fallback(
        enabled=flags.ranking,
        strict=flags.strict,
        function_name="run_ranking_kernel",
        request=serialize_ranking_request(request),
        python_impl=python_impl,
    )
    if backend != "rust":
        return rust_result, backend, reason

    try:
        return decode_ranking_result(rust_result), "rust", None
    except Exception as exc:
        if flags.strict:
            raise RuntimeError(f"rust ranking decode failed: {exc}") from exc
        return python_impl(), "python", f"rust output decode failed: {exc}"


def run_constructor_kernel(
    *,
    request: ConstructorKernelRequest,
    flags: RustKernelFlags,
    python_impl: Callable[[], Any],
) -> tuple[Any, str, str | None]:
    rust_result, backend, reason = _call_rust_or_fallback(
        enabled=flags.constructor,
        strict=flags.strict,
        function_name="run_constructor_kernel",
        request=serialize_constructor_request(request),
        python_impl=python_impl,
    )
    if backend != "rust":
        return rust_result, backend, reason

    try:
        return deserialize_constructor_result(rust_result), "rust", None
    except Exception as exc:
        if flags.strict:
            raise RuntimeError(f"rust constructor decode failed: {exc}") from exc
        return python_impl(), "python", f"rust output decode failed: {exc}"


def run_backtest_kernel(
    *,
    request: BacktestKernelRequest,
    flags: RustKernelFlags,
    python_impl: Callable[[], Any],
) -> tuple[Any, str, str | None]:
    rust_result, backend, reason = _call_rust_or_fallback(
        enabled=flags.backtest,
        strict=flags.strict,
        function_name="run_backtest_kernel",
        request=serialize_backtest_request(request),
        python_impl=python_impl,
    )
    if backend != "rust":
        return rust_result, backend, reason

    try:
        return decode_backtest_result(rust_result), "rust", None
    except Exception as exc:
        if flags.strict:
            raise RuntimeError(f"rust backtest decode failed: {exc}") from exc
        return python_impl(), "python", f"rust output decode failed: {exc}"


__all__ = [
    "run_backtest_kernel",
    "run_constructor_kernel",
    "run_ranking_kernel",
]
