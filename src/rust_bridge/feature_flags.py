"""Feature flags for staged Rust kernel enablement."""

from __future__ import annotations

from dataclasses import dataclass
import os


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class RustKernelFlags:
    ranking: bool = False
    constructor: bool = False
    backtest: bool = False
    strict: bool = False

    @classmethod
    def from_inputs(
        cls,
        *,
        enable_all: bool = False,
        enable_ranking: bool = False,
        enable_constructor: bool = False,
        enable_backtest: bool = False,
        strict: bool = False,
    ) -> RustKernelFlags:
        env_all = _env_flag("QSF_RUST_KERNELS", default=False)
        enabled_all = bool(enable_all or env_all)
        return cls(
            ranking=enabled_all or enable_ranking or _env_flag("QSF_RUST_KERNEL_RANKING", default=False),
            constructor=enabled_all or enable_constructor or _env_flag("QSF_RUST_KERNEL_CONSTRUCTOR", default=False),
            backtest=enabled_all or enable_backtest or _env_flag("QSF_RUST_KERNEL_BACKTEST", default=False),
            strict=strict or _env_flag("QSF_RUST_KERNEL_STRICT", default=False),
        )

    def as_dict(self) -> dict[str, bool]:
        return {
            "ranking": self.ranking,
            "constructor": self.constructor,
            "backtest": self.backtest,
            "strict": self.strict,
        }


__all__ = ["RustKernelFlags"]
