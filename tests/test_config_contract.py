"""Task 1 contract tests for strategy config."""

from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_settings() -> dict[str, object]:
    config_path = PROJECT_ROOT / "src" / "config" / "strategy_config.py"
    namespace: dict[str, object] = {}
    source = config_path.read_text(encoding="utf-8")
    exec(compile(source, str(config_path), "exec"), namespace)

    assert "SETTINGS" in namespace, "SETTINGS missing in strategy_config.py"
    settings_obj = namespace["SETTINGS"]
    assert isinstance(settings_obj, dict), "SETTINGS must be a dictionary"
    raw_settings = cast(dict[object, object], settings_obj)

    settings: dict[str, object] = {}
    for key, value in raw_settings.items():
        assert isinstance(key, str), "SETTINGS keys must be strings"
        settings[key] = value
    return settings


def test_settings_has_required_keys() -> None:
    settings = _load_settings()
    required_keys = {
        "base_currency",
        "rebalance_frequency",
        "factor",
        "target_holdings",
        "max_weight",
        "country_sleeves",
    }
    missing = sorted(required_keys.difference(settings.keys()))
    assert not missing, f"SETTINGS is missing required keys: {missing}"


def test_settings_policy_defaults() -> None:
    settings = _load_settings()
    assert settings["base_currency"] == "KRW"
    assert settings["rebalance_frequency"] == "quarterly"
    assert settings["factor"] == "R&D/Sales(TTM)"
    assert settings["target_holdings"] == 20
    assert settings["max_weight"] == 0.08
    assert settings["country_sleeves"] == {"US": 0.33, "KR": 0.33, "JP": 0.33}
