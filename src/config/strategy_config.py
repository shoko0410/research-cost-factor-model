"""Strategy policy constants for Task 1 scaffold."""

SETTINGS: dict[str, object] = {
    "base_currency": "KRW",
    "rebalance_frequency": "quarterly",
    "factor": "R&D/Sales(TTM)",
    "target_holdings": 20,
    "max_weight": 0.08,
    "country_sleeves": {
        "US": 0.33,
        "KR": 0.33,
        "JP": 0.33,
    },
}
