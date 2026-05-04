"""Hard-rule adjustments for PRISM zero-cost profiling (legacy single-score path)."""

from __future__ import annotations

from statistics import median


def apply_prism_rules(layer_records: list[dict[str, object]]) -> list[dict[str, object]]:
    """
    Apply o_proj / v_proj multipliers before median outlier detection (spec order).
    Outlier: adjusted sensitivity[4] > 2 * median -> forced 4-bit.
    """
    for record in layer_records:
        module_type = str(record["module_type"])
        adjusted = float(record["raw_score"])
        if module_type == "o_proj":
            adjusted *= 0.5
        if module_type == "v_proj":
            adjusted *= 1.5
        record["adjusted_score"] = adjusted
        record["fixed_4bit"] = False
        record["fixed_reason"] = ""

    adjusted_scores = [float(r["adjusted_score"]) for r in layer_records]
    cutoff = 2.0 * median(adjusted_scores) if adjusted_scores else 0.0

    for record in layer_records:
        adjusted = float(record["adjusted_score"])
        if adjusted > cutoff and cutoff > 0:
            record["fixed_4bit"] = True
            record["fixed_reason"] = "median_rule"

    return layer_records
