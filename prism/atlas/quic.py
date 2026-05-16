"""Uncertainty-guided QUIC planning for PRISM-Atlas."""

from __future__ import annotations

from typing import Any


def chosen_action_uncertainties(profile: dict[str, Any], assignment: dict[str, Any]) -> list[tuple[str, float]]:
    chosen = assignment.get("actions", {})
    rows: list[tuple[str, float]] = []
    for layer in profile.get("layers", []):
        layer_name = str(layer["layer_name"])
        action_id = chosen.get(layer_name)
        if action_id is None:
            continue
        response = layer.get("responses", {}).get(action_id)
        if response is None:
            continue
        rows.append((layer_name, float(response.get("uncertainty", 0.0))))
    return rows


def select_uncertain_layers(
    profile: dict[str, Any],
    assignment: dict[str, Any],
    *,
    top_fraction: float = 0.10,
    min_uncertainty: float | None = None,
) -> list[str]:
    rows = chosen_action_uncertainties(profile, assignment)
    if min_uncertainty is not None:
        rows = [(name, value) for name, value in rows if value >= float(min_uncertainty)]
    if not rows:
        return []
    rows = sorted(rows, key=lambda item: item[1], reverse=True)
    count = max(1, int(len(rows) * max(0.0, min(1.0, float(top_fraction)))))
    return [name for name, _ in rows[:count]]
