"""Pareto sweep helpers for PRISM Stage 2."""

from __future__ import annotations

from prism.assign.optimize import assign_bits


def sweep_budgets(profile_artifact, budgets: list[float]) -> list[dict[str, object]]:
    return [assign_bits(profile_artifact, target_average_bits=budget) for budget in budgets]
