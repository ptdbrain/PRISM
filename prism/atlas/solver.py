"""Risk-aware multiple-choice assignment for PRISM-Atlas."""

from __future__ import annotations

import math
from typing import Any

from prism.atlas.schema import ASSIGNMENT_ARTIFACT_TYPE, ASSIGNMENT_SCHEMA_VERSION


def _response_objective(response: dict[str, Any], objective_mode: str, risk_lambda: float) -> float:
    if objective_mode == "oracle/debug":
        if "measured_damage" not in response:
            raise ValueError("objective_mode='oracle/debug' requires measured_damage in every response.")
        return float(response["measured_damage"])
    mean = float(response["mean_damage"])
    uncertainty = float(response.get("uncertainty", 0.0))
    if objective_mode == "mean":
        return mean
    if objective_mode == "risk":
        return mean + float(risk_lambda) * uncertainty
    if objective_mode == "cvar-lite":
        # Normal 90th percentile. This is a tail-risk proxy, not a full CVaR estimator.
        return mean + (float(risk_lambda) + 1.2815515655446004) * uncertainty
    raise ValueError("objective_mode must be one of: mean, risk, cvar-lite, oracle/debug")


def _target_budget_from_average_bits(profile: dict[str, Any], target_average_bits: float) -> float:
    target = float(target_average_bits)
    total = 0.0
    for layer in profile.get("layers", []):
        responses = list(layer["responses"].values())
        exact = [
            response
            for response in responses
            if int(response["action"]["bits"]) == int(round(target))
            and response["action"].get("transform") == "none"
            and int(response["action"].get("group_size", 128)) == 128
        ]
        if exact:
            total += float(exact[0]["memory_cost_bits"])
            continue
        total += min(
            responses,
            key=lambda response: (
                abs(float(response["action"]["bits"]) - target),
                float(response["memory_cost_bits"]),
            ),
        )["memory_cost_bits"]
    return float(total)


def _prune_frontier(
    states: dict[float, tuple[float, tuple[tuple[str, str], ...]]],
    *,
    max_frontier: int,
) -> tuple[dict[float, tuple[float, tuple[tuple[str, str], ...]]], bool]:
    items = sorted(states.items(), key=lambda item: (item[0], item[1][0]))
    pruned: dict[float, tuple[float, tuple[tuple[str, str], ...]]] = {}
    best_obj = math.inf
    for cost, (objective, choices) in items:
        if objective < best_obj - 1e-12:
            pruned[cost] = (objective, choices)
            best_obj = objective
    if len(pruned) <= max_frontier:
        return pruned, True
    kept = sorted(pruned.items(), key=lambda item: (item[1][0], item[0]))[:max_frontier]
    return dict(kept), False


def solve_atlas_assignment(
    profile: dict[str, Any],
    *,
    budget_bits: float | None = None,
    target_average_bits: float | None = None,
    objective_mode: str = "risk",
    risk_lambda: float = 0.25,
    require_valid_actions: bool = True,
    latency_budget: float | None = None,
    max_frontier: int = 50000,
) -> dict[str, Any]:
    if budget_bits is None:
        if target_average_bits is None:
            raise ValueError("Provide either budget_bits or target_average_bits.")
        budget_bits = _target_budget_from_average_bits(profile, target_average_bits)
    budget = float(budget_bits)

    states: dict[float, tuple[float, tuple[tuple[str, str], ...]]] = {0.0: (0.0, ())}
    exact = True
    invalid_actions_filtered = 0
    for layer in profile.get("layers", []):
        layer_name = str(layer["layer_name"])
        next_states: dict[float, tuple[float, tuple[tuple[str, str], ...]]] = {}
        for prev_cost, (prev_obj, prev_choices) in states.items():
            for action_id, response in layer["responses"].items():
                if require_valid_actions and not bool(response.get("valid_action", True)):
                    invalid_actions_filtered += 1
                    continue
                cost = prev_cost + float(response["memory_cost_bits"])
                if cost > budget + 1e-9:
                    continue
                if latency_budget is not None:
                    prev_latency = sum(
                        float(layer_by_action(profile, name, action)["latency_proxy"])
                        for name, action in prev_choices
                    )
                    if prev_latency + float(response.get("latency_proxy", 0.0)) > float(latency_budget) + 1e-9:
                        continue
                objective = prev_obj + _response_objective(response, objective_mode, risk_lambda)
                current = next_states.get(cost)
                if current is None or objective < current[0]:
                    next_states[cost] = (objective, prev_choices + ((layer_name, str(action_id)),))
        if not next_states:
            raise ValueError("Infeasible Atlas budget: no action assignment satisfies the memory constraint.")
        states, pruned_exact = _prune_frontier(next_states, max_frontier=max_frontier)
        exact = exact and pruned_exact

    best_cost, (best_objective, choices) = min(states.items(), key=lambda item: (item[1][0], item[0]))
    actions = dict(choices)
    layer_by_name = {str(layer["layer_name"]): layer for layer in profile.get("layers", [])}
    bits: dict[str, int] = {}
    transforms: dict[str, str] = {}
    selected: dict[str, dict[str, Any]] = {}
    total_params = 0
    weighted_bits = 0
    latency_proxy = 0.0
    for layer_name, action_id in actions.items():
        layer = layer_by_name[layer_name]
        response = layer["responses"][action_id]
        action = response["action"]
        bit = int(action["bits"])
        bits[layer_name] = bit
        transforms[layer_name] = str(action.get("transform", "none"))
        selected[layer_name] = response
        total_params += int(layer.get("num_params", 0))
        weighted_bits += bit * int(layer.get("num_params", 0))
        latency_proxy += float(response.get("latency_proxy", 0.0))

    return {
        "schema_version": ASSIGNMENT_SCHEMA_VERSION,
        "artifact_type": ASSIGNMENT_ARTIFACT_TYPE,
        "method": "prism-atlas-v1",
        "objective_mode": objective_mode,
        "risk_lambda": float(risk_lambda),
        "require_valid_actions": bool(require_valid_actions),
        "invalid_actions_filtered": int(invalid_actions_filtered),
        "actions": actions,
        "bits": bits,
        "transforms": transforms,
        "selected_responses": selected,
        "memory_cost_bits": float(best_cost),
        "budget_memory_bits": budget,
        "latency_proxy": float(latency_proxy),
        "latency_budget": float(latency_budget) if latency_budget is not None else None,
        "average_bits": float(weighted_bits / total_params) if total_params else 0.0,
        "solver": {
            "method": "frontier-dp",
            "objective": float(best_objective),
            "exact": bool(exact),
            "memory_cost_bits": float(best_cost),
            "budget_bits": budget,
        },
    }


def layer_by_action(profile: dict[str, Any], layer_name: str, action_id: str) -> dict[str, Any]:
    for layer in profile.get("layers", []):
        if str(layer["layer_name"]) == layer_name:
            return layer["responses"][action_id]
    raise KeyError(layer_name)
