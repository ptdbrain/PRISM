"""Discrete PRISM bit assignment.

The public function is still named ``solve_lp`` for compatibility with older
callers, but the implementation is now a one-hot discrete optimizer:

    x[layer, bit] in {0, 1}, bit in {2, 3, 4}

It uses SciPy MILP when available and falls back to an exact Pareto-frontier
dynamic program for small/medium profiles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from prism.assignment.memory import BIT_OPTIONS, layer_cost_from_profile, target_memory_budget_bits


@dataclass(frozen=True)
class SolverDiagnostics:
    method: str
    objective: float
    memory_cost_bits: float
    budget_bits: float
    exact: bool

    def to_dict(self) -> dict[str, float | str | bool]:
        return {
            "method": self.method,
            "objective": float(self.objective),
            "memory_cost_bits": float(self.memory_cost_bits),
            "budget_bits": float(self.budget_bits),
            "exact": bool(self.exact),
        }


def _layer_sensitivity(layer: dict, bits: int) -> float:
    sens = layer["sensitivity"]
    if bits in sens:
        return float(sens[bits])
    return float(sens[str(bits)])


def _profile_parts(profile: dict) -> tuple[list[str], dict[str, int], list[str]]:
    layer_names = list(profile.keys())
    forced: dict[str, int] = {}
    for name in layer_names:
        fb = profile[name].get("forced_bits")
        if fb is not None:
            bit = int(fb)
            if bit not in BIT_OPTIONS:
                raise ValueError(f"Unsupported forced bit-width for {name}: {bit}")
            forced[name] = bit
    free_names = [name for name in layer_names if name not in forced]
    return layer_names, forced, free_names


def _solve_with_milp(profile: dict, free_names: list[str], budget_remaining: float) -> dict[str, int]:
    from scipy.optimize import Bounds, LinearConstraint, milp

    n_bits = len(BIT_OPTIONS)
    n_vars = len(free_names) * n_bits
    c = np.array(
        [_layer_sensitivity(profile[name], bit) for name in free_names for bit in BIT_OPTIONS],
        dtype=float,
    )

    eq = np.zeros((len(free_names), n_vars), dtype=float)
    for row in range(len(free_names)):
        eq[row, row * n_bits : (row + 1) * n_bits] = 1.0

    cost = np.array(
        [layer_cost_from_profile(profile[name], bit) for name in free_names for bit in BIT_OPTIONS],
        dtype=float,
    )

    constraints = [
        LinearConstraint(eq, np.ones(len(free_names)), np.ones(len(free_names))),
        LinearConstraint(cost.reshape(1, -1), -np.inf, np.array([float(budget_remaining)])),
    ]
    result = milp(
        c=c,
        integrality=np.ones(n_vars),
        bounds=Bounds(np.zeros(n_vars), np.ones(n_vars)),
        constraints=constraints,
        options={"disp": False},
    )
    if not result.success:
        raise ValueError(f"MILP failed: {result.message}")

    x = np.asarray(result.x).reshape(len(free_names), n_bits)
    return {name: BIT_OPTIONS[int(np.argmax(x[row]))] for row, name in enumerate(free_names)}


def _prune_frontier(
    states: dict[float, tuple[float, tuple[int, ...]]],
    *,
    max_frontier: int,
) -> tuple[dict[float, tuple[float, tuple[int, ...]]], bool]:
    """Remove dominated states. If capped, the returned frontier is approximate."""
    items = sorted(states.items(), key=lambda item: (item[0], item[1][0]))
    pruned: dict[float, tuple[float, tuple[int, ...]]] = {}
    best_loss = float("inf")
    for cost, (loss, choices) in items:
        if loss < best_loss - 1e-12:
            pruned[cost] = (loss, choices)
            best_loss = loss
    if len(pruned) <= max_frontier:
        return pruned, True

    ranked = sorted(pruned.items(), key=lambda item: (item[1][0], item[0]))[:max_frontier]
    return dict(ranked), False


def _solve_with_frontier(
    profile: dict,
    free_names: list[str],
    budget_remaining: float,
    *,
    max_frontier: int = 50000,
) -> tuple[dict[str, int], bool]:
    states: dict[float, tuple[float, tuple[int, ...]]] = {0.0: (0.0, ())}
    exact = True
    for name in free_names:
        next_states: dict[float, tuple[float, tuple[int, ...]]] = {}
        for prev_cost, (prev_loss, prev_choices) in states.items():
            for bit in BIT_OPTIONS:
                cost = prev_cost + layer_cost_from_profile(profile[name], bit)
                if cost > budget_remaining + 1e-9:
                    continue
                loss = prev_loss + _layer_sensitivity(profile[name], bit)
                current = next_states.get(cost)
                if current is None or loss < current[0]:
                    next_states[cost] = (loss, prev_choices + (bit,))
        if not next_states:
            raise ValueError("Infeasible budget: no discrete assignment satisfies the memory constraint.")
        states, pruned_exact = _prune_frontier(next_states, max_frontier=max_frontier)
        exact = exact and pruned_exact

    best_cost, (best_loss, choices) = min(states.items(), key=lambda item: (item[1][0], item[0]))
    del best_cost, best_loss
    return dict(zip(free_names, choices, strict=True)), exact


def _assignment_metrics(profile: dict, bits: dict[str, int], budget_bits: float, method: str, exact: bool) -> SolverDiagnostics:
    objective = sum(_layer_sensitivity(profile[name], bit) for name, bit in bits.items())
    memory_cost = sum(layer_cost_from_profile(profile[name], bit) for name, bit in bits.items())
    if memory_cost > budget_bits + 1e-6:
        raise ValueError("Discrete solver produced an assignment over budget.")
    return SolverDiagnostics(
        method=method,
        objective=float(objective),
        memory_cost_bits=float(memory_cost),
        budget_bits=float(budget_bits),
        exact=exact,
    )


def solve_discrete(
    profile: dict,
    budget_bits: float,
    *,
    prefer_milp: bool = True,
) -> tuple[dict[str, int], SolverDiagnostics]:
    """Solve the one-hot multiple-choice assignment under a memory budget."""
    layer_names, forced, free_names = _profile_parts(profile)
    bits: dict[str, int] = {name: forced[name] for name in forced}

    forced_cost = sum(layer_cost_from_profile(profile[name], bit) for name, bit in bits.items())
    min_free_cost = sum(min(layer_cost_from_profile(profile[name], bit) for bit in BIT_OPTIONS) for name in free_names)
    if forced_cost + min_free_cost > float(budget_bits) + 1e-6:
        raise ValueError("Infeasible budget: forced layers plus minimum free-layer cost exceed budget_bits.")

    if not free_names:
        diagnostics = _assignment_metrics(profile, bits, budget_bits, "forced-only", exact=True)
        return bits, diagnostics

    budget_remaining = float(budget_bits) - forced_cost
    method = "milp"
    exact = True
    try:
        if not prefer_milp:
            raise ImportError("MILP disabled")
        bits.update(_solve_with_milp(profile, free_names, budget_remaining))
    except Exception as exc:
        if prefer_milp and not isinstance(exc, (ImportError, ValueError)):
            raise
        method = "frontier-dp"
        solved, exact = _solve_with_frontier(profile, free_names, budget_remaining)
        bits.update(solved)

    ordered_bits = {name: int(bits[name]) for name in layer_names}
    diagnostics = _assignment_metrics(profile, ordered_bits, budget_bits, method, exact=exact)
    return ordered_bits, diagnostics


def solve_lp(profile: dict, budget_bits: float, *, return_diagnostics: bool = False):
    """
    Backward-compatible entrypoint for Stage 2 assignment.

    Despite the historical name, this no longer relaxes to continuous variables
    and rounds. It solves the discrete one-hot problem exactly when SciPy MILP is
    available, or with a Pareto-frontier DP fallback.
    """
    bits, diagnostics = solve_discrete(profile, budget_bits)
    if return_diagnostics:
        return bits, diagnostics.to_dict()
    return bits


def pareto_configs(
    profile: dict,
    avg_bits_range: tuple[float, float] = (2.0, 4.0),
    steps: int = 20,
) -> list[dict[str, int]]:
    configs: list[dict[str, int]] = []
    for avg in torch.linspace(avg_bits_range[0], avg_bits_range[1], steps).tolist():
        budget = target_memory_budget_bits(profile, float(avg))
        configs.append(solve_lp(profile, budget))
    return configs
