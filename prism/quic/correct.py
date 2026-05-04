"""Greedy budget-neutral correction for QUIC."""

from __future__ import annotations

from prism.assignment.memory import layer_cost_from_profile, memory_costs_by_bit


def greedy_quic_swap(
    assignment: dict[str, object],
    profile_artifact,
    deltas: dict[str, float],
    epsilon: float = 1e-8,
    surprise_low: float = 0.5,
    surprise_high: float = 1.5,
) -> dict[str, object]:
    bits = dict(assignment["bits"])
    scores = {layer.layer_name: layer.adjusted_score for layer in profile_artifact.layers}
    profile = {
        layer.layer_name: {
            "num_params": layer.num_params,
            "memory_cost_bits": {
                int(k): float(v)
                for k, v in (
                    layer.memory_cost_by_bit
                    or {str(k): float(v) for k, v in memory_costs_by_bit(layer.shape).items()}
                ).items()
            },
            "forced_bits": 4 if layer.fixed_4bit else None,
        }
        for layer in profile_artifact.layers
    }
    sizes = {name: int(layer["num_params"]) for name, layer in profile.items()}
    surprise = {name: deltas[name] / (scores[name] + epsilon) for name in deltas}

    budget = float(assignment.get("budget_memory_bits", float(assignment["budget"]) * sum(sizes.values())))
    current_mem = sum(layer_cost_from_profile(profile[name], int(bits[name])) for name in bits)
    under = sorted((name for name in surprise if surprise[name] > surprise_high), key=lambda name: surprise[name], reverse=True)
    over = sorted((name for name in surprise if surprise[name] < surprise_low), key=lambda name: surprise[name])

    swap_count = 0
    for under_name in under:
        if bits[under_name] >= 4 or profile[under_name].get("forced_bits") is not None:
            continue
        for over_name in over:
            if under_name == over_name or bits[over_name] <= 2 or profile[over_name].get("forced_bits") is not None:
                continue
            memory_delta = (
                layer_cost_from_profile(profile[under_name], int(bits[under_name]) + 1)
                - layer_cost_from_profile(profile[under_name], int(bits[under_name]))
                + layer_cost_from_profile(profile[over_name], int(bits[over_name]) - 1)
                - layer_cost_from_profile(profile[over_name], int(bits[over_name]))
            )
            if current_mem + memory_delta <= budget:
                bits[under_name] += 1
                bits[over_name] -= 1
                current_mem += memory_delta
                swap_count += 1
                break

    current_weight_bits = sum(int(bits[name]) * sizes[name] for name in bits)
    total_params = sum(sizes.values())
    average_bits = current_weight_bits / total_params if sizes else 0.0
    return {
        "bits": bits,
        "surprise": surprise,
        "swap_count": swap_count,
        "average_bits": float(average_bits),
        "average_memory_bits_per_param": float(current_mem / total_params) if total_params else 0.0,
        "memory_cost_bits": float(current_mem),
    }
