"""Stage 2 discrete mixed-precision assignment for PRISM."""

from __future__ import annotations

from prism.assignment.lp_solver import solve_lp
from prism.assignment.memory import layer_cost_from_profile, memory_costs_by_bit, target_memory_budget_bits
from prism.data.schemas import ProfileArtifact


def profile_dict_from_artifact(artifact: ProfileArtifact) -> dict:
    profile: dict = {}
    for layer in artifact.layers:
        sens = layer.sensitivity_by_bit
        if not sens:
            s = float(layer.adjusted_score)
            sens_map = {2: s * 2.0, 3: s * 1.25, 4: s}
        else:
            sens_map = {int(k): float(v) for k, v in sens.items()}
        profile[layer.layer_name] = {
            "shape": tuple(layer.shape),
            "num_params": int(layer.num_params),
            "sensitivity": sens_map,
            "memory_cost_bits": {
                int(k): float(v)
                for k, v in (
                    layer.memory_cost_by_bit
                    or {str(k): float(v) for k, v in memory_costs_by_bit(layer.shape).items()}
                ).items()
            },
            "is_outlier": bool(layer.fixed_4bit),
            "forced_bits": 4 if layer.fixed_4bit else None,
        }
    return profile


def assign_bits(profile_artifact: ProfileArtifact, target_average_bits: float) -> dict[str, object]:
    profile = profile_dict_from_artifact(profile_artifact)
    total_params = sum(int(v["num_params"]) for v in profile.values())
    budget_memory_bits = target_memory_budget_bits(profile, target_average_bits)
    bits, solver_info = solve_lp(profile, budget_memory_bits, return_diagnostics=True)
    current_weight_cost = sum(bits[n] * int(profile[n]["num_params"]) for n in bits)
    current_memory_cost = sum(layer_cost_from_profile(profile[n], bits[n]) for n in bits)
    average_bits = current_weight_cost / total_params if total_params else 0.0
    average_memory_bits = current_memory_cost / total_params if total_params else 0.0
    return {
        "bits": bits,
        "average_bits": float(average_bits),
        "average_memory_bits_per_param": float(average_memory_bits),
        "budget": target_average_bits,
        "budget_memory_bits": float(budget_memory_bits),
        "memory_cost_bits": float(current_memory_cost),
        "fixed_4bit_layers": sum(1 for n in profile if profile[n].get("forced_bits") == 4),
        "solver": solver_info,
        "summary": {
            "num_layers": len(profile),
            "total_params": total_params,
            "budget_total_bits": float(target_average_bits) * total_params,
            "budget_memory_bits": float(budget_memory_bits),
            "memory_accounting": "packed_weight_bits + group_scale_bits + alignment + metadata",
        },
    }
