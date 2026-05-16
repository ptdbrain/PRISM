"""Stage 2.5 QUIC correction pipeline."""

from __future__ import annotations

from prism.assign.optimize import profile_dict_from_artifact
from prism.assignment.memory import layer_cost_from_profile
from prism.quantization.rtn import precompute_all
from prism.refinement.quic import compute_surprise, measure_output_perturbation, quic_refine


def run_quic_correction(
    model,
    profile_artifact,
    assignment: dict[str, object],
    hidden_size: int,
    seq_len: int,
    rounds: int = 2,
    group_size: int = 128,
    precomputed: dict | None = None,
) -> dict[str, object]:
    del hidden_size
    profile = profile_dict_from_artifact(profile_artifact)
    if precomputed is None:
        precomputed = precompute_all(model, bits_list=[2, 3, 4], group_size=group_size)
    cfg = {k: int(v) for k, v in assignment["bits"].items()}
    total_params = sum(int(profile[n]["num_params"]) for n in profile)
    budget_bits = float(assignment.get("budget_memory_bits", float(assignment["budget"]) * total_params))
    orig_bits = dict(cfg)
    new_bits = quic_refine(
        model,
        precomputed,
        cfg,
        profile,
        budget_bits=budget_bits,
        max_iters=rounds,
        n_samples=4,
        seq_len=seq_len,
    )
    current_weight_bits = sum(new_bits[n] * int(profile[n]["num_params"]) for n in new_bits)
    current_memory_bits = sum(layer_cost_from_profile(profile[n], new_bits[n]) for n in new_bits)
    average_bits = current_weight_bits / total_params if total_params else 0.0

    deltas = measure_output_perturbation(model, precomputed, new_bits, seq_len=seq_len)
    surprise = compute_surprise(deltas, profile, new_bits)
    swap_count = sum(1 for k in orig_bits if orig_bits[k] != new_bits[k])

    return {
        **assignment,
        "bits": new_bits,
        "average_bits": float(average_bits),
        "average_memory_bits_per_param": float(current_memory_bits / total_params) if total_params else 0.0,
        "memory_cost_bits": float(current_memory_bits),
        "surprise": surprise,
        "deltas": deltas,
        "swap_count": swap_count,
        "original_assignment": assignment,
        "synthetic_input_shape": [],
        "correction_rounds": rounds,
    }
