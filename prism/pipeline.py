"""End-to-end PRISM pipeline (spec entrypoint)."""

from __future__ import annotations

from pathlib import Path

import torch

from prism.assignment.lp_solver import pareto_configs, solve_lp
from prism.assignment.memory import layer_cost_from_profile, target_memory_budget_bits
from prism.inference.runner import PRISMModel
from prism.profiling.meta_learner import load_pretrained_mlp, profile_model as build_profile_dict
from prism.quantization.rtn import precompute_all
from prism.refinement.quic import quic_refine


def run_prism(
    model_name_or_path: str,
    target_avg_bits: float = 3.0,
    group_size: int = 128,
    mlp_path: str = "prism_mlp.pt",
    quic_iters: int = 3,
    quic_samples: int = 4,
    save_precomputed: str | None = None,
    return_pareto: bool = False,
    device: str = "cuda",
) -> dict:
    """
    Full PRISM pipeline: profile → discrete assign → RTN precompute → QUIC → PRISMModel.
    """
    try:
        from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("run_prism requires `transformers` for HuggingFace models.") from exc

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16)
    model.to(device)
    model.eval()

    mlp = load_pretrained_mlp(mlp_path)
    profile = build_profile_dict(model, mlp, group_size=group_size)

    total_params = sum(int(v["num_params"]) for v in profile.values())
    budget_bits = target_memory_budget_bits(profile, target_avg_bits)
    bit_config = solve_lp(profile, budget_bits)

    precomputed = precompute_all(
        model,
        bits_list=[2, 3, 4],
        group_size=group_size,
        save_path=save_precomputed,
    )

    bit_config = quic_refine(
        model,
        precomputed,
        bit_config,
        profile,
        budget_bits=budget_bits,
        max_iters=quic_iters,
        n_samples=quic_samples,
    )

    prism_model = PRISMModel(model, precomputed, bit_config)
    avg_bits = sum(bit_config[k] * int(profile[k]["num_params"]) for k in bit_config) / max(total_params, 1)
    avg_memory_bits = sum(layer_cost_from_profile(profile[k], bit_config[k]) for k in bit_config) / max(total_params, 1)

    out: dict = {
        "model": prism_model,
        "bit_config": bit_config,
        "avg_bits": float(avg_bits),
        "avg_memory_bits_per_param": float(avg_memory_bits),
        "profile": profile,
    }
    if return_pareto:
        out["pareto_configs"] = pareto_configs(profile)
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--bits", type=float, default=3.0)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--mlp_path", type=str, default="prism_mlp.pt")
    parser.add_argument("--quic_iters", type=int, default=3)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--pareto", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    result = run_prism(
        model_name_or_path=args.model,
        target_avg_bits=args.bits,
        group_size=args.group_size,
        mlp_path=args.mlp_path,
        quic_iters=args.quic_iters,
        save_precomputed=args.save,
        return_pareto=args.pareto,
    )

    print(f"Avg bits: {result['avg_bits']:.3f}")
    print(f"Layers: {len(result['bit_config'])}")

    if args.eval:
        from prism.utils import eval_perplexity

        ppl = eval_perplexity(result["model"].inner, tokenizer_id=args.model)
        print(f"Perplexity (wikitext2): {ppl:.2f}")
