"""QUIC — synthetic forward refinement of bit assignment (spec)."""

from __future__ import annotations

import logging

import torch

from prism.assignment.memory import layer_cost_from_profile
from prism.profile.inspect import iter_named_linear_layers
from prism.quantization.rtn import dequantize_layer

logger = logging.getLogger(__name__)


def _hidden_dim(model: torch.nn.Module) -> int:
    for _, module in iter_named_linear_layers(model):
        return int(module.in_features)
    raise ValueError("No linear layers found for hidden size inference.")


def measure_output_perturbation(
    model: torch.nn.Module,
    precomputed: dict,
    config: dict[str, int],
    n_samples: int = 4,
    seq_len: int = 32,
) -> dict[str, float]:
    hidden = _hidden_dim(model)
    x = torch.randn(n_samples, seq_len, hidden, dtype=torch.float32)
    xf = x.reshape(-1, hidden)
    deltas: dict[str, float] = {}
    for layer_name, module in iter_named_linear_layers(model):
        bits = int(config[layer_name])
        pack = precomputed[layer_name][bits]
        W = module.weight.detach().float()
        Wh = dequantize_layer(
            pack["W_int"],
            pack["scale"],
            tuple(pack["shape"]),
            int(pack["group_size"]),
        ).float()
        y_full = xf @ W.t()
        y_quant = xf @ Wh.t()
        num = (y_full - y_quant).pow(2).sum()
        den = y_full.pow(2).sum().clamp_min(1e-12)
        deltas[layer_name] = float((num / den).item())
    return deltas


def compute_surprise(deltas: dict[str, float], profile: dict, config: dict[str, int]) -> dict[str, float]:
    surprise: dict[str, float] = {}
    for name, delta in deltas.items():
        b = int(config[name])
        s = float(profile[name]["sensitivity"][b])
        surprise[name] = float(delta / (s + 1e-8))
    return surprise


def greedy_swap(
    config: dict[str, int],
    surprise: dict[str, float],
    profile: dict,
    budget_bits: float,
    surprise_up_thresh: float = 1.5,
    surprise_down_thresh: float = 0.5,
) -> dict[str, int]:
    current_mem = sum(layer_cost_from_profile(profile[n], config[n]) for n in config)
    bits = dict(config)

    unders = [
        n
        for n in surprise
        if surprise[n] > surprise_up_thresh and bits[n] < 4 and profile[n].get("forced_bits") is None
    ]
    overs = [
        n
        for n in surprise
        if surprise[n] < surprise_down_thresh and bits[n] > 2 and profile[n].get("forced_bits") is None
    ]

    best_pair: tuple[str, str] | None = None
    best_gap = -1.0
    for u in unders:
        for o in overs:
            if u == o:
                continue
            mem_delta = (
                layer_cost_from_profile(profile[u], bits[u] + 1)
                - layer_cost_from_profile(profile[u], bits[u])
                + layer_cost_from_profile(profile[o], bits[o] - 1)
                - layer_cost_from_profile(profile[o], bits[o])
            )
            if current_mem + mem_delta > budget_bits:
                continue
            gap = surprise[u] - surprise[o]
            if gap > best_gap:
                best_gap = gap
                best_pair = (u, o)

    if best_pair is None:
        return bits
    u, o = best_pair
    bits[u] += 1
    bits[o] -= 1
    return bits


def quic_refine(
    model: torch.nn.Module,
    precomputed: dict,
    config: dict[str, int],
    profile: dict,
    budget_bits: float,
    max_iters: int = 3,
    n_samples: int = 4,
) -> dict[str, int]:
    missing_profile = set(config) - set(profile)
    if missing_profile:
        raise ValueError(f"Profile missing layers required by assignment: {sorted(missing_profile)}")

    missing_precomputed = set(config) - set(precomputed)
    if missing_precomputed:
        raise ValueError(
            "Precomputed artifacts missing for layers: "
            f"{sorted(missing_precomputed)}. Rerun RTN precomputation."
        )

    invalid_bits = sorted({int(bit) for bit in config.values() if int(bit) not in (2, 3, 4)})
    if invalid_bits:
        raise ValueError(f"Invalid bit-widths in config: {invalid_bits}. Must be 2, 3, or 4.")

    for layer_name, bit in config.items():
        if int(bit) not in precomputed[layer_name]:
            raise ValueError(
                f"Layer '{layer_name}' not precomputed at {int(bit)}-bit. "
                f"Available: {sorted(precomputed[layer_name])}"
            )

    cfg = dict(config)
    current_memory = sum(layer_cost_from_profile(profile[n], cfg[n]) for n in cfg)
    if current_memory > budget_bits + 1e-3:
        logger.warning(
            "Initial QUIC assignment exceeds budget: %.2f > %.2f bits",
            current_memory,
            budget_bits,
        )
    elif abs(budget_bits - current_memory) > 1e-3:
        logger.debug("Initial QUIC assignment under budget: %.2f < %.2f bits", current_memory, budget_bits)

    for iteration in range(max_iters):
        deltas = measure_output_perturbation(model, precomputed, cfg, n_samples=n_samples)
        surprise = compute_surprise(deltas, profile, cfg)
        new_cfg = greedy_swap(cfg, surprise, profile, budget_bits)
        if new_cfg == cfg:
            logger.debug("QUIC converged at iteration %d", iteration)
            break
        cfg = new_cfg

        current_memory = sum(layer_cost_from_profile(profile[n], cfg[n]) for n in cfg)
        if current_memory > budget_bits + 1e-3:
            logger.warning(
                "QUIC iteration %d exceeded budget: %.2f > %.2f bits",
                iteration,
                current_memory,
                budget_bits,
            )

    final_memory = sum(layer_cost_from_profile(profile[n], cfg[n]) for n in cfg)
    if final_memory > budget_bits + 1e-3:
        raise ValueError(f"QUIC final assignment exceeds budget: {final_memory:.2f} > {budget_bits:.2f} bits")
    return cfg
