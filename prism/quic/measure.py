"""Output-space perturbation measurement for QUIC."""

from __future__ import annotations

import torch

from prism.profile.inspect import iter_named_linear_layers
from prism.rtn.quantize import dequantize_rtn, quantize_rtn


def measure_layer_deltas(
    model,
    assignment: dict[str, object],
    hidden_states: torch.Tensor,
    group_size: int = 128,
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for layer_name, module in iter_named_linear_layers(model):
        bit = int(assignment["bits"][layer_name])
        quantized = quantize_rtn(module.weight.detach(), bits=bit, group_size=group_size)
        dequantized = dequantize_rtn(
            quantized["qweight"],
            quantized["scales"],
            bits=bit,
            group_size=group_size,
            shape=module.weight.shape,
        )
        dense_output = hidden_states @ module.weight.detach().float().t()
        quant_output = hidden_states @ dequantized.t()
        numerator = (dense_output - quant_output).pow(2).sum()
        denominator = dense_output.pow(2).sum().clamp_min(1e-12)
        deltas[layer_name] = float((numerator / denominator).item())
    return deltas
