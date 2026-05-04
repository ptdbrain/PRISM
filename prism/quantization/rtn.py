"""Symmetric RTN precomputation (Marlin-friendly, no zero-point)."""

from __future__ import annotations

from pathlib import Path

import torch

from prism.profile.inspect import iter_named_linear_layers
from prism.rtn.quantize import dequantize_rtn, quantize_rtn


def quantize_symmetric_rtn(weight: torch.Tensor, bits: int, group_size: int) -> dict[str, torch.Tensor]:
    """RTN symmetric quantize; returns int8-packed weights + fp16 scales (spec field names)."""
    q = quantize_rtn(weight, bits=bits, group_size=group_size)
    return {
        "W_int": q["qweight"].to(torch.int8),
        "scale": q["scales"].to(torch.float16),
        "shape": tuple(int(x) for x in weight.shape),
        "group_size": group_size,
        "bits": bits,
    }


def dequantize_layer(
    W_int: torch.Tensor,
    scale: torch.Tensor,
    shape: tuple,
    group_size: int,
) -> torch.Tensor:
    """Restore float16 weights from stored RTN tensors."""
    m = int(W_int.abs().max().item())
    if m <= 1:
        bits = 2
    elif m <= 3:
        bits = 3
    else:
        bits = 4
    return dequantize_rtn(
        W_int.to(torch.float32),
        scale.to(torch.float32),
        bits=bits,
        group_size=group_size,
        shape=shape,
    ).to(torch.float16)


def precompute_all(
    model: torch.nn.Module,
    bits_list: list[int] | None = None,
    group_size: int = 128,
    save_path: str | None = None,
) -> dict:
    """
    For each linear layer, quantize at all bit-widths. Optional torch.save of the full dict.
    """
    if bits_list is None:
        bits_list = [2, 3, 4]
    precomputed: dict[str, dict[int, dict]] = {}
    for layer_name, module in iter_named_linear_layers(model):
        precomputed[layer_name] = {}
        for bits in bits_list:
            precomputed[layer_name][bits] = quantize_symmetric_rtn(module.weight.detach(), bits=bits, group_size=group_size)

    if save_path is not None:
        path = Path(save_path)
        target = path if path.suffix == ".pt" else path / "precomputed.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(precomputed, target)

    return precomputed
