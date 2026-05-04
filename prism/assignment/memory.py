"""Memory accounting helpers for PRISM bit assignment."""

from __future__ import annotations

import math
from typing import Iterable

BIT_OPTIONS = (2, 3, 4)


def _shape2d(shape: Iterable[int]) -> tuple[int, int]:
    dims = [int(x) for x in shape]
    if not dims:
        return 1, 1
    if len(dims) == 1:
        return 1, max(1, dims[0])
    out_features = max(1, dims[0])
    in_features = 1
    for dim in dims[1:]:
        in_features *= max(1, int(dim))
    return out_features, in_features


def layer_memory_cost_bits(
    shape: Iterable[int],
    bits: int,
    *,
    group_size: int = 128,
    scale_bits: int = 16,
    zero_point_bits: int = 0,
    metadata_bits_per_group: int = 0,
    metadata_bits_per_layer: int = 64,
    pack_alignment_bits: int = 32,
) -> int:
    """
    Estimate deployed storage cost for one quantized linear weight tensor.

    The estimate includes packed integer weights, groupwise scale overhead,
    optional zero-points/metadata, and row-level packing alignment. Unquantized
    embeddings, lm_head, KV cache, and temporary buffers should be added at the
    artifact/benchmark level because they are not per-quantized-linear costs.
    """
    if bits not in BIT_OPTIONS:
        raise ValueError(f"bits must be one of {BIT_OPTIONS}")
    rows, cols = _shape2d(shape)
    groups_per_row = int(math.ceil(cols / max(1, group_size)))
    num_groups = rows * groups_per_row
    weight_bits = rows * cols * int(bits)

    aligned_weight_bits = weight_bits
    if pack_alignment_bits > 0:
        row_bits = cols * int(bits)
        aligned_row_bits = int(math.ceil(row_bits / pack_alignment_bits) * pack_alignment_bits)
        aligned_weight_bits = rows * aligned_row_bits

    group_overhead = num_groups * (int(scale_bits) + int(zero_point_bits) + int(metadata_bits_per_group))
    return int(aligned_weight_bits + group_overhead + int(metadata_bits_per_layer))


def memory_costs_by_bit(
    shape: Iterable[int],
    *,
    bits_options: Iterable[int] = BIT_OPTIONS,
    group_size: int = 128,
    scale_bits: int = 16,
    zero_point_bits: int = 0,
    metadata_bits_per_group: int = 0,
    metadata_bits_per_layer: int = 64,
    pack_alignment_bits: int = 32,
) -> dict[int, int]:
    return {
        int(bits): layer_memory_cost_bits(
            shape,
            int(bits),
            group_size=group_size,
            scale_bits=scale_bits,
            zero_point_bits=zero_point_bits,
            metadata_bits_per_group=metadata_bits_per_group,
            metadata_bits_per_layer=metadata_bits_per_layer,
            pack_alignment_bits=pack_alignment_bits,
        )
        for bits in bits_options
    }


def layer_cost_from_profile(layer: dict, bits: int) -> float:
    costs = layer.get("memory_cost_bits") or layer.get("memory_cost_by_bit") or {}
    if str(bits) in costs:
        return float(costs[str(bits)])
    if bits in costs:
        return float(costs[bits])
    return float(int(layer["num_params"]) * int(bits))


def target_memory_budget_bits(profile: dict, target_average_bits: float) -> float:
    """
    Convert a target average bit-width to the matching realistic memory budget.

    For integer targets this equals the sum of per-layer memory costs at that
    bit-width. For fractional targets it linearly interpolates between adjacent
    supported bit costs.
    """
    if not profile:
        return 0.0
    target = float(target_average_bits)
    if target <= BIT_OPTIONS[0]:
        lo = hi = BIT_OPTIONS[0]
    elif target >= BIT_OPTIONS[-1]:
        lo = hi = BIT_OPTIONS[-1]
    else:
        lo = max(bit for bit in BIT_OPTIONS if bit <= target)
        hi = min(bit for bit in BIT_OPTIONS if bit >= target)
    if lo == hi:
        return sum(layer_cost_from_profile(layer, lo) for layer in profile.values())
    alpha = (target - lo) / float(hi - lo)
    return sum(
        (1.0 - alpha) * layer_cost_from_profile(layer, lo) + alpha * layer_cost_from_profile(layer, hi)
        for layer in profile.values()
    )
