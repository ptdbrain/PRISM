"""Data-free matrix feature extraction for PRISM."""

from __future__ import annotations

import torch

from prism.profiling.features import extract_feature_dict


def compute_zero_cost_features(
    weight: torch.Tensor,
    *,
    layer_name: str | None = None,
    layer_index: int | None = None,
    num_layers: int | None = None,
    group_size: int = 128,
) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in extract_feature_dict(
            weight,
            layer_name=layer_name,
            layer_index=layer_index,
            num_layers=num_layers,
            group_size=group_size,
        ).items()
    }
