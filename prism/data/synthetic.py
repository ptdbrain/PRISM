"""Synthetic dataset generation for PRISM training and demos."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from prism.meta.features import compute_zero_cost_features


@dataclass
class TrainingRecord:
    model_id: str
    layer_name: str
    module_type: str
    shape: list[int]
    num_params: int
    features: dict[str, float]
    target_sensitivity: float


def synthetic_sensitivity_target(module_type: str, features: dict[str, float]) -> float:
    base = {
        "v_proj": 8.0,
        "q_proj": 6.0,
        "k_proj": 5.5,
        "gate_proj": 5.0,
        "up_proj": 4.0,
        "down_proj": 3.0,
        "o_proj": 1.5,
    }.get(module_type, 2.0)
    return base + 0.1 * features["kurtosis"] - 0.2 * features["spectral_entropy"]


def make_sensitivity_dataset(model: torch.nn.Module, seed: int = 0) -> list[TrainingRecord]:
    torch.manual_seed(seed)
    records: list[TrainingRecord] = []
    for layer_name, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue
        weight = module.weight.detach().float()
        module_type = layer_name.split(".")[-1]
        features = compute_zero_cost_features(weight)
        records.append(
            TrainingRecord(
                model_id="synthetic-mock-transformer",
                layer_name=layer_name,
                module_type=module_type,
                shape=list(weight.shape),
                num_params=weight.numel(),
                features=features,
                target_sensitivity=synthetic_sensitivity_target(module_type, features),
            )
        )
    return records
