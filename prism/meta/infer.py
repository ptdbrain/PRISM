"""Inference helper for the PRISM meta-learner."""

from __future__ import annotations

import torch

from prism.profiling.meta_learner import SensitivityMLP


def predict_sensitivity(features: dict[str, float], checkpoint: dict[str, object]) -> float:
    """Scalar sensitivity for legacy checkpoints (mean over 2/3/4-bit heads)."""
    feature_order = list(checkpoint["normalizer"]["feature_order"])
    mean = torch.tensor(checkpoint["normalizer"]["mean"], dtype=torch.float32).view(1, -1)
    std = torch.tensor(checkpoint["normalizer"]["std"], dtype=torch.float32).view(1, -1).clamp_min(1e-6)
    inputs = torch.tensor([[features[name] for name in feature_order]], dtype=torch.float32)
    normalized = (inputs - mean) / std

    model = SensitivityMLP(input_dim=len(feature_order))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        prediction = model(normalized)
    return max(float(prediction.mean().item()), 0.0)
