"""Theory-guided and learned scoring modes for PRISM-Atlas."""

from __future__ import annotations

import math
from typing import Any

import torch

from prism.atlas.actions import QuantizationAction
from prism.atlas.response import LoadedResponseSurface, estimate_action_response


def layer_importance_proxy(layer_features: dict[str, float]) -> float:
    """Compute a positive kappa proxy from data-free layer statistics."""
    scale_var = float(layer_features.get("scale_std_4bit", 0.0)) / (
        abs(float(layer_features.get("scale_mean_4bit", 0.0))) + 1e-8
    )
    outliers = float(layer_features.get("outlier_ratio_3sigma", 0.0))
    max_to_mean = math.log1p(max(0.0, float(layer_features.get("max_to_mean_abs", 0.0))))
    spectral = float(layer_features.get("spectral_entropy", 0.0))
    layer_prior = 1.0
    module_type = str(layer_features.get("module_type", ""))
    if module_type in {"v_proj", "down_proj"}:
        layer_prior = 1.15
    elif module_type in {"o_proj"}:
        layer_prior = 0.95
    kappa = layer_prior * (1.0 + 0.35 * scale_var + 2.0 * outliers + 0.05 * max_to_mean + 0.02 * spectral)
    return float(max(kappa, 1e-6))


def analytic_action_response(
    weight: torch.Tensor,
    action: QuantizationAction,
    layer_features: dict[str, Any],
) -> dict[str, float | bool | str]:
    """Score an action as kappa_l times quantizer distortion D_l(action)."""
    estimated = estimate_action_response(weight, action)
    kappa = layer_importance_proxy({str(k): float(v) for k, v in layer_features.items() if _is_number(v)} | {
        "module_type": str(layer_features.get("module_type", ""))
    })
    distortion = float(estimated["mean_damage"])
    mean_damage = float(kappa * distortion)
    uncertainty = float(estimated["uncertainty"]) * math.sqrt(kappa)
    log_variance = math.log(max(uncertainty * uncertainty, 1e-20))
    return {
        "scorer": "analytic",
        "mean_damage": mean_damage,
        "analytic_damage": mean_damage,
        "residual_damage": 0.0,
        "distortion": distortion,
        "kappa": kappa,
        "log_variance": log_variance,
        "uncertainty": uncertainty,
        "ranking_score": -mean_damage,
        "transform_supported": bool(estimated["transform_supported"]),
    }


def learned_action_response(
    response_surface: LoadedResponseSurface,
    layer_features: dict[str, float],
    action: QuantizationAction,
    *,
    transform_supported: bool,
) -> dict[str, float | bool | str]:
    predicted = response_surface.predict(layer_features, action, transform_supported=transform_supported)
    return {
        "scorer": "learned",
        "mean_damage": float(predicted["mean_damage"]),
        "analytic_damage": 0.0,
        "residual_damage": float(predicted["mean_damage"]),
        "distortion": 0.0,
        "kappa": 0.0,
        "log_variance": float(predicted["log_variance"]),
        "uncertainty": float(predicted["uncertainty"]),
        "ranking_score": float(predicted["ranking_score"]),
        "transform_supported": bool(transform_supported),
    }


def hybrid_action_response(
    analytic: dict[str, float | bool | str],
    learned: dict[str, float | bool | str] | None,
) -> dict[str, float | bool | str]:
    if learned is None:
        result = dict(analytic)
        result["scorer"] = "hybrid"
        result["residual_damage"] = 0.0
        return result
    residual = float(learned["mean_damage"])
    mean_damage = max(1e-10, float(analytic["analytic_damage"]) + residual)
    uncertainty = math.sqrt(float(analytic["uncertainty"]) ** 2 + float(learned["uncertainty"]) ** 2)
    result = dict(analytic)
    result.update(
        {
            "scorer": "hybrid",
            "mean_damage": mean_damage,
            "residual_damage": residual,
            "log_variance": math.log(max(uncertainty * uncertainty, 1e-20)),
            "uncertainty": uncertainty,
            "ranking_score": -mean_damage,
        }
    )
    return result


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))
