"""Stage 1 Atlas response-surface profiling."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from prism.atlas.actions import QuantizationAction, build_action_space
from prism.atlas.response import load_response_surface
from prism.atlas.runtime import action_runtime_metadata
from prism.atlas.schema import DAMAGE_DEFINITION, PROFILE_ARTIFACT_TYPE, PROFILE_SCHEMA_VERSION
from prism.atlas.scoring import analytic_action_response, hybrid_action_response, learned_action_response
from prism.data.io import save_json
from prism.profile.inspect import iter_named_linear_layers
from prism.profiling.features import extract_feature_dict
from prism.support.naming import module_type_from_name

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is a project dependency
    def tqdm(iterable, *args, **kwargs):
        return iterable


def build_atlas_profile(
    model: torch.nn.Module,
    *,
    actions: Iterable[QuantizationAction] | None = None,
    atlas_path: str | Path | None = None,
    output_path: str | Path | None = None,
    model_id: str = "profiled-model",
    model_family: str = "unknown",
    scorer: str = "analytic",
) -> dict[str, object]:
    if scorer not in {"analytic", "learned", "hybrid"}:
        raise ValueError("scorer must be one of: analytic, learned, hybrid")
    if scorer == "learned" and atlas_path is None:
        raise ValueError("scorer='learned' requires atlas_path.")
    action_list = list(actions or build_action_space())
    response_surface = load_response_surface(str(atlas_path)) if atlas_path is not None else None
    layers: list[dict[str, object]] = []
    flat_response_surface: list[dict[str, object]] = []
    layer_items = list(iter_named_linear_layers(model))
    for layer_index, (layer_name, module) in enumerate(
        tqdm(layer_items, desc="Stage 1 Atlas profile layers", unit="layer")
    ):
        weight = module.weight.detach()
        features = extract_feature_dict(
            weight,
            layer_name=layer_name,
            layer_index=layer_index,
            num_layers=len(layer_items),
        )
        features_for_scoring = dict(features)
        features_for_scoring["module_type"] = module_type_from_name(layer_name)
        responses: dict[str, dict[str, object]] = {}
        for action in action_list:
            analytic = analytic_action_response(weight, action, features_for_scoring)
            learned = None
            if response_surface is not None:
                learned = learned_action_response(
                    response_surface,
                    features,
                    action,
                    transform_supported=bool(analytic["transform_supported"]),
                )
            if scorer == "analytic":
                predicted = analytic
            elif scorer == "learned":
                assert learned is not None
                predicted = learned
            else:
                predicted = hybrid_action_response(analytic, learned)
            runtime = action_runtime_metadata(
                action,
                transform_supported=bool(predicted["transform_supported"]),
                shape=weight.shape,
            )
            response = {
                "action": action.to_dict(),
                "mean_damage": float(predicted["mean_damage"]),
                "analytic_damage": float(predicted.get("analytic_damage", 0.0)),
                "residual_damage": float(predicted.get("residual_damage", 0.0)),
                "distortion": float(predicted.get("distortion", 0.0)),
                "kappa": float(predicted.get("kappa", 0.0)),
                "log_variance": float(predicted["log_variance"]),
                "uncertainty": float(predicted["uncertainty"]),
                "ranking_score": float(predicted["ranking_score"]),
                "scorer": str(predicted["scorer"]),
                "memory_cost_bits": float(action.memory_cost_bits(weight.shape)),
                "transform_supported": bool(predicted["transform_supported"]),
                **runtime,
            }
            responses[action.action_id] = response
            flat_response_surface.append(
                {
                    "layer_name": layer_name,
                    "action_id": action.action_id,
                    **response,
                }
            )
        layers.append(
            {
                "layer_name": layer_name,
                "module_type": module_type_from_name(layer_name),
                "shape": [int(x) for x in weight.shape],
                "num_params": int(weight.numel()),
                "features": {str(k): float(v) for k, v in features.items()},
                "responses": responses,
            }
        )
    profile: dict[str, object] = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "artifact_type": PROFILE_ARTIFACT_TYPE,
        "method": "prism-atlas-v1",
        "model_id": model_id,
        "model_family": model_family,
        "action_space": [action.to_dict() for action in action_list],
        "layers": layers,
        "response_surface": flat_response_surface,
        "damage_definition": DAMAGE_DEFINITION,
        "metadata": {
            "atlas_path": str(atlas_path) if atlas_path is not None else None,
            "response_source": "checkpoint" if response_surface is not None else "analytic_rtn_distortion",
            "scorer": scorer,
        },
    }
    if output_path is not None:
        save_json(Path(output_path), profile)
    return profile
