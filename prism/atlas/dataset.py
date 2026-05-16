"""Stage 0 response-surface dataset construction for PRISM-Atlas."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from prism.atlas.actions import QuantizationAction, action_feature_names, build_action_space
from prism.atlas.response import estimate_action_response
from prism.profile.inspect import iter_named_linear_layers
from prism.profiling.features import FEATURE_NAMES, extract_feature_dict

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is a project dependency
    def tqdm(iterable, *args, **kwargs):
        return iterable


def build_response_dataset(
    model_entries: Iterable[tuple[str, torch.nn.Module]],
    *,
    actions: Iterable[QuantizationAction] | None = None,
    save_path: str | Path | None = None,
) -> dict[str, object]:
    action_list = list(actions or build_action_space())
    rows_layer: list[torch.Tensor] = []
    rows_action: list[torch.Tensor] = []
    y_mean: list[float] = []
    y_log_variance: list[float] = []
    y_ranking: list[float] = []
    meta: list[dict[str, object]] = []

    for model_name, model in tqdm(list(model_entries), desc="Atlas Stage 0 models", unit="model"):
        layer_items = list(iter_named_linear_layers(model))
        for layer_index, (layer_name, module) in enumerate(
            tqdm(layer_items, desc=f"{model_name}: Atlas layers", unit="layer", leave=False)
        ):
            weight = module.weight.detach()
            features = extract_feature_dict(
                weight,
                layer_name=layer_name,
                layer_index=layer_index,
                num_layers=len(layer_items),
            )
            layer_vec = torch.tensor([features[name] for name in FEATURE_NAMES], dtype=torch.float32)
            for action in action_list:
                response = estimate_action_response(weight, action)
                action_features = action.to_feature_dict(
                    transform_supported=bool(response["transform_supported"])
                )
                action_vec = torch.tensor(
                    [action_features[name] for name in action_feature_names()],
                    dtype=torch.float32,
                )
                rows_layer.append(layer_vec)
                rows_action.append(action_vec)
                y_mean.append(float(response["mean_damage"]))
                y_log_variance.append(float(response["log_variance"]))
                y_ranking.append(float(response["ranking_score"]))
                meta.append(
                    {
                        "model_name": model_name,
                        "layer_name": layer_name,
                        "action": action.to_dict(),
                        "transform_supported": bool(response["transform_supported"]),
                    }
                )

    if not rows_layer:
        raise ValueError("No PRISM-Atlas response rows were produced.")

    bundle: dict[str, object] = {
        "X_layer": torch.stack(rows_layer, dim=0),
        "X_action": torch.stack(rows_action, dim=0),
        "Y_mean": torch.tensor(y_mean, dtype=torch.float32),
        "Y_log_variance": torch.tensor(y_log_variance, dtype=torch.float32),
        "Y_ranking": torch.tensor(y_ranking, dtype=torch.float32),
        "meta": meta,
        "layer_feature_order": FEATURE_NAMES,
        "action_feature_order": action_feature_names(),
        "method": "prism-atlas-v1",
    }
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bundle, path)
    return bundle
