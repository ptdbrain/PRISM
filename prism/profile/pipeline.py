"""Stage 1 low-cost data-free profiling pipeline."""

from __future__ import annotations

from pathlib import Path

from prism.data.io import save_json
from prism.data.schemas import ProfileArtifact, ProfileLayerRecord
from prism.meta.checkpoint import load_checkpoint
from prism.profiling.meta_learner import load_pretrained_mlp, profile_model as spec_profile_model
from prism.assignment.memory import memory_costs_by_bit
from prism.support.naming import module_type_from_name


def build_profile_artifact(
    model,
    mlp_path: Path,
    output_path: Path | None = None,
    model_id: str = "profiled-model",
    model_family: str = "unknown",
    group_size: int = 128,
) -> ProfileArtifact:
    """Run spec `profile_model` and persist as `ProfileArtifact` for downstream CLIs."""
    mlp = load_pretrained_mlp(str(mlp_path))
    profile = spec_profile_model(model, mlp, group_size=group_size)

    layer_records: list[ProfileLayerRecord] = []
    for layer_name, info in profile.items():
        sens = info["sensitivity"]
        layer_records.append(
            ProfileLayerRecord(
                layer_name=layer_name,
                module_type=module_type_from_name(layer_name),
                shape=list(info["shape"]),
                num_params=int(info["num_params"]),
                features={str(k): float(v) for k, v in info.get("features", {}).items()},
                raw_score=float(sens[4]),
                adjusted_score=float(sens[4]),
                fixed_4bit=bool(info.get("forced_bits") == 4),
                fixed_reason="outlier" if info.get("forced_bits") == 4 else "",
                sensitivity_by_bit={str(k): float(v) for k, v in sens.items()},
                memory_cost_by_bit={str(k): float(v) for k, v in info.get("memory_cost_bits", {}).items()},
            )
        )

    artifact = ProfileArtifact(
        model_id=model_id,
        model_family=model_family,
        layers=layer_records,
        metadata={"mlp_path": str(mlp_path), "group_size": int(group_size)},
    )
    if output_path is not None:
        save_json(output_path, artifact.to_dict())
    return artifact


def profile_model_legacy(
    model,
    checkpoint_dir: Path,
    output_path: Path | None = None,
    group_size: int = 128,
) -> ProfileArtifact:
    """Deprecated path: old single-output MLP checkpoint layout."""
    from prism.meta.infer import predict_sensitivity
    from prism.meta.features import compute_zero_cost_features
    from prism.profile.rules import apply_prism_rules

    checkpoint = load_checkpoint(checkpoint_dir)
    layer_records: list[dict[str, object]] = []

    from prism.profile.inspect import iter_named_linear_layers

    layer_items = list(iter_named_linear_layers(model))
    for layer_index, (layer_name, module) in enumerate(layer_items):
        features = compute_zero_cost_features(
            module.weight.detach(),
            layer_name=layer_name,
            layer_index=layer_index,
            num_layers=len(layer_items),
            group_size=group_size,
        )
        raw_score = predict_sensitivity(features, checkpoint)
        layer_records.append(
            {
                "layer_name": layer_name,
                "module_type": module_type_from_name(layer_name),
                "shape": list(module.weight.shape),
                "num_params": module.weight.numel(),
                "features": features,
                "raw_score": raw_score,
            }
        )

    ruled_records = apply_prism_rules(layer_records)
    artifact = ProfileArtifact(
        model_id="mock-transformer",
        model_family="synthetic",
        layers=[
            ProfileLayerRecord(
                layer_name=str(record["layer_name"]),
                module_type=str(record["module_type"]),
                shape=list(record["shape"]),
                num_params=int(record["num_params"]),
                features=dict(record["features"]),
                raw_score=float(record["raw_score"]),
                adjusted_score=float(record["adjusted_score"]),
                fixed_4bit=bool(record["fixed_4bit"]),
                fixed_reason=str(record["fixed_reason"]),
                sensitivity_by_bit=_mono_sensitivity(float(record["adjusted_score"])),
                memory_cost_by_bit={
                    str(k): float(v)
                    for k, v in memory_costs_by_bit(record["shape"], group_size=group_size).items()
                },
            )
            for record in ruled_records
        ],
        metadata={"checkpoint_dir": str(checkpoint_dir), "group_size": int(group_size)},
    )
    if output_path is not None:
        save_json(output_path, artifact.to_dict())
    return artifact


def _mono_sensitivity(score: float) -> dict[str, float]:
    return {"2": score * 2.0, "3": score * 1.25, "4": score}


def profile_model(
    model,
    checkpoint_dir: Path | None = None,
    output_path: Path | None = None,
    mlp_path: Path | None = None,
    model_id: str = "mock-transformer",
    model_family: str = "synthetic",
    group_size: int = 128,
) -> ProfileArtifact:
    """Dispatch profiling: prefer ``mlp_path`` (spec); else ``checkpoint_dir`` (legacy)."""
    if mlp_path is not None:
        return build_profile_artifact(
            model,
            mlp_path=Path(mlp_path),
            output_path=output_path,
            model_id=model_id,
            model_family=model_family,
            group_size=group_size,
        )
    if checkpoint_dir is not None:
        return profile_model_legacy(model, Path(checkpoint_dir), output_path, group_size=group_size)
    raise ValueError("profile_model requires checkpoint_dir or mlp_path")
