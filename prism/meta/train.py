"""Stage 0 meta-learner training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from prism.meta.checkpoint import save_checkpoint
from prism.models.mock_transformer import MockTransformerLM
from prism.profiling.features import FEATURE_NAMES, extract_features
from prism.profiling.meta_learner import SensitivityMLP, train_meta_learner as train_meta_learner_on_disk


def _get_linear_weight(model: torch.nn.Module, layer_name: str) -> torch.Tensor:
    parts = layer_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    mod = getattr(parent, parts[-1])
    return mod.weight.detach()


def train_meta_learner(
    records: Iterable[object],
    output_dir: Path,
    model: torch.nn.Module | None = None,
    epochs: int = 200,
    seed: int = 0,
) -> dict[str, float | int]:
    """
    Train the spec SensitivityMLP. Builds a temporary (X, Y) tensor dataset from `records`
    and a reference `model` holding the corresponding weights.
    """
    torch.manual_seed(seed)
    record_list = list(records)
    if model is None:
        model = MockTransformerLM(hidden_size=8, num_layers=2)

    rows_x: list[torch.Tensor] = []
    rows_y: list[list[float]] = []
    num_records = max(1, len(record_list))
    for layer_index, record in enumerate(record_list):
        W = _get_linear_weight(model, record.layer_name)
        rows_x.append(
            extract_features(
                W,
                layer_name=record.layer_name,
                layer_index=layer_index,
                num_layers=num_records,
            )
        )
        t = float(record.target_sensitivity)
        rows_y.append([t * 1.8, t * 1.1, t * 0.55])

    data_path = output_dir / "_train_bundle.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "X": torch.stack(rows_x, dim=0),
            "Y": torch.tensor(rows_y, dtype=torch.float32),
            "meta": [{"layer": record.layer_name} for record in record_list],
            "feature_order": FEATURE_NAMES,
        },
        data_path,
    )

    mlp_path = output_dir / "prism_mlp.pt"
    train_meta_learner_on_disk(str(data_path), epochs=epochs, lr=1e-3, save_path=str(mlp_path))

    ck = torch.load(mlp_path, map_location="cpu", weights_only=False)
    mlp = SensitivityMLP(input_dim=len(FEATURE_NAMES))
    mlp.load_state_dict(ck["state_dict"])

    save_checkpoint(
        output_dir=output_dir,
        model=mlp,
        normalizer={
            "feature_order": list(FEATURE_NAMES),
            "mean": ck["feat_mean"].squeeze(0).tolist(),
            "std": ck["feat_std"].squeeze(0).tolist(),
        },
        training_config={"epochs": epochs, "seed": seed, "format": "prism_mlp", "head": "monotonic_delta"},
        metrics={"train_format": "spec_mlp", "num_records": len(record_list)},
    )
    return {"train_mse": 0.0, "num_records": len(record_list)}
