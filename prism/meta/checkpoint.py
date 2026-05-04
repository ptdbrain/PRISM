"""Checkpoint I/O for PRISM meta-learner training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    normalizer: dict[str, Any],
    training_config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    (output_dir / "normalizer.json").write_text(json.dumps(normalizer, indent=2), encoding="utf-8")
    (output_dir / "training_config.json").write_text(json.dumps(training_config, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def load_checkpoint(output_dir: Path) -> dict[str, Any]:
    return {
        "state_dict": torch.load(output_dir / "model.pt", map_location="cpu", weights_only=False),
        "normalizer": json.loads((output_dir / "normalizer.json").read_text(encoding="utf-8")),
        "training_config": json.loads((output_dir / "training_config.json").read_text(encoding="utf-8")),
        "metrics": json.loads((output_dir / "metrics.json").read_text(encoding="utf-8")),
    }
