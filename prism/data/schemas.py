"""Dataclass-based artifact schemas used across PRISM stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ProfileLayerRecord:
    layer_name: str
    module_type: str
    shape: list[int]
    num_params: int
    features: dict[str, float]
    raw_score: float
    adjusted_score: float
    fixed_4bit: bool
    fixed_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProfileLayerRecord":
        return cls(**payload)


@dataclass
class ProfileArtifact:
    model_id: str
    model_family: str
    layers: list[ProfileLayerRecord]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_family": self.model_family,
            "layers": [layer.to_dict() for layer in self.layers],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProfileArtifact":
        return cls(
            model_id=payload["model_id"],
            model_family=payload["model_family"],
            layers=[ProfileLayerRecord.from_dict(item) for item in payload["layers"]],
            metadata=payload.get("metadata", {}),
        )
