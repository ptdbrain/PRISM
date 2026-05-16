"""Action schema for PRISM-Atlas response surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from prism.assignment.memory import layer_memory_cost_bits

SUPPORTED_BITS = (2, 3, 4)
SUPPORTED_GROUP_SIZES = (64, 128)
SUPPORTED_TRANSFORMS = ("none", "hadamard")
SUPPORTED_BACKENDS = ("rtn",)

_ACTION_FEATURE_NAMES = (
    "bits",
    "group_size",
    "bits_normalized",
    "group_size_inverse",
    "transform_none",
    "transform_hadamard",
    "backend_rtn",
    "transform_supported",
)


def action_feature_names() -> tuple[str, ...]:
    return _ACTION_FEATURE_NAMES


@dataclass(frozen=True)
class QuantizationAction:
    """One candidate per-layer quantization action."""

    bits: int
    group_size: int = 128
    transform: str = "none"
    backend: str = "rtn"
    target: str = "weight"

    def __post_init__(self) -> None:
        if int(self.bits) not in SUPPORTED_BITS:
            raise ValueError(f"Unsupported bits={self.bits}; expected one of {SUPPORTED_BITS}.")
        if int(self.group_size) not in SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"Unsupported group_size={self.group_size}; expected one of {SUPPORTED_GROUP_SIZES}."
            )
        if str(self.transform) not in SUPPORTED_TRANSFORMS:
            raise ValueError(
                f"Unsupported transform={self.transform!r}; expected one of {SUPPORTED_TRANSFORMS}."
            )
        if str(self.backend) not in SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend={self.backend!r}; expected one of {SUPPORTED_BACKENDS}.")
        if str(self.target) != "weight":
            raise ValueError("PRISM-Atlas v1 only supports target='weight'.")

    @property
    def action_id(self) -> str:
        return f"{self.backend}_b{int(self.bits)}_g{int(self.group_size)}_{self.transform}"

    def to_dict(self) -> dict[str, int | str]:
        return {
            "bits": int(self.bits),
            "group_size": int(self.group_size),
            "transform": str(self.transform),
            "backend": str(self.backend),
            "target": str(self.target),
            "action_id": self.action_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "QuantizationAction":
        return cls(
            bits=int(payload["bits"]),
            group_size=int(payload.get("group_size", 128)),
            transform=str(payload.get("transform", "none")),
            backend=str(payload.get("backend", "rtn")),
            target=str(payload.get("target", "weight")),
        )

    def memory_cost_bits(self, shape: Iterable[int]) -> int:
        return layer_memory_cost_bits(shape, int(self.bits), group_size=int(self.group_size))

    def to_feature_dict(self, *, transform_supported: bool = True) -> dict[str, float]:
        bits = float(self.bits)
        group_size = float(self.group_size)
        return {
            "bits": bits,
            "group_size": group_size,
            "bits_normalized": (bits - min(SUPPORTED_BITS)) / float(max(SUPPORTED_BITS) - min(SUPPORTED_BITS)),
            "group_size_inverse": 1.0 / group_size,
            "transform_none": 1.0 if self.transform == "none" else 0.0,
            "transform_hadamard": 1.0 if self.transform == "hadamard" else 0.0,
            "backend_rtn": 1.0 if self.backend == "rtn" else 0.0,
            "transform_supported": 1.0 if transform_supported else 0.0,
        }


def build_action_space(
    *,
    bits: Iterable[int] = SUPPORTED_BITS,
    group_sizes: Iterable[int] = SUPPORTED_GROUP_SIZES,
    transforms: Iterable[str] = SUPPORTED_TRANSFORMS,
    backend: str = "rtn",
) -> list[QuantizationAction]:
    return [
        QuantizationAction(bits=int(bit), group_size=int(group_size), transform=str(transform), backend=backend)
        for bit in bits
        for group_size in group_sizes
        for transform in transforms
    ]


def action_from_id(action_id: str) -> QuantizationAction:
    parts = action_id.split("_")
    if len(parts) < 4:
        raise ValueError(f"Invalid PRISM-Atlas action id: {action_id!r}")
    backend = parts[0]
    try:
        bits = int(parts[1].removeprefix("b"))
        group_size = int(parts[2].removeprefix("g"))
    except ValueError as exc:
        raise ValueError(f"Invalid PRISM-Atlas action id: {action_id!r}") from exc
    transform = "_".join(parts[3:])
    return QuantizationAction(bits=bits, group_size=group_size, transform=transform, backend=backend)
