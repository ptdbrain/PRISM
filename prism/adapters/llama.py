"""Llama-family adapter for PRISM."""

from __future__ import annotations

from torch import nn

from prism.adapters.base import ModelAdapter


class LlamaAdapter(ModelAdapter):
    family = "llama"

    def matches(self, model: nn.Module) -> bool:
        config = getattr(model, "config", None)
        model_type = str(getattr(config, "model_type", "")).lower()
        if model_type == "llama":
            return True
        architectures = getattr(config, "architectures", None) or []
        return any("llama" in str(arch).lower() for arch in architectures)

    def should_quantize_layer(self, layer_name: str, module: nn.Linear, model: nn.Module) -> bool:
        del module, model
        return layer_name.startswith("model.layers.") or layer_name.startswith("layers.")
