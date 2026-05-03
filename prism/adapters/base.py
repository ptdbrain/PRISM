"""Adapter abstractions for model-family-specific PRISM behavior."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from torch import nn


@dataclass
class LoadedModelBundle:
    model: nn.Module
    adapter: "ModelAdapter"
    model_id: str
    model_family: str
    tokenizer_id: str | None = None
    hidden_size: int | None = None
    num_layers: int | None = None
    is_demo: bool = False


class ModelAdapter:
    family = "generic"

    def matches(self, model: nn.Module) -> bool:
        return False

    def should_quantize_layer(self, layer_name: str, module: nn.Linear, model: nn.Module) -> bool:
        del module, model
        return layer_name != "lm_head"

    def iter_named_linear_layers(self, model: nn.Module) -> Iterable[tuple[str, nn.Linear]]:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and self.should_quantize_layer(name, module, model):
                yield name, module

    def module_type_from_name(self, layer_name: str) -> str:
        return layer_name.split(".")[-1]

    def infer_hidden_size(self, model: nn.Module) -> int | None:
        config = getattr(model, "config", None)
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is not None:
            return int(hidden_size)
        for _name, module in self.iter_named_linear_layers(model):
            return int(module.in_features)
        return None

    def infer_num_layers(self, model: nn.Module) -> int | None:
        config = getattr(model, "config", None)
        num_layers = getattr(config, "num_hidden_layers", None)
        if num_layers is not None:
            return int(num_layers)
        layers = getattr(model, "layers", None)
        if layers is not None:
            try:
                return len(layers)
            except TypeError:
                return None
        backbone = getattr(model, "model", None)
        if backbone is not None and hasattr(backbone, "layers"):
            try:
                return len(backbone.layers)
            except TypeError:
                return None
        return None

    def default_tokenizer_id(self, model: nn.Module, fallback: str | None = None) -> str | None:
        config = getattr(model, "config", None)
        return (
            getattr(config, "name_or_path", None)
            or getattr(config, "_name_or_path", None)
            or fallback
        )


class DemoAdapter(ModelAdapter):
    family = "demo"

    def matches(self, model: nn.Module) -> bool:
        return model.__class__.__name__ == "MockTransformerLM"

    def should_quantize_layer(self, layer_name: str, module: nn.Linear, model: nn.Module) -> bool:
        del layer_name, module, model
        return True


class GenericCausalLMAdapter(ModelAdapter):
    family = "generic"
