"""Model adapters for PRISM."""

from prism.adapters.base import DemoAdapter, GenericCausalLMAdapter, LoadedModelBundle, ModelAdapter
from prism.adapters.llama import LlamaAdapter
from prism.adapters.registry import resolve_adapter

__all__ = [
    "DemoAdapter",
    "GenericCausalLMAdapter",
    "LoadedModelBundle",
    "LlamaAdapter",
    "ModelAdapter",
    "resolve_adapter",
]
