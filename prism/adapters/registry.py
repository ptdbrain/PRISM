"""Adapter registry and family resolution helpers for PRISM."""

from __future__ import annotations

from torch import nn

from prism.adapters.base import DemoAdapter, GenericCausalLMAdapter, ModelAdapter
from prism.adapters.llama import LlamaAdapter


def resolve_adapter(model: nn.Module | None = None, family: str | None = None) -> ModelAdapter:
    normalized = (family or "").strip().lower()
    if normalized in {"demo", "mock", "synthetic"}:
        return DemoAdapter()
    if normalized == "llama":
        return LlamaAdapter()

    if model is not None:
        if DemoAdapter().matches(model):
            return DemoAdapter()
        llama = LlamaAdapter()
        if llama.matches(model):
            return llama

    return GenericCausalLMAdapter()
