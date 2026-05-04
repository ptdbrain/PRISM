"""Layer inspection helpers for PRISM profiling."""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn


def iter_named_linear_layers(model: nn.Module, adapter=None) -> Iterable[tuple[str, nn.Linear]]:
    adapter = adapter or getattr(model, "_prism_adapter", None)
    if adapter is not None:
        yield from adapter.iter_named_linear_layers(model)
        return
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield name, module
