"""Runtime/materialization validity metadata for Atlas actions."""

from __future__ import annotations

from typing import Iterable

from prism.atlas.actions import QuantizationAction


def action_runtime_metadata(
    action: QuantizationAction,
    *,
    transform_supported: bool,
    shape: Iterable[int],
) -> dict[str, float | bool | str | None]:
    dims = [int(x) for x in shape]
    out_features = max(1, dims[0] if dims else 1)
    in_features = max(1, dims[1] if len(dims) > 1 else 1)
    materialization_supported = bool(transform_supported)
    runtime_supported = action.backend == "rtn" and action.transform == "none" and materialization_supported
    fallback_backend = None if runtime_supported else "rtn"
    latency_proxy = (out_features * in_features * int(action.bits)) / 4.0
    if int(action.group_size) == 64:
        latency_proxy *= 1.05
    if action.transform != "none":
        latency_proxy *= 1.15
    return {
        "valid_action": bool(runtime_supported and materialization_supported),
        "runtime_supported": bool(runtime_supported),
        "materialization_supported": bool(materialization_supported),
        "fallback_backend": fallback_backend,
        "latency_proxy": float(latency_proxy),
    }
