"""Stage 4 runtime assembly for PRISM.

Replaces each ``nn.Linear`` in the base model with a quantized wrapper:

* ``RTNMarlinLinear``    — fused dequant+GEMM via Marlin (4-bit only, fastest)
* ``RTNAutoGPTQLinear``  — fused dequant+GEMM via AutoGPTQ (2/3/4-bit, fast)
* ``RTNGemmLinear``      — PyTorch GEMM fallback (all bit-widths, baseline)

Backend selection follows the priority order defined in ``backends.py``.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from prism.profile.inspect import iter_named_linear_layers
from prism.rtn.pack import pack_for_marlin
from prism.runtime.backends import choose_backend
from prism.runtime.gemm import RTNGemmLinear
from prism.runtime.marlin import RTNMarlinLinear
from prism.runtime.autogptq import RTNAutoGPTQLinear
from prism.runtime.rtn_custom import RTNCustomLinear


def _set_module_by_name(root, name: str, module) -> None:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    setattr(parent, parts[-1], module)


def assemble_runtime_model(base_model, manifest: dict[str, object], assignment: dict[str, object], artifact_root: Path):
    """Build a mixed-precision runtime model from precomputed RTN artifacts.

    For 4-bit layers where Marlin is available, weights are repacked into
    Marlin's int32 format and dispatched through the fused kernel.
    For 2/3-bit layers (and 4-bit when Marlin is unavailable), AutoGPTQ
    kernels are used if available.  All other layers fall back to PyTorch GEMM.

    Returns
    -------
    model : nn.Module
        The assembled mixed-precision model.
    summary : dict
        Metadata including per-layer backend selection.
    """
    adapter = getattr(base_model, "_prism_adapter", None)
    model = deepcopy(base_model)
    if adapter is not None:
        setattr(model, "_prism_adapter", adapter)
    backend_by_layer: dict[str, str] = {}

    chosen_assignment = assignment["bits"]
    group_size = int(manifest["group_size"])

    for layer_name, _module in iter_named_linear_layers(model):
        bit = int(chosen_assignment[layer_name])
        layer_entry = manifest["layers"][layer_name][str(bit)]
        qweight = torch.load(artifact_root / layer_entry["qweight_path"], map_location="cpu")
        scales = torch.load(artifact_root / layer_entry["scale_path"], map_location="cpu")

        backend = choose_backend(
            bit=bit,
            marlin_compatible=bool(layer_entry.get("marlin_compatible", False)),
            autogptq_compatible=bool(layer_entry.get("autogptq_compatible", False)),
        )
        backend_by_layer[layer_name] = backend

        shape = list(layer_entry["shape"])

        if backend == "marlin":
            # --- Marlin path: 4-bit fused Tensor-Core GEMM ---
            marlin_pack = pack_for_marlin(
                qweight=qweight,
                scales=scales,
                group_size=group_size,
                shape=shape,
            )
            wrapper = RTNMarlinLinear(
                qweight=qweight,
                scales=scales,
                bits=bit,
                group_size=group_size,
                shape=shape,
                marlin_qweight=marlin_pack["qweight_marlin"],
                marlin_scales=marlin_pack["scales_marlin"],
                workspace=marlin_pack["workspace"],
            )

        elif backend == "rtn_custom":
            # --- RTN Custom path: 2/3/4-bit fused LUT GEMM (No qzeros) ---
            gptq_qweight = torch.load(
                artifact_root / layer_entry["gptq_qweight_path"],
                map_location="cpu",
            )
            gptq_scales = torch.load(
                artifact_root / layer_entry["gptq_scales_path"],
                map_location="cpu",
            )
            wrapper = RTNCustomLinear(
                qweight=qweight,
                scales=scales,
                bits=bit,
                group_size=group_size,
                shape=shape,
                gptq_qweight=gptq_qweight,
                gptq_scales=gptq_scales,
            )

        elif backend == "autogptq":
            # --- AutoGPTQ path: 2/3/4-bit fused LUT+FMA GEMM ---
            gptq_qweight = torch.load(
                artifact_root / layer_entry["gptq_qweight_path"],
                map_location="cpu",
            )
            gptq_scales = torch.load(
                artifact_root / layer_entry["gptq_scales_path"],
                map_location="cpu",
            )
            gptq_qzeros = torch.load(
                artifact_root / layer_entry["gptq_qzeros_path"],
                map_location="cpu",
            )
            wrapper = RTNAutoGPTQLinear(
                qweight=qweight,
                scales=scales,
                bits=bit,
                group_size=group_size,
                shape=shape,
                gptq_qweight=gptq_qweight,
                gptq_scales=gptq_scales,
                gptq_qzeros=gptq_qzeros,
            )

        else:
            # --- GEMM fallback: standard dequant+matmul ---
            wrapper = RTNGemmLinear(
                qweight=qweight,
                scales=scales,
                bits=bit,
                group_size=group_size,
                shape=shape,
            )

        _set_module_by_name(model, layer_name, wrapper)

    setattr(model, "backend_summary", backend_by_layer)
    return model, {"backend_by_layer": backend_by_layer}
