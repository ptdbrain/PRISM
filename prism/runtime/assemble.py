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

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is a project dependency
    def tqdm(iterable, *args, **kwargs):
        return iterable


def _set_module_by_name(root, name: str, module) -> None:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    setattr(parent, parts[-1], module)


def _module_device(module) -> torch.device:
    for tensor in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False)):
        return tensor.device
    return torch.device("cpu")


def _validate_runtime_inputs(
    layer_names: list[str],
    manifest: dict[str, object],
    assignment: dict[str, object],
    artifact_root: Path,
) -> dict[str, int]:
    if "bits" not in assignment:
        raise ValueError("Assignment missing required 'bits' mapping.")
    if "layers" not in manifest:
        raise ValueError("Manifest missing required 'layers' mapping.")
    if "group_size" not in manifest:
        raise ValueError("Manifest missing required 'group_size'.")

    chosen_assignment = {str(name): int(bit) for name, bit in assignment["bits"].items()}
    manifest_layers = manifest["layers"]
    missing_assignment = [name for name in layer_names if name not in chosen_assignment]
    if missing_assignment:
        raise ValueError(f"Assignment missing layer bits for: {missing_assignment[:5]}")

    missing_manifest = [name for name in layer_names if name not in manifest_layers]
    if missing_manifest:
        raise ValueError(f"Manifest missing layer entries for: {missing_manifest[:5]}")

    for layer_name in layer_names:
        bit = chosen_assignment[layer_name]
        if bit not in (2, 3, 4):
            raise ValueError(f"Invalid bit-width for {layer_name}: {bit}")
        bit_key = str(bit)
        if bit_key not in manifest_layers[layer_name]:
            raise ValueError(f"Manifest missing layer '{layer_name}' at {bit}-bit.")
        layer_entry = manifest_layers[layer_name][bit_key]
        for key in ("qweight_path", "scale_path"):
            if key not in layer_entry:
                raise ValueError(f"Manifest entry for {layer_name}/{bit}-bit missing '{key}'.")
            if not (artifact_root / layer_entry[key]).exists():
                raise FileNotFoundError(f"Runtime artifact not found: {artifact_root / layer_entry[key]}")
        if layer_entry.get("bias_path") and not (artifact_root / layer_entry["bias_path"]).exists():
            raise FileNotFoundError(f"Runtime artifact not found: {artifact_root / layer_entry['bias_path']}")

    return chosen_assignment


def assemble_runtime_model(
    base_model,
    manifest: dict[str, object],
    assignment: dict[str, object],
    artifact_root: Path,
    *,
    copy_model: bool = True,
):
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
    if copy_model:
        model = deepcopy(base_model)
        if adapter is not None:
            setattr(model, "_prism_adapter", adapter)
    else:
        model = base_model
    backend_by_layer: dict[str, str] = {}

    layer_items = list(iter_named_linear_layers(model))
    layer_names = [layer_name for layer_name, _module in layer_items]
    chosen_assignment = _validate_runtime_inputs(layer_names, manifest, assignment, artifact_root)
    group_size = int(manifest["group_size"])

    for layer_name, current_module in tqdm(layer_items, desc="Stage 4 assemble layers", unit="layer"):
        bit = int(chosen_assignment[layer_name])
        layer_entry = manifest["layers"][layer_name][str(bit)]
        qweight = torch.load(artifact_root / layer_entry["qweight_path"], map_location="cpu")
        scales = torch.load(artifact_root / layer_entry["scale_path"], map_location="cpu")
        bias = None
        if layer_entry.get("bias_path"):
            bias = torch.load(artifact_root / layer_entry["bias_path"], map_location="cpu")
        target_device = _module_device(current_module)

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
                bias=bias,
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
                bias=bias,
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
                bias=bias,
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
                bias=bias,
            )

        wrapper = wrapper.to(target_device)
        _set_module_by_name(model, layer_name, wrapper)

    setattr(model, "backend_summary", backend_by_layer)
    return model, {"backend_by_layer": backend_by_layer}
