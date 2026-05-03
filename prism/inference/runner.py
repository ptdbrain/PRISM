"""Dual-path weight materialization and PRISM runtime wrapper (spec).

Three inference paths are supported per layer:

1. **Marlin path** (4-bit only, when ``gptq_marlin`` / ``marlin`` is installed):
   Weights stay as packed int32 and the fused dequant-GEMM kernel runs
   directly on the quantized representation — no intermediate float tensor.

2. **AutoGPTQ path** (2/3/4-bit, when AutoGPTQ CUDA kernel compiles):
   Weights packed in GPTQ int32 format with shared-memory LUT dequantization
   and fused multiply-accumulate on GPU.

3. **GEMM fallback** (all bit-widths, always available):
   RTN weights are dequantized to float16 then multiplied via ``torch.matmul``.
"""

from __future__ import annotations

import importlib.util
import logging
from copy import deepcopy

import torch
import torch.nn as nn

from prism.profile.inspect import iter_named_linear_layers
from prism.quantization.rtn import dequantize_layer
from prism.rtn.pack import pack_for_marlin
from prism.rtn.pack_gptq import pack_for_autogptq
from prism.runtime.marlin import RTNMarlinLinear
from prism.runtime.autogptq import RTNAutoGPTQLinear, autogptq_kernel_available
from prism.runtime.gemm import RTNGemmLinear
from prism.runtime.rtn_custom import RTNCustomLinear

logger = logging.getLogger(__name__)

MARLIN_AVAILABLE = (
    importlib.util.find_spec("gptq_marlin") is not None
    or importlib.util.find_spec("marlin") is not None
)
AUTOGPTQ_AVAILABLE = autogptq_kernel_available()


def _set_module_by_name(root: nn.Module, name: str, module: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    setattr(parent, parts[-1], module)


def get_weight_for_inference(
    precomputed: dict,
    layer_name: str,
    bits: int,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    """Dequantize RTN weights to float for GEMM-fallback layers.

    For Marlin/AutoGPTQ-backed layers the weights are *not* dequantized;
    the respective module handles everything internally via its fused kernel.
    This function is only called for the GEMM path.
    """
    pack = precomputed[layer_name][bits]
    return dequantize_layer(
        pack["W_int"],
        pack["scale"],
        tuple(pack["shape"]),
        int(pack["group_size"]),
    ).float()


def _choose_runner_backend(bits: int) -> str:
    """Decide backend for a layer in the PRISMModel constructor."""
    if bits == 4 and MARLIN_AVAILABLE:
        return "marlin"
    if bits in (2, 3, 4) and AUTOGPTQ_AVAILABLE:
        return "autogptq"
    return "gemm"


class PRISMModel(nn.Module):
    """Wrapper: replace Linear weights using RTN precomputed tensors + bit_config.

    For 4-bit layers when Marlin is available, weights are packed into
    Marlin int32 format. For 2/3-bit layers when AutoGPTQ kernel is compiled,
    weights are packed into GPTQ int32 format. All other layers use standard
    PyTorch matmul on dequantized float weights.
    """
    def __init__(
        self,
        base_model: nn.Module,
        precomputed: dict,
        bit_config: dict[str, int],
    ) -> None:
        super().__init__()
        inner = deepcopy(base_model)
        backend_summary: dict[str, str] = {}

        for layer_name, module in list(iter_named_linear_layers(inner)):
            bits = int(bit_config[layer_name])
            pack = precomputed[layer_name][bits]
            backend = _choose_runner_backend(bits)

            if backend == "marlin":
                # --- Marlin: 4-bit fused Tensor-Core GEMM ---
                qweight_raw = pack["W_int"]
                scales_raw = pack["scale"]
                shape = list(pack["shape"])
                group_size = int(pack["group_size"])

                marlin_pack = pack_for_marlin(
                    qweight=qweight_raw,
                    scales=scales_raw,
                    group_size=group_size,
                    shape=shape,
                )
                new_module = RTNMarlinLinear(
                    qweight=qweight_raw,
                    scales=scales_raw,
                    bits=bits,
                    group_size=group_size,
                    shape=shape,
                    marlin_qweight=marlin_pack["qweight_marlin"],
                    marlin_scales=marlin_pack["scales_marlin"],
                    workspace=marlin_pack["workspace"],
                )
                backend_summary[layer_name] = "marlin"
                logger.info("Layer %s → Marlin (4-bit fused GEMM)", layer_name)

            elif backend == "autogptq":
                # --- AutoGPTQ: 2/3/4-bit fused LUT+FMA GEMM ---
                qweight_raw = pack["W_int"]
                scales_raw = pack["scale"]
                shape = list(pack["shape"])
                group_size = int(pack["group_size"])

                gptq_pack = pack_for_autogptq(
                    qweight=qweight_raw,
                    scales=scales_raw,
                    bits=bits,
                    group_size=group_size,
                    shape=shape,
                )
                new_module = RTNAutoGPTQLinear(
                    qweight=qweight_raw,
                    scales=scales_raw,
                    bits=bits,
                    group_size=group_size,
                    shape=shape,
                    gptq_qweight=gptq_pack["qweight_gptq"],
                    gptq_scales=gptq_pack["scales_gptq"],
                    gptq_qzeros=gptq_pack["qzeros_gptq"],
                )
                backend_summary[layer_name] = "autogptq"
                logger.info("Layer %s → AutoGPTQ (%d-bit fused GEMM)", layer_name, bits)

            else:
                # --- GEMM fallback: dequantize → matmul ---
                w = get_weight_for_inference(
                    precomputed,
                    layer_name,
                    bits,
                    batch_size=1,
                    seq_len=1,
                ).to(device=module.weight.device, dtype=module.weight.dtype)

                bias = module.bias is not None
                new_module = nn.Linear(
                    module.in_features,
                    module.out_features,
                    bias=bias,
                    device=module.weight.device,
                    dtype=w.dtype,
                )
                with torch.no_grad():
                    new_module.weight.copy_(w)
                    if bias and module.bias is not None:
                        new_module.bias.copy_(module.bias)
                backend_summary[layer_name] = "gemm"

            _set_module_by_name(inner, layer_name, new_module)

        self.add_module("inner", inner)
        self._backend_summary = backend_summary

    @property
    def backend_summary(self) -> dict[str, str]:
        """Per-layer backend selection: ``'marlin'``, ``'autogptq'``, or ``'gemm'``."""
        return self._backend_summary

    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs)

    def generate(self, *args, **kwargs):
        gen = getattr(self.inner, "generate", None)
        if gen is None:
            raise AttributeError("Underlying model has no generate()")
        return gen(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)
