"""AutoGPTQ-accelerated 2/3/4-bit linear layer for PRISM runtime.

When the AutoGPTQ CUDA kernel is compiled and available, this module calls
the fused dequant-GEMM kernel for 2/3/4-bit layers — achieving 2-3× speedup
over the PyTorch GEMM fallback path (which must dequantize to FP16 first).

If the kernel is not available (no CUDA, compilation failed), the module
transparently falls back to dequantize-then-matmul via RTN primitives.

Key difference from Marlin:
  - Marlin: 4-bit only, Tensor-Core optimized, highest throughput
  - AutoGPTQ: 2/3/4-bit, shared-memory LUT + FMA, broader bit-width support
"""

from __future__ import annotations

import logging
import math

import torch
from torch import nn

from prism.rtn.quantize import dequantize_rtn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to load the AutoGPTQ kernel at module-load time
# ---------------------------------------------------------------------------
_autogptq_kernel = None

try:
    from prism.kernels.autogptq.build import get_kernel

    _autogptq_kernel = get_kernel()
    logger.debug("AutoGPTQ kernel loaded successfully")
except ImportError:
    logger.debug("AutoGPTQ kernel is not installed; RTNAutoGPTQLinear will use GEMM fallback")
except RuntimeError as exc:
    logger.warning("AutoGPTQ CUDA kernel load failed: %s; falling back to GEMM", exc)
except Exception as exc:  # noqa: BLE001
    logger.error("Unexpected AutoGPTQ kernel load error: %s: %s", type(exc).__name__, exc)


def autogptq_kernel_available() -> bool:
    """Return True if the AutoGPTQ CUDA kernel is loaded and ready."""
    return _autogptq_kernel is not None


class RTNAutoGPTQLinear(nn.Module):
    """2/3/4-bit linear layer backed by AutoGPTQ fused dequant-GEMM kernel.

    Parameters
    ----------
    qweight : Tensor
        Raw RTN quantized weight (int8/int16) for GEMM fallback.
    scales : Tensor
        Per-group FP16 scale factors (RTN format).
    bits : int
        Bit-width (2, 3, or 4).
    group_size : int
        Number of weights per scale group.
    shape : list[int]
        Original ``[out_features, in_features]``.
    gptq_qweight : Tensor | None
        Packed int32 in AutoGPTQ layout.
    gptq_scales : Tensor | None
        Float32 scales in GPTQ layout ``(num_groups, out_features)``.
    gptq_qzeros : Tensor | None
        Float32 zeros ``(num_groups, out_features)``.  For symmetric RTN
        this equals ``scale * qmax``.
    """

    def __init__(
        self,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        bits: int,
        group_size: int,
        shape: list[int],
        *,
        gptq_qweight: torch.Tensor | None = None,
        gptq_scales: torch.Tensor | None = None,
        gptq_qzeros: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        # --- Always keep raw RTN tensors for GEMM fallback ---
        self.register_buffer("qweight", qweight)
        self.register_buffer("scales", scales)
        self.bits = bits
        self.group_size = group_size
        self.shape = shape

        # --- AutoGPTQ-specific buffers ---
        self._has_autogptq = False
        if (
            gptq_qweight is not None
            and _autogptq_kernel is not None
            and bits in (2, 3, 4)
        ):
            self.register_buffer("gptq_qweight", gptq_qweight)
            self.register_buffer("gptq_scales", gptq_scales)
            self.register_buffer("gptq_qzeros", gptq_qzeros)
            self._has_autogptq = True
            logger.debug(
                "RTNAutoGPTQLinear: AutoGPTQ kernel active (bits=%d)", bits
            )
        else:
            self.register_buffer("gptq_qweight", torch.empty(0))
            self.register_buffer("gptq_scales", torch.empty(0))
            self.register_buffer("gptq_qzeros", torch.empty(0))
            logger.debug("RTNAutoGPTQLinear: falling back to PyTorch GEMM")

    # ------------------------------------------------------------------ #
    #  Forward dispatch                                                   #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._has_autogptq:
            return self._forward_autogptq(x)
        return self._forward_gemm(x)

    def _forward_autogptq(self, x: torch.Tensor) -> torch.Tensor:
        """Fused dequant + matvec via AutoGPTQ CUDA kernel."""
        assert _autogptq_kernel is not None
        out_features, in_features = self.shape

        orig_shape = x.shape
        # AutoGPTQ kernel expects: vec (batch, in_features) as half
        x_2d = x.reshape(-1, in_features).to(torch.float16)
        batch = x_2d.shape[0]

        # vec_height = in_features // 2 (half2 packing of input vector)
        vec_height = in_features // 2

        # Output buffer
        mul = torch.zeros(batch, out_features, dtype=torch.float32, device=x.device)

        # Dispatch to the correct bit-width kernel
        if self.bits == 2:
            _autogptq_kernel.vecquant2matmul_faster_old(
                x_2d, self.gptq_qweight, mul,
                self.gptq_scales, self.gptq_qzeros,
                self.group_size, vec_height,
            )
        elif self.bits == 3:
            _autogptq_kernel.vecquant3matmul_faster_old(
                x_2d, self.gptq_qweight, mul,
                self.gptq_scales, self.gptq_qzeros,
                self.group_size, vec_height,
            )
        elif self.bits == 4:
            _autogptq_kernel.vecquant4matmul_faster_old(
                x_2d, self.gptq_qweight, mul,
                self.gptq_scales, self.gptq_qzeros,
                self.group_size, vec_height,
            )

        # Convert back to half and reshape
        return mul.to(torch.float16).reshape(*orig_shape[:-1], out_features)

    def _forward_gemm(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback: dequantize RTN weights to float, then standard matmul."""
        weight = dequantize_rtn(
            self.qweight,
            self.scales,
            bits=self.bits,
            group_size=self.group_size,
            shape=self.shape,
        )
        return x @ weight.t()

    def extra_repr(self) -> str:
        mode = f"autogptq_{self.bits}bit" if self._has_autogptq else "gemm_fallback"
        return (
            f"in={self.shape[1]}, out={self.shape[0]}, "
            f"bits={self.bits}, group_size={self.group_size}, "
            f"backend={mode}"
        )
