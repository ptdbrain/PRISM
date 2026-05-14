"""Custom RTN-accelerated 2/3/4-bit linear layer for PRISM runtime.

This module utilizes the custom RTN CUDA kernel, which is a specialized
version of AutoGPTQ. It operates on the exact same packed int32 weight format
(GPTQ layout) but does NOT require the `qzeros` tensor, as the symmetric RTN
sign-shifting is handled directly in the GPU Shared Memory LUT.

This achieves higher performance than AutoGPTQ by eliminating memory reads
for `qzeros` and replacing FMA with a simple multiply inside the inner loop.
"""

from __future__ import annotations

import logging
import math

import torch
from torch import nn

from prism.rtn.quantize import dequantize_rtn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to load the RTN kernel at module-load time
# ---------------------------------------------------------------------------
_rtn_kernel = None

try:
    from prism.kernels.rtn.build import get_kernel
    _rtn_kernel = get_kernel()
except Exception:  # noqa: BLE001
    pass


def rtn_kernel_available() -> bool:
    """Return True if the RTN CUDA kernel is loaded and ready."""
    return _rtn_kernel is not None


class RTNCustomLinear(nn.Module):
    """2/3/4-bit linear layer backed by custom RTN fused dequant-GEMM kernel.

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
    """

    def __init__(
        self,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        bits: int,
        group_size: int,
        shape: list[int],
        bias: torch.Tensor | None = None,
        *,
        gptq_qweight: torch.Tensor | None = None,
        gptq_scales: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        # --- Always keep raw RTN tensors for GEMM fallback ---
        self.register_buffer("qweight", qweight)
        self.register_buffer("scales", scales)
        self.register_buffer("bias", None if bias is None else bias.detach().clone())
        self.bits = bits
        self.group_size = group_size
        self.shape = shape

        # --- Custom RTN-specific buffers ---
        self._has_rtn_kernel = False
        if (
            gptq_qweight is not None
            and _rtn_kernel is not None
            and bits in (2, 3, 4)
        ):
            self.register_buffer("gptq_qweight", gptq_qweight)
            self.register_buffer("gptq_scales", gptq_scales)
            self._has_rtn_kernel = True
            logger.debug(
                "RTNCustomLinear: RTN kernel active (bits=%d)", bits
            )
        else:
            self.register_buffer("gptq_qweight", torch.empty(0))
            self.register_buffer("gptq_scales", torch.empty(0))
            logger.debug("RTNCustomLinear: falling back to PyTorch GEMM")

    # ------------------------------------------------------------------ #
    #  Forward dispatch                                                   #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._has_rtn_kernel:
            return self._forward_rtn_kernel(x)
        return self._forward_gemm(x)

    def _forward_rtn_kernel(self, x: torch.Tensor) -> torch.Tensor:
        """Fused dequant + matvec via Custom RTN CUDA kernel."""
        assert _rtn_kernel is not None
        out_features, in_features = self.shape

        orig_shape = x.shape
        x_2d = x.reshape(-1, in_features).to(torch.float16)
        batch = x_2d.shape[0]

        vec_height = in_features // 2

        mul = torch.zeros(batch, out_features, dtype=torch.float32, device=x.device)

        # Dispatch to the correct bit-width kernel (no qzeros passed)
        if self.bits == 2:
            _rtn_kernel.vecquant2matmul_rtn(
                x_2d, self.gptq_qweight, mul,
                self.gptq_scales, self.group_size, vec_height,
            )
        elif self.bits == 3:
            _rtn_kernel.vecquant3matmul_rtn(
                x_2d, self.gptq_qweight, mul,
                self.gptq_scales, self.group_size, vec_height,
            )
        elif self.bits == 4:
            _rtn_kernel.vecquant4matmul_rtn(
                x_2d, self.gptq_qweight, mul,
                self.gptq_scales, self.group_size, vec_height,
            )

        return self._apply_bias(mul.to(torch.float16).reshape(*orig_shape[:-1], out_features))

    def _forward_gemm(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback: dequantize RTN weights to float, then standard matmul."""
        weight = dequantize_rtn(
            self.qweight,
            self.scales,
            bits=self.bits,
            group_size=self.group_size,
            shape=self.shape,
        )
        return self._apply_bias(x @ weight.to(device=x.device, dtype=x.dtype).t())

    def _apply_bias(self, output: torch.Tensor) -> torch.Tensor:
        if self.bias is None:
            return output
        return output + self.bias.to(device=output.device, dtype=output.dtype)

    def extra_repr(self) -> str:
        mode = f"rtn_custom_{self.bits}bit" if self._has_rtn_kernel else "gemm_fallback"
        return (
            f"in={self.shape[1]}, out={self.shape[0]}, "
            f"bits={self.bits}, group_size={self.group_size}, "
            f"backend={mode}"
        )
