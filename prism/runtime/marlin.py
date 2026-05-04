"""Marlin-accelerated 4-bit linear layer for PRISM runtime.

When the ``gptq_marlin`` (or ``marlin``) package is installed AND the weights
have been pre-packed in Marlin layout, this module calls the fused
dequant-GEMM kernel — achieving 3-4× speedup over FP16 for 4-bit layers.

If the kernel is not available the module transparently falls back to the
PyTorch GEMM path (dequantize → matmul), preserving numerical correctness.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from prism.rtn.quantize import dequantize_rtn
from prism.runtime.backends import marlin_available

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import the Marlin GEMM entry-point at module-load time so that
# ``forward()`` can dispatch without per-call import overhead.
# ---------------------------------------------------------------------------
_marlin_gemm = None
_MARLIN_KERNEL = "none"

try:
    import gptq_marlin  # type: ignore[import-not-found]
    _marlin_gemm = gptq_marlin.gemm
    _MARLIN_KERNEL = "gptq_marlin"
except ImportError:
    try:
        import marlin as _marlin_mod  # type: ignore[import-not-found]
        _marlin_gemm = _marlin_mod.mul
        _MARLIN_KERNEL = "marlin"
    except ImportError:
        pass


class RTNMarlinLinear(nn.Module):
    """4-bit linear layer backed by Marlin fused dequant-GEMM kernel.

    Parameters
    ----------
    qweight : Tensor
        Quantized weight — either Marlin-packed int32 or raw int16 from RTN.
    scales : Tensor
        Per-group FP16 scale factors.
    bits : int
        Bit-width (must be 4 for true Marlin acceleration).
    group_size : int
        Number of weights sharing one scale factor.
    shape : list[int]
        Original ``[out_features, in_features]`` of the weight.
    marlin_qweight : Tensor | None
        Pre-packed int32 in Marlin thread-order layout. If ``None`` the layer
        will attempt naive dequant+matmul fallback.
    marlin_scales : Tensor | None
        Scales rearranged for Marlin. Defaults to ``scales`` if not provided.
    workspace : Tensor | None
        Scratch buffer required by the Marlin kernel.
    """

    def __init__(
        self,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        bits: int,
        group_size: int,
        shape: list[int],
        *,
        marlin_qweight: torch.Tensor | None = None,
        marlin_scales: torch.Tensor | None = None,
        workspace: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        # --- Always keep the raw RTN tensors for fallback dequant path ---
        self.register_buffer("qweight", qweight)
        self.register_buffer("scales", scales)
        self.bits = bits
        self.group_size = group_size
        self.shape = shape

        # --- Marlin-specific buffers ---
        self._has_marlin = False
        if marlin_qweight is not None and _marlin_gemm is not None and bits == 4:
            self.register_buffer("marlin_qweight", marlin_qweight)
            self.register_buffer("marlin_scales", marlin_scales if marlin_scales is not None else scales)
            self.register_buffer("workspace", workspace if workspace is not None else torch.zeros(shape[0], dtype=torch.int32))
            self._has_marlin = True
            logger.debug("RTNMarlinLinear: Marlin kernel active (%s)", _MARLIN_KERNEL)
        else:
            # Register as None so state_dict remains consistent
            self.register_buffer("marlin_qweight", torch.empty(0))
            self.register_buffer("marlin_scales", torch.empty(0))
            self.register_buffer("workspace", torch.empty(0))
            logger.debug("RTNMarlinLinear: falling back to PyTorch GEMM")

    # --------------------------------------------------------------------- #
    #  Forward: Marlin path (fast) → GEMM fallback (always correct)         #
    # --------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._has_marlin:
            return self._forward_marlin(x)
        return self._forward_gemm(x)

    def _forward_marlin(self, x: torch.Tensor) -> torch.Tensor:
        """Fused dequant + GEMM via Marlin kernel — 3-4× faster than FP16."""
        assert _marlin_gemm is not None
        out_features, in_features = self.shape

        # Marlin expects x as 2-D  (batch*seq, hidden)
        orig_shape = x.shape
        x_2d = x.reshape(-1, in_features).to(torch.float16)

        if _MARLIN_KERNEL == "gptq_marlin":
            # gptq_marlin.gemm(a, b_q_weight, b_scales, b_zeros, g_idx,
            #                   perm, workspace, num_bits, ...)
            # For symmetric quant (no zero-point) we pass zeros tensor and
            # identity permutation.
            num_groups = self.marlin_scales.shape[0] if self.marlin_scales.ndim == 2 else 1
            b_zeros = torch.empty(0, dtype=torch.int32, device=x.device)
            g_idx = torch.empty(0, dtype=torch.int32, device=x.device)
            perm = torch.empty(0, dtype=torch.int32, device=x.device)

            output = _marlin_gemm(
                x_2d,
                self.marlin_qweight,
                self.marlin_scales,
                b_zeros,
                g_idx,
                perm,
                self.workspace,
                num_bits=4,
                size_m=x_2d.shape[0],
                size_n=out_features,
                size_k=in_features,
                is_k_full=True,
            )
        else:
            # Standalone marlin.mul(A, B, s, workspace, thread_k, thread_n, sms, max_par)
            output = _marlin_gemm(
                x_2d,
                self.marlin_qweight,
                self.marlin_scales,
                self.workspace,
                -1,   # thread_k: auto
                -1,   # thread_n: auto
                -1,   # sms: auto
                8,    # max_par
            )

        return output.reshape(*orig_shape[:-1], out_features)

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
        mode = "marlin" if self._has_marlin else "gemm_fallback"
        return (
            f"in={self.shape[1]}, out={self.shape[0]}, "
            f"bits={self.bits}, group_size={self.group_size}, "
            f"backend={mode}"
        )
