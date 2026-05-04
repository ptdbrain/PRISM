"""PyTorch GEMM fallback runtime modules for PRISM."""

from __future__ import annotations

import torch
from torch import nn

from prism.rtn.quantize import dequantize_rtn


class RTNGemmLinear(nn.Module):
    def __init__(self, qweight, scales, bits: int, group_size: int, shape: list[int]) -> None:
        super().__init__()
        self.register_buffer("qweight", qweight)
        self.register_buffer("scales", scales)
        self.bits = bits
        self.group_size = group_size
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = dequantize_rtn(
            self.qweight,
            self.scales,
            bits=self.bits,
            group_size=self.group_size,
            shape=self.shape,
        )
        return x @ weight.t()
