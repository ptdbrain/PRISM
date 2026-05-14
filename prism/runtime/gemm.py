"""PyTorch GEMM fallback runtime modules for PRISM."""

from __future__ import annotations

import torch
from torch import nn

from prism.rtn.quantize import dequantize_rtn


class RTNGemmLinear(nn.Module):
    def __init__(
        self,
        qweight,
        scales,
        bits: int,
        group_size: int,
        shape: list[int],
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("qweight", qweight)
        self.register_buffer("scales", scales)
        self.register_buffer("bias", None if bias is None else bias.detach().clone())
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
        output = x @ weight.to(device=x.device, dtype=x.dtype).t()
        if self.bias is not None:
            output = output + self.bias.to(device=output.device, dtype=output.dtype)
        return output
