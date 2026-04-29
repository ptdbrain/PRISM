"""Symmetric RTN quantization primitives used by PRISM."""

from __future__ import annotations

import torch


def quantize_rtn(weight: torch.Tensor, bits: int, group_size: int) -> dict[str, torch.Tensor]:
    matrix = weight.detach().float()
    rows = matrix.reshape(matrix.shape[0], -1)
    qmax = max(2 ** (bits - 1) - 1, 1)
    qweight_rows = []
    scale_rows = []

    for row in rows:
        row_groups = []
        row_scales = []
        for start in range(0, row.numel(), group_size):
            group = row[start : start + group_size]
            scale = group.abs().max().clamp_min(1e-6) / qmax
            quantized = torch.round(group / scale).clamp(-qmax, qmax).to(torch.int16)
            row_groups.append(quantized)
            row_scales.append(scale)
        qweight_rows.append(torch.cat(row_groups))
        scale_rows.append(torch.stack(row_scales))

    return {"qweight": torch.stack(qweight_rows), "scales": torch.stack(scale_rows)}


def dequantize_rtn(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    bits: int,
    group_size: int,
    shape: tuple[int, ...] | list[int],
) -> torch.Tensor:
    del bits
    rows = []
    for row, row_scales in zip(qweight, scales):
        chunks = []
        for group_index, scale in enumerate(row_scales):
            start = group_index * group_size
            end = min(start + group_size, row.numel())
            chunks.append(row[start:end].float() * scale)
        rows.append(torch.cat(chunks))
    return torch.stack(rows).reshape(tuple(shape))
