"""Small deterministic transform helpers for PRISM-Atlas v1."""

from __future__ import annotations

import math

import torch


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def hadamard_supported(weight: torch.Tensor) -> bool:
    if weight.dim() < 2:
        return False
    return is_power_of_two(int(weight.shape[-1]))


def hadamard_matrix(size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not is_power_of_two(size):
        raise ValueError("Hadamard transform requires a power-of-two input dimension.")
    matrix = torch.ones((1, 1), dtype=torch.float32, device=device)
    while matrix.shape[0] < size:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return (matrix / math.sqrt(float(size))).to(dtype=dtype)


def transform_weight(weight: torch.Tensor, transform: str) -> tuple[torch.Tensor, bool]:
    matrix = weight.detach().float()
    if transform == "none":
        return matrix, True
    if transform != "hadamard":
        raise ValueError(f"Unsupported transform: {transform}")
    if not hadamard_supported(matrix):
        return matrix, False
    h = hadamard_matrix(int(matrix.shape[-1]), device=matrix.device, dtype=matrix.dtype)
    return matrix @ h, True
