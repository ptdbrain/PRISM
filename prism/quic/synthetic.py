"""Synthetic activation generation for data-free QUIC refinement."""

from __future__ import annotations

import torch


def make_synthetic_hidden_states(hidden_size: int, seq_len: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(seq_len, hidden_size, generator=generator)
