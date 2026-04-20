"""Tiny transformer-like model used for PRISM demo and tests."""

from __future__ import annotations

import torch
from torch import nn


class MockSelfAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)


class MockMLP(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, hidden_size, bias=False)


class MockBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.self_attn = MockSelfAttention(hidden_size)
        self.mlp = MockMLP(hidden_size)


class MockTransformerLM(nn.Module):
    def __init__(self, hidden_size: int = 16, num_layers: int = 4) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList([MockBlock(hidden_size) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer.self_attn.o_proj(layer.self_attn.v_proj(x))
            x = x + layer.mlp.down_proj(layer.mlp.up_proj(x))
        return x
