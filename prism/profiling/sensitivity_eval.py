"""Heavy evaluation helpers for offline sensitivity (transformers + wikitext2)."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from functools import lru_cache

import torch
from torch import nn

from prism.rtn.quantize import dequantize_rtn, quantize_rtn


def eval_wikitext2_perplexity(
    model: torch.nn.Module,
    tokenizer_id: str | None = None,
    n_samples: int = 128,
    seq_len: int = 512,
    input_ids: torch.Tensor | None = None,
) -> float:
    """Token-level perplexity on wikitext-2 test split (HF causal LM)."""
    cfg = getattr(model, "config", None)
    tid = tokenizer_id or getattr(cfg, "name_or_path", None) or getattr(cfg, "_name_or_path", None) or "gpt2"
    prepared_input_ids = input_ids if input_ids is not None else prepare_wikitext2_input_ids(
        tid,
        n_samples=n_samples,
        seq_len=seq_len,
    )
    device = next(model.parameters()).device
    eval_input_ids = prepared_input_ids.to(device)

    model.eval()
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, eval_input_ids.shape[1], seq_len):
            chunk = eval_input_ids[:, start : start + seq_len]
            if chunk.shape[1] < 2:
                break
            out = model(chunk, labels=chunk)
            total_nll += float(out.loss.item()) * chunk.numel()
            total_tokens += chunk.numel()
            if total_tokens >= n_samples * seq_len:
                break
    mean_nll = total_nll / max(total_tokens, 1)
    return float(torch.exp(torch.tensor(mean_nll)).item())


@lru_cache(maxsize=16)
def _load_wikitext2_text() -> str:
    from datasets import load_dataset  # type: ignore[import-not-found]

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join(t["text"] for t in dataset if len(t["text"].strip()) > 0)


@lru_cache(maxsize=16)
def prepare_wikitext2_input_ids(
    tokenizer_id: str,
    n_samples: int = 128,
    seq_len: int = 512,
) -> torch.Tensor:
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(
        _load_wikitext2_text(),
        return_tensors="pt",
        max_length=seq_len * n_samples,
        truncation=True,
    )
    return enc["input_ids"].cpu()


def quantize_single_linear_inplace(model: nn.Module, layer_name: str, bits: int, group_size: int) -> None:
    _, mod = resolve_linear_module(model, layer_name)
    q = quantize_rtn(mod.weight.data, bits=bits, group_size=group_size)
    dq = dequantize_rtn(q["qweight"], q["scales"], bits=bits, group_size=group_size, shape=mod.weight.shape)
    mod.weight.data.copy_(dq.to(dtype=mod.weight.dtype, device=mod.weight.device))


def resolve_linear_module(model: nn.Module, layer_name: str) -> tuple[nn.Module, nn.Linear]:
    parts = layer_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    mod = getattr(parent, parts[-1])
    if not isinstance(mod, nn.Linear):
        raise TypeError(f"{layer_name} is not Linear")
    return parent, mod


@contextmanager
def temporarily_quantized_linear(model: nn.Module, layer_name: str, bits: int, group_size: int):
    _, mod = resolve_linear_module(model, layer_name)
    original_weight = mod.weight.detach().clone()
    try:
        quantize_single_linear_inplace(model, layer_name, bits=bits, group_size=group_size)
        yield model
    finally:
        mod.weight.data.copy_(original_weight)


def clone_model_for_eval(model: torch.nn.Module) -> torch.nn.Module:
    return copy.deepcopy(model)
