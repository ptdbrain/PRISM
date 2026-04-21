"""Performance measurement utilities for evaluating PRISM models."""

import gc
import time
from typing import Literal

import torch
from torch import nn


def cleanup_memory() -> None:
    """Clear GPU cache and run garbage collector."""
    torch.cuda.empty_cache()
    gc.collect()


def get_memory_footprint(module: nn.Module, return_buffers: bool = True) -> float:
    """Calculate the memory footprint of a PyTorch module.

    Returns memory in Gigabytes (GB).
    """
    if not isinstance(module, nn.Module):
        raise TypeError("Input must be a PyTorch Module")

    mem_bytes = sum(param.nelement() * param.element_size() for param in module.parameters())
    if return_buffers:
        mem_bytes += sum(buf.nelement() * buf.element_size() for buf in module.buffers())

    return mem_bytes / (1024 ** 3)


@torch.no_grad()
def benchmark_tps(
    model: nn.Module,
    tokenizer,
    batch_size: int = 1,
    prompt_len: int = 64,
    gen_len: int = 128,
    warmup: int = 1,
    iterations: int = 3,
    device: str = "cuda",
) -> dict[str, float]:
    """Measure Tokens Per Second (TPS) during auto-regressive generation.

    Parameters
    ----------
    model : nn.Module
        The language model to benchmark.
    tokenizer : PreTrainedTokenizer
        The tokenizer for generating dummy inputs.
    batch_size : int
        Number of sequences in a batch.
    prompt_len : int
        Length of the dummy context prompt.
    gen_len : int
        Number of tokens to generate.
    warmup : int
        Number of warmup iterations before recording time.
    iterations : int
        Number of benchmarking iterations to average.
    device : str
        Target device for generation.

    Returns
    -------
    dict
        A dictionary containing ``"TPS"``, ``"TotalTime"``, and ``"Tokens"``.
    """
    model.eval()
    
    # Create dummy input
    dummy_input_ids = torch.randint(100, 1000, (batch_size, prompt_len), device=device)
    
    # Warmup
    for _ in range(warmup):
        model.generate(
            dummy_input_ids,
            max_new_tokens=gen_len,
            min_new_tokens=gen_len,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
    
    cleanup_memory()

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for _ in range(iterations):
        model.generate(
            dummy_input_ids,
            max_new_tokens=gen_len,
            min_new_tokens=gen_len,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    total_new_tokens = batch_size * gen_len * iterations
    tps = total_new_tokens / total_time

    return {
        "TPS": round(tps, 2),
        "TotalTime": round(total_time, 4),
        "Tokens": total_new_tokens,
    }
