"""JIT-compile the AutoGPTQ CUDA kernel and expose Python entry-points.

The kernel provides fused dequantize-then-multiply for 2/3/4-bit weights
packed in the GPTQ int32 layout.  It is compiled once on first import
using ``torch.utils.cpp_extension.load`` and cached under
``~/.cache/prism/autogptq_kernel/``.

The three entry-points exposed after a successful build are:

* ``vecquant2matmul_faster_old(vec, mat, mul, scales, zeros, groupsize, vec_height)``
* ``vecquant3matmul_faster_old(vec, mat, mul, scales, zeros, groupsize, vec_height)``
* ``vecquant4matmul_faster_old(vec, mat, mul, scales, zeros, groupsize, vec_height)``

All tensors must reside on the same CUDA device.
"""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_kernel_module = None  # populated by _load_kernel()


@functools.lru_cache(maxsize=1)
def _load_kernel():
    """JIT-compile the AutoGPTQ CUDA extension.  Returns the module or *None*."""
    global _kernel_module

    try:
        import torch
        from torch.utils.cpp_extension import load as _cpp_load
    except ImportError:
        logger.debug("PyTorch not found — AutoGPTQ kernel unavailable.")
        return None

    if not torch.cuda.is_available():
        logger.debug("CUDA not available — AutoGPTQ kernel skipped.")
        return None

    cu_file = Path(__file__).with_name("auto_gptq_kernel.cu")
    if not cu_file.exists():
        logger.warning("AutoGPTQ kernel source not found at %s", cu_file)
        return None

    cache_dir = Path.home() / ".cache" / "prism" / "autogptq_kernel"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        _kernel_module = _cpp_load(
            name="prism_autogptq",
            sources=[str(cu_file)],
            build_directory=str(cache_dir),
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-std=c++17",
            ],
            verbose=False,
        )
        logger.info("AutoGPTQ kernel compiled successfully.")
        return _kernel_module
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to JIT-compile AutoGPTQ kernel: %s", exc)
        return None


def is_available() -> bool:
    """Return *True* if the AutoGPTQ kernel was compiled and loaded."""
    return _load_kernel() is not None


def get_kernel():
    """Return the compiled kernel module, or *None*."""
    return _load_kernel()
