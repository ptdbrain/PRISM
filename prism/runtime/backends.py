"""Backend selection for PRISM runtime assembly.

Priority order (highest to lowest):
  1. **Marlin**    — 4-bit only, Tensor-Core fused GEMM (3-4× speedup)
  2. **AutoGPTQ**  — 2/3/4-bit, shared-memory LUT + FMA (2-3× speedup)
  3. **GEMM**      — all bit-widths, PyTorch dequantize + matmul (baseline)
"""

from __future__ import annotations

import functools
import importlib.util
import logging

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def marlin_available() -> bool:
    """Check whether a Marlin-compatible GEMM kernel is importable.

    Supports two common packages:
      - ``gptq_marlin``  (vLLM / AutoGPTQ ecosystem)
      - ``marlin``       (standalone Marlin wheel)
    """
    for pkg in ("gptq_marlin", "marlin"):
        if importlib.util.find_spec(pkg) is not None:
            return True
    return False


@functools.lru_cache(maxsize=1)
def autogptq_available() -> bool:
    """Check whether the AutoGPTQ CUDA kernel was compiled and loaded.

    Uses PRISM's built-in JIT compilation (``prism.kernels.autogptq.build``).
    Returns True only if CUDA is available and the kernel compiles successfully.
    """
    try:
        from prism.kernels.autogptq.build import is_available

        return is_available()
    except ImportError:
        logger.debug("AutoGPTQ kernel build helper is not importable.")
        return False
    except RuntimeError as exc:
        logger.warning("AutoGPTQ CUDA kernel availability check failed: %s", exc)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected AutoGPTQ availability error: %s: %s", type(exc).__name__, exc)
        return False


@functools.lru_cache(maxsize=1)
def rtn_custom_available() -> bool:
    """Check whether the Custom RTN CUDA kernel was compiled and loaded.

    Uses PRISM's built-in JIT compilation (``prism.kernels.rtn.build``).
    Returns True only if CUDA is available and the kernel compiles successfully.
    """
    try:
        from prism.kernels.rtn.build import is_available

        return is_available()
    except ImportError:
        logger.debug("RTN custom kernel build helper is not importable.")
        return False
    except RuntimeError as exc:
        logger.warning("RTN custom CUDA kernel availability check failed: %s", exc)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected RTN custom availability error: %s: %s", type(exc).__name__, exc)
        return False



def choose_backend(bit: int, marlin_compatible: bool = False, autogptq_compatible: bool = False) -> str:
    """Decide which kernel backend a layer should use.

    Decision matrix
    ---------------
    +---------+--------+------------+-----------+----------+
    | bit     | Marlin | RTN Custom | AutoGPTQ  | Fallback |
    +=========+========+============+===========+==========+
    | 4-bit   |   ✓ (1)| ✓ (2)      | ✓ (3)     | GEMM     |
    | 2/3-bit |   ✗    | ✓ (1)      | ✓ (2)     | GEMM     |
    +---------+--------+------------+-----------+----------+

    Parameters
    ----------
    bit : int
        Quantization bit-width (2, 3, or 4).
    marlin_compatible : bool
        Whether Marlin-packed buffers exist for this layer.
    autogptq_compatible : bool
        Whether AutoGPTQ/RTN-packed buffers exist for this layer.
        (RTN Custom uses the exact same GPTQ int32 layout but without qzeros).

    Returns
    -------
    str
        One of ``"marlin"``, ``"rtn_custom"``, ``"autogptq"``, ``"gemm"``.
    """
    # Priority 1 for 4-bit: Marlin (highest throughput)
    if bit == 4 and marlin_compatible and marlin_available():
        return "marlin"

    # Priority 2 for 4-bit, Priority 1 for 2/3-bit: RTN Custom
    # (No qzeros, shifted LUT, slightly faster than AutoGPTQ)
    if bit in (2, 3, 4) and autogptq_compatible and rtn_custom_available():
        return "rtn_custom"

    # Priority 3 for 4-bit, Priority 2 for 2/3-bit: AutoGPTQ
    if bit in (2, 3, 4) and autogptq_compatible and autogptq_available():
        return "autogptq"

    # Fallback: PyTorch GEMM fallback (always works)
    return "gemm"
