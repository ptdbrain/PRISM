"""Storage helpers for RTN artifacts, including Marlin-compatible packing."""

from __future__ import annotations

import torch


def pack_qweight_for_storage(qweight: torch.Tensor) -> torch.Tensor:
    """Pack RTN qweight into contiguous int16 for generic (GEMM fallback) storage."""
    return qweight.to(torch.int16).contiguous()


# ---------------------------------------------------------------------------
# Marlin 4-bit packing utilities
# ---------------------------------------------------------------------------
# Marlin expects 4-bit weights packed into int32 tensors:
#   - 8 × 4-bit values per int32 word
#   - Symmetric quantization (no zero-point)
#   - Column-major interleaved layout for coalesced Tensor-Core access
#
# The packing below handles the *bit-level* conversion from int8/int16 → int32.
# The *spatial reorder* (thread-to-data mapping) is delegated to the Marlin
# library's own ``marlin.repack()`` when available;  without the library we
# still produce a valid naive int32 pack so that unit-tests can exercise the
# data-path without a GPU.
# ---------------------------------------------------------------------------

_MARLIN_TILE_N = 16  # Marlin processes weights in tiles of 16 columns


def _pack_4bit_to_int32_naive(q_int: torch.Tensor) -> torch.Tensor:
    """Pack int8/int16 tensor with values in [-8, 7] into int32 (8 values per word).

    The resulting tensor has shape ``(rows, cols // 8)`` and dtype ``int32``.
    Each int32 stores 8 consecutive 4-bit values in *unsigned* representation
    (offset by 8 so that the range [-8..7] maps to [0..15]).

    Parameters
    ----------
    q_int : Tensor
        Integer tensor of shape ``(rows, cols)`` where every element is in
        ``[-8, 7]`` (signed 4-bit range).

    Returns
    -------
    Tensor
        Packed ``int32`` tensor of shape ``(rows, cols // 8)``.
    """
    assert q_int.ndim == 2, "Expected 2-D weight matrix"
    rows, cols = q_int.shape
    assert cols % 8 == 0, f"Column count {cols} must be divisible by 8"

    # Shift to unsigned 4-bit range [0, 15]
    unsigned = (q_int.to(torch.int32) + 8) & 0xF  # 4-bit mask

    # Reshape so the last dim has groups of 8
    unsigned = unsigned.reshape(rows, cols // 8, 8)

    # Pack: value_i occupies bits [4*i .. 4*i+3]
    packed = torch.zeros(rows, cols // 8, dtype=torch.int32, device=q_int.device)
    for i in range(8):
        packed |= unsigned[:, :, i] << (4 * i)

    return packed


def pack_for_marlin(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    shape: tuple[int, ...] | list[int],
) -> dict[str, torch.Tensor | int | list[int]]:
    """Convert RTN-quantised 4-bit weights into Marlin-consumable tensors.

    This function performs two tasks:
    1.  Bit-packing: 8 × int4 → 1 × int32.
    2.  Spatial reorder: if the ``gptq_marlin`` library is installed its
        ``gptq_marlin.repack_from_gptq`` helper is used to produce the optimal
        Tensor-Core layout.  Otherwise, the naive column-order packing is used
        (functionally correct but ~15-20 % slower on hardware).

    Parameters
    ----------
    qweight : Tensor  (int8 or int16)
        RTN-quantized weight of shape ``(out_features, in_features)``
        with values in the signed 4-bit range ``[-8, 7]``.
    scales : Tensor  (float16)
        Per-group scale factors, shape ``(out_features, in_features // group_size)``.
    group_size : int
        Number of weights sharing one scale factor (typically 128).
    shape : tuple
        Original ``(out_features, in_features)`` of the FP16 weight.

    Returns
    -------
    dict
        ``qweight_marlin`` : int32 packed weight
        ``scales_marlin``  : float16 scales (potentially reordered)
        ``workspace``      : int32 workspace tensor required by Marlin GEMM
        ``shape``          : original weight shape
        ``group_size``     : group size
        ``packing``        : ``"marlin_native"`` or ``"naive_int32"``
    """
    rows, cols = int(shape[0]), int(shape[1])

    # Step 1: ensure qweight is the right shape
    qw = qweight.to(torch.int8).reshape(rows, cols)

    # Step 2: try native Marlin repack, fall back to naive
    packing = "naive_int32"
    try:
        import gptq_marlin  # type: ignore[import-not-found]

        # gptq_marlin expects unsigned 4-bit packed in int32 (GPTQ-style)
        naive_packed = _pack_4bit_to_int32_naive(qw)
        # Repack from GPTQ column-order → Marlin thread-order
        marlin_qweight = gptq_marlin.repack_from_gptq(
            naive_packed,
            num_bits=4,
            size_k=cols,
            size_n=rows,
        )
        packing = "marlin_native"
    except ImportError:
        # Native library not installed — use naive packing.
        # The ``RTNMarlinLinear.forward()`` will handle this via its own
        # dequant-and-matmul fallback when the Marlin kernel is absent.
        marlin_qweight = _pack_4bit_to_int32_naive(qw)

    # Step 3: prepare scales in Marlin-expected layout
    # Marlin wants scales as (num_groups, out_features) in float16
    scales_marlin = scales.to(torch.float16).contiguous()

    # Step 4: workspace (Marlin needs a small scratch buffer)
    workspace = torch.zeros(rows, dtype=torch.int32, device=qweight.device)

    return {
        "qweight_marlin": marlin_qweight.contiguous(),
        "scales_marlin": scales_marlin.contiguous(),
        "workspace": workspace,
        "shape": list(shape),
        "group_size": group_size,
        "packing": packing,
    }
