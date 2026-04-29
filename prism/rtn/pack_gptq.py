"""Pack RTN-quantised weights into AutoGPTQ's int32 layout.

AutoGPTQ kernel expects:
  - ``qweight``: int32 tensor, ``(in_features // pack_factor, out_features)``
      where pack_factor = 32 // bits  (e.g. 16 for 2-bit, ~10 for 3-bit, 8 for 4-bit).
  - ``scales``:  float32 tensor, ``(num_groups, out_features)``
  - ``qzeros``:  float32 tensor, ``(num_groups, out_features)``
      For PRISM's *symmetric* RTN, qzeros ≡ 0 everywhere (no zero-point).

The key insight is:
  GPTQ dequant formula:  W = scale * (q_unsigned - zero)
  RTN  dequant formula:  W = scale * q_signed

Since RTN is symmetric, q_signed ∈ [-qmax, qmax] and there is no zero-point.
To adapt: we store q_unsigned = q_signed + qmax (offset to unsigned range)
and set qzeros = scale * qmax so that:  scale * (q_unsigned - qmax) = scale * q_signed  ✓

Packing order (column-major, transposed):
  AutoGPTQ stores weights as (height, width) = (in_features // pack_factor, out_features).
  Inside each int32 word, consecutive N-bit values from the *input* dimension are packed.

Bit layouts:
  2-bit: 16 values × 2 bits = 32 bits per int32
  3-bit: 10 values × 3 bits = 30 bits per int32 (2 bits unused)
  4-bit: 8  values × 4 bits = 32 bits per int32
"""

from __future__ import annotations

import torch


def _compute_pack_factor(bits: int) -> int:
    """Number of quantized values packed per int32 word."""
    if bits == 2:
        return 16
    elif bits == 3:
        return 10  # 10 × 3 = 30 bits, 2 wasted
    elif bits == 4:
        return 8
    else:
        raise ValueError(f"Unsupported bit-width for AutoGPTQ packing: {bits}")


def pack_for_autogptq(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    bits: int,
    group_size: int,
    shape: tuple[int, ...] | list[int],
) -> dict[str, torch.Tensor | int | list[int]]:
    """Convert RTN-quantised weights into AutoGPTQ-consumable tensors.

    Parameters
    ----------
    qweight : Tensor (int8/int16)
        RTN-quantized weight of shape ``(out_features, in_features)``
        with values in the signed range ``[-qmax, qmax]``.
    scales : Tensor (float16/float32)
        Per-group scale factors, shape ``(out_features, num_groups)``.
    bits : int
        Bit-width: 2, 3, or 4.
    group_size : int
        Number of weights per scale group (typically 128).
    shape : tuple
        Original ``(out_features, in_features)`` of the FP16 weight.

    Returns
    -------
    dict with keys:
        ``qweight_gptq`` : int32, shape ``(in_features // pack_factor, out_features)``
        ``scales_gptq``  : float32, shape ``(num_groups, out_features)``
        ``qzeros_gptq``  : float32, shape ``(num_groups, out_features)``
        ``shape``        : original weight shape
        ``group_size``   : group size
        ``bits``         : bit-width
        ``pack_factor``  : values per int32
    """
    out_features, in_features = int(shape[0]), int(shape[1])
    qmax = 2 ** (bits - 1) - 1
    pack_factor = _compute_pack_factor(bits)
    bit_mask = (1 << bits) - 1  # e.g. 0x3 for 2-bit, 0x7 for 3-bit, 0xF for 4-bit

    # --- Step 1: Reshape qweight to (out_features, in_features) ---
    qw = qweight.to(torch.int32).reshape(out_features, in_features)

    # --- Step 2: Shift from signed [-qmax, qmax] to unsigned [0, 2*qmax] ---
    # This makes values non-negative for packing
    qw_unsigned = qw + qmax

    # --- Step 3: Transpose to (in_features, out_features) ---
    # AutoGPTQ stores weights transposed: each int32 column packs along in_features
    qw_t = qw_unsigned.t().contiguous()  # (in_features, out_features)

    # --- Step 4: Pack N-bit values into int32 ---
    # After transposing, we group along dim=0 (in_features dimension)
    # Pad in_features if not divisible by pack_factor
    padded_in = ((in_features + pack_factor - 1) // pack_factor) * pack_factor
    if padded_in > in_features:
        pad = torch.zeros(padded_in - in_features, out_features, dtype=torch.int32, device=qw_t.device)
        qw_t = torch.cat([qw_t, pad], dim=0)

    packed_height = padded_in // pack_factor
    packed = torch.zeros(packed_height, out_features, dtype=torch.int32, device=qw_t.device)

    for i in range(pack_factor):
        idx = i  # which sub-value within the int32
        row_offset = torch.arange(packed_height, device=qw_t.device) * pack_factor + i
        # Mask to ensure we don't exceed actual in_features
        valid_mask = row_offset < in_features
        values = qw_t[row_offset.clamp(max=padded_in - 1)]  # (packed_height, out_features)
        # Zero out invalid entries
        values = values * valid_mask.unsqueeze(1).to(torch.int32)
        packed |= (values & bit_mask) << (bits * i)

    # --- Step 5: Prepare scales in GPTQ layout ---
    # RTN scales shape: (out_features, num_groups) → GPTQ: (num_groups, out_features)
    num_groups = (in_features + group_size - 1) // group_size
    scales_gptq = scales.to(torch.float32).reshape(out_features, num_groups).t().contiguous()

    # --- Step 6: Compute qzeros ---
    # For symmetric RTN: dequant = scale * q_signed = scale * (q_unsigned - qmax)
    # AutoGPTQ dequant: W_j = scale_g * (q_unsigned_j - zero_g)
    # So zero = qmax (in unsigned integer space), but AutoGPTQ stores zeros
    # as float: zero_float = scale * qmax
    # Actually, looking at the kernel code:
    #   half2 zero = __float2half2_rn(-(zero_f));
    #   res2 = __hfma2(__hfma2(deq2[val], scale, zero), blockvec[k], res2)
    # This computes: (deq2[val] * scale + (-zero_f)) * blockvec
    #              = (deq2[val] * scale - zero_f) * blockvec
    # So the kernel does: q_unsigned * scale - zero_f
    # For RTN symmetric: W = scale * q_signed = scale * (q_unsigned - qmax)
    #                      = scale * q_unsigned - scale * qmax
    # Therefore: zero_f = scale * qmax
    qzeros_gptq = scales_gptq * qmax  # (num_groups, out_features)

    return {
        "qweight_gptq": packed.contiguous(),
        "scales_gptq": scales_gptq.contiguous(),
        "qzeros_gptq": qzeros_gptq.contiguous(),
        "shape": list(shape),
        "group_size": group_size,
        "bits": bits,
        "pack_factor": pack_factor,
    }


def unpack_autogptq_to_signed(
    packed: torch.Tensor,
    bits: int,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Unpack int32 GPTQ-packed weights back to signed integer tensor.

    This is the inverse of the packing step (useful for testing roundtrips).

    Returns
    -------
    Tensor : int32, shape ``(out_features, in_features)``
        Values in signed range ``[-qmax, qmax]``.
    """
    qmax = 2 ** (bits - 1) - 1
    pack_factor = _compute_pack_factor(bits)
    bit_mask = (1 << bits) - 1
    packed_height = packed.shape[0]

    # Unpack: (packed_height, out_features) → (in_features_padded, out_features)
    unpacked_rows = []
    for i in range(pack_factor):
        vals = (packed >> (bits * i)) & bit_mask  # (packed_height, out_features)
        unpacked_rows.append(vals)

    # Interleave: row 0 from position 0, row 1 from position 1, ...
    unpacked = torch.zeros(packed_height * pack_factor, out_features, dtype=torch.int32, device=packed.device)
    for i in range(pack_factor):
        unpacked[i::pack_factor] = unpacked_rows[i]

    # Trim to actual in_features and transpose back to (out_features, in_features)
    unpacked = unpacked[:in_features].t().contiguous()

    # Shift back to signed range
    return unpacked - qmax
