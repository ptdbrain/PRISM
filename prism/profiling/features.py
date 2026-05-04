"""Data-free layer features for PRISM sensitivity prediction."""

from __future__ import annotations

import math
import re

import torch

EPS = 1e-8

LEGACY_FEATURE_NAMES = (
    "kurtosis",
    "spectral_entropy",
    "rank_ratio",
    "nuclear_norm_normalized",
)

LAYER_TYPE_BUCKETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "other",
)

FEATURE_NAMES = (
    "log_kurtosis",
    "spectral_entropy",
    "rank_ratio",
    "log_nuclear_norm_normalized",
    "layer_depth_normalized",
    "layer_type_q_proj",
    "layer_type_k_proj",
    "layer_type_v_proj",
    "layer_type_o_proj",
    "layer_type_gate_proj",
    "layer_type_up_proj",
    "layer_type_down_proj",
    "layer_type_other",
    "shape_ratio",
    "log_num_params",
    "max_to_mean_abs",
    "outlier_ratio_3sigma",
    "scale_mean_4bit",
    "scale_std_4bit",
    "scale_max_4bit",
    "singular_value_decay_top4",
    "rtn_rel_mse_2bit",
    "rtn_rel_mse_3bit",
    "rtn_rel_mse_4bit",
)


def _as_matrix(W: torch.Tensor) -> torch.Tensor:
    W = W.detach().float()
    if W.dim() == 0:
        return W.reshape(1, 1)
    if W.dim() == 1:
        return W.reshape(1, -1)
    if W.dim() > 2:
        return W.reshape(W.shape[0], -1)
    return W


def _safe_svdvals(W: torch.Tensor) -> torch.Tensor:
    try:
        return torch.linalg.svdvals(W)
    except RuntimeError:
        return torch.linalg.svdvals(W.cpu())


def _infer_layer_type(layer_name: str | None) -> str:
    if not layer_name:
        return "other"
    leaf = layer_name.split(".")[-1]
    return leaf if leaf in LAYER_TYPE_BUCKETS[:-1] else "other"


def _infer_depth(layer_name: str | None, layer_index: int | None, num_layers: int | None) -> float:
    if layer_index is None and layer_name:
        matches = [int(m) for m in re.findall(r"\.(\d+)\.", f".{layer_name}.")]
        if matches:
            layer_index = matches[0]
    if layer_index is None:
        return 0.0
    if num_layers is None or num_layers <= 1:
        return 0.0
    return float(max(0.0, min(1.0, layer_index / float(num_layers - 1))))


def _groupwise_view(W: torch.Tensor, group_size: int) -> torch.Tensor:
    if group_size <= 0:
        group_size = W.shape[1]
    rows, cols = W.shape
    padded_cols = int(math.ceil(cols / group_size) * group_size)
    if padded_cols != cols:
        pad = torch.zeros(rows, padded_cols - cols, dtype=W.dtype, device=W.device)
        W = torch.cat([W, pad], dim=1)
    return W.reshape(rows, padded_cols // group_size, group_size)


def _scale_stats(W: torch.Tensor, bits: int, group_size: int) -> tuple[float, float, float]:
    qmax = float(2 ** (bits - 1) - 1)
    groups = _groupwise_view(W, group_size)
    scales = groups.abs().amax(dim=-1) / max(qmax, 1.0)
    scales = scales.flatten()
    if scales.numel() == 0:
        return 0.0, 0.0, 0.0
    return (
        float(scales.mean().item()),
        float(scales.std(unbiased=False).item()),
        float(scales.max().item()),
    )


def rtn_relative_mse(W: torch.Tensor, bits: int, group_size: int = 128) -> float:
    """Relative MSE of groupwise symmetric RTN dequantization for a weight matrix."""
    if bits not in (2, 3, 4):
        raise ValueError("bits must be 2, 3, or 4")
    W = _as_matrix(W)
    cols = W.shape[1]
    qmax = float(2 ** (bits - 1) - 1)
    groups = _groupwise_view(W, group_size)
    scales = (groups.abs().amax(dim=-1, keepdim=True) / max(qmax, 1.0)).clamp_min(EPS)
    q = torch.round(groups / scales).clamp(-qmax, qmax)
    deq = (q * scales).reshape(W.shape[0], -1)[:, :cols]
    numerator = (W - deq).pow(2).mean()
    denominator = W.pow(2).mean().clamp_min(EPS)
    return float((numerator / denominator).item())


def kurtosis(W: torch.Tensor) -> float:
    """Excess-free kurtosis E[(W - mu)^4] / sigma^4 on flattened weights."""
    W_flat = W.detach().float().flatten()
    mu = W_flat.mean()
    sigma = W_flat.std(unbiased=False)
    return float((((W_flat - mu) ** 4).mean() / (sigma**4 + EPS)).item())


def spectral_entropy(W: torch.Tensor) -> float:
    """Entropy (nats) of normalized squared singular values."""
    S = _safe_svdvals(_as_matrix(W))
    denom = (S**2).sum().clamp_min(EPS)
    p = S**2 / denom
    p = p[p > 1e-10]
    return float((-(p * torch.log(p)).sum()).item())


def rank_ratio(W: torch.Tensor) -> float:
    """||W||_F^2 / (||W||_2^2 * min(m, n)); in (0, 1] for typical matrices."""
    W = _as_matrix(W)
    frob_sq = (W**2).sum()
    S = _safe_svdvals(W)
    spectral_sq = S[0] ** 2 if S.numel() else torch.tensor(0.0, dtype=W.dtype, device=W.device)
    m, n = W.shape
    return float((frob_sq / (spectral_sq * min(m, n) + EPS)).item())


def nuclear_norm_normalized(W: torch.Tensor) -> float:
    """||W||_* / min(m, n) = sum(singular values) / min(m, n)."""
    W = _as_matrix(W)
    S = _safe_svdvals(W)
    m, n = W.shape
    return float((S.sum() / max(1, min(m, n))).item())


def extract_feature_dict(
    W: torch.Tensor,
    *,
    layer_name: str | None = None,
    layer_index: int | None = None,
    num_layers: int | None = None,
    group_size: int = 128,
) -> dict[str, float]:
    """
    Return data-free features for one weight matrix.

    The dictionary includes legacy raw feature names for old checkpoints plus the
    expanded default feature set used by new PRISM checkpoints.
    """
    W = _as_matrix(W)

    S = _safe_svdvals(W)
    m, n = W.shape

    W_flat = W.flatten()
    mu = W_flat.mean()
    sigma = W_flat.std(unbiased=False)
    kurt = float((((W_flat - mu) ** 4).mean() / (sigma**4 + EPS)).item())

    p = S**2 / (S**2).sum().clamp_min(EPS)
    p_safe = p[p > 1e-10]
    h = float((-(p_safe * torch.log(p_safe)).sum()).item())

    frob_sq = float((W**2).sum().item())
    spectral_sq = float((S[0] ** 2).item()) if S.numel() else 0.0
    rr = frob_sq / (spectral_sq * min(m, n) + EPS)

    nn = float(S.sum().item()) / max(1, min(m, n))
    mean_abs = float(W_flat.abs().mean().item())
    max_abs = float(W_flat.abs().max().item()) if W_flat.numel() else 0.0
    outlier_ratio = float((W_flat.abs() > 3.0 * sigma.clamp_min(EPS)).float().mean().item())
    scale_mean, scale_std, scale_max = _scale_stats(W, bits=4, group_size=group_size)
    sv_decay = 0.0
    if S.numel() >= 2:
        top_idx = min(3, S.numel() - 1)
        sv_decay = float(((S[0] - S[top_idx]) / S[0].clamp_min(EPS)).item())

    layer_type = _infer_layer_type(layer_name)
    features: dict[str, float] = {
        "kurtosis": kurt,
        "spectral_entropy": h,
        "rank_ratio": rr,
        "nuclear_norm_normalized": nn,
        "log_kurtosis": math.log1p(max(kurt, 0.0)),
        "log_nuclear_norm_normalized": math.log1p(max(nn, 0.0)),
        "layer_depth_normalized": _infer_depth(layer_name, layer_index, num_layers),
        "shape_ratio": float(m / max(n, 1)),
        "log_num_params": math.log1p(float(W.numel())),
        "max_to_mean_abs": max_abs / (mean_abs + EPS),
        "outlier_ratio_3sigma": outlier_ratio,
        "scale_mean_4bit": scale_mean,
        "scale_std_4bit": scale_std,
        "scale_max_4bit": scale_max,
        "singular_value_decay_top4": sv_decay,
        "rtn_rel_mse_2bit": rtn_relative_mse(W, bits=2, group_size=group_size),
        "rtn_rel_mse_3bit": rtn_relative_mse(W, bits=3, group_size=group_size),
        "rtn_rel_mse_4bit": rtn_relative_mse(W, bits=4, group_size=group_size),
    }
    for bucket in LAYER_TYPE_BUCKETS:
        features[f"layer_type_{bucket}"] = 1.0 if layer_type == bucket else 0.0
    return features


def extract_features(
    W: torch.Tensor,
    *,
    layer_name: str | None = None,
    layer_index: int | None = None,
    num_layers: int | None = None,
    group_size: int = 128,
    feature_names: tuple[str, ...] | list[str] | None = None,
) -> torch.Tensor:
    """Return feature tensor ordered by ``feature_names`` or ``FEATURE_NAMES``."""
    names = tuple(feature_names) if feature_names is not None else FEATURE_NAMES
    features = extract_feature_dict(
        W,
        layer_name=layer_name,
        layer_index=layer_index,
        num_layers=num_layers,
        group_size=group_size,
    )
    return torch.tensor([features[name] for name in names], dtype=torch.float32)
