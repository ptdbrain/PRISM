"""Sensitivity MLP — offline-trained PRISM meta-learner."""

from __future__ import annotations

from statistics import median

import torch
import torch.nn as nn
import torch.nn.functional as F

from prism.profile.inspect import iter_named_linear_layers
from prism.profiling.features import FEATURE_NAMES, LEGACY_FEATURE_NAMES, extract_feature_dict, extract_features
from prism.assignment.memory import memory_costs_by_bit

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is a project dependency
    def tqdm(iterable, *args, **kwargs):
        return iterable


class SensitivityMLP(nn.Module):
    """
    Input: data-free layer features.
    Output: predicted sensitivity at [2-bit, 3-bit, 4-bit].

    The output head is monotonic by construction:
    s4 = softplus(a), s3 = s4 + softplus(d3), s2 = s3 + softplus(d2).
    """

    def __init__(self, input_dim: int | None = None) -> None:
        super().__init__()
        self.input_dim = int(input_dim or len(FEATURE_NAMES))
        self.feature_order = _feature_order_for_dim(self.input_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        s4 = F.softplus(raw[:, 0])
        s3 = s4 + F.softplus(raw[:, 1])
        s2 = s3 + F.softplus(raw[:, 2])
        return torch.stack((s2, s3, s4), dim=1)


def _feature_order_for_dim(input_dim: int, checkpoint: dict | None = None) -> tuple[str, ...]:
    if checkpoint and "feature_order" in checkpoint:
        order = tuple(str(x) for x in checkpoint["feature_order"])
        if len(order) == input_dim:
            return order
    if input_dim == len(LEGACY_FEATURE_NAMES):
        return LEGACY_FEATURE_NAMES
    if input_dim == len(FEATURE_NAMES):
        return FEATURE_NAMES
    return tuple(FEATURE_NAMES[:input_dim])


def _ranking_loss(log_pred: torch.Tensor, log_target: torch.Tensor, max_pairs: int = 4096) -> torch.Tensor:
    n = log_pred.shape[0]
    if n < 2:
        return log_pred.new_tensor(0.0)
    losses: list[torch.Tensor] = []
    for col in range(log_pred.shape[1]):
        i, j = torch.triu_indices(n, n, offset=1, device=log_pred.device)
        if i.numel() > max_pairs:
            step = max(1, i.numel() // max_pairs)
            i = i[::step][:max_pairs]
            j = j[::step][:max_pairs]
        target_diff = log_target[i, col] - log_target[j, col]
        mask = target_diff.abs() > 1e-8
        if not bool(mask.any()):
            continue
        sign = target_diff[mask].sign()
        pred_diff = log_pred[i[mask], col] - log_pred[j[mask], col]
        losses.append(F.softplus(-sign * pred_diff).mean())
    if not losses:
        return log_pred.new_tensor(0.0)
    return torch.stack(losses).mean()


def train_meta_learner(
    dataset_path: str = "prism_train_data.pt",
    epochs: int = 500,
    lr: float = 1e-3,
    save_path: str = "prism_mlp.pt",
    ranking_weight: float = 0.05,
) -> SensitivityMLP:
    """Train MLP on (X, Y) from build_training_dataset output."""
    bundle = torch.load(dataset_path, map_location="cpu", weights_only=False)
    X = bundle["X"].float()
    Y = bundle["Y"].float()
    eps = 1e-10
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Y = torch.nan_to_num(Y, nan=eps, posinf=1e6, neginf=eps).clamp_min(eps)
    feature_order = tuple(bundle.get("feature_order") or _feature_order_for_dim(X.shape[1]))

    feat_mean = X.mean(dim=0, keepdim=True)
    feat_std = X.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    Xn = (X - feat_mean) / feat_std

    mlp = SensitivityMLP(input_dim=X.shape[1])
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    mlp.train()
    for _ in tqdm(range(epochs), desc="Stage 0 train MLP", unit="epoch"):
        opt.zero_grad()
        pred = mlp(Xn)
        log_pred = torch.log(pred + eps)
        log_target = torch.log(Y)
        loss = F.mse_loss(log_pred, log_target)
        if ranking_weight > 0:
            loss = loss + float(ranking_weight) * _ranking_loss(log_pred, log_target)
        loss.backward()
        opt.step()

    torch.save(
        {
            "state_dict": mlp.state_dict(),
            "feat_mean": feat_mean,
            "feat_std": feat_std,
            "feature_order": feature_order,
            "head": "monotonic_delta",
        },
        save_path,
    )
    mlp.feat_mean = feat_mean
    mlp.feat_std = feat_std
    mlp.feature_order = feature_order
    mlp.eval()
    return mlp


def load_pretrained_mlp(mlp_path: str = "prism_mlp.pt") -> SensitivityMLP:
    checkpoint = torch.load(mlp_path, map_location="cpu", weights_only=False)
    nonfinite = [
        name
        for name, tensor in checkpoint["state_dict"].items()
        if not torch.isfinite(tensor.float()).all()
    ]
    if nonfinite:
        names = ", ".join(nonfinite[:5])
        raise ValueError(
            f"Invalid PRISM MLP checkpoint contains non-finite tensors: {names}. "
            "Retrain prism_mlp.pt with the current Stage 0 trainer."
        )
    if (
        not torch.isfinite(checkpoint["feat_mean"].float()).all()
        or not torch.isfinite(checkpoint["feat_std"].float()).all()
    ):
        raise ValueError("Invalid PRISM MLP checkpoint contains non-finite feature normalization tensors.")
    input_dim = int(checkpoint["state_dict"]["net.0.weight"].shape[1])
    feature_order = _feature_order_for_dim(input_dim, checkpoint)
    mlp = SensitivityMLP(input_dim=input_dim)
    mlp.load_state_dict(checkpoint["state_dict"])
    mlp.feat_mean = checkpoint["feat_mean"]
    mlp.feat_std = checkpoint["feat_std"]
    mlp.feature_order = feature_order
    mlp.eval()
    return mlp


def predict_sensitivity(mlp: SensitivityMLP, W: torch.Tensor, bits: int) -> float:
    if bits not in (2, 3, 4):
        raise ValueError("bits must be 2, 3, or 4")
    feature_order = tuple(getattr(mlp, "feature_order", FEATURE_NAMES))
    feat = extract_features(W, feature_names=feature_order).unsqueeze(0)
    feat_norm = (feat - mlp.feat_mean) / mlp.feat_std
    with torch.no_grad():
        pred = mlp(feat_norm)
    idx = {2: 0, 3: 1, 4: 2}[bits]
    return pred[0, idx].item()


def profile_model(model: torch.nn.Module, mlp: SensitivityMLP, group_size: int = 128) -> dict:
    """
    Build sensitivity and memory profile using the frozen MLP.
    """
    layer_rows: list[tuple[str, dict[int, float], tuple[int, int], int, dict[str, float], dict[int, int]]] = []
    layer_items = list(iter_named_linear_layers(model))
    feature_order = tuple(getattr(mlp, "feature_order", FEATURE_NAMES))
    for layer_index, (layer_name, module) in enumerate(
        tqdm(layer_items, desc="Stage 1 profile layers", unit="layer")
    ):
        W = module.weight.detach()
        feat = extract_features(
            W,
            layer_name=layer_name,
            layer_index=layer_index,
            num_layers=len(layer_items),
            group_size=group_size,
            feature_names=feature_order,
        ).unsqueeze(0)
        feat_norm = (feat - mlp.feat_mean) / mlp.feat_std
        with torch.no_grad():
            pred = mlp(feat_norm)[0]
        sens: dict[int, float] = {
            2: float(pred[0].item()),
            3: float(pred[1].item()),
            4: float(pred[2].item()),
        }
        if "o_proj" in layer_name:
            sens = {b: v * 0.5 for b, v in sens.items()}
        if "v_proj" in layer_name:
            sens = {b: v * 1.5 for b, v in sens.items()}
        shp = tuple(int(x) for x in W.shape)
        shape2d = (shp[0], shp[1] if len(shp) >= 2 else 1)
        features = extract_feature_dict(
            W,
            layer_name=layer_name,
            layer_index=layer_index,
            num_layers=len(layer_items),
            group_size=group_size,
        )
        mem_cost = memory_costs_by_bit(shape2d, group_size=group_size)
        layer_rows.append((layer_name, sens, shape2d, int(W.numel()), features, mem_cost))

    s4_values = [row[1][4] for row in layer_rows]
    med_s4 = median(s4_values) if s4_values else 0.0

    profile: dict = {}
    for layer_name, sens, shape2d, n_params, features, mem_cost in layer_rows:
        is_outlier = bool(med_s4 > 0 and sens[4] > 2.0 * med_s4)
        profile[layer_name] = {
            "shape": shape2d,
            "num_params": n_params,
            "sensitivity": sens,
            "features": features,
            "memory_cost_bits": mem_cost,
            "is_outlier": bool(is_outlier),
            "forced_bits": 4 if is_outlier else None,
        }
    return profile
