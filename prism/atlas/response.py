"""Action-conditioned response surface model for PRISM-Atlas."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from prism.atlas.actions import QuantizationAction, action_feature_names
from prism.profiling.features import FEATURE_NAMES, rtn_relative_mse
from prism.atlas.transforms import transform_weight

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is a project dependency
    def tqdm(iterable, *args, **kwargs):
        return iterable


class ResponseSurfaceMLP(nn.Module):
    """Predict mean damage, log variance, and ranking score for one layer/action row."""

    def __init__(self, layer_feature_dim: int | None = None, action_feature_dim: int | None = None) -> None:
        super().__init__()
        self.layer_feature_dim = int(layer_feature_dim or len(FEATURE_NAMES))
        self.action_feature_dim = int(action_feature_dim or len(action_feature_names()))
        input_dim = self.layer_feature_dim + self.action_feature_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 3),
        )

    def forward(self, layer_features: torch.Tensor, action_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.net(torch.cat((layer_features, action_features), dim=-1))
        mean_damage = F.softplus(raw[:, 0]) + 1e-10
        log_variance = raw[:, 1].clamp(min=-20.0, max=10.0)
        ranking_score = raw[:, 2]
        return mean_damage, log_variance, ranking_score


@dataclass
class LoadedResponseSurface:
    model: ResponseSurfaceMLP
    feat_mean: torch.Tensor
    feat_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor
    layer_feature_order: tuple[str, ...]
    action_feature_order: tuple[str, ...]

    def predict(
        self,
        layer_features: dict[str, float],
        action: QuantizationAction,
        *,
        transform_supported: bool,
    ) -> dict[str, float]:
        layer_vec = torch.tensor(
            [[float(layer_features[name]) for name in self.layer_feature_order]],
            dtype=torch.float32,
        )
        action_features = action.to_feature_dict(transform_supported=transform_supported)
        action_vec = torch.tensor(
            [[float(action_features[name]) for name in self.action_feature_order]],
            dtype=torch.float32,
        )
        layer_vec = (layer_vec - self.feat_mean) / self.feat_std
        action_vec = (action_vec - self.action_mean) / self.action_std
        self.model.eval()
        with torch.no_grad():
            mean_damage, log_variance, ranking_score = self.model(layer_vec, action_vec)
        uncertainty = torch.exp(0.5 * log_variance).clamp_min(0.0)
        return {
            "mean_damage": float(mean_damage[0].item()),
            "log_variance": float(log_variance[0].item()),
            "uncertainty": float(uncertainty[0].item()),
            "ranking_score": float(ranking_score[0].item()),
        }


def estimate_action_response(weight: torch.Tensor, action: QuantizationAction) -> dict[str, float | bool]:
    transformed, supported = transform_weight(weight, action.transform)
    mean = rtn_relative_mse(transformed, bits=int(action.bits), group_size=int(action.group_size))
    if action.transform != "none" and not supported:
        mean *= 1.05
    base_uncertainty = max(mean, 1e-8) * (0.10 + 0.08 * (4 - int(action.bits)))
    if int(action.group_size) == 64:
        base_uncertainty *= 0.9
    if action.transform != "none":
        base_uncertainty *= 1.35
    if not supported:
        base_uncertainty += max(mean, 1e-8) * 0.5
    log_variance = math.log(max(base_uncertainty * base_uncertainty, 1e-20))
    return {
        "mean_damage": float(max(mean, 1e-10)),
        "log_variance": float(log_variance),
        "uncertainty": float(base_uncertainty),
        "ranking_score": float(-mean),
        "transform_supported": bool(supported),
    }


def _ranking_loss(pred_rank: torch.Tensor, target: torch.Tensor, max_pairs: int = 4096) -> torch.Tensor:
    n = pred_rank.shape[0]
    if n < 2:
        return pred_rank.new_tensor(0.0)
    i, j = torch.triu_indices(n, n, offset=1, device=pred_rank.device)
    if i.numel() > max_pairs:
        step = max(1, i.numel() // max_pairs)
        i = i[::step][:max_pairs]
        j = j[::step][:max_pairs]
    target_diff = target[i] - target[j]
    mask = target_diff.abs() > 1e-10
    if not bool(mask.any()):
        return pred_rank.new_tensor(0.0)
    sign = -target_diff[mask].sign()
    pred_diff = pred_rank[i[mask]] - pred_rank[j[mask]]
    return F.softplus(-sign * pred_diff).mean()


def train_response_surface(
    *,
    dataset_path: str,
    epochs: int = 100,
    lr: float = 1e-3,
    save_path: str,
    ranking_weight: float = 0.05,
) -> LoadedResponseSurface:
    bundle = torch.load(dataset_path, map_location="cpu", weights_only=False)
    x_layer = torch.nan_to_num(bundle["X_layer"].float(), nan=0.0, posinf=0.0, neginf=0.0)
    x_action = torch.nan_to_num(bundle["X_action"].float(), nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.nan_to_num(bundle["Y_mean"].float(), nan=1e-10, posinf=1e6, neginf=1e-10).clamp_min(1e-10)

    feat_mean = x_layer.mean(dim=0, keepdim=True)
    feat_std = x_layer.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    action_mean = x_action.mean(dim=0, keepdim=True)
    action_std = x_action.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    x_layer_n = (x_layer - feat_mean) / feat_std
    x_action_n = (x_action - action_mean) / action_std

    model = ResponseSurfaceMLP(layer_feature_dim=x_layer.shape[1], action_feature_dim=x_action.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    log_target = torch.log(y)
    model.train()
    for _ in tqdm(range(int(epochs)), desc="Stage 0 Atlas train", unit="epoch"):
        opt.zero_grad()
        mean, log_var, ranking = model(x_layer_n, x_action_n)
        log_mean = torch.log(mean)
        inv_var = torch.exp(-log_var)
        loss = 0.5 * (inv_var * (log_mean - log_target).pow(2) + log_var).mean()
        if ranking_weight > 0:
            loss = loss + float(ranking_weight) * _ranking_loss(ranking, y)
        loss.backward()
        opt.step()

    layer_feature_order = tuple(bundle.get("layer_feature_order") or FEATURE_NAMES)
    action_feature_order = tuple(bundle.get("action_feature_order") or action_feature_names())
    checkpoint = {
        "state_dict": model.state_dict(),
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "layer_feature_order": layer_feature_order,
        "action_feature_order": action_feature_order,
        "head": "mean_logvariance_ranking",
        "method": "prism-atlas-v1",
    }
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    model.eval()
    return LoadedResponseSurface(
        model=model,
        feat_mean=feat_mean,
        feat_std=feat_std,
        action_mean=action_mean,
        action_std=action_std,
        layer_feature_order=layer_feature_order,
        action_feature_order=action_feature_order,
    )


def load_response_surface(path: str) -> LoadedResponseSurface:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    input_dim = int(checkpoint["state_dict"]["net.0.weight"].shape[1])
    action_order = tuple(checkpoint.get("action_feature_order") or action_feature_names())
    layer_order = tuple(checkpoint.get("layer_feature_order") or FEATURE_NAMES)
    layer_dim = input_dim - len(action_order)
    model = ResponseSurfaceMLP(layer_feature_dim=layer_dim, action_feature_dim=len(action_order))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return LoadedResponseSurface(
        model=model,
        feat_mean=checkpoint["feat_mean"],
        feat_std=checkpoint["feat_std"],
        action_mean=checkpoint["action_mean"],
        action_std=checkpoint["action_std"],
        layer_feature_order=layer_order,
        action_feature_order=action_order,
    )
