from pathlib import Path

import torch

from prism.atlas.actions import build_action_space
from prism.atlas.dataset import build_response_dataset
from prism.atlas.response import ResponseSurfaceMLP, load_response_surface, train_response_surface
from prism.models.mock_transformer import MockTransformerLM


def test_response_model_outputs_mean_variance_and_ranking() -> None:
    model = ResponseSurfaceMLP(layer_feature_dim=4, action_feature_dim=3)

    mean, log_var, ranking = model(torch.zeros(2, 4), torch.zeros(2, 3))

    assert mean.shape == (2,)
    assert log_var.shape == (2,)
    assert ranking.shape == (2,)
    assert torch.isfinite(mean).all()


def test_response_dataset_and_training_checkpoint(tmp_path: Path) -> None:
    mock = MockTransformerLM(hidden_size=8, num_layers=1)
    actions = build_action_space(bits=(2, 4), group_sizes=(64,), transforms=("none", "hadamard"))
    dataset_path = tmp_path / "atlas_data.pt"

    bundle = build_response_dataset([("mock", mock)], actions=actions, save_path=dataset_path)

    assert bundle["X_layer"].shape[0] == len(bundle["meta"])
    assert bundle["Y_mean"].shape[0] == len(bundle["meta"])

    ckpt = tmp_path / "atlas.pt"
    train_response_surface(dataset_path=str(dataset_path), epochs=2, save_path=str(ckpt))
    loaded = load_response_surface(str(ckpt))

    assert loaded.layer_feature_order
    assert loaded.action_feature_order
