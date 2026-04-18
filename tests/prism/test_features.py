import torch

from prism.meta.features import compute_zero_cost_features


def test_compute_zero_cost_features_returns_expected_keys() -> None:
    weight = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)

    features = compute_zero_cost_features(weight)

    assert set(features) == {
        "kurtosis",
        "spectral_entropy",
        "effective_rank_ratio",
        "nuclear_norm_normalized",
    }
