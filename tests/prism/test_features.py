import torch

from prism.meta.features import compute_zero_cost_features
from prism.profiling.features import FEATURE_NAMES, extract_features, rank_ratio, rtn_relative_mse, spectral_entropy


def test_compute_zero_cost_features_returns_expected_keys() -> None:
    weight = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)

    features = compute_zero_cost_features(weight)

    assert set(FEATURE_NAMES).issubset(features)
    assert {"rtn_rel_mse_2bit", "rtn_rel_mse_3bit", "rtn_rel_mse_4bit"}.issubset(features)


def test_extract_features_shape_and_dtype() -> None:
    w = torch.randn(64, 128)
    f = extract_features(w)
    assert f.shape == (len(FEATURE_NAMES),)
    assert f.dtype == torch.float32


def test_spectral_entropy_non_negative() -> None:
    w = torch.randn(32, 48)
    assert spectral_entropy(w) >= 0.0


def test_rank_ratio_identity() -> None:
    n = 8
    eye = torch.eye(n)
    rr = rank_ratio(eye)
    assert 0.99 <= rr <= 1.01


def test_rtn_reconstruction_error_decreases_with_bits() -> None:
    w = torch.randn(16, 32)
    assert rtn_relative_mse(w, 2, group_size=8) > rtn_relative_mse(w, 3, group_size=8)
    assert rtn_relative_mse(w, 3, group_size=8) > rtn_relative_mse(w, 4, group_size=8)
