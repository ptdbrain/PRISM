from prism.data.synthetic import make_sensitivity_dataset
from prism.models.mock_transformer import MockTransformerLM


def test_synthetic_dataset_makes_v_proj_more_sensitive_than_o_proj() -> None:
    model = MockTransformerLM(hidden_size=8, num_layers=2)

    records = make_sensitivity_dataset(model, seed=0)

    v_proj_scores = [item.target_sensitivity for item in records if item.module_type == "v_proj"]
    o_proj_scores = [item.target_sensitivity for item in records if item.module_type == "o_proj"]

    assert min(v_proj_scores) > max(o_proj_scores)
