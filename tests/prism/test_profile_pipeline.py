from pathlib import Path

from prism.data.io import load_json
from prism.data.synthetic import make_sensitivity_dataset
from prism.meta.train import train_meta_learner
from prism.models.mock_transformer import MockTransformerLM
from prism.profile.pipeline import profile_model


def test_profile_model_applies_prism_rules(tmp_path: Path) -> None:
    model = MockTransformerLM(hidden_size=8, num_layers=2)
    records = make_sensitivity_dataset(model, seed=0)
    checkpoint_dir = tmp_path / "checkpoint"
    train_meta_learner(records, output_dir=checkpoint_dir, epochs=5, seed=0)

    output_path = tmp_path / "profile.json"
    artifact = profile_model(model=model, checkpoint_dir=checkpoint_dir, output_path=output_path)

    v_proj_layers = [layer for layer in artifact.layers if layer.module_type == "v_proj"]
    o_proj_layers = [layer for layer in artifact.layers if layer.module_type == "o_proj"]

    assert all(layer.adjusted_score >= layer.raw_score for layer in v_proj_layers)
    assert all(layer.fixed_4bit for layer in o_proj_layers)
    assert load_json(output_path)["model_id"] == "mock-transformer"
