from pathlib import Path

from prism.models.mock_transformer import MockTransformerLM
from prism.rtn.precompute import precompute_model_rtn


def test_precompute_model_rtn_writes_manifest_and_layer_artifacts(tmp_path: Path) -> None:
    model = MockTransformerLM(hidden_size=8, num_layers=1)

    manifest = precompute_model_rtn(
        model=model,
        output_dir=tmp_path,
        group_size=8,
        bits=(2, 3, 4),
        model_id="unit-test-model",
    )

    assert manifest["model_id"] == "unit-test-model"
    assert manifest["group_size"] == 8
    assert "layers.0.self_attn.v_proj" in manifest["layers"]
    entry = manifest["layers"]["layers.0.self_attn.v_proj"]["2"]
    assert (tmp_path / entry["qweight_path"]).exists()
    assert (tmp_path / entry["scale_path"]).exists()
