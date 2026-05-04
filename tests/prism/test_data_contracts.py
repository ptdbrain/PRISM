from pathlib import Path

from prism.data.io import load_json, save_json
from prism.data.schemas import ProfileArtifact, ProfileLayerRecord


def test_profile_artifact_round_trip(tmp_path: Path) -> None:
    artifact = ProfileArtifact(
        model_id="mock-transformer",
        model_family="synthetic",
        layers=[
            ProfileLayerRecord(
                layer_name="layers.0.self_attn.v_proj",
                module_type="v_proj",
                shape=[8, 8],
                num_params=64,
                features={
                    "kurtosis": 1.2,
                    "spectral_entropy": 0.8,
                    "rank_ratio": 0.5,
                    "nuclear_norm_normalized": 0.1,
                },
                raw_score=2.0,
                adjusted_score=3.0,
                fixed_4bit=False,
                fixed_reason="",
            )
        ],
        metadata={"group_size": 128},
    )

    output_path = tmp_path / "profile.json"
    save_json(output_path, artifact.to_dict())
    loaded = ProfileArtifact.from_dict(load_json(output_path))

    assert loaded.model_id == "mock-transformer"
    assert loaded.layers[0].module_type == "v_proj"
    assert loaded.layers[0].adjusted_score == 3.0
