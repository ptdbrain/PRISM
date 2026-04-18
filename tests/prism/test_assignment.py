from prism.assign.optimize import assign_bits
from prism.data.schemas import ProfileArtifact, ProfileLayerRecord


def test_assign_bits_respects_budget_and_fixed_layers() -> None:
    artifact = ProfileArtifact(
        model_id="mock-transformer",
        model_family="synthetic",
        layers=[
            ProfileLayerRecord(
                layer_name="layers.0.self_attn.o_proj",
                module_type="o_proj",
                shape=[8, 8],
                num_params=64,
                features={},
                raw_score=2.0,
                adjusted_score=2.0,
                fixed_4bit=True,
                fixed_reason="o_proj_rule",
            ),
            ProfileLayerRecord(
                layer_name="layers.0.self_attn.v_proj",
                module_type="v_proj",
                shape=[8, 8],
                num_params=64,
                features={},
                raw_score=3.0,
                adjusted_score=4.5,
                fixed_4bit=False,
                fixed_reason="",
            ),
            ProfileLayerRecord(
                layer_name="layers.0.mlp.down_proj",
                module_type="down_proj",
                shape=[8, 8],
                num_params=64,
                features={},
                raw_score=1.0,
                adjusted_score=1.0,
                fixed_4bit=False,
                fixed_reason="",
            ),
        ],
        metadata={},
    )

    result = assign_bits(artifact, target_average_bits=3.0)

    assert result["bits"]["layers.0.self_attn.o_proj"] == 4
    assert result["average_bits"] <= 3.0
