from prism.assign.optimize import assign_bits
from prism.models.mock_transformer import MockTransformerLM
from prism.profile.pipeline import build_profile_artifact
from prism.quic.pipeline import run_quic_correction


def test_quic_correction_preserves_budget_and_updates_surprise_scores(tmp_path, tiny_prism_mlp_path) -> None:
    model = MockTransformerLM(hidden_size=32, num_layers=1)
    mlp_path = tiny_prism_mlp_path
    profile = build_profile_artifact(
        model=model,
        mlp_path=mlp_path,
        model_id="t",
        model_family="mock",
        group_size=8,
    )
    assignment = assign_bits(profile, target_average_bits=3.0)

    corrected = run_quic_correction(
        model=model,
        profile_artifact=profile,
        assignment=assignment,
        hidden_size=32,
        seq_len=4,
        group_size=8,
    )

    assert corrected["average_bits"] <= assignment["budget"] + 1e-5
    assert corrected["swap_count"] >= 0
    assert corrected["surprise"]
