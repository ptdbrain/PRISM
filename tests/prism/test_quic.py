from prism.assign.optimize import assign_bits
from prism.data.synthetic import make_sensitivity_dataset
from prism.meta.train import train_meta_learner
from prism.models.mock_transformer import MockTransformerLM
from prism.profile.pipeline import profile_model
from prism.quic.pipeline import run_quic_correction


def test_quic_correction_preserves_budget_and_updates_surprise_scores(tmp_path) -> None:
    model = MockTransformerLM(hidden_size=8, num_layers=1)
    checkpoint_dir = tmp_path / "checkpoint"
    train_meta_learner(make_sensitivity_dataset(model), output_dir=checkpoint_dir, epochs=3, seed=0)
    profile = profile_model(model=model, checkpoint_dir=checkpoint_dir)
    assignment = assign_bits(profile, target_average_bits=3.0)

    corrected = run_quic_correction(
        model=model,
        profile_artifact=profile,
        assignment=assignment,
        hidden_size=8,
        seq_len=4,
    )

    assert corrected["average_bits"] <= assignment["budget"]
    assert corrected["swap_count"] >= 0
    assert corrected["surprise"]
