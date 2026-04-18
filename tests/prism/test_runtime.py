from pathlib import Path

import torch

from prism.assign.optimize import assign_bits
from prism.data.synthetic import make_sensitivity_dataset
from prism.meta.train import train_meta_learner
from prism.models.mock_transformer import MockTransformerLM
from prism.profile.pipeline import profile_model
from prism.rtn.precompute import precompute_model_rtn
from prism.runtime.assemble import assemble_runtime_model


def test_assemble_runtime_model_falls_back_to_gemm_when_marlin_is_unavailable(tmp_path: Path) -> None:
    model = MockTransformerLM(hidden_size=8, num_layers=1)
    checkpoint_dir = tmp_path / "checkpoint"
    train_meta_learner(make_sensitivity_dataset(model), output_dir=checkpoint_dir, epochs=3, seed=0)
    profile = profile_model(model=model, checkpoint_dir=checkpoint_dir)
    assignment = assign_bits(profile, target_average_bits=3.0)
    manifest = precompute_model_rtn(model=model, output_dir=tmp_path / "rtn", group_size=8)

    runtime_model, summary = assemble_runtime_model(
        base_model=model,
        manifest=manifest,
        assignment=assignment,
        artifact_root=tmp_path / "rtn",
    )

    output = runtime_model(torch.randn(1, 8))

    assert output.shape == (1, 8)
    assert "gemm" in set(summary["backend_by_layer"].values())
