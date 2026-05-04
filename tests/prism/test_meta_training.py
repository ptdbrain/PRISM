from pathlib import Path

from prism.data.synthetic import make_sensitivity_dataset
from prism.meta.checkpoint import load_checkpoint
from prism.meta.train import train_meta_learner
from prism.models.mock_transformer import MockTransformerLM
from prism.profiling.features import FEATURE_NAMES


def test_train_meta_learner_writes_checkpoint(tmp_path: Path) -> None:
    model = MockTransformerLM(hidden_size=8, num_layers=2)
    records = make_sensitivity_dataset(model, seed=0)

    output_dir = tmp_path / "checkpoint"
    result = train_meta_learner(records, output_dir=output_dir, model=model, epochs=40, seed=0)
    checkpoint = load_checkpoint(output_dir)

    assert result["num_records"] == len(records)
    assert checkpoint["normalizer"]["feature_order"] == list(FEATURE_NAMES)
    assert (output_dir / "model.pt").exists()
    assert (output_dir / "metrics.json").exists()
