"""CLI for Stage 0 PRISM meta-learner training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prism.profiling.meta_learner import train_meta_learner
from prism.profiling.sensitivity import build_training_dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the PRISM meta-learner.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--model-names", nargs="*", default=None)
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset_path) if args.dataset_path else output_dir / "prism_train_data.pt"

    if args.dataset_path is None:
        model_names = args.model_names or ["__prism_mock_ten_layers__"]
        build_training_dataset(
            model_name_list=list(model_names),
            group_size=args.group_size,
            save_path=str(dataset_path),
        )

    mlp_path = output_dir / "prism_mlp.pt"
    train_meta_learner(
        dataset_path=str(dataset_path),
        epochs=args.epochs,
        save_path=str(mlp_path),
    )
    print(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "mlp_path": str(mlp_path),
                "epochs": args.epochs,
            },
            indent=2,
        )
    )
