"""CLI for PRISM-Atlas Stage 0 response-surface training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prism.atlas.actions import build_action_space
from prism.atlas.dataset import build_response_dataset
from prism.atlas.response import train_response_surface
from prism.support.model_loading import load_model_bundle


def _split_csv(values: list[str] | None, default: tuple[str, ...]) -> list[str]:
    if not values:
        return list(default)
    items: list[str] = []
    for value in values:
        items.extend(part.strip() for part in value.split(","))
    return [item for item in items if item]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the PRISM-Atlas response surface.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-names", nargs="*", default=None)
    parser.add_argument("--bits", nargs="*", default=["2", "3", "4"])
    parser.add_argument("--group-sizes", nargs="*", default=["64", "128"])
    parser.add_argument("--transforms", nargs="*", default=["none", "hadamard"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--family", default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    actions = build_action_space(
        bits=[int(x) for x in _split_csv(args.bits, ("2", "3", "4"))],
        group_sizes=[int(x) for x in _split_csv(args.group_sizes, ("64", "128"))],
        transforms=_split_csv(args.transforms, ("none", "hadamard")),
    )
    model_names = args.model_names or [None]
    entries = []
    for model_name in model_names:
        bundle = load_model_bundle(
            model_id_or_path=model_name,
            family=None if args.family == "auto" else args.family,
            device=args.device,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            trust_remote_code=args.trust_remote_code,
        )
        entries.append((bundle.model_id, bundle.model))

    dataset_path = output_dir / "atlas_train_data.pt"
    checkpoint_path = output_dir / "prism_atlas.pt"
    build_response_dataset(entries, actions=actions, save_path=dataset_path)
    train_response_surface(dataset_path=str(dataset_path), epochs=args.epochs, save_path=str(checkpoint_path))
    print(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "atlas_path": str(checkpoint_path),
                "num_actions": len(actions),
                "epochs": int(args.epochs),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
