"""CLI for Stage 1 low-cost data-free profiling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prism.profile.pipeline import build_profile_artifact, profile_model_legacy
from prism.support.model_loading import load_model_bundle


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run PRISM low-cost data-free profiling.")
    parser.add_argument("--mlp-path", type=str, default=None, help="Path to prism_mlp.pt (spec meta-learner).")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Legacy checkpoint directory.")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model-id-or-path", type=str, default=None)
    parser.add_argument("--family", type=str, default="auto")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args(argv)

    bundle = load_model_bundle(
        model_id_or_path=args.model_id_or_path,
        family=None if args.family == "auto" else args.family,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    )
    if args.mlp_path:
        artifact = build_profile_artifact(
            bundle.model,
            mlp_path=Path(args.mlp_path),
            output_path=Path(args.output_path),
            model_id=bundle.model_id,
            model_family=bundle.model_family,
            group_size=args.group_size,
        )
    elif args.checkpoint_dir:
        artifact = profile_model_legacy(
            bundle.model,
            checkpoint_dir=Path(args.checkpoint_dir),
            output_path=Path(args.output_path),
            group_size=args.group_size,
        )
    else:
        parser.error("Provide either --mlp-path (recommended) or --checkpoint-dir (legacy).")

    print(json.dumps(artifact.to_dict(), indent=2))
