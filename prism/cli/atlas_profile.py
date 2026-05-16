"""CLI for PRISM-Atlas Stage 1 response-surface profiling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prism.atlas.actions import build_action_space
from prism.atlas.profile import build_atlas_profile
from prism.support.model_loading import load_model_bundle


def _split_csv(values: list[str] | None, default: tuple[str, ...]) -> list[str]:
    if not values:
        return list(default)
    items: list[str] = []
    for value in values:
        items.extend(part.strip() for part in value.split(","))
    return [item for item in items if item]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Profile a model with PRISM-Atlas.")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--atlas-path", default=None)
    parser.add_argument("--model-id-or-path", default=None)
    parser.add_argument("--family", default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--bits", nargs="*", default=["2", "3", "4"])
    parser.add_argument("--group-sizes", nargs="*", default=["64", "128"])
    parser.add_argument("--transforms", nargs="*", default=["none", "hadamard"])
    parser.add_argument("--scorer", choices=["analytic", "learned", "hybrid"], default="analytic")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(argv)

    bundle = load_model_bundle(
        model_id_or_path=args.model_id_or_path,
        family=None if args.family == "auto" else args.family,
        device=args.device,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        trust_remote_code=args.trust_remote_code,
    )
    actions = build_action_space(
        bits=[int(x) for x in _split_csv(args.bits, ("2", "3", "4"))],
        group_sizes=[int(x) for x in _split_csv(args.group_sizes, ("64", "128"))],
        transforms=_split_csv(args.transforms, ("none", "hadamard")),
    )
    profile = build_atlas_profile(
        bundle.model,
        actions=actions,
        atlas_path=args.atlas_path,
        output_path=Path(args.output_path),
        model_id=bundle.model_id,
        model_family=bundle.model_family,
        scorer=args.scorer,
    )
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
