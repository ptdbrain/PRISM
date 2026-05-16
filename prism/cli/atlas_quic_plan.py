"""CLI for PRISM-Atlas uncertainty-guided QUIC planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prism.atlas.quic import select_uncertain_layers
from prism.data.io import load_json, save_json


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Select PRISM-Atlas layers for uncertainty-guided QUIC.")
    parser.add_argument("--profile-path", required=True)
    parser.add_argument("--assignment-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--min-uncertainty", type=float, default=None)
    args = parser.parse_args(argv)

    profile = load_json(Path(args.profile_path))
    assignment = load_json(Path(args.assignment_path))
    layers = select_uncertain_layers(
        profile,
        assignment,
        top_fraction=args.top_fraction,
        min_uncertainty=args.min_uncertainty,
    )
    payload = {
        "method": "prism-atlas-v1",
        "stage": "2.5",
        "correction": "uncertainty-guided-quic-plan",
        "selected_layers": layers,
        "num_selected_layers": len(layers),
        "top_fraction": float(args.top_fraction),
        "min_uncertainty": args.min_uncertainty,
    }
    save_json(Path(args.output_path), payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
