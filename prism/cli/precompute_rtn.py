"""CLI for Stage 3 RTN precomputation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prism.rtn.precompute import precompute_model_rtn
from prism.support.model_loading import load_model_bundle


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Precompute PRISM RTN artifacts.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id-or-path", type=str, default=None)
    parser.add_argument("--family", type=str, default="auto")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=4)
    args = parser.parse_args(argv)

    bundle = load_model_bundle(
        model_id_or_path=args.model_id_or_path,
        family=None if args.family == "auto" else args.family,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    )
    manifest = precompute_model_rtn(
        model=bundle.model,
        output_dir=Path(args.output_dir),
        group_size=args.group_size,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
