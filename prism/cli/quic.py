"""CLI for Stage 2.5 QUIC correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prism.data.io import load_json, save_json
from prism.data.schemas import ProfileArtifact
from prism.quic.pipeline import run_quic_correction
from prism.support.model_loading import load_model_bundle


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run PRISM QUIC correction.")
    parser.add_argument("--profile-path", required=True)
    parser.add_argument("--assignment-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model-id-or-path", type=str, default=None)
    parser.add_argument("--family", type=str, default="auto")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=8)
    args = parser.parse_args(argv)

    bundle = load_model_bundle(
        model_id_or_path=args.model_id_or_path,
        family=None if args.family == "auto" else args.family,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        hidden_size=args.hidden_size,
        num_layers=4,
    )
    corrected = run_quic_correction(
        model=bundle.model,
        profile_artifact=ProfileArtifact.from_dict(load_json(Path(args.profile_path))),
        assignment=load_json(Path(args.assignment_path)),
        hidden_size=args.hidden_size,
        seq_len=args.seq_len,
    )
    save_json(Path(args.output_path), corrected)
    print(json.dumps(corrected, indent=2))
