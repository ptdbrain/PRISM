"""CLI for Stage 4 runtime assembly and dry-run execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prism.data.io import load_json
from prism.rtn.manifest import load_manifest
from prism.runtime.assemble import assemble_runtime_model
from prism.runtime.runner import run_forward, run_generate
from prism.support.model_loading import load_model_bundle, load_tokenizer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Assemble and run a PRISM runtime model.")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--assignment-path", required=True)
    parser.add_argument("--model-id-or-path", type=str, default=None)
    parser.add_argument("--family", type=str, default="auto")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--summary-path", type=str, default=None, help="Optional JSON file for runtime summary.")
    args = parser.parse_args(argv)

    artifact_root = Path(args.artifact_root)
    manifest_path = artifact_root / "manifest.json"
    if not manifest_path.exists():
        manifest_path = artifact_root / "manifest.pt"

    bundle = load_model_bundle(
        model_id_or_path=args.model_id_or_path,
        family=None if args.family == "auto" else args.family,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    )
    manifest = load_manifest(manifest_path)
    assignment = load_json(Path(args.assignment_path))
    runtime_model, summary = assemble_runtime_model(
        bundle.model,
        manifest,
        assignment,
        artifact_root,
        copy_model=False,
    )

    if args.execute:
        if bundle.is_demo:
            output = run_forward(runtime_model, hidden_size=args.hidden_size)
            summary["output_shape"] = list(output.shape)
        else:
            tokenizer = load_tokenizer(
                bundle.tokenizer_id or bundle.model_id,
                trust_remote_code=args.trust_remote_code,
            )
            summary["generated_text"] = run_generate(
                runtime_model,
                tokenizer,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
            )

    if args.summary_path:
        summary_path = Path(args.summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
