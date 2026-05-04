"""End-to-end synthetic demo for the full PRISM pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prism.api import PRISM
from prism.models.mock_transformer import MockTransformerLM


def run_demo(output_root: Path) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    quantizer = PRISM(
        model=MockTransformerLM(hidden_size=8, num_layers=2),
        family="demo",
        artifact_dir=output_root,
    )
    runtime_model = quantizer.run(target_bits=3.0, group_size=8)
    runtime_summary = quantizer.last_run["runtime_summary"]

    return {
        "profile_path": str(output_root / "profile.json"),
        "assignment_path": str(output_root / "assignment.json"),
        "quic_assignment_path": str(output_root / "quic_assignment.json"),
        "rtn_dir": str(output_root / "rtn"),
        "runtime_summary": runtime_summary,
        "backend_summary": runtime_model.backend_summary,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the synthetic end-to-end PRISM demo.")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_demo(Path(args.output_root)), indent=2))
