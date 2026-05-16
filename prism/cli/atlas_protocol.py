"""CLI for writing PRISM-Atlas paper protocol scaffolds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prism.atlas.protocols import benchmark_protocol, cleanup_plan, transfer_protocol
from prism.data.io import save_json


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write PRISM-Atlas benchmark/transfer protocol JSON.")
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args(argv)

    payload = {
        "method": "prism-atlas-v1",
        "benchmark": benchmark_protocol(),
        "transfer": transfer_protocol(),
        "cleanup": cleanup_plan(),
    }
    save_json(Path(args.output_path), payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
