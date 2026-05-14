"""CLI for Stage 2 PRISM assignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prism.assign.optimize import assign_bits
from prism.assign.pareto import sweep_budgets
from prism.data.io import load_json, save_json
from prism.data.schemas import ProfileArtifact


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run PRISM mixed-precision assignment.")
    parser.add_argument("--profile-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--budget", type=float)
    parser.add_argument("--budgets", nargs="*", type=float)
    args = parser.parse_args(argv)

    profile = ProfileArtifact.from_dict(load_json(Path(args.profile_path)))
    if args.budget is not None:
        result = assign_bits(profile, target_average_bits=args.budget)
    else:
        result = {"pareto_front": sweep_budgets(profile, budgets=args.budgets or [])}

    save_json(Path(args.output_path), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
