"""CLI for PRISM-Atlas Stage 2 risk-aware assignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prism.atlas.solver import solve_atlas_assignment
from prism.data.io import load_json, save_json


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Solve PRISM-Atlas action assignments.")
    parser.add_argument("--profile-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--budget-bits", type=float, default=None)
    parser.add_argument("--budget", type=float, default=None, help="Target average bit budget.")
    parser.add_argument("--objective-mode", choices=["mean", "risk", "cvar-lite", "oracle/debug"], default="risk")
    parser.add_argument("--risk-lambda", type=float, default=0.25)
    parser.add_argument("--allow-invalid-actions", action="store_true")
    parser.add_argument("--latency-budget", type=float, default=None)
    args = parser.parse_args(argv)

    profile = load_json(Path(args.profile_path))
    result = solve_atlas_assignment(
        profile,
        budget_bits=args.budget_bits,
        target_average_bits=args.budget,
        objective_mode=args.objective_mode,
        risk_lambda=args.risk_lambda,
        require_valid_actions=not args.allow_invalid_actions,
        latency_budget=args.latency_budget,
    )
    save_json(Path(args.output_path), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
