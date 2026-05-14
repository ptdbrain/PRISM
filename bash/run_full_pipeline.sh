#!/usr/bin/env bash
set -euo pipefail

# Thin shell launcher for the shared Python PRISM full-pipeline runner.
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
exec python3 scripts/run_full_pipeline.py "$@"
