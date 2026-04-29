"""Manifest I/O helpers for PRISM RTN artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def save_manifest(manifest: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    torch.save(manifest, output_dir / "manifest.pt")


def load_manifest(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return torch.load(path, map_location="cpu")
