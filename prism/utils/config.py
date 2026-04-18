"""Configuration defaults shared across PRISM stages."""

from __future__ import annotations

from pathlib import Path


DEFAULT_GROUP_SIZE = 128
DEFAULT_BITS = (2, 3, 4)
DEFAULT_ARTIFACT_ROOT = Path("artifacts") / "prism"
