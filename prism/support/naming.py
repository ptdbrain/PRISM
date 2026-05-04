"""Naming helpers for PRISM layer normalization and classification."""

from __future__ import annotations


def module_type_from_name(layer_name: str) -> str:
    return layer_name.split(".")[-1]
