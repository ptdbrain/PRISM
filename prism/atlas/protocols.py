"""Paper-facing benchmark, transfer, and cleanup protocols for PRISM-Atlas."""

from __future__ import annotations


def benchmark_protocol() -> dict[str, list[str]]:
    return {
        "systems": [
            "Uniform RTN",
            "PRISM-original",
            "PRISM-Atlas analytic mean-only",
            "PRISM-Atlas learned mean-only",
            "PRISM-Atlas hybrid mean-only",
            "PRISM-Atlas hybrid risk-aware",
            "PRISM-Atlas no-transform",
            "PRISM-Atlas no-uncertainty",
        ],
        "budgets": ["2.0", "2.5", "3.0", "3.5", "4.0"],
        "metrics": [
            "WikiText-2 PPL",
            "C4 PPL",
            "ARC-Easy",
            "ARC-Challenge",
            "HellaSwag",
            "PIQA",
            "BoolQ",
            "WinoGrande",
            "MMLU",
            "GSM8K",
            "real average bits",
            "real memory",
            "decode TPS",
            "profiling/search time",
            "materialization time",
        ],
    }


def transfer_protocol() -> dict[str, list[str]]:
    return {
        "source_small_models": ["OPT-125M", "Pythia-160M", "TinyLlama"],
        "target_unseen_models": ["Qwen2.5-1.5B", "Qwen2.5-7B", "Llama-3.x-1B", "Llama-3.x-8B", "Mistral-7B", "Gemma"],
        "settings": ["zero-target-label", "light-target-calibration", "full-target-label-oracle"],
        "claim_boundary": [
            "Main transfer claim uses zero-target-label.",
            "Light target calibration is an auxiliary setting.",
            "Full target labels are oracle/debug only.",
        ],
    }


def cleanup_plan() -> dict[str, list[str]]:
    return {
        "phase_1": [
            "archive prism/meta/",
            "remove legacy --checkpoint-dir from mainline CLI",
            "unify assign/assignment namespace",
            "unify profile/profiling namespace",
            "add environment.yml or lockfile",
            "expand README reproduction path",
            "add real-model regression test",
        ],
        "non_goals": [
            "Do not remove PRISM-original baseline reproducibility.",
            "Do not claim Hadamard runtime speedup until optimized runtime exists.",
        ],
    }
