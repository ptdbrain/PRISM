"""Shared helpers (spec `utils.py` entrypoint)."""

from __future__ import annotations

import torch


def eval_perplexity(model: torch.nn.Module, tokenizer_id: str | None = None) -> float:
    """
    Evaluate causal LM perplexity on wikitext-2 (test split).
    Requires `transformers` and `datasets`.
    """
    from prism.profiling.sensitivity_eval import eval_wikitext2_perplexity

    inner = getattr(model, "inner", model)
    tid = tokenizer_id or getattr(getattr(inner, "config", None), "name_or_path", None) or getattr(
        getattr(inner, "config", None), "_name_or_path", None
    )
    return eval_wikitext2_perplexity(inner, tokenizer_id=tid)
