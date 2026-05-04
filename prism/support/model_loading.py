"""Model loading helpers for PRISM demo and runtime paths."""

from __future__ import annotations

from pathlib import Path

import torch

from prism.adapters import LoadedModelBundle, resolve_adapter
from prism.models.mock_transformer import MockTransformerLM


def load_demo_model(hidden_size: int = 16, num_layers: int = 4) -> MockTransformerLM:
    return MockTransformerLM(hidden_size=hidden_size, num_layers=num_layers)


def _resolve_torch_dtype(torch_dtype: str | torch.dtype | None):
    if torch_dtype is None:
        return None
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype

    normalized = str(torch_dtype).strip().lower()
    if normalized in {"", "auto", "none"}:
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
    return mapping[normalized]


def _attach_adapter(model, adapter) -> None:
    setattr(model, "_prism_adapter", adapter)


def _build_bundle(
    *,
    model,
    adapter,
    model_id: str,
    is_demo: bool,
) -> LoadedModelBundle:
    _attach_adapter(model, adapter)
    return LoadedModelBundle(
        model=model,
        adapter=adapter,
        model_id=model_id,
        model_family=adapter.family,
        tokenizer_id=adapter.default_tokenizer_id(model, fallback=model_id),
        hidden_size=adapter.infer_hidden_size(model),
        num_layers=adapter.infer_num_layers(model),
        is_demo=is_demo,
    )


def load_model_bundle(
    model_id_or_path: str | None = None,
    *,
    model=None,
    family: str | None = None,
    device: str | None = None,
    torch_dtype: str | torch.dtype | None = "float16",
    trust_remote_code: bool = False,
    hidden_size: int = 16,
    num_layers: int = 4,
) -> LoadedModelBundle:
    if model is None and model_id_or_path is None:
        demo_model = load_demo_model(hidden_size=hidden_size, num_layers=num_layers)
        return _build_bundle(
            model=demo_model,
            adapter=resolve_adapter(model=demo_model, family="demo"),
            model_id="mock-transformer",
            is_demo=True,
        )

    if model is not None:
        adapter = resolve_adapter(model=model, family=family)
        inferred_model_id = (
            model_id_or_path
            or getattr(getattr(model, "config", None), "name_or_path", None)
            or getattr(getattr(model, "config", None), "_name_or_path", None)
            or model.__class__.__name__
        )
        return _build_bundle(
            model=model,
            adapter=adapter,
            model_id=str(inferred_model_id),
            is_demo=adapter.family == "demo",
        )

    try:
        from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Real-model loading requires `transformers`.") from exc

    load_kwargs = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    resolved_dtype = _resolve_torch_dtype(torch_dtype)
    if resolved_dtype is not None:
        load_kwargs["torch_dtype"] = resolved_dtype

    loaded_model = AutoModelForCausalLM.from_pretrained(model_id_or_path, **load_kwargs)
    if device:
        loaded_model = loaded_model.to(device)
    loaded_model.eval()

    return load_model_bundle(
        model_id_or_path=model_id_or_path,
        model=loaded_model,
        family=family,
        device=device,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )


def load_tokenizer(tokenizer_id: str | Path, *, trust_remote_code: bool = False):
    try:
        from transformers import AutoTokenizer  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Tokenizer loading requires `transformers`.") from exc

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_id), trust_remote_code=trust_remote_code)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer
