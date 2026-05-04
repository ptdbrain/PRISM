"""Offline ground-truth collection for meta-learner training (not used in inference)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from prism.profile.inspect import iter_named_linear_layers
from prism.profiling.features import FEATURE_NAMES, extract_features

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is a project dependency
    def tqdm(iterable, *args, **kwargs):
        del args, kwargs
        return iterable


def _progress(iterable, *, show_progress: bool, **kwargs):
    if not show_progress:
        return iterable
    return tqdm(iterable, **kwargs)


def measure_layer_sensitivity(
    model_name: str,
    layer_name: str,
    bits: int,
    calibration_data: Any,
    group_size: int = 128,
    _model: torch.nn.Module | None = None,
    _baseline_ppl: float | None = None,
    _eval_input_ids: torch.Tensor | None = None,
) -> float:
    """
    Normalized perplexity increase when only `layer_name` is RTN-quantized.

    calibration_data: ignored placeholder for API parity (wikitext2 used internally).
    Optional internal arguments avoid recomputing model-level eval state for every layer.
    """
    del calibration_data
    try:
        from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]
        from prism.profiling.sensitivity_eval import (
            eval_wikitext2_perplexity,
            temporarily_quantized_linear,
        )
    except ImportError as exc:  # pragma: no cover - optional stack
        raise RuntimeError(
            "measure_layer_sensitivity requires `transformers` and `datasets`; "
            "install prism with optional [train] extras."
        ) from exc

    if _model is None:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    else:
        model = _model
    model.eval()
    ppl_base = (
        float(_baseline_ppl)
        if _baseline_ppl is not None
        else eval_wikitext2_perplexity(model, tokenizer_id=model_name, input_ids=_eval_input_ids)
    )
    with temporarily_quantized_linear(model, layer_name, bits=bits, group_size=group_size):
        ppl_q = eval_wikitext2_perplexity(model, tokenizer_id=model_name, input_ids=_eval_input_ids)
    return float((ppl_q - ppl_base) / (ppl_base + 1e-8))


def build_training_dataset(
    model_name_list: list[str],
    bits_list: list[int] | None = None,
    group_size: int = 128,
    save_path: str = "prism_train_data.pt",
    show_progress: bool = True,
) -> None:
    """
    Build (X, Y, meta) for meta-learner training. Extremely GPU-heavy on real models.

    For unit tests, include the sentinel name ``\"__prism_mock_ten_layers__\"`` which
    builds 10 linear layers without HuggingFace.
    """
    if bits_list is None:
        bits_list = [2, 3, 4]

    rows_X: list[torch.Tensor] = []
    rows_Y: list[list[float]] = []
    meta: list[dict[str, str]] = []

    for model_name in _progress(model_name_list, show_progress=show_progress, desc="Models"):
        if model_name == "__prism_mock_ten_layers__":
            mock = _TenLinearMock()
            layer_items = list(iter_named_linear_layers(mock))
            for layer_index, (layer_name, module) in enumerate(_progress(
                layer_items,
                show_progress=show_progress,
                desc=f"{model_name}: layers",
                leave=False,
            )):
                W = module.weight.detach()
                X = extract_features(
                    W,
                    layer_name=layer_name,
                    layer_index=layer_index,
                    num_layers=len(layer_items),
                    group_size=group_size,
                )
                ys = [
                    0.05 * (5 - b) + 0.01 * X[0].item()
                    for b in _progress(
                        bits_list,
                        show_progress=show_progress,
                        desc=f"{layer_name}: bits",
                        leave=False,
                    )
                ]
                rows_X.append(X)
                rows_Y.append(ys)
                meta.append({"model_name": model_name, "layer_name": layer_name})
            continue

        try:
            from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]
            from prism.profiling.sensitivity_eval import eval_wikitext2_perplexity, prepare_wikitext2_input_ids
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("build_training_dataset requires transformers for real models.") from exc

        loaded = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        loaded.eval()
        eval_input_ids = prepare_wikitext2_input_ids(model_name)
        ppl_base = eval_wikitext2_perplexity(loaded, tokenizer_id=model_name, input_ids=eval_input_ids)
        layer_items = list(iter_named_linear_layers(loaded))
        for layer_index, (layer_name, module) in enumerate(_progress(
            layer_items,
            show_progress=show_progress,
            desc=f"{model_name}: layers",
            leave=False,
        )):
            W = module.weight.detach()
            X = extract_features(
                W,
                layer_name=layer_name,
                layer_index=layer_index,
                num_layers=len(layer_items),
                group_size=group_size,
            )
            ys = []
            for b in _progress(
                bits_list,
                show_progress=show_progress,
                desc=f"{layer_name}: bits",
                leave=False,
            ):
                ys.append(
                    measure_layer_sensitivity(
                        model_name,
                        layer_name,
                        b,
                        calibration_data=None,
                        group_size=group_size,
                        _model=loaded,
                        _baseline_ppl=ppl_base,
                        _eval_input_ids=eval_input_ids,
                    )
                )
            rows_X.append(X)
            rows_Y.append(ys)
            meta.append({"model_name": model_name, "layer_name": layer_name})

    X_t = torch.stack(rows_X, dim=0)
    Y_t = torch.tensor(rows_Y, dtype=torch.float32)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"X": X_t, "Y": Y_t, "meta": meta, "feature_order": FEATURE_NAMES}, save_path)


class _TenLinearMock(torch.nn.Module):
    """10 independent linear layers (for dataset schema tests)."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(32, 32, bias=False) for _ in range(10)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lin in self.layers:
            x = lin(x)
        return x
