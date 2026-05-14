"""Summarize a completed PRISM pipeline run.

The report is intentionally based only on JSON/artifact metadata. It answers
"what happened in this run" without loading the model or importing torch.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REQUIRED_RESEARCH_METRICS = [
    {
        "id": "wikitext2_perplexity",
        "category": "perplexity",
        "task": "wikitext2",
        "display": "WikiText-2 perplexity",
        "metric": "perplexity",
        "higher_is_better": False,
    },
    {
        "id": "c4_perplexity",
        "category": "perplexity",
        "task": "c4",
        "display": "C4 perplexity",
        "metric": "perplexity",
        "higher_is_better": False,
    },
    {
        "id": "mmlu_accuracy",
        "category": "zero_shot",
        "task": "mmlu",
        "display": "MMLU accuracy",
        "metric": "accuracy",
        "higher_is_better": True,
    },
    {
        "id": "gsm8k_accuracy",
        "category": "zero_shot",
        "task": "gsm8k",
        "display": "GSM8K accuracy",
        "metric": "accuracy",
        "higher_is_better": True,
    },
    {
        "id": "arc_challenge_accuracy",
        "category": "zero_shot",
        "task": "arc_challenge",
        "display": "ARC-Challenge accuracy",
        "metric": "accuracy",
        "higher_is_better": True,
    },
    {
        "id": "arc_easy_accuracy",
        "category": "zero_shot",
        "task": "arc_easy",
        "display": "ARC-Easy accuracy",
        "metric": "accuracy",
        "higher_is_better": True,
    },
    {
        "id": "hellaswag_accuracy",
        "category": "zero_shot",
        "task": "hellaswag",
        "display": "HellaSwag accuracy",
        "metric": "accuracy",
        "higher_is_better": True,
    },
]

ACCURACY_KEYS = (
    "accuracy",
    "acc_norm,none",
    "acc,none",
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "exact_match,none",
    "exact_match",
    "acc_norm",
    "acc",
)
PERPLEXITY_KEYS = (
    "perplexity",
    "word_perplexity,none",
    "perplexity,none",
    "ppl",
    "word_perplexity",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


def _find_assignment_paths(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("assignment_*.json")) + sorted(run_dir.glob("quic_assignment_*.json"))


def _normalize_task_name(task_name: str) -> str:
    normalized = task_name.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized.startswith("mmlu"):
        return "mmlu"
    if "gsm8k" in normalized:
        return "gsm8k"
    if "hellaswag" in normalized:
        return "hellaswag"
    if normalized in {"arc", "arc_c", "arc_challenge", "ai2_arc_challenge"} or "arc_challenge" in normalized:
        return "arc_challenge"
    if normalized in {"arc_e", "arc_easy", "ai2_arc_easy"} or "arc_easy" in normalized:
        return "arc_easy"
    if "wikitext" in normalized or normalized in {"wiki_text_2", "wikitext2"}:
        return "wikitext2"
    if normalized == "c4" or normalized.startswith("c4_") or "allenai_c4" in normalized:
        return "c4"
    return normalized


def _infer_variant_from_path(path: Path) -> str:
    stem = path.stem.lower()
    if any(token in stem for token in ("baseline", "fp16", "float", "base")):
        return "baseline_fp16"
    if any(token in stem for token in ("prism", "quant", "w2", "w3", "w4")):
        return "prism"
    return path.stem


def _empty_research_variant(source_path: str | None = None) -> dict[str, Any]:
    return {
        "perplexity": {},
        "zero_shot": {},
        "downstream_accuracy": {},
        "efficiency": {},
        "config": {},
        "source_paths": [source_path] if source_path else [],
    }


def _extract_metric_entry(
    value: Any,
    keys: tuple[str, ...],
    *,
    metric: str,
    higher_is_better: bool,
) -> dict[str, Any] | None:
    if isinstance(value, (int, float)) and _is_finite(value):
        return {"value": float(value), "metric": metric, "higher_is_better": higher_is_better}
    if not isinstance(value, dict):
        return None

    selected_key = None
    selected_value = None
    for key in keys:
        if key in value and _is_finite(value[key]):
            selected_key = key
            selected_value = value[key]
            break
    if selected_key is None:
        for key, candidate in value.items():
            if isinstance(key, str) and any(key.startswith(prefix) for prefix in keys) and _is_finite(candidate):
                selected_key = key
                selected_value = candidate
                break
    if selected_key is None:
        return None

    entry = {
        "value": float(selected_value),
        "metric": metric,
        "source_metric": selected_key,
        "higher_is_better": higher_is_better,
    }
    for optional_key in ("shots", "num_fewshot", "num_samples", "stderr", "alias"):
        if optional_key in value:
            entry[optional_key] = value[optional_key]
    return entry


def _merge_metric(
    target: dict[str, Any],
    category: str,
    task: str,
    entry: dict[str, Any],
) -> None:
    if category not in target:
        target[category] = {}
    existing = target[category].get(task)
    if existing is None:
        target[category][task] = entry
        return
    values = [float(existing["value"]), float(entry["value"])]
    merged = dict(existing)
    merged["value"] = sum(values) / len(values)
    merged["num_merged_values"] = int(existing.get("num_merged_values", 1)) + 1
    target[category][task] = merged


def _merge_variant(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    for category in ("perplexity", "zero_shot", "downstream_accuracy", "efficiency"):
        existing.setdefault(category, {}).update(incoming.get(category, {}))
    existing.setdefault("config", {}).update(incoming.get("config", {}))
    existing.setdefault("source_paths", []).extend(incoming.get("source_paths", []))
    return existing


def _normalize_lm_eval_results(results: dict[str, Any], variant: dict[str, Any]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for raw_task, metrics in results.items():
        task = _normalize_task_name(str(raw_task))
        ppl_entry = _extract_metric_entry(
            metrics,
            PERPLEXITY_KEYS,
            metric="perplexity",
            higher_is_better=False,
        )
        if ppl_entry is not None and task in {"wikitext2", "c4"}:
            grouped[("perplexity", task)].append(ppl_entry)
            continue

        acc_entry = _extract_metric_entry(
            metrics,
            ACCURACY_KEYS,
            metric="accuracy",
            higher_is_better=True,
        )
        if acc_entry is not None:
            grouped[("zero_shot", task)].append(acc_entry)

    for (category, task), entries in grouped.items():
        merged = dict(entries[0])
        merged["value"] = sum(float(entry["value"]) for entry in entries) / len(entries)
        if len(entries) > 1:
            merged["num_merged_values"] = len(entries)
        _merge_metric(variant, category, task, merged)


def _normalize_named_metrics(
    raw_metrics: dict[str, Any],
    *,
    category: str,
    metric: str,
    higher_is_better: bool,
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    keys = PERPLEXITY_KEYS if metric == "perplexity" else ACCURACY_KEYS
    for raw_task, raw_value in raw_metrics.items():
        task = _normalize_task_name(str(raw_task))
        entry = _extract_metric_entry(raw_value, keys, metric=metric, higher_is_better=higher_is_better)
        if entry is not None:
            normalized[task] = entry
    return normalized


def _normalize_research_variant(name: str, payload: dict[str, Any], source_path: Path) -> dict[str, Any]:
    variant = _empty_research_variant(str(source_path))
    if isinstance(payload.get("config"), dict):
        variant["config"].update(payload["config"])
    for key in ("model_id", "evaluator", "date", "seed", "num_fewshot"):
        if key in payload:
            variant["config"][key] = payload[key]

    if isinstance(payload.get("results"), dict):
        _normalize_lm_eval_results(payload["results"], variant)

    if isinstance(payload.get("perplexity"), dict):
        variant["perplexity"].update(
            _normalize_named_metrics(
                payload["perplexity"],
                category="perplexity",
                metric="perplexity",
                higher_is_better=False,
            )
        )
    if isinstance(payload.get("zero_shot"), dict):
        variant["zero_shot"].update(
            _normalize_named_metrics(
                payload["zero_shot"],
                category="zero_shot",
                metric="accuracy",
                higher_is_better=True,
            )
        )
    if isinstance(payload.get("downstream_accuracy"), dict):
        variant["downstream_accuracy"].update(
            _normalize_named_metrics(
                payload["downstream_accuracy"],
                category="downstream_accuracy",
                metric="accuracy",
                higher_is_better=True,
            )
        )
    if isinstance(payload.get("efficiency"), dict):
        variant["efficiency"].update(payload["efficiency"])

    variant["name"] = name
    return variant


def _load_research_eval_variants(paths: list[Path]) -> dict[str, dict[str, Any]]:
    variants: dict[str, dict[str, Any]] = {}
    for path in paths:
        blob = _load_json(path)
        if isinstance(blob.get("variants"), dict):
            for raw_name, payload in blob["variants"].items():
                if isinstance(payload, dict):
                    name = str(raw_name)
                    normalized = _normalize_research_variant(name, payload, path)
                    variants[name] = _merge_variant(variants.get(name, _empty_research_variant()), normalized)
            continue

        name = str(blob.get("variant") or _infer_variant_from_path(path))
        normalized = _normalize_research_variant(name, blob, path)
        variants[name] = _merge_variant(variants.get(name, _empty_research_variant()), normalized)
    return variants


def _select_variant(variants: dict[str, dict[str, Any]], kind: str) -> str | None:
    if not variants:
        return None
    keywords = {
        "baseline": ("baseline", "fp16", "float", "full_precision", "base"),
        "prism": ("prism", "quant", "mixed", "w2", "w3", "w4"),
    }[kind]
    for name in variants:
        if any(keyword in name.lower() for keyword in keywords):
            return name
    names = list(variants)
    if kind == "baseline" and len(names) >= 2:
        return names[0]
    if kind == "prism" and len(names) >= 2:
        return names[1]
    return None


def _lookup_research_metric(variant: dict[str, Any] | None, requirement: dict[str, Any]) -> dict[str, Any] | None:
    if variant is None:
        return None
    task = requirement["task"]
    category = requirement["category"]
    candidates = [variant.get(category, {}).get(task)]
    if category == "zero_shot":
        candidates.append(variant.get("downstream_accuracy", {}).get(task))
    for candidate in candidates:
        if isinstance(candidate, dict) and _is_finite(candidate.get("value")):
            return candidate
    return None


def _coverage_status(has_baseline: bool, has_prism: bool) -> str:
    if has_baseline and has_prism:
        return "present"
    if has_baseline:
        return "missing_prism"
    if has_prism:
        return "missing_baseline"
    return "missing_both"


def _build_research_evaluation(eval_results_paths: list[Path]) -> dict[str, Any]:
    variants = _load_research_eval_variants(eval_results_paths) if eval_results_paths else {}
    baseline_name = _select_variant(variants, "baseline")
    prism_name = _select_variant(variants, "prism")
    baseline = variants.get(baseline_name) if baseline_name else None
    prism = variants.get(prism_name) if prism_name else None

    coverage: dict[str, dict[str, Any]] = {}
    comparison: list[dict[str, Any]] = []
    for requirement in REQUIRED_RESEARCH_METRICS:
        baseline_metric = _lookup_research_metric(baseline, requirement)
        prism_metric = _lookup_research_metric(prism, requirement)
        has_baseline = baseline_metric is not None
        has_prism = prism_metric is not None
        coverage[requirement["id"]] = {
            "display": requirement["display"],
            "category": requirement["category"],
            "task": requirement["task"],
            "metric": requirement["metric"],
            "baseline_present": has_baseline,
            "prism_present": has_prism,
            "status": _coverage_status(has_baseline, has_prism),
        }
        if has_baseline and has_prism:
            baseline_value = float(baseline_metric["value"])
            prism_value = float(prism_metric["value"])
            delta = prism_value - baseline_value
            comparison.append(
                {
                    "id": requirement["id"],
                    "task": requirement["task"],
                    "display": requirement["display"],
                    "metric": requirement["metric"],
                    "higher_is_better": requirement["higher_is_better"],
                    "baseline": baseline_value,
                    "prism": prism_value,
                    "delta": delta,
                    "relative_change_pct": (100.0 * delta / baseline_value) if baseline_value else None,
                    "prism_better": delta > 0 if requirement["higher_is_better"] else delta < 0,
                }
            )

    return {
        "provided": bool(eval_results_paths),
        "input_paths": [str(path) for path in eval_results_paths],
        "baseline_variant": baseline_name,
        "prism_variant": prism_name,
        "variants": variants,
        "coverage": coverage,
        "comparison": comparison,
        "required_metric_ids": [item["id"] for item in REQUIRED_RESEARCH_METRICS],
        "notes": [
            "Perplexity is lower-is-better; accuracy/exact-match metrics are higher-is-better.",
            "For paper claims, report both FP16/baseline and PRISM on the same evaluator, seed, prompt template, and few-shot setting.",
        ],
    }


def _choose_assignment_path(run_dir: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    quic_paths = sorted(run_dir.glob("quic_assignment_*.json"), key=lambda path: path.stat().st_mtime)
    if quic_paths:
        return quic_paths[-1]
    assignment_paths = sorted(run_dir.glob("assignment_*.json"), key=lambda path: path.stat().st_mtime)
    if assignment_paths:
        return assignment_paths[-1]
    raise FileNotFoundError(f"No assignment_*.json or quic_assignment_*.json found under {run_dir}")


def _summarize_assignment(
    assignment: dict[str, Any],
    layers_by_name: dict[str, dict[str, Any]],
    *,
    baseline_bits: float,
) -> dict[str, Any]:
    bits = {str(name): int(bit) for name, bit in assignment.get("bits", {}).items()}
    bit_counts = {str(bit): 0 for bit in (2, 3, 4)}
    param_bits = {str(bit): 0 for bit in (2, 3, 4)}
    module_counts: dict[str, Counter[str]] = defaultdict(Counter)

    weight_bits = 0.0
    total_params = 0
    missing_profile_layers: list[str] = []
    for layer_name, bit in bits.items():
        bit_key = str(bit)
        bit_counts.setdefault(bit_key, 0)
        param_bits.setdefault(bit_key, 0)
        bit_counts[bit_key] += 1
        layer = layers_by_name.get(layer_name)
        if layer is None:
            missing_profile_layers.append(layer_name)
            continue
        params = int(layer.get("num_params", 0))
        total_params += params
        param_bits[bit_key] += params
        weight_bits += float(bit) * params
        module_counts[str(layer.get("module_type", "unknown"))][bit_key] += 1

    memory_cost_bits = _safe_float(assignment.get("memory_cost_bits"))
    if memory_cost_bits <= 0:
        memory_cost_bits = sum(
            _safe_float(layers_by_name[name].get("memory_cost_by_bit", {}).get(str(bit)))
            for name, bit in bits.items()
            if name in layers_by_name
        )

    fp_baseline_bits = float(total_params) * float(baseline_bits)
    compression = fp_baseline_bits / memory_cost_bits if memory_cost_bits > 0 else None
    weight_only_average_bits = weight_bits / total_params if total_params else 0.0

    return {
        "path": "",
        "budget": assignment.get("budget"),
        "average_bits": _safe_float(assignment.get("average_bits"), weight_only_average_bits),
        "average_memory_bits_per_param": _safe_float(assignment.get("average_memory_bits_per_param")),
        "memory_cost_bits": memory_cost_bits,
        "budget_memory_bits": _safe_float(assignment.get("budget_memory_bits")),
        "weight_only_average_bits": weight_only_average_bits,
        "linear_fp_baseline_bits": fp_baseline_bits,
        "linear_compression_ratio_vs_fp": compression,
        "bit_counts": {str(bit): bit_counts.get(str(bit), 0) for bit in (2, 3, 4)},
        "param_counts_by_bit": {str(bit): param_bits.get(str(bit), 0) for bit in (2, 3, 4)},
        "module_bit_counts": {
            module: {str(bit): counter.get(str(bit), 0) for bit in (2, 3, 4)}
            for module, counter in sorted(module_counts.items())
        },
        "solver": assignment.get("solver", {}),
        "missing_profile_layers": missing_profile_layers,
    }


def _summarize_profile(profile: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    layers = list(profile.get("layers", []))
    layers_by_name = {str(layer["layer_name"]): layer for layer in layers if "layer_name" in layer}
    scores = [_safe_float(layer.get("adjusted_score")) for layer in layers if _is_finite(layer.get("adjusted_score"))]
    nonfinite_score_layers = [
        str(layer.get("layer_name"))
        for layer in layers
        if not _is_finite(layer.get("adjusted_score")) or not _is_finite(layer.get("raw_score"))
    ]
    nonfinite_sensitivity_layers = []
    for layer in layers:
        sens = layer.get("sensitivity_by_bit", {})
        if any(not _is_finite(value) for value in sens.values()):
            nonfinite_sensitivity_layers.append(str(layer.get("layer_name")))

    module_type_counts = Counter(str(layer.get("module_type", "unknown")) for layer in layers)
    top_sensitive = sorted(
        (
            {
                "layer_name": str(layer.get("layer_name")),
                "module_type": str(layer.get("module_type", "unknown")),
                "adjusted_score": _safe_float(layer.get("adjusted_score")),
            }
            for layer in layers
            if _is_finite(layer.get("adjusted_score"))
        ),
        key=lambda item: item["adjusted_score"],
        reverse=True,
    )[:10]

    summary = {
        "model_id": profile.get("model_id"),
        "model_family": profile.get("model_family"),
        "layer_count": len(layers),
        "total_params": sum(int(layer.get("num_params", 0)) for layer in layers),
        "module_type_counts": dict(sorted(module_type_counts.items())),
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "score_mean": sum(scores) / len(scores) if scores else None,
        "top_sensitive_layers": top_sensitive,
        "nonfinite_score_layers": nonfinite_score_layers,
        "nonfinite_sensitivity_layers": nonfinite_sensitivity_layers,
    }
    return summary, layers_by_name


def _summarize_runtime(runtime: dict[str, Any]) -> dict[str, Any]:
    backends = runtime.get("backend_by_layer", {})
    return {
        "backend_counts": dict(sorted(Counter(str(value) for value in backends.values()).items())),
        "generated_text": runtime.get("generated_text"),
        "has_generated_text": bool(runtime.get("generated_text")),
        "layer_count": len(backends),
    }


def _summarize_assignments_sweep(
    assignment_paths: list[Path],
    layers_by_name: dict[str, dict[str, Any]],
    *,
    baseline_bits: float,
) -> list[dict[str, Any]]:
    rows = []
    for path in assignment_paths:
        assignment = _load_json(path)
        if "bits" not in assignment:
            continue
        row = _summarize_assignment(assignment, layers_by_name, baseline_bits=baseline_bits)
        row["path"] = str(path)
        rows.append(row)
    return sorted(rows, key=lambda item: _safe_float(item.get("budget")))


def summarize_run(
    run_dir: str | Path,
    *,
    assignment_path: str | Path | None = None,
    baseline_bits: float = 16.0,
    eval_results_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    profile_path = run_dir / "profile.json"
    runtime_path = run_dir / "runtime_summary.json"
    manifest_path = run_dir / "rtn" / "manifest.json"
    selected_assignment_path = _choose_assignment_path(
        run_dir,
        Path(assignment_path) if assignment_path is not None else None,
    )

    profile = _load_json(profile_path)
    profile_summary, layers_by_name = _summarize_profile(profile)
    final_assignment = _summarize_assignment(
        _load_json(selected_assignment_path),
        layers_by_name,
        baseline_bits=baseline_bits,
    )
    final_assignment["path"] = str(selected_assignment_path)

    runtime = _summarize_runtime(_load_json(runtime_path)) if runtime_path.exists() else {}
    manifest = _load_json(manifest_path) if manifest_path.exists() else {}
    sweep = _summarize_assignments_sweep(
        _find_assignment_paths(run_dir),
        layers_by_name,
        baseline_bits=baseline_bits,
    )

    missing_runtime_layers = []
    runtime_layers = set((runtime or {}).get("backend_counts", {}).keys())
    del runtime_layers
    if runtime_path.exists():
        runtime_raw = _load_json(runtime_path)
        backend_layers = set(runtime_raw.get("backend_by_layer", {}).keys())
        assigned_layers = set(_load_json(selected_assignment_path).get("bits", {}).keys())
        missing_runtime_layers = sorted(assigned_layers - backend_layers)

    manifest_model_id = manifest.get("model_id")
    manifest_layer_count = len(manifest.get("layers", {}))
    artifact_size_bytes = _dir_size_bytes(run_dir / "rtn")
    profile_model_id = profile_summary["model_id"]
    research_evaluation = _build_research_evaluation(
        [Path(path) for path in eval_results_paths] if eval_results_paths else []
    )
    summary = {
        "run_dir": str(run_dir),
        "model_id": profile_model_id,
        "model_family": profile_summary["model_family"],
        "layer_count": profile_summary["layer_count"],
        "total_params": profile_summary["total_params"],
        "profile": profile_summary,
        "final_assignment": final_assignment,
        "assignment_sweep": sweep,
        "runtime": runtime,
        "research_evaluation": research_evaluation,
        "rtn_manifest": {
            "exists": manifest_path.exists(),
            "model_id": manifest_model_id,
            "group_size": manifest.get("group_size"),
            "layer_count": manifest_layer_count,
            "artifact_size_bytes": artifact_size_bytes,
            "artifact_size_human": _format_bytes(artifact_size_bytes),
            "note": "RTN directory stores all precomputed bit variants; it is not the final deployed model size.",
        },
        "quality_checks": {
            "has_nonfinite_scores": bool(profile_summary["nonfinite_score_layers"]),
            "has_nonfinite_sensitivity": bool(profile_summary["nonfinite_sensitivity_layers"]),
            "manifest_model_matches_profile": manifest_model_id == profile_model_id,
            "manifest_layer_count_matches_profile": manifest_layer_count == profile_summary["layer_count"],
            "missing_profile_layers_in_assignment": final_assignment["missing_profile_layers"],
            "missing_runtime_layers": missing_runtime_layers,
            "has_runtime_summary": runtime_path.exists(),
            "has_rtn_manifest": manifest_path.exists(),
        },
    }
    return summary


def _pct(value: float, denominator: float) -> float:
    return 100.0 * float(value) / float(denominator) if denominator else 0.0


def _format_number(value: Any, precision: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)) and _is_finite(value):
        return f"{float(value):.{precision}f}"
    return str(value)


def _format_present(value: bool) -> str:
    return "yes" if value else "missing"


def _render_research_evaluation(summary: dict[str, Any]) -> list[str]:
    research = summary.get("research_evaluation", {})
    lines = [
        "",
        "## Research Evaluation",
        "",
        f"- Eval result files provided: `{research.get('provided', False)}`",
        f"- Baseline variant: `{research.get('baseline_variant')}`",
        f"- PRISM variant: `{research.get('prism_variant')}`",
    ]
    input_paths = research.get("input_paths", [])
    if input_paths:
        lines.append(f"- Eval result paths: `{', '.join(input_paths)}`")
    else:
        lines.append("- Eval result paths: `none`")

    lines.extend(
        [
            "",
            "### Required Metric Coverage",
            "",
            "| Metric | Baseline | PRISM | Status |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for requirement in REQUIRED_RESEARCH_METRICS:
        row = research.get("coverage", {}).get(requirement["id"], {})
        lines.append(
            f"| {requirement['display']} | {_format_present(bool(row.get('baseline_present')))} | "
            f"{_format_present(bool(row.get('prism_present')))} | `{row.get('status', 'missing_both')}` |"
        )

    comparison = research.get("comparison", [])
    lines.extend(
        [
            "",
            "### Baseline vs PRISM",
            "",
            "| Metric | Direction | Baseline | PRISM | Delta | Relative change | PRISM better |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    if comparison:
        for row in comparison:
            direction = "higher is better" if row.get("higher_is_better") else "lower is better"
            relative = row.get("relative_change_pct")
            relative_text = f"{relative:.2f}%" if relative is not None else ""
            lines.append(
                f"| {row['display']} | {direction} | {_format_number(row.get('baseline'))} | "
                f"{_format_number(row.get('prism'))} | {_format_number(row.get('delta'))} | "
                f"{relative_text} | `{row.get('prism_better')}` |"
            )
    else:
        lines.append("| No comparable baseline/PRISM metric pairs yet |  |  |  |  |  |  |")

    variants = research.get("variants", {})
    if variants:
        lines.extend(["", "### Per-Variant Metrics", ""])
        for variant_name, variant in variants.items():
            lines.extend([f"#### {variant_name}", ""])
            perplexity = variant.get("perplexity", {})
            if perplexity:
                lines.extend(["Perplexity:", "", "| Dataset | Value | Metric |", "| --- | ---: | --- |"])
                for task, metric in sorted(perplexity.items()):
                    lines.append(f"| {task} | {_format_number(metric.get('value'))} | {metric.get('metric', '')} |")
                lines.append("")
            zero_shot = variant.get("zero_shot", {})
            downstream = variant.get("downstream_accuracy", {})
            combined_accuracy = {**downstream, **zero_shot}
            if combined_accuracy:
                lines.extend(["Zero-shot / downstream accuracy:", "", "| Benchmark | Value | Metric |", "| --- | ---: | --- |"])
                for task, metric in sorted(combined_accuracy.items()):
                    lines.append(f"| {task} | {_format_number(metric.get('value'))} | {metric.get('metric', '')} |")
                lines.append("")
            efficiency = variant.get("efficiency", {})
            if efficiency:
                lines.extend(["Efficiency:", "", "| Metric | Value |", "| --- | ---: |"])
                for key, value in sorted(efficiency.items()):
                    lines.append(f"| {key} | {_format_number(value)} |")
                lines.append("")
    else:
        lines.extend(
            [
                "",
                "No research evaluation results were attached. Add JSON files with `--eval-results` to report perplexity, zero-shot accuracy, downstream accuracy, MMLU, GSM8K, ARC, and HellaSwag.",
            ]
        )

    lines.extend(
        [
            "",
            "Recommended paper table columns: model, method, average bits, compression, WikiText-2 PPL, C4 PPL, MMLU, GSM8K, ARC-Challenge, ARC-Easy, HellaSwag, tokens/sec, peak memory.",
        ]
    )
    return lines


def render_markdown(summary: dict[str, Any]) -> str:
    final_assignment = summary["final_assignment"]
    total_params = int(summary["total_params"])
    lines = [
        "# PRISM Run Summary",
        "",
        f"- Run dir: `{summary['run_dir']}`",
        f"- Model: `{summary.get('model_id')}`",
        f"- Quantized linear layers: `{summary['layer_count']}`",
        f"- Quantized linear params: `{total_params:,}`",
        "",
        "## Final Assignment",
        "",
        f"- Assignment file: `{final_assignment['path']}`",
        f"- Target budget: `{final_assignment.get('budget')}`",
        f"- Average weight bits: `{final_assignment['average_bits']:.4f}`",
        f"- Average memory bits/param: `{final_assignment['average_memory_bits_per_param']:.4f}`",
        f"- Linear FP baseline bits: `{final_assignment['linear_fp_baseline_bits']:.0f}`",
        f"- Assigned memory bits: `{final_assignment['memory_cost_bits']:.0f}`",
        f"- Compression vs FP{int(final_assignment['linear_fp_baseline_bits'] / total_params) if total_params else 16} linear weights: `{final_assignment['linear_compression_ratio_vs_fp']:.3f}x`",
        "",
        "| Bit | Layers | Params | Param % |",
        "| --- | ---: | ---: | ---: |",
    ]
    for bit in ("2", "3", "4"):
        params = int(final_assignment["param_counts_by_bit"].get(bit, 0))
        lines.append(
            f"| {bit} | {final_assignment['bit_counts'].get(bit, 0)} | {params:,} | {_pct(params, total_params):.2f}% |"
        )

    lines.extend(["", "## Backend", "", "| Backend | Layers |", "| --- | ---: |"])
    for backend, count in summary.get("runtime", {}).get("backend_counts", {}).items():
        lines.append(f"| {backend} | {count} |")

    generated_text = summary.get("runtime", {}).get("generated_text")
    if generated_text:
        lines.extend(["", "## Generated Text", "", "```text", str(generated_text), "```"])

    lines.extend(["", "## Assignment Sweep", "", "| File | Budget | Avg bits | Avg memory bits/param | Compression |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in summary.get("assignment_sweep", []):
        compression = row.get("linear_compression_ratio_vs_fp")
        compression_text = f"{compression:.3f}x" if compression is not None else ""
        lines.append(
            f"| `{Path(row['path']).name}` | {row.get('budget')} | {row['average_bits']:.4f} | {row['average_memory_bits_per_param']:.4f} | {compression_text} |"
        )

    lines.extend(_render_research_evaluation(summary))

    checks = summary["quality_checks"]
    lines.extend(
        [
            "",
            "## Quality Checks",
            "",
            f"- Non-finite scores: `{checks['has_nonfinite_scores']}`",
            f"- Non-finite sensitivity: `{checks['has_nonfinite_sensitivity']}`",
            f"- RTN manifest model: `{summary['rtn_manifest'].get('model_id')}`",
            f"- RTN manifest model matches profile: `{checks['manifest_model_matches_profile']}`",
            f"- RTN manifest layer count matches profile: `{checks['manifest_layer_count_matches_profile']}`",
            f"- Runtime summary exists: `{checks['has_runtime_summary']}`",
            f"- RTN manifest exists: `{checks['has_rtn_manifest']}`",
            f"- RTN artifact directory size: `{summary['rtn_manifest']['artifact_size_human']}`",
            "",
            "Note: artifact metadata is computed from the PRISM run. Research metrics are reported only when external evaluation JSON files are attached with `--eval-results`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize a completed PRISM pipeline run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--assignment-path", default=None)
    parser.add_argument("--baseline-bits", type=float, default=16.0)
    parser.add_argument(
        "--eval-results",
        nargs="*",
        default=None,
        help="Optional research evaluation JSON files for baseline/PRISM perplexity and zero-shot metrics.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--quiet", action="store_true", help="Only print output paths, not the full JSON summary.")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    summary = summarize_run(
        run_dir,
        assignment_path=Path(args.assignment_path) if args.assignment_path else None,
        baseline_bits=args.baseline_bits,
        eval_results_paths=[Path(path) for path in args.eval_results] if args.eval_results else None,
    )
    output_json = Path(args.output_json) if args.output_json else run_dir / "summary_stats.json"
    output_md = Path(args.output_md) if args.output_md else run_dir / "summary_stats.md"
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(render_markdown(summary), encoding="utf-8")
    if not args.quiet:
        print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nWrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
