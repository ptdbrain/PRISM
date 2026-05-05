#!/usr/bin/env python3
"""Generic PRISM multi-GPU worker for Kaggle notebook orchestration."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("sensitivity_dataset", "profile_artifact", "rtn_precompute"))
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--bits", default="2,3,4")
    parser.add_argument("--num-shards", type=int, default=0)
    parser.add_argument("--shards", default="")
    parser.add_argument("--layer-names-json", default="")
    parser.add_argument("--mlp-path", default="")
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    return parser.parse_args()


def _resolve_torch_dtype(value: str):
    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(str(value).strip().lower(), torch.float16)


def _load_selected_layers(path: str) -> list[str] | None:
    if not path:
        return None
    return list(json.loads(Path(path).read_text(encoding="utf-8")))


def _patch_layer_iterator(selected_layers: list[str] | None, num_shards: int, shard_id: int):
    import prism.profile.inspect as inspect_mod
    import prism.profiling.meta_learner as meta_learner_mod
    import prism.profiling.sensitivity as sensitivity_mod

    original_iter = inspect_mod.iter_named_linear_layers

    def filtered_iter(model, adapter=None):
        items = list(original_iter(model, adapter=adapter))
        if selected_layers is not None:
            allow = set(selected_layers)
            for item in items:
                if item[0] in allow:
                    yield item
            return
        if num_shards > 0:
            for idx, item in enumerate(items):
                if idx % num_shards == shard_id:
                    yield item
            return
        yield from items

    inspect_mod.iter_named_linear_layers = filtered_iter
    sensitivity_mod.iter_named_linear_layers = filtered_iter
    meta_learner_mod.iter_named_linear_layers = filtered_iter
    return inspect_mod, sensitivity_mod, meta_learner_mod, original_iter


def _restore_layer_iterator(inspect_mod, sensitivity_mod, meta_learner_mod, original_iter) -> None:
    inspect_mod.iter_named_linear_layers = original_iter
    sensitivity_mod.iter_named_linear_layers = original_iter
    meta_learner_mod.iter_named_linear_layers = original_iter


def _patch_eval_sample_budget(n_samples: int, seq_len: int):
    import prism.profiling.sensitivity_eval as sensitivity_eval_mod

    original_prepare = sensitivity_eval_mod.prepare_wikitext2_input_ids

    def small_prepare(tokenizer_id, n_samples=n_samples, seq_len=seq_len):
        return original_prepare(tokenizer_id, n_samples=n_samples, seq_len=seq_len)

    sensitivity_eval_mod.prepare_wikitext2_input_ids = small_prepare
    return sensitivity_eval_mod, original_prepare


def _restore_eval_sample_budget(sensitivity_eval_mod, original_prepare) -> None:
    sensitivity_eval_mod.prepare_wikitext2_input_ids = original_prepare


def _load_model(model_name: str, torch_dtype):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def _run_sensitivity_dataset(args: argparse.Namespace, torch_dtype) -> None:
    from prism.profiling.sensitivity import build_training_dataset

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_ids = [int(s) for s in args.shards.split(",") if s]

    sensitivity_eval_mod, original_prepare = _patch_eval_sample_budget(args.n_samples, args.seq_len)
    try:
        for shard_id in shard_ids:
            inspect_mod, sensitivity_mod, meta_learner_mod, original_iter = _patch_layer_iterator(
                selected_layers=None,
                num_shards=args.num_shards,
                shard_id=shard_id,
            )
            try:
                target = out_dir / f"train_shard_{shard_id:02d}_of_{args.num_shards}.pt"
                if target.exists():
                    print(f"[GPU {args.gpu}] skip existing shard {shard_id}", flush=True)
                    continue
                print(f"[GPU {args.gpu}] build shard {shard_id}", flush=True)
                build_training_dataset(
                    model_name_list=[args.model],
                    bits_list=[int(x) for x in args.bits.split(",") if x],
                    group_size=args.group_size,
                    save_path=str(target),
                    show_progress=True,
                )
                print(f"[GPU {args.gpu}] saved {target.name}", flush=True)
            finally:
                _restore_layer_iterator(inspect_mod, sensitivity_mod, meta_learner_mod, original_iter)
    finally:
        _restore_eval_sample_budget(sensitivity_eval_mod, original_prepare)


def _run_profile_artifact(args: argparse.Namespace, torch_dtype, selected_layers: list[str]) -> None:
    from prism.profile.pipeline import build_profile_artifact

    if not args.mlp_path:
        raise ValueError("--mlp-path is required for stage=profile_artifact")

    model = _load_model(args.model, torch_dtype)
    inspect_mod, sensitivity_mod, meta_learner_mod, original_iter = _patch_layer_iterator(
        selected_layers=selected_layers,
        num_shards=0,
        shard_id=0,
    )
    try:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"profile_gpu{args.gpu}.json"
        build_profile_artifact(
            model,
            mlp_path=Path(args.mlp_path),
            output_path=output_path,
            model_id=args.model,
            model_family="unknown",
            group_size=args.group_size,
        )
        print(f"[GPU {args.gpu}] saved {output_path}", flush=True)
    finally:
        _restore_layer_iterator(inspect_mod, sensitivity_mod, meta_learner_mod, original_iter)


def _run_rtn_precompute(args: argparse.Namespace, torch_dtype, selected_layers: list[str]) -> None:
    from prism.rtn.precompute import precompute_model_rtn

    model = _load_model(args.model, torch_dtype)
    inspect_mod, sensitivity_mod, meta_learner_mod, original_iter = _patch_layer_iterator(
        selected_layers=selected_layers,
        num_shards=0,
        shard_id=0,
    )
    try:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stage_dir = out_dir / f"gpu{args.gpu}"
        precompute_model_rtn(
            model,
            output_dir=stage_dir,
            group_size=args.group_size,
            bits=tuple(int(x) for x in args.bits.split(",") if x),
        )
        print(f"[GPU {args.gpu}] saved RTN artifacts in {stage_dir}", flush=True)
    finally:
        _restore_layer_iterator(inspect_mod, sensitivity_mod, meta_learner_mod, original_iter)


def main() -> None:
    args = _parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, args.repo_dir)
    os.chdir(args.repo_dir)

    import torch

    torch_dtype = _resolve_torch_dtype(args.torch_dtype)
    selected_layers = _load_selected_layers(args.layer_names_json)

    importlib.invalidate_caches()

    if args.stage == "sensitivity_dataset":
        _run_sensitivity_dataset(args, torch_dtype)
    elif args.stage == "profile_artifact":
        _run_profile_artifact(args, torch_dtype, selected_layers or [])
    else:
        _run_rtn_precompute(args, torch_dtype, selected_layers or [])

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
