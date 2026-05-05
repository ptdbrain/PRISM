#!/usr/bin/env python3
"""Worker script: processes a subset of shards on a specific GPU."""
import argparse, os, sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shards", type=str, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--repo-dir", type=str, required=True)
    args = parser.parse_args()

    # Set GPU BEFORE any CUDA init
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.path.insert(0, args.repo_dir)
    os.chdir(args.repo_dir)

    from pathlib import Path
    import torch, importlib
    import prism.profiling.sensitivity as sensitivity_mod
    import prism.profiling.sensitivity_eval as sensitivity_eval_mod
    import prism.profile.inspect as inspect_mod

    sensitivity_eval_mod = importlib.reload(sensitivity_eval_mod)
    inspect_mod = importlib.reload(inspect_mod)
    sensitivity_mod = importlib.reload(sensitivity_mod)

    # Patch: reduce eval data
    _orig_prepare = sensitivity_eval_mod.prepare_wikitext2_input_ids
    def small_prepare(tokenizer_id, n_samples=16, seq_len=512):
        return _orig_prepare(tokenizer_id, n_samples=16, seq_len=512)
    sensitivity_eval_mod.prepare_wikitext2_input_ids = small_prepare

    # Patch: filter layers by shard
    _orig_iter = inspect_mod.iter_named_linear_layers
    num_shards = args.num_shards
    _current_sid = [0]

    def shard_iter(model, adapter=None):
        layers = list(_orig_iter(model, adapter=adapter))
        for i, item in enumerate(layers):
            if i % num_shards == _current_sid[0]:
                yield item

    inspect_mod.iter_named_linear_layers = shard_iter
    sensitivity_mod.iter_named_linear_layers = shard_iter
    from prism.profiling.sensitivity import build_training_dataset

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_ids = [int(s) for s in args.shards.split(",")]

    print(f"[GPU {args.gpu}] Shards to process: {shard_ids}", flush=True)

    for sid in shard_ids:
        _current_sid[0] = sid
        path = out_dir / f"train_shard_{sid:02d}_of_{num_shards}.pt"
        if path.exists():
            print(f"[GPU {args.gpu}] Shard {sid} exists, skip.", flush=True)
            continue
        print(f"[GPU {args.gpu}] Building shard {sid}...", flush=True)
        build_training_dataset(
            model_name_list=[args.model],
            bits_list=[2, 3, 4],
            group_size=128,
            save_path=str(path),
            show_progress=True,
        )
        print(f"[GPU {args.gpu}] Saved shard {sid}", flush=True)

    inspect_mod.iter_named_linear_layers = _orig_iter
    sensitivity_mod.iter_named_linear_layers = _orig_iter
    print(f"[GPU {args.gpu}] All done!", flush=True)

if __name__ == "__main__":
    main()
