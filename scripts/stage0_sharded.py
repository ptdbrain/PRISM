"""Build and merge sharded PRISM Stage 0 sensitivity datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import prism.profile.inspect as inspect_mod
import prism.profiling.meta_learner as meta_learner_mod
import prism.profiling.sensitivity as sensitivity_mod
from prism.profiling.meta_learner import train_meta_learner
from prism.profiling.sensitivity import build_training_dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sharded PRISM Stage 0 dataset generation.")
    parser.add_argument("--models", required=True, help="Comma-separated model names.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--end-shard", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-train-until-complete", action="store_true", default=True)
    return parser.parse_args()


def _patch_shard_iterator(num_shards: int, shard_id: int):
    original_iter = inspect_mod.iter_named_linear_layers

    def shard_iter(model, adapter=None):
        items = list(original_iter(model, adapter=adapter))
        for idx, item in enumerate(items):
            if idx % num_shards == shard_id:
                yield item

    inspect_mod.iter_named_linear_layers = shard_iter
    sensitivity_mod.iter_named_linear_layers = shard_iter
    meta_learner_mod.iter_named_linear_layers = shard_iter
    return original_iter


def _restore_iterator(original_iter) -> None:
    inspect_mod.iter_named_linear_layers = original_iter
    sensitivity_mod.iter_named_linear_layers = original_iter
    meta_learner_mod.iter_named_linear_layers = original_iter


def _merge_and_train(out_dir: Path, num_shards: int, epochs: int) -> None:
    paths = [out_dir / f"train_shard_{idx:02d}_of_{num_shards}.pt" for idx in range(num_shards)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        print("Stage 0 shards are incomplete; skipping merge/train.")
        print("Missing shards:")
        for path in missing:
            print(f"  {path}")
        return

    xs, ys, metas = [], [], []
    feature_order = None
    for path in paths:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        xs.append(bundle["X"].cpu())
        ys.append(bundle["Y"].cpu())
        metas.extend(bundle["meta"])
        feature_order = bundle.get("feature_order", feature_order)

    merged = {
        "X": torch.cat(xs, dim=0),
        "Y": torch.cat(ys, dim=0),
        "meta": metas,
        "feature_order": feature_order,
    }
    merged_path = out_dir / "prism_train_data.pt"
    torch.save(merged, merged_path)
    train_meta_learner(str(merged_path), epochs=epochs, save_path=str(out_dir / "prism_mlp.pt"))
    print(
        {
            "dataset_path": str(merged_path),
            "mlp_path": str(out_dir / "prism_mlp.pt"),
            "records": int(merged["X"].shape[0]),
        }
    )


def main() -> None:
    args = _parse_args()
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    if not models:
        raise ValueError("--models must contain at least one model name.")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    end_shard = args.num_shards if args.end_shard is None else args.end_shard

    for shard_id in range(args.start_shard, end_shard):
        if shard_id < 0 or shard_id >= args.num_shards:
            raise ValueError(f"shard id {shard_id} is outside [0, {args.num_shards}).")
        shard_path = out_dir / f"train_shard_{shard_id:02d}_of_{args.num_shards}.pt"
        if args.skip_existing and shard_path.exists():
            print(f"skip existing {shard_path}")
            continue

        original_iter = _patch_shard_iterator(args.num_shards, shard_id)
        try:
            build_training_dataset(
                model_name_list=models,
                group_size=args.group_size,
                save_path=str(shard_path),
                show_progress=True,
            )
        finally:
            _restore_iterator(original_iter)

    _merge_and_train(out_dir, args.num_shards, args.epochs)


if __name__ == "__main__":
    main()
