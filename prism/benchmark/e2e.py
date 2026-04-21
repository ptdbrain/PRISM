"""End-to-End benchmarking script for PRISM framework.

Measures and compares:
1. Search Time (Profiling + Assignment) vs AMQ.
2. Memory Footprint (FP16 Base vs Quantized PRISM).
3. Tokens Per Second (TPS).
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from prism.api import PRISM
from prism.benchmark.speed import benchmark_tps, cleanup_memory, get_memory_footprint

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("PRISM-Benchmark")


def main():
    parser = argparse.ArgumentParser(description="PRISM End-to-End Benchmark")
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model ID or path")
    parser.add_argument("--target_bits", type=float, default=4.0, help="Target average bit-width")
    parser.add_argument("--dataset", type=str, default="wikitext2", help="Calibration dataset")
    parser.add_argument("--output_file", type=str, default="prism_benchmark.json", help="Result JSON file")
    
    # TPS params
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prompt_len", type=int, default=64)
    parser.add_argument("--gen_len", type=int, default=128)
    parser.add_argument("--skip_baseline_tps", action="store_true", help="Skip running TPS on unquantized model")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    report = {
        "model_id": args.model_id,
        "target_bits": args.target_bits,
        "device": device,
        "metrics": {}
    }

    # =========================================================================
    # Phase 1: Search Time & Compression (PRISM Pipeline)
    # =========================================================================
    logger.info("=== Phase 1: PRISM Quantization Pipeline ===")
    
    # We measure how long it takes to profile and assign bits
    start_search = time.perf_counter()
    
    # Initialize PRISM
    # We use a dummy temp directory for benchmarking
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        quantizer = PRISM(
            model_id=args.model_id,
            dataset=args.dataset,
            artifact_dir=tmp_path,
        )
        
        # Run End-to-End Pipeline
        quantized_model = quantizer.run(target_bits=args.target_bits)
        
        search_time = time.perf_counter() - start_search
        report["metrics"]["search_time_seconds"] = round(search_time, 2)
        logger.info("Search & Packing Time: %.2f seconds", search_time)

        # Print model backend distribution
        logger.info("Backend Summary: %s", quantized_model.backend_summary)
        report["backend_summary"] = quantized_model.backend_summary

        # Move to GPU for benchmarking
        quantized_model = quantized_model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # =========================================================================
        # Phase 2: Memory & TPS for Quantized Model
        # =========================================================================
        logger.info("=== Phase 2: Benchmarking Quantized Model ===")
        q_mem = get_memory_footprint(quantized_model)
        report["metrics"]["quantized_memory_gb"] = round(q_mem, 2)
        logger.info("Quantized Model Memory: %.2f GB", q_mem)

        logger.info("Warming up TPS benchmark...")
        q_tps_res = benchmark_tps(
            quantized_model, tokenizer,
            batch_size=args.batch_size, prompt_len=args.prompt_len,
            gen_len=args.gen_len, device=device
        )
        report["metrics"]["quantized_tps"] = q_tps_res
        logger.info("Quantized TPS: %.2f tokens/sec", q_tps_res["TPS"])

        # Release VRAM
        del quantized_model
        del quantizer
        cleanup_memory()

    # =========================================================================
    # Phase 3: Baseline Memory & TPS (FP16)
    # =========================================================================
    logger.info("=== Phase 3: Benchmarking FP16 Baseline ===")
    logger.info("Loading original FP16 model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    base_model.eval()

    base_mem = get_memory_footprint(base_model)
    report["metrics"]["base_memory_gb"] = round(base_mem, 2)
    logger.info("Baseline Model Memory: %.2f GB", base_mem)
    
    if not args.skip_baseline_tps:
        logger.info("Warming up Baseline TPS benchmark...")
        b_tps_res = benchmark_tps(
            base_model, tokenizer,
            batch_size=args.batch_size, prompt_len=args.prompt_len,
            gen_len=args.gen_len, device=device
        )
        report["metrics"]["base_tps"] = b_tps_res
        logger.info("Baseline TPS: %.2f tokens/sec", b_tps_res["TPS"])
    else:
        report["metrics"]["base_tps"] = None

    del base_model
    cleanup_memory()

    # =========================================================================
    # Phase 4: Output Report
    # =========================================================================
    logger.info("=== Benchmark Report ===")
    print(json.dumps(report, indent=4))
    
    out_path = Path(args.output_file)
    with out_path.open("w") as f:
        json.dump(report, f, indent=4)
    logger.info("Report saved to %s", out_path.resolve())


if __name__ == "__main__":
    main()
