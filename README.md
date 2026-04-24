# PRISM

PRISM is a zero-cost and largely data-free mixed-precision quantization pipeline with:

- Stage 0 meta-learner training
- Stage 1 zero-cost profiling
- Stage 2 budgeted LP-style bit assignment
- Stage 2.5 QUIC output-space correction
- Stage 3 RTN precomputation
- Stage 4 runtime assembly with Marlin preference and PyTorch GEMM fallback

## Demo

```bash
prism-demo --output-root artifacts/prism/demo
```

## Real Model Workflow

Train or prepare a meta-learner checkpoint:

```bash
prism-train-meta --output-dir artifacts/prism/meta
```

Profile a real Hugging Face model:

```bash
prism-profile \
  --model-id-or-path meta-llama/Llama-2-7b-hf \
  --family llama \
  --mlp-path artifacts/prism/meta/prism_mlp.pt \
  --output-path artifacts/prism/profile.json
```

Precompute RTN artifacts:

```bash
prism-precompute-rtn \
  --model-id-or-path meta-llama/Llama-2-7b-hf \
  --family llama \
  --output-dir artifacts/prism/rtn
```

Assemble and optionally execute the runtime:

```bash
prism-run \
  --model-id-or-path meta-llama/Llama-2-7b-hf \
  --family llama \
  --artifact-root artifacts/prism/rtn \
  --assignment-path artifacts/prism/quic_assignment.json \
  --execute
```
