# PRISM

PRISM is a low-cost, deployment-time calibration-free mixed-precision quantization pipeline with:

- Stage 0 meta-learner training
- Stage 1 data-free profiling with layer/shape/outlier/RTN-error features
- Stage 2 budgeted discrete one-hot bit assignment under realistic memory accounting
- Stage 2.5 synthetic-only QUIC output-space correction
- Stage 3 RTN precomputation
- Stage 4 runtime assembly with Marlin preference and PyTorch GEMM fallback

PRISM-Atlas is the action-aware research path layered on top of the baseline pipeline. It predicts or scores a
response surface over `{bits, group_size, transform, backend, target}` actions and solves a risk-aware assignment
objective:

```text
mean_damage + lambda * uncertainty
```

The v1 action space supports `bits={2,3,4}`, `group_size={64,128}`, `transform={none,hadamard}`, `backend=rtn`,
and `target=weight`. Atlas assignments also emit a legacy-compatible `bits` map so existing RTN/runtime tooling
can still consume the result.

Atlas profile artifacts are versioned as `prism-atlas-profile-v2` / `ResponseSurfaceArtifactV2`; assignment
artifacts are `prism-atlas-assignment-v2` / `AssignmentV2`. Each layer/action response records:

- `mean_damage`, `uncertainty`, `ranking_score`
- analytic components `kappa`, `distortion`, and optional residual damage
- `memory_cost_bits`, `latency_proxy`
- `valid_action`, `runtime_supported`, `materialization_supported`, and `fallback_backend`

Scorer modes:

- `analytic`: `damage = kappa_l * D_l(action)`
- `learned`: response MLP predicts `mean_damage`, `log_variance`, and `ranking_score`
- `hybrid`: analytic score plus learned residual if a checkpoint is supplied

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
  --group-size 128 \
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

## PRISM-Atlas Workflow

Train an action-conditioned response surface:

```bash
prism-atlas-train \
  --output-dir artifacts/prism_atlas/stage0 \
  --model-names __prism_mock_ten_layers__ \
  --epochs 20
```

Profile a target model into a layer/action response surface:

```bash
prism-atlas-profile \
  --model-id-or-path meta-llama/Llama-2-7b-hf \
  --family llama \
  --atlas-path artifacts/prism_atlas/stage0/prism_atlas.pt \
  --scorer hybrid \
  --output-path artifacts/prism_atlas/profile.json
```

Solve a risk-aware assignment:

```bash
prism-atlas-assign \
  --profile-path artifacts/prism_atlas/profile.json \
  --budget 3.0 \
  --objective-mode risk \
  --risk-lambda 0.25 \
  --output-path artifacts/prism_atlas/assignment.json
```

Select only high-uncertainty layers for optional QUIC correction:

```bash
prism-atlas-quic-plan \
  --profile-path artifacts/prism_atlas/profile.json \
  --assignment-path artifacts/prism_atlas/assignment.json \
  --top-fraction 0.10 \
  --output-path artifacts/prism_atlas/quic_plan.json
```

Write the required benchmark, transfer, and repo-cleanup protocol:

```bash
prism-atlas-protocol --output-path artifacts/prism_atlas/protocol.json
```

In v1, `transform=hadamard` is represented in profiling and assignment and can affect predicted damage/risk.
Runtime execution still falls back to the existing RTN path unless a model-family-specific transform runtime is
added. Do not claim Hadamard runtime speedup until that runtime path exists.

Paper-facing evaluation should use WikiText-2/C4 perplexity, ARC-Easy, ARC-Challenge, HellaSwag, PIQA, BoolQ,
WinoGrande, optional MMLU/GSM8K, real average bits, real memory, decode TPS, profiling/search time, and
materialization time. The central transfer protocol trains or calibrates Stage 0 on small models such as
OPT-125M, Pythia-160M, and TinyLlama, then evaluates zero-target-label transfer on unseen Qwen/Llama/Mistral/Gemma
families.
