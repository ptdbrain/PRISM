# PRISM-Atlas v1 Design

## Goal

Turn PRISM mainline into PRISM-Atlas v1 while preserving PRISM-original as a runnable baseline. The v1 scope is action-aware, risk-aware mixed precision with a minimal transform axis that can be evaluated on small models and transferred to larger unseen models.

## Architecture

PRISM-Atlas adds a new `prism.atlas` package instead of rewriting the existing Stage 0-4 modules. Existing PRISM CLIs keep working. New Atlas CLIs produce Atlas-specific artifacts and also emit legacy-compatible `bits` maps so the current RTN/runtime path can still run.

The main data unit is a quantization action:

```text
{bits in 2/3/4, group_size in 64/128, transform in none/hadamard, backend=rtn, target=weight}
```

Each layer receives a response surface over actions:

```text
mean_damage[layer, action]
log_variance[layer, action]
uncertainty[layer, action]
ranking_score[layer, action]
memory_cost_bits[layer, action]
latency_proxy[layer, action]
valid_action[layer, action]
```

Atlas uses versioned artifacts:

```text
ProfileArtifactV1 = PRISM-original bit-only profile
ResponseSurfaceArtifactV2 = PRISM-Atlas profile, schema_version=prism-atlas-profile-v2
AssignmentV1 = PRISM-original bit-only assignment
AssignmentV2 = PRISM-Atlas action assignment, schema_version=prism-atlas-assignment-v2
```

## Stages

Stage 0 has three subpaths:

- Stage 0A: analytic/direct scorer, `damage = kappa_l * D_l(action)`.
- Stage 0B: optional small-model response-surface dataset with one row per `(model, layer, action)`.
- Stage 0C: optional residual/uncertainty predictor if measured labels exist.

Scorer modes are:

```text
analytic = kappa_l * D_l(action)
learned = MLP(layer_features, action_features) -> mean, log_variance, ranking_score
hybrid = analytic + residual_MLP(layer_features, action_features)
```

`kappa_l` is a data-free layer-importance proxy from curvature/outlier/scale/layer-type features. `D_l(action)` is the RTN distortion curve after the selected group size and transform. Low-resource smoke tests can use RTN relative MSE, but paper-grade labels should use `delta_nll_or_jsd_logits` as the primary damage label.

Stage 1 profiles a target model into an Atlas profile JSON. If an Atlas checkpoint is supplied, Stage 1 can use `learned` or `hybrid`; otherwise it uses `analytic`. Each layer/action response includes validity metadata:

```text
runtime_supported
materialization_supported
valid_action
fallback_backend
latency_proxy
```

Stage 2 solves a multiple-choice assignment over actions under realistic memory accounting and action validity. Objective modes are `mean`, `risk`, `cvar-lite`, and `oracle/debug`. The risk objective is:

```text
mean_damage + lambda * uncertainty
```

`cvar-lite` uses a normal-tail multiplier for the worst 10 percent approximation.

Stage 2.5 no longer treats QUIC as a global black-box correction. Atlas selects high-uncertainty layers from the chosen action response and returns a correction plan. Existing QUIC can still run, but Atlas gates it to the uncertain subset.

Stage 3/4 remain compatible with the existing runtime path. `transform=none` maps directly to RTN artifacts. `transform=hadamard` is represented in the profile/assignment and can influence damage estimates, but v1 runtime must explicitly mark fallback if no transform kernel path is available. Hadamard v1 is an evaluation/materialization artifact; Hadamard v2 is the optimized runtime/kernel path. Do not claim transform speedup until v2 exists.

## Calibration And Evaluation

Uncertainty must be evaluated, not merely output. Required calibration metrics are:

- NLL calibration
- Spearman correlation between predicted and measured damage
- ECE-style calibration over risk bins
- top-k risky layer recall

Required benchmark systems:

- Uniform RTN
- PRISM-original
- PRISM-Atlas analytic mean-only
- PRISM-Atlas learned mean-only
- PRISM-Atlas hybrid mean-only
- PRISM-Atlas hybrid risk-aware
- PRISM-Atlas no-transform
- PRISM-Atlas no-uncertainty

Budget sweep: `2.0`, `2.5`, `3.0`, `3.5`, `4.0` average bits.

Metrics: WikiText-2 PPL, C4 PPL, ARC-Easy, ARC-Challenge, HellaSwag, PIQA, BoolQ, WinoGrande, optional MMLU/GSM8K, real average bits, real memory, decode TPS, profiling/search time, and materialization time.

Transfer protocol:

- Train/calibrate on small models: OPT-125M, Pythia-160M, TinyLlama.
- Evaluate zero-shot on unseen target families: Qwen2.5-1.5B/7B, Llama-3.x 1B/8B, Mistral-7B, Gemma.
- Report `zero-target-label`, `light-target-calibration`, and `full-target-label-oracle` settings separately.

## Repo Cleanup Requirements

Phase 1 cleanup remains required before paper packaging:

- archive `prism/meta/`
- remove legacy `--checkpoint-dir` from mainline CLI
- unify `assign/assignment`
- unify `profile/profiling`
- add environment lockfile or `environment.yml`
- expand README reproduction path
- add real-model regression test

## Reference Code Use

QuaRot and SpinQuant justify the Hadamard axis and show that rotation reduces outlier pressure but is architecture-sensitive. FlatQuant justifies a future learned transform axis, but v1 intentionally does not train FlatQuant matrices because that would violate the low-resource transfer scope.

## Tests

The implementation is validated by CPU-only contract tests:

- action schema and action features
- action-conditioned response model output shapes and positive uncertainty
- Atlas profile construction on `MockTransformerLM`
- risk-aware assignment changing choices when uncertainty is penalized
- validity-mask filtering in the solver
- analytic scorer components and schema v2 fields
- calibration/protocol scaffolds
- uncertainty-guided QUIC layer selection
- CLI entrypoint main guards

## Non-Goals For v1

V1 does not claim full runtime Hadamard execution for every model family. It records transform choices and supports response/assignment over transforms. Runtime transform execution requires a later model-family-specific kernel or wrapper pass.
