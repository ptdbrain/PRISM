#!/usr/bin/env bash
set -euo pipefail

# PRISM full-pipeline runner for Linux, WSL, Kaggle, and Colab shells.
# Defaults are intentionally conservative: Stage 0 is expensive and is disabled
# unless --run-stage0 or RUN_STAGE0=1 is provided.

usage() {
  cat <<'USAGE'
Usage:
  bash run_full_pipeline.sh [options]

Common examples:
  bash run_full_pipeline.sh --model Qwen/Qwen2.5-1.5B --mlp-path artifacts/prism/Qwen_Qwen2.5-1.5B_merged/prism_mlp.pt

  bash run_full_pipeline.sh --run-stage0 --stage0-models "facebook/opt-125m,EleutherAI/pythia-160m" --model Qwen/Qwen2.5-1.5B

  bash run_full_pipeline.sh --run-stage0 --stage0-use-shards --stage0-models "Qwen/Qwen2.5-1.5B" --num-shards 16 --model Qwen/Qwen2.5-1.5B

Options:
  --model NAME                 Target model for Stage 1-4. Default: MODEL_NAME or Qwen/Qwen2.5-1.5B
  --stage0-models LIST         Comma-separated Stage 0 training models.
  --mlp-path PATH              Existing prism_mlp.pt. If omitted and Stage 0 runs, uses Stage 0 output.
  --out-root DIR               Output root. Default: artifacts/prism
  --run-stage0                 Run Stage 0 before Stage 1.
  --stage0-use-shards          Build Stage 0 dataset as train_shard_XX_of_NN.pt then merge/train.
  --num-shards N               Number of Stage 0 shards. Default: 16
  --start-shard N              First shard id, inclusive. Default: 0
  --end-shard N                Last shard id, exclusive. Default: NUM_SHARDS
  --epochs N                   Stage 0 MLP train epochs. Default: 100
  --group-size N               Quantization group size. Default: 128
  --hidden-size N              Demo/mock hidden size. Default: 16
  --num-layers N               Demo/mock layer count. Default: 4
  --seq-len N                  QUIC synthetic sequence length. Default: 8
  --device DEVICE              Device passed to CLI: cuda, cpu, cuda:0, etc. Default: cuda
  --family NAME                Model family adapter, or auto. Default: auto
  --budget B                   Main Stage 2 budget. Default: 3.0
  --budgets LIST               Comma-separated extra budgets to sweep. Default: 2.5,2.75,3.0,3.25,3.5
  --skip-stage1                Do not run profiling.
  --skip-stage2                Do not run assignment.
  --skip-quic                  Do not run Stage 2.5.
  --skip-rtn                   Do not run Stage 3.
  --skip-run                   Do not run Stage 4.
  --execute                    Execute a small generation/forward pass in Stage 4.
  --trust-remote-code          Pass --trust-remote-code to HF loaders.
  --prompt TEXT                Stage 4 prompt. Default: Hello
  --max-new-tokens N           Stage 4 generation tokens. Default: 16
  --dry-run                    Print commands without executing.
  -h, --help                   Show this help.

Environment overrides:
  MODEL_NAME, STAGE0_MODELS, MLP_PATH, OUT_ROOT, RUN_STAGE0, STAGE0_USE_SHARDS,
  NUM_SHARDS, START_SHARD, END_SHARD, EPOCHS, GROUP_SIZE, DEVICE, FAMILY,
  HIDDEN_SIZE, NUM_LAYERS, SEQ_LEN,
  BUDGET, BUDGETS, RUN_STAGE1, RUN_STAGE2, RUN_QUIC, RUN_RTN, RUN_STAGE4,
  EXECUTE, TRUST_REMOTE_CODE, PROMPT, MAX_NEW_TOKENS, DRY_RUN
USAGE
}

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B}"
STAGE0_MODELS="${STAGE0_MODELS:-$MODEL_NAME}"
OUT_ROOT="${OUT_ROOT:-artifacts/prism}"
RUN_STAGE0="${RUN_STAGE0:-0}"
STAGE0_USE_SHARDS="${STAGE0_USE_SHARDS:-0}"
NUM_SHARDS="${NUM_SHARDS:-16}"
START_SHARD="${START_SHARD:-0}"
END_SHARD="${END_SHARD:-$NUM_SHARDS}"
EPOCHS="${EPOCHS:-100}"
GROUP_SIZE="${GROUP_SIZE:-128}"
HIDDEN_SIZE="${HIDDEN_SIZE:-16}"
NUM_LAYERS="${NUM_LAYERS:-4}"
SEQ_LEN="${SEQ_LEN:-8}"
DEVICE="${DEVICE:-cuda}"
FAMILY="${FAMILY:-auto}"
BUDGET="${BUDGET:-3.0}"
BUDGETS="${BUDGETS:-2.5,2.75,3.0,3.25,3.5}"
RUN_STAGE1="${RUN_STAGE1:-1}"
RUN_STAGE2="${RUN_STAGE2:-1}"
RUN_QUIC="${RUN_QUIC:-1}"
RUN_RTN="${RUN_RTN:-1}"
RUN_STAGE4="${RUN_STAGE4:-1}"
EXECUTE="${EXECUTE:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
PROMPT="${PROMPT:-Hello}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
DRY_RUN="${DRY_RUN:-0}"

sanitize_model_name() {
  printf '%s' "$1" | sed 's#[/:\\ ]#_#g'
}

SANITIZED_MODEL="$(sanitize_model_name "$MODEL_NAME")"
STAGE0_DIR="${STAGE0_DIR:-$OUT_ROOT/${SANITIZED_MODEL}_stage0}"
RUN_DIR="${RUN_DIR:-$OUT_ROOT/${SANITIZED_MODEL}_full}"
MLP_PATH="${MLP_PATH:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_NAME="$2"; shift 2 ;;
    --stage0-models) STAGE0_MODELS="$2"; shift 2 ;;
    --mlp-path) MLP_PATH="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --run-stage0) RUN_STAGE0=1; shift ;;
    --stage0-use-shards) STAGE0_USE_SHARDS=1; shift ;;
    --num-shards) NUM_SHARDS="$2"; shift 2 ;;
    --start-shard) START_SHARD="$2"; shift 2 ;;
    --end-shard) END_SHARD="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --group-size) GROUP_SIZE="$2"; shift 2 ;;
    --hidden-size) HIDDEN_SIZE="$2"; shift 2 ;;
    --num-layers) NUM_LAYERS="$2"; shift 2 ;;
    --seq-len) SEQ_LEN="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --family) FAMILY="$2"; shift 2 ;;
    --budget) BUDGET="$2"; shift 2 ;;
    --budgets) BUDGETS="$2"; shift 2 ;;
    --skip-stage1) RUN_STAGE1=0; shift ;;
    --skip-stage2) RUN_STAGE2=0; shift ;;
    --skip-quic) RUN_QUIC=0; shift ;;
    --skip-rtn) RUN_RTN=0; shift ;;
    --skip-run) RUN_STAGE4=0; shift ;;
    --execute) EXECUTE=1; shift ;;
    --trust-remote-code) TRUST_REMOTE_CODE=1; shift ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

SANITIZED_MODEL="$(sanitize_model_name "$MODEL_NAME")"
STAGE0_DIR="${STAGE0_DIR:-$OUT_ROOT/${SANITIZED_MODEL}_stage0}"
RUN_DIR="${RUN_DIR:-$OUT_ROOT/${SANITIZED_MODEL}_full}"
PROFILE_PATH="$RUN_DIR/profile.json"
ASSIGNMENT_PATH="$RUN_DIR/assignment_${BUDGET}.json"
QUIC_PATH="$RUN_DIR/quic_assignment_${BUDGET}.json"
RTN_DIR="$RUN_DIR/rtn"

trust_flag=()
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  trust_flag=(--trust-remote-code)
fi

run_cmd() {
  echo
  printf '+'
  printf ' %q' "$@"
  echo
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

run_stage0_shards() {
  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$STAGE0_DIR"
  fi
  run_cmd python3 scripts/stage0_sharded.py \
    --models "$STAGE0_MODELS" \
    --output-dir "$STAGE0_DIR" \
    --num-shards "$NUM_SHARDS" \
    --start-shard "$START_SHARD" \
    --end-shard "$END_SHARD" \
    --group-size "$GROUP_SIZE" \
    --epochs "$EPOCHS"
}

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$RUN_DIR"
fi

echo "PRISM full pipeline"
echo "repo: $repo_dir"
echo "target model: $MODEL_NAME"
echo "run dir: $RUN_DIR"

if [[ "$RUN_STAGE0" == "1" ]]; then
  if [[ "$STAGE0_USE_SHARDS" == "1" ]]; then
    run_stage0_shards
  else
    IFS=',' read -r -a stage0_models_array <<< "$STAGE0_MODELS"
    run_cmd python3 -m prism.cli.train_meta \
      --output-dir "$STAGE0_DIR" \
      --epochs "$EPOCHS" \
      --group-size "$GROUP_SIZE" \
      --model-names "${stage0_models_array[@]}"
  fi
  MLP_PATH="$STAGE0_DIR/prism_mlp.pt"
fi

if [[ -z "$MLP_PATH" ]]; then
  MLP_PATH="$STAGE0_DIR/prism_mlp.pt"
fi

if [[ "$RUN_STAGE1" == "1" ]]; then
  run_cmd python3 -m prism.cli.profile \
    --model-id-or-path "$MODEL_NAME" \
    --family "$FAMILY" \
    --mlp-path "$MLP_PATH" \
    --group-size "$GROUP_SIZE" \
    --hidden-size "$HIDDEN_SIZE" \
    --num-layers "$NUM_LAYERS" \
    --device "$DEVICE" \
    --output-path "$PROFILE_PATH" \
    "${trust_flag[@]}"
fi

if [[ "$RUN_STAGE2" == "1" ]]; then
  run_cmd python3 -m prism.cli.assign \
    --profile-path "$PROFILE_PATH" \
    --budget "$BUDGET" \
    --output-path "$ASSIGNMENT_PATH"

  IFS=',' read -r -a budget_array <<< "$BUDGETS"
  for budget in "${budget_array[@]}"; do
    [[ -z "$budget" || "$budget" == "$BUDGET" ]] && continue
    run_cmd python3 -m prism.cli.assign \
      --profile-path "$PROFILE_PATH" \
      --budget "$budget" \
      --output-path "$RUN_DIR/assignment_${budget}.json"
  done
fi

if [[ "$RUN_QUIC" == "1" ]]; then
  run_cmd python3 -m prism.cli.quic \
    --model-id-or-path "$MODEL_NAME" \
    --family "$FAMILY" \
    --device "$DEVICE" \
    --profile-path "$PROFILE_PATH" \
    --assignment-path "$ASSIGNMENT_PATH" \
    --output-path "$QUIC_PATH" \
    --hidden-size "$HIDDEN_SIZE" \
    --seq-len "$SEQ_LEN" \
    "${trust_flag[@]}"
  FINAL_ASSIGNMENT="$QUIC_PATH"
else
  FINAL_ASSIGNMENT="$ASSIGNMENT_PATH"
fi

if [[ "$RUN_RTN" == "1" ]]; then
  run_cmd python3 -m prism.cli.precompute_rtn \
    --model-id-or-path "$MODEL_NAME" \
    --family "$FAMILY" \
    --device "$DEVICE" \
    --group-size "$GROUP_SIZE" \
    --hidden-size "$HIDDEN_SIZE" \
    --num-layers "$NUM_LAYERS" \
    --output-dir "$RTN_DIR" \
    "${trust_flag[@]}"
fi

if [[ "$RUN_STAGE4" == "1" ]]; then
  run_args=(
    python3 -m prism.cli.run
    --model-id-or-path "$MODEL_NAME"
    --family "$FAMILY"
    --device "$DEVICE"
    --artifact-root "$RTN_DIR"
    --assignment-path "$FINAL_ASSIGNMENT"
    --hidden-size "$HIDDEN_SIZE"
    --num-layers "$NUM_LAYERS"
    --prompt "$PROMPT"
    --max-new-tokens "$MAX_NEW_TOKENS"
  )
  if [[ "$EXECUTE" == "1" ]]; then
    run_args+=(--execute)
  fi
  run_args+=("${trust_flag[@]}")
  run_cmd "${run_args[@]}"
fi

echo
echo "Done."
echo "Profile: $PROFILE_PATH"
echo "Assignment: $ASSIGNMENT_PATH"
echo "Final assignment: ${FINAL_ASSIGNMENT:-$ASSIGNMENT_PATH}"
echo "RTN artifacts: $RTN_DIR"
