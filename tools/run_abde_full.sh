#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
  echo "[WARN] 当前没有激活 conda 环境。推荐先执行: conda activate xph_env"
fi

DATASETS="${DATASETS:-assist_09,junyi}"
SEEDS="${SEEDS:-42}"
PROFILES="${PROFILES:-best}"
ABLATIONS="${ABLATIONS:-full,no_A,no_B,no_D,no_E}"
COMPONENT_SET="${COMPONENT_SET:-single}"
GPUS="${GPUS:-0,1,2}"
MAX_CONCURRENT="${MAX_CONCURRENT:-3}"
MAX_PER_GPU="${MAX_PER_GPU:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_CLEAN="${SKIP_CLEAN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  fi
fi

echo "[INFO] repo=$ROOT_DIR"
echo "[INFO] datasets=$DATASETS"
echo "[INFO] seeds=$SEEDS"
echo "[INFO] profiles=$PROFILES"
echo "[INFO] ablations=$ABLATIONS"
echo "[INFO] component_set=$COMPONENT_SET"
echo "[INFO] gpus=$GPUS max_concurrent=$MAX_CONCURRENT max_per_gpu=$MAX_PER_GPU"

if [[ "$SKIP_CLEAN" == "1" ]]; then
  echo "[SKIP] keeping existing logs/results/checkpoints contents"
else
  echo "[CLEAN] removing old logs/results/checkpoints contents"
  rm -rf logs/* results/* checkpoints/*
  mkdir -p logs results checkpoints
fi

CMD=(
  "$PYTHON_BIN" run_abce_ablation.py
  --datasets "$DATASETS"
  --seeds "$SEEDS"
  --profiles "$PROFILES"
  --component_set "$COMPONENT_SET"
  --ablations "$ABLATIONS"
  --gpus "$GPUS"
  --max_concurrent "$MAX_CONCURRENT"
  --max_per_gpu "$MAX_PER_GPU"
  --generate_diagnosis
)

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=($EXTRA_ARGS)
  CMD+=("${EXTRA_ARR[@]}")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=(--dry_run)
fi

echo "[RUN] ${CMD[*]}"
"${CMD[@]}"
