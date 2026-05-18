#!/usr/bin/env bash
set -euo pipefail

# Run full-model baselines for the public benchmark datasets.
# Usage:
#   bash tools/run_public_full.sh
#
# The script starts at most one job per listed GPU and waits for the currently
# assigned job before reusing that GPU. It is intentionally plain shell so the
# command line is auditable on the server.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-public_full_$(date +%Y%m%d_%H%M%S)}"
GPUS_CSV="${GPUS:-0,2,3}"
DATASETS_CSV="${DATASETS:-frcsub,math2,assist_15,nips34,assist_12,ednet_kt1}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-5}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if [[ -x "/home/zsh/anaconda3/envs/xph_env/bin/python" ]]; then
    PYTHON_BIN="/home/zsh/anaconda3/envs/xph_env/bin/python"
  else
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 127
  fi
fi

IFS=',' read -r -a GPUS_ARR <<< "$GPUS_CSV"
IFS=',' read -r -a DATASETS <<< "$DATASETS_CSV"

mkdir -p "logs/${RUN_ID}" "checkpoints/${RUN_ID}" "results/${RUN_ID}"

echo "run_id=${RUN_ID}"
echo "gpus=${GPUS_CSV}"
echo "datasets=${DATASETS[*]}"
echo "epochs=${EPOCHS} patience=${PATIENCE} num_workers=${NUM_WORKERS}"
echo "python=${PYTHON_BIN}"

declare -A GPU_PID=()
declare -A GPU_DATASET=()

wait_for_gpu() {
  local gpu="$1"
  local pid="${GPU_PID[$gpu]:-}"
  if [[ -n "$pid" ]]; then
    local dataset="${GPU_DATASET[$gpu]}"
    echo "[wait] gpu=${gpu} dataset=${dataset} pid=${pid}"
    if wait "$pid"; then
      echo "[done] gpu=${gpu} dataset=${dataset} pid=${pid}"
    else
      local code="$?"
      echo "[failed] gpu=${gpu} dataset=${dataset} pid=${pid} exit=${code}" >&2
    fi
    unset "GPU_PID[$gpu]"
    unset "GPU_DATASET[$gpu]"
  fi
}

launch_job() {
  local dataset="$1"
  local gpu="$2"
  local log_dir="logs/${RUN_ID}/${dataset}_full"
  local ckpt_dir="checkpoints/${RUN_ID}/${dataset}_full"
  local stdout_log="logs/${RUN_ID}/${dataset}_full.stdout.log"
  mkdir -p "$log_dir" "$ckpt_dir"
  echo "[launch] dataset=${dataset} gpu=${gpu} log=${stdout_log}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" main.py \
    --dataset_name "$dataset" \
    --model_variant full \
    --epochs "$EPOCHS" \
    --early_stop_patience "$PATIENCE" \
    --patience "$PATIENCE" \
    --num_workers "$NUM_WORKERS" \
    --save_dir "$ckpt_dir" \
    --log_dir "$log_dir" \
    --generate_diagnosis False \
    > "$stdout_log" 2>&1 &
  GPU_PID[$gpu]="$!"
  GPU_DATASET[$gpu]="$dataset"
  echo "[pid] dataset=${dataset} gpu=${gpu} pid=${GPU_PID[$gpu]}"
}

gpu_index=0
for dataset in "${DATASETS[@]}"; do
  gpu="${GPUS_ARR[$gpu_index]}"
  wait_for_gpu "$gpu"
  launch_job "$dataset" "$gpu"
  gpu_index=$(( (gpu_index + 1) % ${#GPUS_ARR[@]} ))
done

for gpu in "${GPUS_ARR[@]}"; do
  wait_for_gpu "$gpu"
done

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
run_id = Path("logs").glob("public_full_*")
print("[summary] runs are under logs/<RUN_ID>, checkpoints/<RUN_ID>, results/<RUN_ID>")
PY
