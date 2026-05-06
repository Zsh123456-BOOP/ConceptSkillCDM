#!/usr/bin/env bash
set -euo pipefail
cd /home/zsh/ConceptSkillCDM
pkill -f '/home/zsh/ConceptSkillCDM/.*(main.py|run_abce_ablation.py)' || true
git fetch origin master
git checkout master
git merge --ff-only origin/master
mkdir -p logs results server_logs
find logs -mindepth 1 -maxdepth 1 -exec rm -rf {} +
find results -mindepth 1 -maxdepth 1 -exec rm -rf {} +
source ~/anaconda3/etc/profile.d/conda.sh
conda activate xph_env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[launcher] repo=$(git rev-parse --short HEAD)"
echo "[launcher] waiting for GPU 2"
while true; do
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2 | tr -d ' ')
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 2 | tr -d ' ')
  if [ "$mem" -lt 2000 ] && [ "$util" -lt 10 ]; then
    break
  fi
  echo "[launcher] gpu2 busy: mem=${mem}MiB util=${util}% ; sleep 60s"
  sleep 60
done
echo "[launcher] gpu2 ready, start run_abce_ablation"
python run_abce_ablation.py --datasets assist_09,junyi --profiles best --ablations full,no_A,no_E --include_matched_no_e --generate_diagnosis --gpus 2 --max_concurrent 1 --max_per_gpu 1 --run_id remote_struct_v2_20260419_86dfed4
