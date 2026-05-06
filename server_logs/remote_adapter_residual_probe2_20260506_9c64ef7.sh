#!/usr/bin/env bash
set -euo pipefail
cd ~/ConceptSkillCDM
/home/zsh/anaconda3/envs/xph_env/bin/python run_abce_ablation.py \
  --datasets assist_09,junyi \
  --profiles best \
  --seeds 42 \
  --ablations full,no_A,no_E \
  --epochs 12 \
  --early_stop_patience 2 \
  --max_train_batches 24 \
  --max_val_batches 6 \
  --max_test_batches 6 \
  --run_id remote_adapter_residual_probe2_20260506_9c64ef7 \
  --gpus 2,3 \
  --max_concurrent 2 \
  --max_per_gpu 1 \
  --poll_interval 10
