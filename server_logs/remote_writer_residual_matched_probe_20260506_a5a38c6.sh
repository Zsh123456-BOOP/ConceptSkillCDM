#!/usr/bin/env bash
set -euo pipefail
cd /home/zsh/ConceptSkillCDM
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home/zsh/anaconda3/envs/xph_env/bin/python run_abce_ablation.py \
  --datasets assist_09,junyi \
  --profiles best \
  --seeds 42 \
  --ablations full,no_A,no_E \
  --include_matched_no_e \
  --epochs 12 \
  --early_stop_patience 2 \
  --max_train_batches 24 \
  --max_val_batches 6 \
  --max_test_batches 6 \
  --run_id remote_writer_residual_matched_probe_20260506_a5a38c6 \
  --gpus 2,3,0 \
  --max_concurrent 3 \
  --max_per_gpu 1 \
  --poll_interval 10
