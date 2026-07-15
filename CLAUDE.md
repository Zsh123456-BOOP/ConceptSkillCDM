# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working conventions

- **Respond in 简体中文 by default** (see `AGENTS.md`). Code, docstrings, and log strings are English; prose/reports are Chinese.
- `AGENTS.md` holds the full collaboration rules. The load-bearing ones: explore before editing, reuse existing helpers, compare ≥2 options before non-trivial changes, and verify (run smoke tests / a minimal loop) before declaring done. Do not roll back or commit unrelated uncommitted work from other windows.
- Before editing any CRG/LCRF or cognitive-diagnosis paper draft under `docs/paper_review_2025_2026/`, read `docs/paper_review_2025_2026/codex_cd_writing_guardrails.md` first.

## Commands

There is no build step and no pytest suite. Smoke files are standalone executable scripts.

```bash
# Static compile check (run after any edit under src/, tools/, tests/, or the root scripts)
python -m compileall -q main.py experiment_configs.py run_graph_ablation.py src tests tools

# Run one smoke test (each is a plain script with a main(); no test runner)
python tests/smoke_label_isolation.py        # verifies train-only LOO leakage safety
python tests/smoke_prediction_head.py         # 2PL head contract
python tests/smoke_graph_propagation_alpha.py # message-passing on/off invariants
# ...the rest live in tests/smoke_*.py; README lists the full ordered set.

# Minimal CPU training loop (fast end-to-end check)
python main.py --dataset_name assist_09 --model_variant full --epochs 1 \
  --batch_size 128 --max_train_batches 2 --max_val_batches 1 --run_mode train \
  --num_workers 0 --no_cuda \
  --save_dir checkpoints/local_graph_irt_smoke --log_dir logs/local_graph_irt_smoke

# Structural ablation runner (dry-run prints the exact per-job main.py commands)
python run_graph_ablation.py --datasets assist_09 --ablations full,no_response_evidence \
  --seeds 42 --run_mode train --gpus 0 --dry_run
```

To exercise the whole model change end-to-end, prefer `main.py ... --run_mode train` with `--max_*_batches` over reading tests alone.

## The train/test seal (critical operational contract)

Model selection and testing are deliberately separated to prevent test-set leakage:

- **`--run_mode train`** selects a checkpoint on validation AUC only. It never opens `test.csv`; the checkpoint is bound to the train/valid corpus via `data_dir` + train/valid SHA-256 (`_build_data_identity`). Writes `validation_result.json`.
- **`--run_mode test`** runs against an existing sealed checkpoint (matched by `--run_id`/`--save_dir`) and never trains. The first test read atomically writes `test_seal.json` (`_claim_test_seal`, `O_EXCL`). After that the directory refuses a second test *and* refuses re-training that would overwrite the checkpoint (`main.py` and `train_one_experiment` both check for `test_seal.json`/`test_results.json`).
- Practical consequence: if you change the model/config, you must retrain into a **new `--save_dir`**; you cannot re-test a sealed directory. If a metric drops after removing a shortcut, record it as evidence — do not restore the shortcut to recover the number.

## Architecture: single-path Response-Evidence Graph-IRT (`graph_irt_v10`)

The repo intentionally implements exactly **one** prediction path. Legacy branches (personal posterior, roadmap/tutor residuals, student–item collaborative graph, per-concept item difficulty, extra logit residuals, Hadamard/low-rank interaction) are **physically deleted**, not toggled off. Preserve this: do not reintroduce a second prediction branch or a label-derived lookup consumed at prediction time.

Forward path (`src/model_cdm.py::CognitiveDiagnosisModel.forward`):

```
train-only label-free concept-graph priors  (item co-occurrence + student co-exposure)
  + two-channel leave-one-out train response evidence
      -> MultiHeadRelationLearning         (row-stochastic concept graphs)      src/model_graph.py
      -> StudentKnowledgeEncoder           (evidence-init state + GNN hops)     src/model_graph.py
      -> CognitiveDiagnosisHead            (Q-masked scalar ability θ_e,        src/prediction_head.py
                                            θ_c anchored on direct + graph-
                                            propagated LOO evidence, non-
                                            negative channel weights)
      -> a * (θ_e - b)                     (single scalar-difficulty 2PL logit)
```

Architectural invariant, asserted in diagnostics: `details["logits"] == details["irt_logit"]`. There is no residual or calibration term added to the IRT logit; the evidence anchor shifts θ_c *before* the single 2PL readout.

### Leave-one-out response evidence (the anti-leakage core)

Response sufficient statistics (`src/dataset.py::build_student_concept_response_stats`) are built from **train only** as raw counts: `student_concept_count/correct`, a difficulty-adjusted `student_concept_residual_sum`, per-concept and global totals, and a sorted `student_item_keys → expected_correct` table.

At training time `_build_response_evidence` (in `model_cdm.py`) subtracts the *current row's* label, count, and residual from every statistic that feeds its own prediction (exact leave-one-out), so a target label can never be copied into its own input. Validation and test rows read the complete, unmodified train statistics. `outcome_to_exclude=labels` is passed **only** for training batches (`trainer.py::_run_epoch`). When touching evidence code, keep this LOO contract exact — `smoke_label_isolation.py` guards it.

### Model variants map to fixed, interpretable ablations

`--model_variant` (in `main.py::MODEL_VARIANTS`) is not a free hyperparameter knob; each variant deterministically sets the underlying switches via `_apply_model_variant`:

- `full` — concept graph + evidence anchor (direct + residual + graph-propagated) + fixed BCE + pairwise-AUC objective.
- `no_response_evidence` — removes the train response buffers/projection entirely.
- `no_evidence_anchor` — evidence feeds only the initial state; θ has no anchor channels (v9 behaviour).
- `no_evidence_propagation` — anchor keeps direct rate + residual channels but drops the graph-propagated channel.
- `no_pairwise_loss` — identical model, objective reverts to pure BCE.
- `ema_bce` — pure BCE with a fixed 0.9 per-epoch weight EMA for validation/checkpoint selection (inference stays single-model).
- `no_message_passing` — sets `graph_propagation_alpha=0`; output state equals the initial state.
- `item_only` / `exposure_only` — keep only one graph-prior evidence source (`graph_prior_mode`).
- `degree_random` — row-degree-matched random support graph (relation-identity control).

`pairwise_auc_weight` and `ema_decay` are **fixed by the variant** and cross-checked in `trainer.py` (`_resolve_pairwise_auc_weight`, `_resolve_ema_decay`) — supplying a conflicting value raises. Constants live in `src/config.py` (`PAIRWISE_AUC_WEIGHT=0.5`, `EMA_DECAY=0.9`).

### Module map

| File | Role |
|------|------|
| `main.py` | CLI: arg parsing, `--model_variant` → switches, dataset-default application, seal checks, dispatch to trainer. |
| `src/config.py` | `DATASET_DEFAULTS` — per-dataset hyperparameters; `apply_dataset_defaults` only fills args not explicitly passed on the CLI. |
| `experiment_configs.py` | Wraps `DATASET_DEFAULTS` for the ablation launcher; `DEFAULT_SEEDS`. |
| `src/dataset.py` | Train-only ID maps, Q-matrix, graph priors (item co-occurrence / student co-exposure / degree-random), response-evidence stats, dataloaders, cold-start + train-seen filtering. |
| `src/model.py` | Public re-export of `CognitiveDiagnosisModel`, `GRAPH_IRT_ARCHITECTURE`. |
| `src/model_cdm.py` | The model: buffers, LOO response evidence, the sole forward path, `get_student_diagnosis`. |
| `src/model_graph.py` | `MultiHeadRelationLearning` (prior→row-stochastic graph, top-k support), `ConceptGraphConv` (dense/sparse SpMM by concept count), `StudentKnowledgeEncoder`. |
| `src/prediction_head.py` | `ExerciseDifficultyEncoder` (scalar b, softplus a), `CognitiveDiagnosisHead` (Q-masked θ, 2PL logit). |
| `src/model_regularization.py` | Graph entropy/diag/uniform + prediction-L2 regularization components. |
| `src/trainer.py` | `train_one_experiment`, `run_inference`, epoch loop, grad-guard/clip, checkpointing, EMA, test-seal machinery, component-analysis export. |
| `src/experiment_utils.py` | Logging, `compute_metrics` (auc/acc/rmse), device selection, `_config_hash`, atomic summary-CSV append. |
| `src/module_activity.py` | Post-hoc module-activity diagnostics/report. |
| `gpu_utils.py` | GPU memory probing and round-robin slot assignment for multi-job launches. |
| `run_graph_ablation.py` | Fan-out launcher: builds one `main.py` command per (dataset, ablation, seed), schedules across GPUs. |

### Data layout

Each `data/<dataset>/` holds `train.csv`, `valid.csv`, `test.csv` with columns `stu_id, exer_id, cpt_seq, label`. `cpt_seq` is a quoted comma-separated concept-id list (e.g. `"1,24,44"`); `label` is 0/1. **All** ID maps, the Q-matrix, graph priors, and response stats are derived from `train.csv` only; valid/test rows are then filtered to train-seen students/items. Datasets are registered in `src/config.py::DATASET_DEFAULTS` (add a new entry there to make `--dataset_name` accept it). Generated public-benchmark CSVs are git-ignored (only manifests are tracked); `tools/` contains the adapters that regenerate them.

## Server / git rules (from AGENTS.md — enforced)

Sync to the training server `10.154.22.11` **only through git**: commit locally (stage only this task's files), push, then `git pull` under `/home/zsh/ConceptSkillCDM`. **Never** wrap commands in Base64 (`echo … | base64 -d | bash`) or inject multi-line scripts over SSH — this triggers the network-security alert the rules exist to avoid. Do not commit checkpoints, full logs, or large intermediates. Before cleaning `logs/`/`results/`/`checkpoints/`, confirm the path is inside the project and scoped to the current run id, and extract a summary CSV / best-AUC record first.
