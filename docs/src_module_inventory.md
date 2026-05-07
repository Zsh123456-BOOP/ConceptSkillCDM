# src module inventory

This repository now keeps `src/model.py` as the public model API and splits the heavy runtime into focused modules. The number of files is intentional: the previous single-model style became too large to debug safely during A/E ablation work.

## Runtime entry and training

| File | Status | Role |
| --- | --- | --- |
| `src/config.py` | used | Dataset defaults and explicit CLI-argument tracking. |
| `src/dataset.py` | used | Dataset objects, Q-matrix loading, train-only prior inputs, dataloaders. |
| `src/experiment_utils.py` | used | Metrics, config hashes, summary CSV writing, logging helpers. |
| `src/trainer.py` | used | Main training, validation, inference, ablation diagnostics, result persistence. It is still long, but it is live training code rather than dead code. |

## Model API and GLICR-AE

| File | Status | Role |
| --- | --- | --- |
| `src/model.py` | used | Thin public re-export layer for tests, scripts, and external callers. |
| `src/model_cdm.py` | used | Top-level CDM model: GLICR-AE plus fixed prediction head and AE logit residual assembly. |
| `src/model_cdm_forward.py` | used | Forward-pass helper for the top-level CDM model. |
| `src/model_structure.py` | used | GLICR-AE assembly: wires A, E, and the student knowledge encoder together. |
| `src/model_structure_forward.py` | used | Forward-pass helper for GLICR-AE. |
| `src/model_graph.py` | used | A module: evidence-guided global concept relation graph and graph encoder. |
| `src/model_personal.py` | used | E module: student-conditioned local posterior reweighting and gate logic. |
| `src/model_ops.py` | used | Shared sparse-support tensor operations used by A/E. |
| `src/model_regularization.py` | used | Graph and posterior regularization terms. |
| `src/prediction_head.py` | used | Fixed cognitive diagnosis prediction head. Kept separate so A/E changes do not silently become prediction-head changes. |
| `src/module_activity.py` | used | Diagnostic summaries showing which modules are active in each run. |

## Removed legacy files

| File | Reason |
| --- | --- |
| `src/analysis.py` | No current runtime, tool, or test imports it. Component plotting is now handled by `plot_component_analysis.py` and `src/trainer.py::save_component_analysis_data`. |
| `src/utils.py` | Old seed/logger/device helpers. Current entrypoints use `src/experiment_utils.py`, `gpu_utils.py`, and explicit training setup instead. |

## Cleanup decision

Two options were considered:

1. Physically refactor `src/trainer.py` and `src/model_cdm.py` into more files now.
2. Remove confirmed dead legacy files and document the current split.

The second option is the safer current choice. It reduces real clutter without changing the validated training path or invalidating the completed GLICR-AE results.
