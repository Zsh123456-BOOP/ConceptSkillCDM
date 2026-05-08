# ConceptSkillCDM

本仓库当前用于验证一个可解释的认知诊断模型：固定预测头作为主干，A/E 结构模块负责概念关系建模与个性化局部重加权。

当前不要给 A/E 模块使用固定论文名。`A` 和 `E` 仍是临时占位符，后续需要结合代码、CD/KT 论文和真实教育场景重新命名。

## 1. 当前核心结果

最近一次干净实验：

`ae_reliability_gpu23_20260507_170717`

| Dataset | Variant | Test AUC | Best Val AUC | Epoch |
| --- | --- | ---: | ---: | ---: |
| assist_09 | full | 0.779646 | 0.778956 | 6 |
| assist_09 | no_A | 0.681722 | 0.681624 | 1 |
| assist_09 | no_E | 0.692737 | 0.693441 | 3 |
| junyi | full | 0.829135 | 0.824817 | 20 |
| junyi | no_A | 0.635569 | 0.641430 | 1 |
| junyi | no_E | 0.798560 | 0.789544 | 1 |

目标检查：

- `assist_09` full test AUC 达到 `0.779646`，超过目标 `0.778`。
- `junyi` full test AUC 达到 `0.829135`，超过目标 `0.823`。

归档产物：

- `results/experiment_results.csv`
- `results/abce_ablation_diagnosis.csv`
- `results/abce_ablation_summary.csv`
- `results/abce_ablation_summary_mean.csv`
- `logs/abce_diag/ae_reliability_gpu23_20260507_170717/`
- `server_logs/ae_reliability_gpu23_20260507_170717.out`

仓库外备份：

`C:\Users\zsh\Desktop\test_xph\ConceptSkillCDM_artifact_backups\ae_reliability_gpu23_20260507_170717_20260508_073034.zip`

SHA256:

`258E1A21860368B5B5A870EBB664494163BA62A92023E6689BCFD9306A57903E`

## 2. 模型结构

模型主入口是 `src/model.py` 暴露的 `CognitiveDiagnosisModel`。

当前结构分为两层：

1. 固定认知诊断预测头  
   `src/prediction_head.py` 中实现，保留 IRT/题目难度/概念掌握度等可解释预测路径。

2. A/E 结构模块  
   `src/model_structure.py` 装配，具体由：
   - `A`：全局概念关系图，使用 train-only item co-occurrence、sequence transition、自环保持和少量可解释校正。
   - `E`：个性化局部 posterior reweighting，只在 A 给出的 support 上调整边权，不生成任意新边。
   - AE joint residual：只在 A 与 E 同时启用时生效，使用 train-only evidence reliability 特征，不使用 valid/test 信息。

预测头不是本轮达标的主要改动点；本轮有效增益来自 A/E 结构模块中的可解释 train-only reliability 信号。

## 3. src 目录是否都是有用的

基于当前 import 链、脚本入口和 smoke 验证，`src` 目录剩余文件在文件级都是有用途的，不建议继续删除。

已删除的旧文件：

- `src/analysis.py`：没有当前入口、工具或测试引用；组件分析由 `plot_component_analysis.py` 和 `src/trainer.py::save_component_analysis_data` 负责。
- `src/utils.py`：旧 logger/seed/device helper；当前入口使用 `src/experiment_utils.py`、`gpu_utils.py` 和训练流程内的显式设置。

当前 `src` 文件职责：

| File | Status | Role |
| --- | --- | --- |
| `src/config.py` | used | Dataset defaults and explicit CLI-argument tracking. |
| `src/dataset.py` | used | Dataset objects, Q-matrix loading, train-only prior inputs, dataloaders. |
| `src/experiment_utils.py` | used | Metrics, config hashes, summary CSV writing, logging helpers. |
| `src/trainer.py` | used | Main training, validation, inference, ablation diagnostics, result persistence. It is long but live training code. |
| `src/model.py` | used | Thin public re-export layer for tests, scripts, and external callers. |
| `src/model_cdm.py` | used | Top-level CDM model: A/E structure module plus fixed prediction head and AE logit residual assembly. |
| `src/model_cdm_forward.py` | used | Forward-pass helper for the top-level CDM model. |
| `src/model_structure.py` | used | A/E assembly: wires A, E, and the student knowledge encoder together. |
| `src/model_structure_forward.py` | used | Forward-pass helper for A/E. |
| `src/model_graph.py` | used | A module: evidence-guided global concept relation graph and graph encoder. |
| `src/model_personal.py` | used | E module: student-conditioned local posterior reweighting and gate logic. |
| `src/model_ops.py` | used | Shared sparse-support tensor operations used by A/E. |
| `src/model_regularization.py` | used | Graph and posterior regularization terms. |
| `src/prediction_head.py` | used | Fixed cognitive diagnosis prediction head. Kept separate so A/E changes do not silently become prediction-head changes. |
| `src/module_activity.py` | used | Diagnostic summaries showing which modules are active in each run. |

需要注意：`src/trainer.py` 和 `src/model_cdm.py` 仍然较长，但它们是当前真实训练链路，不是垃圾代码。现在不建议为了“看起来短”继续大拆，否则会增加已经验证过的实验链路回归风险。

## 4. 关键运行入口

单次训练：

```bash
python main.py --dataset_name assist_09 --model_variant assist_09_abce_best_full
```

A/E 消融：

```bash
python run_abce_ablation.py --datasets assist_09,junyi --seeds 42 --profiles best --gpus 2,3 --max_concurrent 2 --max_per_gpu 1 --ablations full,no_A,no_E
```

错误分析：

```bash
python tools/analyze_ae_errors.py --help
```

## 5. 验证方式

本仓库 smoke 文件是可直接执行脚本，不以 pytest collection 为主。

常用验证：

```bash
python -m py_compile main.py run_abce_ablation.py src\config.py src\dataset.py src\experiment_utils.py src\model.py src\model_cdm.py src\model_cdm_forward.py src\model_graph.py src\model_ops.py src\model_personal.py src\model_regularization.py src\model_structure.py src\model_structure_forward.py src\module_activity.py src\prediction_head.py src\trainer.py
python tests\smoke_ae_reliability_features.py
python tests\smoke_interpretable_ae.py
python tests\smoke_ablation_flags.py
python tests\smoke_prediction_head.py
python tests\smoke_sequence_prior.py
python tests\smoke_runtime_regressions.py
python tests\smoke_ae_rescue_regressions.py