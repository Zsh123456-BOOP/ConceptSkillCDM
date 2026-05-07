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
```

## 6. 给 GPT Pro 的提示词

下面这段可以直接交给 GPT Pro。它的任务不是继续改代码，而是结合仓库代码、当前 CD/KT 论文和现实教育测评场景，为 A、E 分别取名，并设计后续实验。

```text
你是一个严谨的认知诊断（Cognitive Diagnosis, CD）和知识追踪（Knowledge Tracing, KT）论文方法设计顾问。请你基于我提供的 Git 仓库代码、实验结果、以及近年 CD/KT 论文，帮助我为当前代码中的 A、E 两个模块分别命名，并把它们组织成一个有现实教育测评意义、科学问题递进清晰、可通过实验验证的论文方法故事。

请重点阅读这些文件，不要只根据摘要判断：
- README.md
- src/model.py
- src/model_cdm.py
- src/model_structure.py
- src/model_graph.py
- src/model_personal.py
- src/model_structure_forward.py
- src/model_cdm_forward.py
- src/prediction_head.py
- src/trainer.py
- src/dataset.py
- run_abce_ablation.py
- tools/analyze_ae_errors.py
- results/experiment_results.csv
- results/abce_ablation_diagnosis.csv

当前模型由固定认知诊断预测头和 A/E 结构模块组成。预测头不是我希望命名和包装的重点。

A 当前技术含义：
- train-only 全局概念关系 substrate；
- 使用 item-level concept co-occurrence；
- 使用 student trajectory 中的 sequence transition；
- 使用 self-loop 状态保持；
- 使用少量可解释的 receiver/self-loop 校正；
- 进入 concept graph propagation 和 relation-supported prediction residual；
- 目标不是为某个数据集打补丁，而是用统一公式处理 multi-concept 与 single-concept 数据。

E 当前技术含义：
- student-conditioned local posterior reweighting；
- 不生成任意新边；
- 不扩展 A 的 support；
- 只在 A 给出的 support 上，根据学生当前 concept-state contrast 对边权做局部 posterior 调整；
- 不读 student-id embedding 作为 shortcut；
- 不使用 MLP 黑盒学生图。

AE joint reliability 当前技术含义：
- 只在 A 和 E 同时开启时生效；
- 使用 train-only count/evidence reliability 特征；
- 特征包括 student count、exercise count、concept count、student-concept count 等；
- 它是 named feature 的线性校正，不是黑盒分支；
- no_A 和 no_E 中该项禁用。

当前实验结果：

| Dataset | Variant | Test AUC | Best Val AUC |
| --- | --- | ---: | ---: |
| assist_09 | full | 0.779646 | 0.778956 |
| assist_09 | no_A | 0.681722 | 0.681624 |
| assist_09 | no_E | 0.692737 | 0.693441 |
| junyi | full | 0.829135 | 0.824817 |
| junyi | no_A | 0.635569 | 0.641430 |
| junyi | no_E | 0.798560 | 0.789544 |

约束：
1. 不要把 A/E 命名成临时拼装味很重的名字。
2. 不要把 sequence transition 描述成严格 prerequisite，除非你能给出论文依据；更稳妥是 empirical transition、learning-path relation、temporal succession 等。
3. 不要设计 dataset-specific story，例如“Junyi 用一套、ASSIST09 用一套”。
4. 不要提出作弊式实验，例如利用 valid/test 信息、按测试集错误调结构、或对不同数据集使用不同规则。
5. 不要建议引入不可解释的 MLP/Transformer 黑盒分支来解释 A/E。
6. A 必须是主问题，E 必须是副问题；两者需要是递进关系，而不是两个平行堆叠模块。
7. 需要结合 CD/KT 论文和真实教育测评场景，不要只做工程解释。
8. 命名必须能在论文中自然展开，最好有英文全称、缩写、中文名、公式对应和现实问题对应。

请输出以下内容：

1. 文献与现实问题定位
- 总结传统 CDM 如何使用 Q-matrix、concept relation、student knowledge state；
- 总结近年图式 CD/KT 方法如何处理概念间关系；
- 说明当前方法与这些工作的差异；
- 提出 A 解决的主科学问题；
- 提出 E 解决的副科学问题；
- 两个问题必须环环相扣，A 为主，E 为副。

2. 为 A 和 E 分别取名
请分别给出 3 组候选名称，每组包括：
- 英文全称；
- 缩写；
- 中文名；
- 对应公式或代码含义；
- 名字优点；
- 可能被 reviewer 质疑的点；
- 最推荐的一组。
不要给整个 A/E 模块先取总名，先把 A 和 E 各自命名清楚。

3. 方法故事线
请组织成递进式故事：
- 现实教育测评问题；
- 传统方法不足；
- A 如何解决主问题；
- A 解决后还留下什么学生层面的异质性问题；
- E 如何在不破坏 A 可解释 support 的前提下解决副问题；
- 为什么 AE joint reliability 是合理的 train-only evidence reliability 校正，而不是作弊或黑盒。

4. 三个实验设计
请设计 3 个后续实验，实验后续会交给 Codex 跑。每个实验必须包括：
- 实验目的；
- 对比组；
- 数据集；
- 需要保存的指标；
- 预期结论；
- 结果不符合预期时如何解释；
- 需要 Codex 修改或新增的日志/CSV 字段；
- 适合绘制的新颖图形。

实验必须覆盖：
- A 是否真的学到了 train-only evidence relation，而不是任意图传播；
- E 是否真的做了 student-conditioned local posterior，而不是重复 A；
- A/E 的递进关系，即 E 的收益是否依赖 A 先提供合理 support。

请至少考虑这些候选对照：
- full；
- no_A；
- no_E；
- A_uniform；
- A_item_only；
- A_seq_only；
- A_self_only；
- E_frozen_alpha；
- E_prior_only；
- A_full_E_random_or_uniform_support；
- full without reliability residual。

5. 新颖实验图设计
请不要只建议普通柱状图。请提出 3 类更有信息量的图，例如：
- train-only evidence source survival Sankey/flow；
- A prior vs E posterior entropy/KL ridge plot；
- student-specific local reweighting case study heatmap；
- concept relation support overlap chord diagram；
- per-concept frequency vs A/E gain reliability calibration plot。

每个图要说明：
- 横纵轴或节点/边含义；
- 数据从哪些日志/CSV 字段来；
- 如何证明 A 或 E；
- 可能的 reviewer 质疑和防御说法。

6. 最终建议
最后给出：
- 推荐采用的 A 名称；
- 推荐采用的 E 名称；
- 一段可放进论文 introduction 的问题描述；
- 一段可放进 method 的模块描述；
- 一段可放进 experiment 的消融说明；
- 哪些实验必须先跑，哪些可以作为 appendix。

输出请具体，不要泛泛而谈。
```
