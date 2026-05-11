# ConceptSkillCDM

本仓库当前用于验证一个可解释认知诊断模型。最新结构不再把 A/E 当成两个松散 residual，而是明确成递进式地图系统：

- **A：Global Curriculum Roadmap，全局学习路线图。**
- **E：Personalized Tutoring Navigator，个性化辅导导航图。**

旧实验日志、结果表和 checkpoint 已按要求清理，不再作为当前代码说明的一部分保留。

## 1. 核心设计

### A：全局学习路线图

A 解决主问题：从训练集可观测证据中构建全体学生共享的概念路线图，给出某个知识点附近的可解释路线节点。

当前 A 的边权只来自 train-only evidence：

- item co-occurrence：同一题目中概念共同出现的关系；
- sequence transition：同一学生训练序列中的概念接续关系；
- self-retention：概念自身状态保持；
- receiver/source reliability：概念作为路线节点的全局统计可靠性。

A 的输出是 row-stochastic concept relation graph。它负责定义大地图和候选 support，不再被解释成任意神经网络学出的黑盒图。

### E：个性化辅导导航图

E 解决副问题：在 A 给出的局部路线 support 内，根据当前学生的掌握状态选择更适合这个学生的局部辅导方向。

当前 E 的局部打分来自：

- 当前题目概念的学生掌握度；
- 当前题目概念的近期掌握度；
- A 路线邻居概念的学生掌握度；
- A 路线邻居概念的近期掌握度；
- 当前概念与路线邻居之间的 readiness gap；
- train-only 学生-概念观测次数形成的可靠性门控。

E 不生成新边，不读 valid/test，不使用 student-id embedding 作为 shortcut，也不使用 MLP 生成任意个性化图。它只在 A 给出的“小地图”里做学生级导航。当前实现会显式记录并使用 `personal posterior - global A route` 的路线偏移：如果 E 把当前学生从 A 的平均路线导向更薄弱的支撑概念，预测 logit 会受到有界惩罚；如果导向更稳的支撑概念，则会给出有界加成。

## 2. 当前代码入口

模型主入口：

- `src/model.py`：公开 `CognitiveDiagnosisModel`。
- `src/model_cdm.py`：顶层 CDM 模型、路线图/辅导图 logit 组装。
- `src/model_cdm_forward.py`：主 forward 路径。
- `src/model_graph.py`：A，全局学习路线图。
- `src/model_personal.py`：E，个性化辅导导航图。
- `src/model_structure.py` / `src/model_structure_forward.py`：A/E 与 knowledge encoder 的装配。
- `src/prediction_head.py`：固定认知诊断预测头。

重要诊断键：

- A：`roadmap_macro_logit_abs_mean`、`roadmap_difficulty_logit_abs_mean`、`roadmap_reliability_logit_abs_mean`。
- E：`tutor_local_navigation_logit_abs_mean`、`tutor_current_mastery_logit_abs_mean`、`tutor_route_mastery_logit_abs_mean`、`tutor_gap_penalty_logit_abs_mean`。
- E posterior：`ae_posterior_prior_logit_abs_mean`、`ae_posterior_prior_delta_abs_mean`、`tutor_posterior_mastery_shift_logit_abs_mean`。
- Support：`support_item_survival_rate`、`support_seq_survival_rate`、`support_self_retention_rate`。

## 3. 消融语义

- `full`：A 全局路线图 + E 个性化辅导导航。
- `no_A`：移除全局路线图，E 失去路线 support。
- `no_E`：保留 A 的全局路线图，移除学生级局部导航。
- `A_uniform`：保留图形态和参数量控制，但路线不使用 train-only evidence。
- `E_shuffle_student`：打乱学生个性化证据，检查 E 是否真的依赖学生历史。

后续实验必须优先检查：

```text
full > no_A
full > A_uniform
full > no_E
full > E_shuffle_student
```

如果 `E_shuffle_student` 接近或超过 `full`，说明 E 仍然不是有效的个性化小地图，不能只靠 full AUC 宣称 E 成立。

## 4. 验证方式

本仓库 smoke 文件是可直接执行脚本，不以 pytest collection 为主。

常用验证：

```bash
python -m py_compile main.py run_abce_ablation.py src/config.py src/dataset.py src/experiment_utils.py src/model.py src/model_cdm.py src/model_cdm_forward.py src/model_graph.py src/model_ops.py src/model_personal.py src/model_regularization.py src/model_structure.py src/model_structure_forward.py src/module_activity.py src/prediction_head.py src/trainer.py
python tests/smoke_interpretable_ae.py
python tests/smoke_ae_reliability_features.py
python tests/smoke_runtime_regressions.py
```

最小真实训练 smoke：

```bash
python main.py --dataset_name assist_09 --model_variant assist_09_abce_best_full --epochs 1 --batch_size 128 --max_train_batches 2 --max_val_batches 1 --max_test_batches 1 --num_workers 0 --no_cuda --save_dir checkpoints/local_route_map_smoke --log_dir logs/local_route_map_smoke
```

该 smoke 只用于确认训练链路、日志和 checkpoint 写入正常，不代表正式指标。

## 5. 远程服务器规范

服务器 `10.154.22.11` 上的代码同步必须走 git：

```bash
cd /home/zsh/ConceptSkillCDM
git pull
```

禁止使用 `echo <base64> | base64 -d | bash` 或其他不可审计的编码包装命令。临时命令必须是可读命令，清理 logs/results/checkpoints 前必须确认路径位于 `/home/zsh/ConceptSkillCDM` 内。
