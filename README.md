# ConceptSkillCDM

本仓库当前用于验证一个可解释认知诊断模型。论文叙事采用 **Concept Reachability under Sparse Response Evidence**：学生在测试题上的目标概念不一定直接出现在个人历史中，但可以通过训练集中的概念关系从历史概念“到达”。模型因此被解释为递进式的“全局路线图 + 个性化过滤器”：

- **CRG：Concept Reachability Graph，概念可达图。**
- **LCRF：Learner-Conditioned Reachability Filter，学习者条件化可达性过滤器。**

旧实验日志、结果表和 checkpoint 已按要求清理，不再作为当前代码说明的一部分保留。

## 1. 核心设计

### CRG：概念可达图

CRG 解决主问题：从训练集可观测证据中构建全体学生共享的概念可达图，判断当前题目概念能否从学生历史概念通过全局关系到达，并给出可解释的路线节点。

当前 CRG 的边权只来自 train-only evidence：

- item co-occurrence：同一题目中概念共同出现的关系；
- sequence transition：同一学生训练序列中的概念接续关系；
- self-retention：概念自身状态保持；
- receiver/source reliability：概念作为可达路线节点的全局统计可靠性。

CRG 的输出是 row-stochastic concept relation graph。它负责定义全局大地图和候选 support，不再被解释成任意神经网络学出的黑盒图。

### LCRF：学习者条件化可达性过滤器

LCRF 解决副问题：CRG 只说明“有路”，但有路不等于对当前学生可靠。LCRF 在 CRG 给出的局部可达 support 内，根据当前学生的掌握状态、近期表现和历史邻居证据，过滤出更适合该学生的局部辅导方向。

当前 LCRF 的局部打分来自：

- 当前题目概念的学生掌握度；
- 当前题目概念的近期掌握度；
- CRG 路线邻居概念的学生掌握度；
- CRG 路线邻居概念的近期掌握度；
- 当前概念与路线邻居之间的 readiness gap；
- train-only 学生-概念观测次数形成的可靠性门控。

LCRF 不生成新边，不读 valid/test，不使用 student-id embedding 作为 shortcut，也不使用 MLP 生成任意个性化图。它只在 CRG 给出的“小地图”里做学生级过滤。当前实现会显式记录并使用 `personal posterior - global CRG route` 的路线偏移：如果 LCRF 把当前学生从 CRG 的平均路线导向更薄弱的支撑概念，预测 logit 会受到有界惩罚；如果导向更稳的支撑概念，则会给出有界加成。

## 2. 当前代码入口

模型主入口：

- `src/model_cdm.py`：顶层 CDM 模型、CRG/LCRF logit 组装。
- `src/model_cdm_forward.py`：主 forward 路径。
- `src/model_graph.py`：CRG，概念可达图。
- `src/model_personal.py`：LCRF，学习者条件化可达性过滤器。
- `src/model_structure.py` / `src/model_structure_forward.py`：CRG/LCRF 与 knowledge encoder 的装配。
- `src/prediction_head.py`：固定认知诊断预测头。

重要诊断键：

- CRG：`roadmap_macro_logit_abs_mean`、`roadmap_difficulty_logit_abs_mean`、`roadmap_reliability_logit_abs_mean`。
- LCRF：`tutor_local_navigation_logit_abs_mean`、`tutor_current_mastery_logit_abs_mean`、`tutor_route_mastery_logit_abs_mean`、`tutor_gap_penalty_logit_abs_mean`。
- LCRF posterior：`ae_posterior_prior_logit_abs_mean`、`ae_posterior_prior_delta_abs_mean`、`tutor_posterior_mastery_shift_logit_abs_mean`。
- Support：`support_item_survival_rate`、`support_seq_survival_rate`、`support_self_retention_rate`。

## 3. 消融语义

- `full`：CRG 概念可达图 + LCRF 个性化过滤。
- `no_CRG`：移除概念可达图，LCRF 失去路线 support。
- `no_LCRF`：保留 CRG 的全局可达图，移除学生级局部过滤。
- `CRG_uniform`：保留图形态和参数量控制，但路线不使用 train-only evidence。
- `LCRF_shuffle_student`：打乱学生个性化证据，检查 LCRF 是否真的依赖学生历史。

后续实验必须优先检查：

```text
full > no_CRG
full > CRG_uniform
full > no_LCRF
full > LCRF_shuffle_student
```

如果 `LCRF_shuffle_student` 接近或超过 `full`，说明 LCRF 仍然不是有效的个性化过滤器，不能只靠 full AUC 宣称 LCRF 成立。

## 4. 小实验链路

四类机制实验按递进关系组织：

1. **数据现象表**：统计 direct seen、bridgeable@K、item co-occurrence、sequence density 和学生历史长度，证明数据集中确实存在“目标概念需要从历史概念到达”的问题。
2. **CRG Held-out Reachability Retrieval**：只用 train-only CRG 检索 valid/test 的后续概念，指标为 Hit@K、NDCG@K、MRR，用于证明 CRG 的关系证据充分。
3. **CRG Support Corruption**：固定 checkpoint，在 inference 时逐步破坏 CRG support，验证可达边是否必要。
4. **LCRF Counterfactual + Case**：固定 CRG 和 checkpoint，对比 actual / shuffled / mean / no_LCRF，并筛选同一 query 不同学生的 case，验证个性化过滤是否来自学生状态。

## 5. 验证方式

常用静态验证：

```bash
python -m py_compile main.py run_abce_ablation.py src/config.py src/dataset.py src/experiment_utils.py src/model_cdm.py src/model_cdm_forward.py src/model_graph.py src/model_ops.py src/model_personal.py src/model_regularization.py src/model_structure.py src/model_structure_forward.py src/module_activity.py src/prediction_head.py src/trainer.py
```

## 6. 远程服务器规范

服务器 `10.154.22.11` 上的代码同步必须走 git：

```bash
cd /home/zsh/ConceptSkillCDM
git pull
```

禁止使用 `echo <base64> | base64 -d | bash` 或其他不可审计的编码包装命令。临时命令必须是可读命令，清理 logs/results/checkpoints 前必须确认路径位于 `/home/zsh/ConceptSkillCDM` 内。
