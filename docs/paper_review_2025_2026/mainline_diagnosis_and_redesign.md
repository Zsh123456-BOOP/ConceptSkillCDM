# CRG/LCRF 论文主线诊断与重构方案

> 生成于 2026-06-23，分支 `codex/recover-pre-cleanup-e92e448`。
> 本文件是内部诊断/规划文档，不并入论文正文。所有数字来自 `results/` 下既有实验产物的**再聚合**（见 `tools/diagnose_mainline_evidence.py` 与 `results/mainline_evidence_ledger/`），未改动任何实验数值。

---

## 0. 一句话结论

师兄的质疑**基本成立**（三条独立审稿视角一致判定 `yes_mostly`，high confidence）：
**当前论文用 CRG 解释“到达未观测知识点 / concept reachability”过强。** CRG 的 **AUC** 增益主要来自“图骨干/度结构”这一通用先验（与 degree-matched random 支持几乎无差别），并不来自它声称的“为未观测目标概念提供特定路线支持”。

但数据里**确实存在一个更窄、更真、能被证明的贡献**：在“目标概念缺乏直接历史证据”的子群里，**CRG 的具体 train-only 关系携带了超出度结构的“校准（BCE）”信息**（assist_09 +0.032、assist_17 +0.071 BCE，相对 degree-random）。这应当成为新主线，而不是 ranking/AUC 叙事。

---

## 1. 当前论文核心漏洞诊断

### 漏洞 A：问题规模与叙事不匹配（headline 是 3% 的尾部）
`dataset_story_cards`：direct-unseen rate = **3.10%**（assist_09）、**2.78%**（assist_17）、**100%**（junyi，单概念退化，指标无意义）。
论文把“concept evidence gap（目标概念未出现在学生历史中）”当主问题，但在两个 assist 主数据集上它只覆盖约 3% 的测试样本。一个 3% 的尾部无法解释 Table 1 里全局 AUC 第一的结果。

### 漏洞 B：难度前提不成立 / 跨数据集不一致
“未观测 = 更难”这一前提只在 assist_17 成立，在 assist_09 反转：

| dataset | direct_seen clean AUC/BCE | direct_unseen clean AUC/BCE | 是否更难 |
|---|---|---|---|
| assist_09 | 0.7726 / 0.5350 | **0.8180 / 0.3946** | 否（unseen 反而更易） |
| assist_17 | 0.7863 / 0.5502 | 0.7359 / 0.6178 | 是 |

在 assist_09 上，未观测样本 AUC、BCE 都更好。把“未观测是失效场景”当通用前提站不住。

### 漏洞 C：no_CRG 消融测的是“整张图骨干”，不是“支持模块”（最关键）
代码确认 CRG 的邻接 `A` **一物两用**：既是 GNN 消息传递邻接，又是 LCRF 的支持集来源。
- `src/model_cdm.py:134-143`：`ablate_module1`/no_CRG 同时强制 `use_concept_graph=False` 且 `use_personal_graph=False`。
- `src/model_structure_forward.py:52,55,57`：`use_concept_graph=False` 时把 `relation_matrices` 换成 identity，直接喂给 `knowledge_encoder`（GNN）。
- `src/model_graph.py:355`：同一 `relation_matrices` 用于 `ConceptGraphConv` 消息传递。

因此 **no_CRG 的 1–2% AUC 跌落（assist_09 0.0112、assist_17 0.0200）= “有图 CD vs 无图 CD”，而非“有无未观测路线支持”。** 论文却把它解读成路线支持的作用。

### 漏洞 D：增益落在“错误的子群”
coverage-conditioned（`results/main_problem_experiments_20260523`，ΔAUC = full − variant，CI 见 ledger）：
- 显著的 full_vs_no_CRG 增益落在 **high_route_mass**（assist_09 +0.0081 CI[0.0036,0.0124]；assist_17 +0.0151 CI[0.0123,0.0178]）和 **direct_seen**，
- 而真正的 **direct_unseen_bridgeable** 子群：assist_09 no_CRG **−0.0049（符号相反，CI 跨 0）**、no_LCRF +0.0001；assist_17 no_LCRF **−0.0055**。

即：方法主要在“概念已被观测 / 路线质量高”时起作用，与“桥接未观测概念”的叙事相反。

### 漏洞 E：degree-matched 控制让“特定关系”叙事在 AUC 上塌掉
`tools/analyze_crg_support_corruption_controls.py:343-404`，full corruption（ratio=1.0）下 `evidence_minus_degree_random_auc`：
- assist_09 ≈ **0.0002**（corpus）、junyi ≈ **−0.0009**：删掉“真实 top 证据边”与删掉“同度随机边”掉得一样多 → 特定关系在 AUC 上不重要，只有度结构重要。
- assist_17 = **+0.0079**（corpus）是唯一在 AUC 上有特定性的数据集。

### 漏洞 F：LCRF 反事实的大跌幅是“学生 ID 混淆”，不是 LCRF 过滤
`mean_state`/`shuffle_state` 沿**学生维度**置换/平均 student-indexed buffers（`model_cdm.py:1128-1162`，student_concept_prior/recent/count 与 ae_student_prior_logit 都按 student_id 索引）。所以 0.14–0.20 的 AUC 跌落测的是“预测对每个学生身份信号的依赖”，不是“LCRF 对 CRG 支持的重加权”。
**干净的 LCRF 专属控制是 `no_filter`**：assist_09 仅 **0.0149**、assist_17 仅 **0.0018**、junyi 0.0005 —— 比 mean/shuffle 小 1–2 个数量级。论文用 mean/shuffle 当 LCRF 必要性证据是夸大。

### 漏洞 G：跨数据集“两根支柱反相关”
唯一支持“特定关系有用”的数据集是 assist_17（漏洞 E），而 assist_17 恰好是 **history-to-query 检索失败**的数据集（Hit@10 0.078/0.093 < random 0.120）。反过来，检索成功的 assist_09/junyi 没有 AUC 特定性。**没有任何单一数据集端到端地同时支撑“路线检索 + 关系特定性”整条故事。**

### 漏洞 H（次要）：Exp3 是缓冲态遮罩，非原始历史重算
`run_main_problem_experiments.py:516-530` 遮罩的是聚合后的 student-concept 状态 buffer，代码自述非 raw-history 重算。stress test（`run_gap_stress_and_lcrf_strata.py:477-481`）的直接证据遮罩只在 `direct_cov_cnt>0`（已观测）样本上做，根本没覆盖未观测regime。

---

## 2. CRG 作为 train-only 统计结构，最多能支撑什么 claim

按数据，CRG 能诚实支撑的**最大** claim：

1. **（图骨干）** 一个 train-only 概念关系图作为结构先验，能整体提升 CD 的 ranking（AUC）——但这部分本质是度/连通结构，**不是**特定可达路线，且并非本文独有创新（图 CD 已有）。
2. **（关系特定性，仅校准）** 在“缺乏直接证据”的子群里，CRG 的**具体**关系携带超出度结构的**校准（BCE）**信息：用 degree-matched random 替换会更严重地恶化 BCE（assist_09 +0.032、assist_17 +0.071，见下表），两个 assist 数据集一致。
3. **（检索）** train-only 关系在 2/3 数据集（assist_09、junyi）上能以远超随机的 Hit@10 从历史概念检索到目标概念——这是图统计的真实性质，但**不等于**它改善了对未观测概念的*诊断*。
4. **（个性化，弱）** 去掉 learner-state 校正（干净的 no_filter）在 assist_09 有小而非零的影响（0.015 AUC）；assist_17/junyi 接近 0。

不能支撑：**“CRG 解决/桥接未观测知识点、实现 concept reachability、并由此带来主表 AUC 提升”**——这是过强 claim。

### 关键证据表（Q3，full corruption，evidence − degree-matched random）

| dataset | subgroup | ΔAUC(ev−deg) | ΔBCE(ev−deg) |
|---|---|---|---|
| assist_09 | direct_seen | −0.0013 | 0.0006 |
| assist_09 | **direct_unseen** | 0.0007 | **0.0322** |
| assist_17 | direct_seen | 0.0055 | 0.0014 |
| assist_17 | **direct_unseen** | 0.0086 | **0.0710** |
| junyi | all | −0.0009 | 0.0140 |

→ **AUC 几乎全靠度结构；唯一稳定、跨数据集、且落在未观测子群的“关系特定”效应在校准（BCE）上。**

---

## 3. 新论文主线（problem / method / experiment 自洽）

### 主线一句话
> **概念证据缺口下的“校准”问题**：当目标概念缺乏学生的直接历史证据时，CD 模型仍能给出可用的*排序*，但其掌握度估计**校准失真**（缺少可锚定的直接观测）。我们用一个 train-only 概念支持先验（CRG）在该regime 提供**超出通用共现/度结构**的校准信号，并用 learner-conditioned 过滤（LCRF）按学生状态调节该支持。

### 为什么这个问题真实存在（数据已能证明）
- 缺口存在且可度量（direct-unseen / weak-direct-evidence 子群可定义）。
- 在该子群里，**特定** CRG 关系对校准有因果可验证的贡献（degree-matched 控制：BCE +0.032 / +0.071，两数据集一致）——这是 degree/backbone 解释**不能**覆盖的部分。
- 该 claim 是窄的、可证伪的、跨数据集一致的——正好避开师兄的“只是 train-only 统计图”反驳：是的，对 AUC 它接近通用统计；但对“缺口下的校准”它携带了特定关系信息。

### 与旧主线的区别
- 旧：unobserved concept reachability + AUC 主叙事（被 B/C/D/E/G 证伪）。
- 新：concept-evidence-gap **calibration** + 度控对照证明“关系特定性”（被 Q3 证明）。AUC 主表降级为“具竞争力的整体性能”，不再充当机制证据。

### 备选/收敛轴：用 `high_route_mass` 作为“前置条件成立时方法生效”
显著且跨数据集的 CRG 增益落在 high_route_mass（§1 漏洞 D）。可作为“当且仅当存在足够路线支持时，CRG 起作用”的内在一致性证据，与校准主线互补。

---

## 4. 方法重构方案：保留 / 弱化 / 替换

| 模块 | 处置 | 理由 |
|---|---|---|
| CRG（图骨干角色） | **保留**，但**降级表述**为“train-only 概念关系结构先验”，不再叫“可达图/桥接未观测” | AUC 增益是度结构，诚实即可，不是创新点 |
| CRG（支持集角色） | **重构：与骨干解耦**（核心改动） | 当前 A 一物两用，导致无法干净隔离“支持”作用（漏洞 C/E） |
| LCRF | **弱化 + 正名** | 干净效应（no_filter）很小且 dataset-dependent；mean/shuffle 是学生身份混淆，应改作“student-identity dependence 诊断”，不作 LCRF 必要性证据 |

### 核心方法改动（让主线可被证明）：支持集与消息传递图解耦
**问题**：现在 LCRF 的支持集 = GNN 邻接 = 同一 `A`，所以任何对支持的扰动都同时动了表征骨干，无法回答“支持本身是否有用”。
**改法（精炼、边界清楚）**：让 LCRF 消费一个**独立的支持索引张量** `S_support`（可与 `A` 同源初始化，但在前向中作为单独输入，可被单独冻结/扰动/置随机），而 GNN 仍用真实 `A` 传播。这样：
- `freeze A, randomize only S_support` 成为干净的“支持作用”对照（当前缺失，三审稿一致点名为最该补的控制）。
- 不改变默认训练行为（默认 `S_support` 来自 `A` 的 top-K，等价当前模型）。
- 边界清楚：CRG 产生 (A 用于表征, S_support 用于 LCRF)；LCRF 只在 S_support 上做后验，不新增边。

> 实现位置建议：在 `MultiHeadRelationLearning` 暴露 `support_index`（已有 `item_support_mask`/`sequence_support_mask`），在 `model_cdm.py` 的 LCRF 路径里改为读取该独立索引而非从 `relation_matrices` 再推；新增 `--decouple_support` 开关，默认 off（=当前行为，保数值不变）。这是**唯一必要的模型内改动**，其余都是实验脚本与表述。

---

## 5. 实验闭环（每个实验回答什么 / 怎么跑 / 判据）

> 设计原则：先证缺口存在 → 再证普通 CD 在缺口样本更难（校准） → 再证 CRG 特定支持有效（度控） → 再证 LCRF 学生级过滤 → 最后 deletion/corruption/shuffle/counterfactual 证支持非装饰。多数可**复用现有 full/no_CRG/no_LCRF checkpoint，纯推理**。

| # | 实验 | 回答的问题 | 怎么跑 | 成功判据 |
|---|---|---|---|---|
| E0 | 缺口画像 + 难度 | 缺口是否存在、是否更难 | `tools/diagnose_mainline_evidence.py`（已可本地跑，复用 CSV） | 报告 prevalence + 按子群 clean BCE；**诚实写跨数据集差异**（assist_17 更难、assist_09 校准也依赖支持） |
| E1 | 整体性能（主表） | 是否具竞争力 | 现有 Table 1（已有） | full AUC 三数据集第一即可；**不再当机制证据** |
| E2 | 缺口校准曲线 | 普通 CD 在缺口样本是否校准更差，CRG 是否恢复 | 复用 checkpoint，按 query 目标概念历史计数分桶(0/1-2/3-5/>5)报告 **BCE**：full vs no_CRG vs degree_random | 计数→0 时 BCE 上升；full 相对 no_CRG/degree_random 在低计数桶 BCE 改善（CI 不跨 0） |
| E3 | **关系特定性（度控，主力机制证据）** | CRG 的特定关系是否超出度结构、且在缺口子群 | 现有 `analyze_crg_support_corruption_controls.py`，**按 direct_seen/direct_unseen 分层报告 evidence_minus_degree 的 BCE** | direct_unseen 子群 ΔBCE(ev−deg) 显著>0 且 > direct_seen（已有：+0.032/+0.071） |
| E4 | **支持解耦对照（新，需小改模型）** | “支持”作用能否与“图骨干”分离 | `--decouple_support` 训练 full；推理时 `freeze A, randomize S_support` | 随机化 S_support 在缺口子群 BCE 上升且 > 随机化等量 degree-matched；证明支持非骨干副产物 |
| E5 | LCRF 干净反事实 | learner-state 过滤是否有效（去身份混淆） | `run_gap_stress_and_lcrf_strata.py` 取 **no_filter**，按 direct_cov 分桶 | no_filter ΔAUC/ΔBCE 在目标子群>0；**mean/shuffle 仅作 student-identity 诊断附录** |
| E6 | same-query 个案 | 同题同支持，不同学生是否得到不同后验 | 现有 `export_lcrf_same_query_posterior.py` | 后验权重随 support mastery/count 共变（已有图） |
| E7 | 检索（降级为辅助） | train-only 关系是否含路线信号 | 现有 Exp1 | 报告为 dataset-dependent（assist_09/junyi 成功、assist_17 失败），**不作主问题证据** |

deletion/corruption/shuffle/counterfactual 已被 E3（corruption + degree + shuffle + self-only）、E4（randomize support）、E5（state counterfactual）覆盖。

---

## 6. 现有结果可直接用 / 不可支撑主线

**可直接用（reframe 后）**
- `table_main_ablation_core3` / Table 1 整体性能 → 作“具竞争力性能”，不作机制证据。
- `crg_subgroup_support_dependence_core3` 的 **BCE** 分层 evidence−degree（E3）→ **新主力机制证据**。
- `lcrf_counterfactual` 的 **no_filter** 行（E5）→ LCRF 干净证据。
- `crg_retrieval` / Exp1（E7）→ 辅助、dataset-dependent。
- same-query 个案（E6）→ 个性化定性证据。

**不可支撑主线 / 需降级或改写**
- 用 no_CRG 消融 AUC 跌幅论证“路线支持作用” → 改为“图骨干作用”，机制结论删。
- coverage-conditioned 在 direct_unseen_bridgeable 的 AUC 增益 → 弱/反号，**降级为负向/null 结果诚实报告**，不要硬解释。
- LCRF mean_state/shuffle_state 大跌幅 → 改作 student-identity dependence 诊断（附录），不作 LCRF 必要性。
- 检索的 assist_17 → 明确写失败，作对照而非正例。
- “reach unobserved concepts / concept reachability” 标题与叙事 → 替换为 calibration under concept-evidence gaps。

---

## 7. 待办（按依赖排序）
1. 纯推理即可的 E0/E3-分层/E5/E7 聚合 → 复用 checkpoint，无需重训（见 §8 Codex 提示词）。
2. E2 校准曲线分桶聚合 → 复用 checkpoint 推理。
3. E4 解耦对照 → 唯一需要小改模型 + 重训 full 的实验（`--decouple_support`）。
4. 论文重写：标题/摘要/引言/问题定义/方法表述/实验组织，按上表 reframe（遵循 `codex_cd_writing_guardrails.md`）。
