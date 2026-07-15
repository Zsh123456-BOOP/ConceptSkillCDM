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

## 6.5 Codex 实验闭环回收（run `mainline_evidence_20260623_065509`，已核对原始 CSV）

> ⚠️ `results/.../mainline_review_packet.md` 标“全 pass”不可信：它的 gate 过宽（E4 只要 `bce_increase>0` 即 pass，连 4e-05 也算；E2 只要 no_CRG **或** degree_random 任一动了就 pass）。可信的是各实验自带的 `success_gate_summary.csv` 与原始 metrics CSV。下面结论全部基于原始数。

**核心新发现：连“校准靠特定支持”这个收窄后的主线也被 E4 证伪。支持集与 LCRF 重加权对预测基本是装饰；CRG 的全部贡献来自图骨干（度结构驱动）。**

| 实验 | 干净判据结果 | 原始关键数 | 含义 |
|---|---|---|---|
| **E4 解耦支持（决定性）** | 实质 **FAIL**（packet 误标 pass） | 冻结 A、只扰动解耦支持集：direct_unseen ΔBCE = **0.003 / 0.0005 / 0.0004**（09/junyi/17），且 `evidence_minus_degree ≈ 0`，CI 多跨 0（17: ci_low −0.0004） | 支持集**非承载**。E3 的校准效应来自 A 的骨干角色，不是支持 |
| E2 缺口校准 | no_CRG 显著、degree_random≈0 | bucket0 ΔBCE：no_CRG +0.0176/+0.0191（CI>0），degree_random +0.0015/+0.0005（CI 跨 0） | 同一结论：动**骨干**有效，动**支持**无效 |
| E3 关系特定性（纠缠） | support_pass=True，但 posterior_pass=False，overall FAIL | direct_unseen ev−deg ΔBCE +0.0337/+0.0710 | 纠缠测试（mask 同时是 A+支持）→ 效应被 E4 归因到骨干 |
| E5 LCRF 干净 | **FAIL**（三数据集） | 标签被重排：新 `no_LCRF`(=旧干净 no_filter) 在 unseen ΔAUC≈**0.001**、ΔBCE≈0.033；新 `no_filter` AUC=**0.4367<随机**（退化变体，无效）；mean/shuffle 仍是学生身份混淆 | LCRF 干净效应小且**非缺口定向**；大数字是混淆/退化 |

**两条独立干净对照（E4 解耦扰动、E2 degree_random 支持腐蚀）一致表明支持集不承载**；E5 严格 gate 表明 LCRF 过滤非缺口定向。师兄的质疑被进一步坐实，且我先前提出的“校准收窄主线”也不成立。

### 决策点（下一步）
- **路径 1（诚实降级，低风险/低新意）**：放弃 CRG-as-reachability 与 LCRF-as-filtering 作为贡献，写成“train-only 概念图结构先验改善 CD（尤其低证据样本的校准）”。支持集/LCRF 退为“试过但非承载”的组件或删除。新意薄。
- **路径 2（让命题为真后再证，高投入/真贡献，推荐）**：当前任务让 per-student/per-concept embedding + 骨干就能记住一切，所以路线支持从不被需要 → 支持/LCRF 必然退化。要让支持承载，须改**任务+方法**：
  - 任务：构造真正的 concept-evidence-gap split（把目标概念从该学生的**训练历史**中挖空 / cold-concept-per-student / cold-concept 迁移），使模型无法靠直接目标概念 embedding 作答；
  - 方法：让目标概念掌握度**必须经由** LCRF 加权支持聚合得到（限制骨干对目标概念的直接编码），使支持成为唯一跨概念通路；
  - 复跑 E4：若解耦支持扰动这时出现稳健、超越 degree 的效应 → 命题被真正证明。

## 6.6 已落地的修复与 Path 2 第一步（2026-06-23，本地已验证）

**修复 1 — review packet 诚实化**（`tools/build_mainline_review_packet.py`）：gate 改为要求 幅度阈值(`--min-bce-effect`,默认 0.005)+CI 下界>0+(支持类)击败 degree 对照；E2 把 backbone(no_CRG) 与 support(degree_random) 分开报；E4 要求 击败 degree 才算 pass；E5 排除退化变体(AUC<0.55)且要求有效 overall 基线。复跑既有 run 的诚实结论：**E2=backbone-only(support 装饰)；E4=fail；E3=pass(需 E4 确认,纠缠)；E5=仅 assist_17 用 retrained no_LCRF 有 gap-specific 弱信号**。

**修复 2 — E5 变体纠正**（`tools/run_gap_stress_and_lcrf_strata.py`）：删除退化的 `no_filter`（在训练时开启 personal graph 的 checkpoint 上推理期禁用 → 训练/推理失配 → AUC 0.44<随机）；新增干净的 `lcrf_state_off`（推理期把 `personal_delta_scale=0`，只去掉 z_state、保留 CRG 先验支持混合，见 `src/model_structure_forward.py:329`）；`mean_state/shuffle_state` 明确标注为学生身份置换诊断。**需服务器用 checkpoint 重生成 E5 CSV。**

**Path 2 第一步 — cold-concept-per-student split**（`tools/build_cold_concept_split.py`，本地已生成并验证）：
- 把某概念从“该学生”训练历史整体挖空（其它学生仍保留 → CRG 仍能为该概念建支持边），保留该学生测试查询 → 制造真正“目标概念零训练历史”的缺口。
- 结果(holdout_frac=0.30, seed42)：assist_09 cold-query rate **32.9%**(17285)、assist_17 **27.9%**(21552)、junyi **0%**(其本就 native-cold)；概念全局保留 112/112、79/79。产物在 `data/<ds>_coldconcept/`。
- ⚠️ **junyi 警示**：junyi 本就 100% direct-unseen 且 E4 仍显示支持装饰 → 仅改任务可能不足以让支持承载。因此采用**增量验证**：先在 cold split 上复跑 E4，若支持仍装饰，再做方法改动。

### Path 2 增量计划（方法改动已实现）
1. （已完成）造 cold-concept split：`tools/build_cold_concept_split.py`。
2. （已完成，待服务器验证）方法改动 `--support_only_unseen`：对 direct-unseen 查询行（该学生该目标概念训练观测数=0），把目标概念表征替换为 LCRF 支持加权聚合 `post_local_full`(=Σ P·value(support))，强制预测经支持路由；默认 off=数值不变。实现：[model_cdm_forward.py](../../src/model_cdm_forward.py)(替换逻辑)、[model_cdm.py](../../src/model_cdm.py)(flag + `post_local_full` 返回)、main/trainer 透传，冒烟测试 `tests/smoke_support_only_unseen.py`。
3. （服务器）两组对照训练（cold split，超参沿用基数据集）：
   - **Run A（仅任务）**：full `--decouple_support True`。
   - **Run B（任务+方法）**：full `--decouple_support True --support_only_unseen True`。
4. 判读（区分两类证据，避免循环论证）：
   - **充分性（Run B 的非循环核心指标）**：Run B 在 cold-unseen 查询上的 AUC/BCE 是否与 Run A 相当（不崩）。相当 → 支持是缺口下**充分**的诊断信号（命题成立）；崩 → 支持不足以替代直接证据（命题证伪，诚实写）。
   - **承载性（E4）**：Run A 上解耦支持扰动是否 ΔBCE≥0.005、CI>0、超 degree → 任务本身是否已让支持承载。（Run B 上 E4 必然受影响，属构造性，只用来验证“特定关系>degree”，不单独当承载证据。）
   - **特定性**：两组 E4 的 `evidence_minus_degree` 是否>0（特定 train-only 关系优于同度随机）。
   - junyi 不参与（native-cold，且单概念无可桥接支持）。

## 6.7 Codex 服务器运行提示词（Path 2：cold-concept 任务 + 方法改动 + 两项修复重生成）

```text
背景:分支 codex/recover-pre-cleanup-e92e448。目标=验证“缺口下 CRG 支持是否真的承载并充分”。
本轮已在本地完成的代码改动(只需在服务器运行,勿重写):
  - tools/build_cold_concept_split.py(造 cold-concept-per-student split)
  - 方法改动 --support_only_unseen(model_cdm_forward.py / model_cdm.py / trainer.py / main.py),默认 off=数值不变
  - 冒烟测试 tests/smoke_support_only_unseen.py
  - 修复:tools/build_mainline_review_packet.py(诚实 gate)、tools/run_gap_stress_and_lcrf_strata.py(lcrf_state_off 替换退化 no_filter)
硬约束:不改既有实验数值;新产物落新目录(带日期戳);真实 checkpoint 在 /home/zsh/ConceptSkillCDM/checkpoints/...。

步骤 0(冒烟,必须先过):
  python tests/smoke_support_only_unseen.py        # 验证 off==baseline 且 unseen 时 on 改变预测且有限
  python tests/smoke_ablation_flags.py             # 既有 flag 冒烟
  若 smoke 失败,先报错停下,不要继续训练。

步骤 1(数据):
  python tools/build_cold_concept_split.py --dataset assist_09 assist_17 --holdout-frac 0.30 --seed 42
  检查 summary:cold_concepts_lost_from_global_train=0,cold-query rate≈0.25~0.35。junyi 跳过(native-cold)。

步骤 2(训练,GPU,tools/select_idle_gpus.py 选卡;超参沿用基数据集,--data_dir 指向 data/<ds>_coldconcept,日志确认路径):
  Run A(仅任务):  full  --decouple_support True
  Run B(任务+方法):full  --decouple_support True  --support_only_unseen True
  各数据集(assist_09/assist_17) seed 42(可加 43/44)。两组 checkpoint 分目录,勿覆盖旧。

步骤 3(推理评测,仅 cold-query 子群 test_cold_annotation.is_cold_query==True;Run A、Run B 各跑):
  - 充分性(Run B 核心):在 cold-unseen 查询上算 Run B 的 AUC/BCE,与 Run A 比较(同子群)。
  - E4:tools/run_decoupled_support_control.py(冻结 A、只扰动解耦支持集;evidence vs degree_matched_random)。
  - E2:tools/run_gap_calibration_curve.py;E5:tools/run_gap_stress_and_lcrf_strata.py(已修,产 lcrf_state_off);E3:tools/analyze_crg_support_corruption_controls.py。

步骤 4(汇总,诚实 gate):
  python tools/build_mainline_review_packet.py --run-root <新run目录> --min-bce-effect 0.005
  python tools/diagnose_mainline_evidence.py --new-run-root <新run目录>

判据(区分两类,避免循环论证):
  - 充分性[非循环,主判据]:Run B 在 cold-unseen 上 AUC/BCE 是否与 Run A 相当(差距 ΔAUC≤~0.01)。
    * 相当 → 支持是缺口下充分诊断信号,命题成立,主线=“缺口下经支持路由的诊断”。
    * 崩(ΔAUC≫0.01 或 BCE 飙升)→ 支持不足,命题证伪,如实写,不硬解释。
  - 承载性 E4@RunA:解耦支持扰动 ΔBCE 在 cold 子群 ≥0.005、CI>0、超 degree → 仅任务即让支持承载。
  - 特定性:两组 E4 的 minus_degree_random_bce_increase>0(特定关系优于同度随机)。
  - Run B 的 E4 受影响是构造性的,只用于验证“特定>degree”,不单独当承载证据。
  - junyi 不参与;assist_09/assist_17 分开下结论。不通过判据一律标 fail(遵循 codex_cd_writing_guardrails.md)。
```

## 7. 待办（按依赖排序）
1. 纯推理即可的 E0/E3-分层/E5/E7 聚合 → 复用 checkpoint，无需重训（见 §8 Codex 提示词）。
2. E2 校准曲线分桶聚合 → 复用 checkpoint 推理。
3. E4 解耦对照 → 唯一需要小改模型 + 重训 full 的实验（`--decouple_support`）。
4. 论文重写：标题/摘要/引言/问题定义/方法表述/实验组织，按上表 reframe（遵循 `codex_cd_writing_guardrails.md`）。
