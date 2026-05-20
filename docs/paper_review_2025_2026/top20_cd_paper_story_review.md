# 2025-2026 CD 顶会顶刊论文叙事与实验设计复盘

本文档用于决定 ConceptSkillCDM 后续实验是继续当前重构、回溯到强消融版本，还是重新定义 CRG/LCRF 模块边界。主表阅读对象限定为 2025-2026 年 cognitive diagnosis 或强相关智能教育诊断论文，准入标准收紧为 CCF-A 会议或中科院一区期刊；不满足该标准的论文只能作为附录灵感，不能作为主叙事依据。JCR Q1、SJR Q1 或普通影响因子排名只能作为辅助信息，不能替代中科院一区。

需要先说明边界：严格只算 2025-2026 年 CCF-A/一区且完整可下载的 CD 论文，数量可能不足 20 篇。因此本文档采用“核心深读论文 + 指标参考论文 + 附录灵感论文”的分层结构。每篇都必须保留来源路径、venue 等级依据、PDF/代码链接、数据集与指标可比性判断；未确认正式录用或等级不满足的条目必须降级为附录。

## 重读执行总控清单

更新规则：每完成一个阶段，先在本清单中把对应项从 `[ ]` 改为 `[x]`，再继续下一阶段。任何没有完成准入审计、公式拆解和实验拆解的论文，不得进入最终核心深读表。

- [x] 阶段 0：把重读方案与执行约束写入本文档。
- [x] 阶段 1A：建立候选论文审计表骨架，列出当前已有 PDF、年份、venue 和待核验状态。
- [x] 阶段 1B：逐篇核验 venue 等级来源：CCF-A 会议或中科院一区期刊；不满足者标记为降级。
- [x] 阶段 1C：逐篇核验正式发表状态、PDF 来源、代码链接、任务相关性和是否可用于主叙事。
- [x] 阶段 2：将候选论文分为三类：核心深读论文、指标参考论文、附录灵感论文；不再为了凑 20 篇把弱相关或低等级论文混入主表。
- [x] 阶段 3：为每篇核心深读论文补全固定复盘模板：现实问题、研究缺口、符号定义、公式推理、模块边界、算法流程、实验设计、结果写法、可视化证据、可模仿点和不可模仿点。
- [x] 阶段 4：逐篇拆公式：基础 CD/KT 公式是什么，新变量在哪里引入，loss 每一项对应什么机制，消融删除的是公式里的哪一项，训练与推理是否一致。
- [x] 阶段 5：逐篇拆写作结构：Introduction 每段如何铺垫，contribution 如何组织，method 如何承接问题，experiment 如何从主表过渡到机制实验。
- [x] 阶段 6：建立数据集指标对照表，记录 ASSIST09、Junyi、ASSIST17、NIPS34、FrcSub、EdNet-KT1、ASSIST12、Math2 等数据集在顶会顶刊论文中的 reported result、split/cleaning、metric 和是否可直接比较。
- [x] 阶段 7：从论文中抽取可迁移实验模板，映射到 CRG/LCRF：CRG retrieval、CRG support corruption、LCRF actual/shuffle/mean、same query different learners、reachability subgroup/case。
- [x] 阶段 8：输出对 ConceptSkillCDM 的最终决策：保留稳定代码、轻改命名/日志、补机制实验，还是重构模块边界。

### 核心深读模板

每篇核心论文必须按下面结构重写，不能只写摘要：

1. 基本信息：论文名、年份、venue、等级依据、PDF、代码、任务相关性、数据集、指标。
2. 现实问题：作者从哪个教育场景或数据现象切入，是否有统计或案例支撑。
3. 研究缺口：现有 CD/KT 方法缺什么，缺口是否被具体化成可验证问题。
4. 符号与问题定义：学生、题目、知识点、Q 矩阵、作答记录、预测目标和新任务设置。
5. 公式推理：基础公式、新变量、模块公式、loss/regularization、训练推理一致性、消融对应项。
6. 模块边界：主模块/副模块各解决什么，二者是并列、递进、过滤、校正还是蒸馏关系。
7. 算法流程：是否有 algorithm block，构图/预训练/反事实/检索/推理阶段如何组织。
8. 实验设计：主表、baseline、消融、小实验、鲁棒性、敏感性、case、热力图、t-SNE、效率。
9. 结果写法：提升大或小时作者如何解释，是否使用场景分组、显著性检验或机制图补强。
10. 对 CRG/LCRF 的启发：能直接借鉴什么，哪些不能借鉴，需要哪些数据统计才能支撑。

### 数据集指标记录模板

| dataset | paper | venue/level | split/cleaning | metric | reported result | comparable to ours? | note |
|---|---|---|---|---|---:|---|---|

不同 split、不同过滤规则、不同任务定义的结果只能作为参考区间，不能直接写成超过 SOTA。

### 候选论文审计表

当前表是本轮审计后的可执行分类。会议等级按 CCF 2026 目录记录；期刊如果没有可审计的中科院一区证据，不进入核心深读，只保留为附录灵感或条件参考。

| ID | 论文简称 | 年份 | venue/期刊 | 当前 PDF | 等级核验状态 | 正式发表/代码状态 | 初步处理 |
|---|---|---:|---|---|---|---|---|
| NCDLA | Noise-Aware Graph-based CD Through Low-Rank Alignment | 2026 | AAAI | `aaai26_ncdla.pdf` | CCF 2026: A | PDF 已本地化；正式 AAAI 论文 | 核心深读 |
| DBCD | Debiased Cognitive Diagnosis | 2026 | AAAI | `aaai26_dbcd.pdf` | CCF 2026: A | PDF 已本地化；正式 AAAI 论文 | 核心深读 |
| KCD | Knowledge is Power | 2025 | AAAI | `aaai25_kcd.pdf` | CCF 2026: A | PDF 已本地化；代码/数据链接见论文 | 核心深读 |
| DMC-CDM | Multi-Perspective Consolidation | 2025 | AAAI | `aaai25_dmccdm.pdf` | CCF 2026: A | PDF 已本地化；代码链接见论文 | 核心深读 |
| AD4CD | Causal-Guided Anomaly Detection | 2025 | AAAI | `aaai25_ad4cd.pdf` | CCF 2026: A | PDF 已本地化；任务强相关 | 核心深读 |
| FACD | Fast-Adaptive Cognitive Diagnosis | 2025 | IJCAI | `ijcai25_facd.pdf` | CCF 2026: B | PDF 已本地化；若按旧 CCF 目录可恢复 | 条件参考 |
| KAN2CD | KAN for Neural CD | 2025 | IJCAI | `ijcai25_kan2cd.pdf` | CCF 2026: B | PDF 已本地化；若按旧 CCF 目录可恢复 | 条件参考 |
| OSCD | One-Shot NAS for Robust CD | 2026 | KDD/arXiv | `kdd26_oscd.pdf` | CCF 2026: A | 正式 KDD proceedings 未完成核验 | 条件参考 |
| DFCD | Dual-Fusion Cognitive Diagnosis | 2025 | KDD | `kdd25_dfcd.pdf` | CCF 2026: A | PDF 已本地化；KDD CD 强相关 | 核心深读 |
| ISG-CD | Informative and Stable Graph CD | 2025 | KDD | `kdd25_isgcd.pdf` | CCF 2026: A | PDF 已本地化；KDD graph-CD 强相关 | 核心深读 |
| LRCD | Language Representation Cognitive Diagnosis | 2025 | KDD | `kdd25_lrcd.pdf` | CCF 2026: A | PDF 已本地化；KDD zero-shot CD 强相关 | 核心深读 |
| LLM4CD | LLM4CD | 2025 | CIKM 候选 | `cikm25_llm4cd.pdf` | CCF 2026: B | PDF 已本地化；不满足 CCF-A | 附录灵感 |
| ESR-CD | Enhancing Student Representations | 2025 | Frontiers of Computer Science | `fcs25_esrcd.pdf` | 未取得中科院一区证据 | PDF 已本地化 | 附录灵感 |
| FineCD | Foundation Model Enhanced Derivative-Free CD | 2025 | Frontiers of Computer Science | `fcs25_fdecd.pdf` | 未取得中科院一区证据 | PDF 已本地化 | 附录灵感 |
| LCST | LLM-Guided Cognitive State Transfer | 2025 | Frontiers of Digital Education | `fde25_lcst.pdf` | 未取得中科院一区证据 | PDF 已本地化 | 附录灵感 |
| PromptCD | PromptCD | 2025 | IEEE TCSS | `tcss25_promptcd.pdf` | 未取得中科院一区证据 | PDF 已本地化 | 附录灵感 |
| Generative CD | Generative Cognitive Diagnosis | 2026 | IEEE TLT | `tlt26_generative_cd.pdf` | 未取得中科院一区证据 | PDF 已本地化 | 附录灵感 |
| Transfer-Q | Transfer Learning with Q-matrix Constraints | 2026 | Acta Psychologica Sinica | `psyacta26_transfer_qmatrix.pdf` | 未取得中科院一区证据 | PDF 已本地化 | 附录灵感 |
| DiaCDM | Dialogue-based CDM | 2026 | ICASSP | `icassp26_diacdm.pdf` | CCF 2026: B | PDF 已本地化；不满足 CCF-A | 附录灵感 |
| Exploratory DeepCDM | Exploratory DeepCDM | 2025/2026 | Psychometrika | `psychometrika25_exploratory_deepcdm.pdf` | 未取得中科院一区证据 | PDF 已本地化 | 附录灵感 |

## 最终审计与深读结论

本节是 2026-05-19 的最终执行版，后面的旧版逐篇摘要保留为原始材料，但论文主叙事、准入和实验决策以本节为准。

### 对 ConceptSkillCDM 的叙事修正：从数据现象到 CRG/LCRF

原来的现实问题“CDM 需要诊断学生对概念的掌握，但现实平台里概念覆盖稀疏、单概念题多、学生路径不同”不够精确，尤其“单概念题多”不能套到所有数据集。修正后的问题应写成：

> 现实学习平台中的概念证据并不总是以同一题内多概念共现的形式出现。在 ASSIST09、Junyi 和 ASSIST17 中，题目大多是单概念或 item co-occurrence 很稀疏，当前题目的概念证据常需要从学生历史概念通过训练集可观测的学习路径到达。NIPS34 是多概念密集对照，当前 same-query posterior 证据未过阈值，因此不进入核心主线，只保留为附录/历史对照。

因此论文故事不是“所有数据集都单概念”，而是“在核心数据集里，概念关系证据常需要通过全局路线图从历史概念桥接到当前概念”。CRG 和 LCRF 的递进关系如下：

- CRG, Concept Reachability Graph：用 train-only item co-occurrence、sequence transition 和 self retention 构造全局概念可达图，解决“当前概念能否从学生历史概念通过全局学习路线到达”的主问题。
- LCRF, Learner-Conditioned Reachability Filter：在 CRG 给出的同一 support 上，用学生自身历史、近期表现和邻居概念状态过滤路线，解决“可达路线是否适合当前学生”的副问题。

数据统计支持如下，来源为 `results/crg_lcrf_small_core_20260519_compact/data_phenomenon/crg_lcrf_data_readiness.csv`。

| dataset | single-concept rate | multi-concept rate | item edge density | sequence edge density | test direct-unseen rate | bridge-only reachable | 叙事角色 |
|---|---:|---:|---:|---:|---:|---:|---|
| ASSIST09 | 82.8% | 17.2% | 0.9% | 64.3% | 3.1% | 3.1% | 平衡主例：item 共现稀疏，但 sequence 路线强，CRG/LCRF 都有证据。 |
| Junyi | 100.0% | 0.0% | 0.0% | 25.2% | 100.0% | 100.0% | 最强 CRG 现象：没有 item 共现，当前概念必须靠 sequence 路线桥接。 |
| ASSIST17 | 78.3% | 21.7% | 6.2% | 76.3% | 2.8% | 2.8% | CRG 适合，LCRF 全局消融较弱但 counterfactual 强。 |
| NIPS34 | 0.0% | 100.0% | 6.3% | 98.2% | 0.0% | 0.0% | 附录对照：多概念密集，不进入 09/Junyi/17 核心主线。 |

这张表直接修正了“单概念题多”的说法：ASSIST09、Junyi、ASSIST17 可以支持该现象，NIPS34 不能。当前论文核心数据集应只保留 ASSIST09、Junyi、ASSIST17；NIPS34 不承担核心 claim。

本轮使用 R 重新绘制了论文图，脚本为 `tools/plot_crg_lcrf_mechanism_figures.R`，输出目录为 `results/crg_lcrf_small_core_20260519_compact/paper_figures/`。重画前额外看了本地 PDF 中的 figure page：AAAI/KDD 的机制图通常不是大标题海报式图，而是短标签、小面板、少 legend、长解释放 caption；因此新图采用 compact small-multiple、line/dot counterfactual 和 posterior heatmap，而不是把所有解释塞进图内。

| 图 | 文件 | 证明点 | 当前判断 |
|---|---|---|---|
| Figure 1 | `fig1_mechanism_crg_lcrf.png` | 方法机制：train-only evidence -> CRG roadmap -> fixed support -> LCRF posterior -> prediction。 | 可用；作为方法总览图，不承担结果证明。 |
| Figure 2 | `fig2_data_and_crg_retrieval.png` | 数据现象 + CRG 充分性：稀疏可达现象与 held-out route retrieval。 | 可用；CRG 的主证据应放在 09/Junyi/17。 |
| Figure 3 | `fig3_crg_support_necessity_controls.png` | CRG 必要性：四类 support corruption control。 | 可用但要精确表述：ASSIST17 最干净，ASSIST09 证明 support-dependence，Junyi 弱。 |
| Figure 4 | `fig4_lcrf_counterfactual_delta_auc.png` | LCRF 必要性：shuffle/mean learner state 明显伤害 AUC。 | 可用；核心写 09/17，Junyi 谨慎补充，NIPS34 仅附录。 |
| Figure 5 | `fig5_lcrf_same_query_posterior.png` | LCRF 充分性：同一 CRG support 被不同学生过滤成不同 posterior。 | 可用；主例用 ASSIST09/ASSIST17，不使用 NIPS34 same-query。 |

按图和 CSV 的结论，后续论文不能写成“CRG/LCRF 在四个数据集上都同等强”。更稳的写法是：

- CRG 的充分性：ASSIST09、Junyi、ASSIST17 的 held-out transition retrieval 均明显强于 degree-random/self-only。
- CRG 的必要性：新增 control 后，ASSIST17 最干净，100% evidence support corruption AUC drop 约 0.011、BCE increase 约 0.022，且明显强于 degree-random；ASSIST09 也有约 0.015 AUC drop，但 degree-random 接近或略强，因此只能写成模型依赖 support substrate，而不能写成 evidence edge 独占有效；Junyi 弱。
- LCRF 的必要性：ASSIST09、ASSIST17 的 shuffle/mean counterfactual 明显崩，说明个性化状态不是固定补丁；NIPS34 可作为附录对照，不进入核心三数据集叙事。
- LCRF 的充分性：same-query posterior case 在 ASSIST09 与 ASSIST17 过阈值，mean pairwise L1 分别约 0.173 与 0.710；NIPS34 的 same-query posterior 最高 L1 约 0.053，没有过阈值，不能作为该图主例。
- LCRF 在 Junyi 上不能夸大：Junyi 主要证明 CRG，因为它是 100% bridge-only；LCRF 在 Junyi 只作为谨慎补充。

### 交给 GPT Pro 的仓库审图提示词

下面这段可以直接交给 GPT Pro。目标不是让它重写方法，而是让它先阅读 git 仓库中的代码和实验数据，再决定哪些图最能证明 CRG/LCRF。如果它判断数据不足，需要输出给 Codex 的补充实验提示词。

```text
你需要作为 cognitive diagnosis 顶会论文的实验设计 reviewer，先完整阅读这个 git 仓库中的代码、README 和实验结果，再决定如何绘制论文机制图。

仓库中重点阅读：
1. src/ 中和 CRG/LCRF 对应的模型实现，确认 CRG 是否只用 train-only item cooccurrence、sequence transition、self retention，LCRF 是否只在 CRG support 上做 learner-conditioned filtering。
2. tools/plot_crg_lcrf_mechanism_figures.R，确认当前 R 图是否符合 AAAI/KDD 论文常见风格：小面板、少标题、短标签、caption 承担解释。
3. results/crg_lcrf_small_core_20260519_compact/README.md。
4. results/crg_lcrf_small_core_20260519_compact/data_phenomenon/crg_lcrf_data_readiness.csv。
5. results/crg_lcrf_small_core_20260519_compact/crg_retrieval/*/crg_transition_retrieval.csv。
6. results/crg_lcrf_small_core_20260519_compact/crg_support_corruption/*/crg_support_corruption_aggregate.csv。
7. results/crg_lcrf_small_core_20260519_compact/lcrf_case_studies/*/metrics_check.csv。
8. results/crg_lcrf_small_core_20260519_compact/paper_figures/paper_figure_summary.csv。
9. docs/paper_review_2025_2026/top20_cd_paper_story_review.md 中的“最终审计与深读结论”。

请完成四件事：
1. 判断当前 Figure 1-5 是否足以证明两个模块：
   - CRG 的充分性：train-only concept reachability 是否能检索 held-out concept transition，并强于 random/self。
   - CRG 的必要性：support corruption 是否能证明模型依赖 evidence support，而不是任意图。
   - LCRF 的必要性：actual/shuffle/mean/no-filter counterfactual 是否证明个性化状态不可替代。
   - LCRF 的充分性：是否还缺少 same query different learners 或 posterior heatmap/case 证据。
2. 如果图不够，请明确指出要替换哪张图，为什么，应该画成什么类型：heatmap、dumbbell、slope chart、case diagram、subgroup curve、support map、posterior map 等。
3. 如果需要补充实验，请写成可以直接交给 Codex 的提示词，要求包含：
   - 需要读取哪些 checkpoint/result CSV；
   - 是否需要重训，还是只做 inference/counterfactual；
   - 输出哪些 CSV；
   - 用 R 画哪些图；
   - 成功/失败判据是什么。
4. 输出必须非常具体，不要泛泛说“画更好看的图”。每个建议都要对应 CRG 或 LCRF 的一个可检验 claim。
```

### 准入审计结论

主表不再追求凑满 20 篇。严格采用“CCF-A 或中科院一区”后，当前可稳定支撑主叙事的是 AAAI/KDD 论文；IJCAI、CIKM、ICASSP 和未核验一区期刊只能作为附录或条件参考。

| 类别 | 论文 | 处理结论 | 原因 |
|---|---|---|---|
| 核心深读 | NCDLA, DBCD, KCD, DMC-CDM, AD4CD | 保留 | AAAI，按 CCF 2026 为 A；PDF 已本地化，任务强相关。 |
| 核心深读 | DFCD, ISG-CD, LRCD | 保留 | KDD，按 CCF 2026 为 A；PDF 已本地化，CD/图/跨域诊断强相关。 |
| 条件参考 | OSCD | 暂不进核心主表 | 标称 KDD 2026，但正式录用与 proceedings 仍需核验；可用于鲁棒性实验灵感。 |
| 条件参考 | FACD, KAN2CD | 不进严格主表，保留为写法参考 | IJCAI 在 CCF 2026 为 B；若学校按旧 CCF 目录认可，可恢复为核心参考。 |
| 附录灵感 | LLM4CD, DiaCDM | 降级 | CIKM/ICASSP 在 CCF 2026 为 B，不满足当前硬约束。 |
| 条件参考或附录 | ESR-CD, FineCD, PromptCD, Generative CD, Transfer-Q, Exploratory DeepCDM | 暂不进核心主表 | 没有可审计的中科院一区证据前，不作为主叙事依据；只能借鉴写法或可视化。 |
| 附录灵感 | LCST | 降级 | Frontiers of Digital Education 未作为当前顶会顶刊主准入。 |

### 核心深读论文

#### NCDLA, AAAI 2026

现实问题：图式 CDM 容易被错误日志、猜测、失误和交互噪声污染。作者先用 NeurIPS2020 的奇异值/子空间现象说明：噪声更容易积累在低奇异成分，而主子空间更稳定。

写作链条：Introduction 不是先报模块，而是先给“图 CD 有噪声、GNN 会传播噪声、低秩主空间更稳定”的三段式问题链。贡献写成“噪声发现 -> 低秩重构 -> spectral anchor regularisation -> 多 backbone 验证”。

公式推理：基础 CD 仍是响应预测 BCE。NCDLA 先把正确/错误响应构成两个图，再做低秩重构得到去噪邻接，之后用 self-supervised alignment 和 spectral anchor 约束保留主子空间。最终目标写为：

```text
L = L_BCE + alpha * L_SSL + beta * L_SAR
```

其中 `L_BCE` 对应诊断预测，`L_SSL` 对齐原图与低秩图，`L_SAR` 固定主奇异空间，防止低秩重构丢掉有效信号。

实验设计：Assist17、NeurIPS2020、Junyi，7:1:2 split；指标 ACC/AUC/F1/DOA；比较 IRT/MIRT/NCD/MFKC/KaNCD 以及 LightGCN、ORCDF、ISGCD；再做噪声注入、消融、奇异值/子空间图。它的关键不是 clean AUC，而是“噪声越强，稳定图越有价值”。

可模仿点：CRG 的机制实验应加入 support corruption 或 edge noise，证明全局路线图不是任意图，而是在边被破坏时性能/检索能力会下降。

#### DBCD, AAAI 2026

现实问题：学生不是随机作答，题目选择和缺失机制会造成 MNAR bias。普通 CDM 容易把“谁被观察到”误认为“谁掌握了”。

写作链条：先从平台作答选择偏差切入，再引出 missing-not-at-random 和反事实设定，最后提出 debiasing framework。作者不是说“加一个 VAE”，而是说“要把反事实样本和潜在混杂变量显式建模”。

公式推理：基础数据是学生-题目响应矩阵和 Q 矩阵。DBCD 构造 factual data `D_f` 与 counterfactual data `D_cf`，使用 beta-VAE 学潜在外生混杂，再通过 gating 融合到学生表示。训练目标：

```text
L_total = L_factual + L_info
L_info = alpha * L_fc + gamma * L_p
```

其中 `L_factual` 是 BCE，`L_fc` 约束 factual/counterfactual 一致性，`L_p` 是潜在混杂先验项。

实验设计：ASSIST09、ASSIST17、Junyi；Full/Random/Uniform 三种 test construction；指标 ACC/RMSE/AUC；在 MIRT/DINA/NCD 等 backbone 上都验证；另有 difficulty variant、ablation、hyperparameter 和 gating 可视化。

结果写法：作者承认复杂模型提升较小，但强调简单模型和偏置测试集提升明显。它把“模块主次贡献不均衡”写成符合偏置机制的结果。

可模仿点：LCRF 的必要性不应该只用全 test no_LCRF，而应做 actual/shuffle/mean 反事实，证明收益来自真实学生状态，而不是固定补丁。

#### KCD, AAAI 2025

现实问题：冷启动学生、低频题目和低频概念缺少行为证据，但 LLM 具有学科语义知识。问题不是“直接用 LLM”，而是“语义空间和行为诊断空间不一致”。

写作链条：作者先讲教育平台低频场景，再讲 LLM 的语义能力，接着指出二者空间不一致，需要 cognitive level alignment。贡献写成“LLM 语义增强 + 行为空间/语义空间对齐 + 冷启动验证”。

公式推理：学生-题目-概念响应仍是基础 CD 公式。KCD 用 LLM 生成文本描述并编码为语义向量，再通过 InfoNCE 做 global/local contrast alignment，并用 dynamic mask reconstruction 保持概念语义可恢复。整体目标可以概括为：

```text
L = L_CDM + alpha * L_global + beta * L_local + lambda * L_recon
```

实验设计：PTADisc 的 Python/Linux/Database/Literature 四门课；8:1:1 split；指标 AUC/ACC/RMSE；七个 CD backbone 上加 KCD；冷/暖场景拆分、dropout ratio、t-SNE、case study。

结果写法：它不只报平均 AUC，还强调冷启动场景改善更明显。消融表中去掉 collaborative information、local/global contrast、dynamic mask 都会下降。

可模仿点：CRG/LCRF 也应该按“该发挥的场景”拆分，例如 query concept 未直接见过但可由历史 concept 桥接、high reachability、high state contrast，而不是只看全 test 均值。

#### DMC-CDM, AAAI 2025

现实问题：CD 是从有限作答观测恢复 latent cognitive state 的逆问题，单一观察会造成 information loss 和 ill-posed recovery。

写作链条：作者先把 CD 定义为逆问题，再把信息损失分成 under-expressive interaction function 和 incomplete observation，最后提出 multi-perspective consolidation。这个链条很适合模仿：先定义科学问题，再让模块自然出现。

公式推理：目标是从响应日志最大化认知状态后验 `p(Theta | R)`。作者用 mutual information 解释多视角观测为什么能减少信息损失，再用 diffusion consolidation 聚合潜在观察；训练由 prediction loss、diffusion loss 和 reconstruction loss 共同组成。

实验设计：MoocRadar、Ifly、Junyi；过滤交互少于 30 的题目/概念/学生；每个学生 80/10/10 split；指标 AUC/ACC/F1；主表、w/o M、w/o S、w/o both、K 值敏感性、稀疏鲁棒性和教育推荐应用。

结果写法：作者明确指出 multi-perspective consolidation 比 semantic extractor 更关键，没有强行说每个模块贡献一样大。

可模仿点：我们应该把 CRG 定为主问题模块，LCRF 定为副问题模块。Junyi/assist17 如果 LCRF drop 小，不需要硬说 LCRF 是主贡献。

#### AD4CD, AAAI 2025

现实问题：学生会猜对、失误，响应时间异常也会污染诊断。普通 CDM 把这些异常作答当作真实掌握证据。

写作链条：先给教育场景中的猜测/失误/异常响应时间，再用因果图定义混杂路径，最后把 anomaly detection 接到 CDM。它的写法是“因果问题 -> 异常表示 -> 无偏预测”。

公式推理：响应日志是 `(s_i, e_j, r_ij, t_ij)`。AD4CD 学学生异常、题目异常和响应时间异常三类表示，再用 attention 融合得到 guessing/slipping：

```text
y*_ij = M(h_s, h_e, h_c)
y_hat_ij = x_Guess * (1 - y*_ij) + (1 - x_Slip) * y*_ij
L = L_CD + L_ELBO
```

其中 `M` 可以是任意基础 CDM，`x_Guess/x_Slip` 是异常状态对真实诊断的校正。

实验设计：ASSIST09、ASSIST17、Junyi；过滤少于 10 条响应的学生；指标 ACC/RMSE/AUC；IRT/DINA/MIRT/NCD/KSCD/KANCD 都加 “+”；ablation 去 guess/slip/fusion；还有异常热力图和噪声鲁棒性。

结果写法：AD4CD 在所有 backbone 上方向一致提升，即使个别指标提升不大，也通过多 backbone 一致性和异常图支撑故事。

可模仿点：LCRF 如果要证明个体过滤，不必要求所有数据集大幅 no_LCRF drop；更应该证明在多个 case/子群中 actual state 比 shuffle/mean 更可靠。

#### DFCD, KDD 2025

现实问题：开放学习环境会出现 unseen students、unseen exercises、unseen concepts，纯 ID embedding 无法泛化，纯文本语义又缺少行为结构。

写作链条：先定义 open student learning environment，再拆成 unseen student/exercise/concept 三种场景，不把所有样本混成一个平均 AUC。方法部分承接“文本 + response matrix + graph”三种信息源。

公式推理：开放场景下实体集合被拆成 observed/unseen。DFCD 用 LLM refine 题目/概念文本，构造 text feature 和 response feature，再用 dual-fusion attention 与 graph encoder 得到开放实体表示，最终仍通过 CDM interaction function 和 BCE 训练。

实验设计：NeurIPS2020、XES3G5M、MOOCRadar；开放场景主表、standard scenario appendix、ablation、test size sensitivity、cold-start、embedding model 比较、t-SNE 和诊断结果可视化。

结果写法：作者不把所有数据混写，而是按 unseen 类型解释提升来源。

可模仿点：CRG 的实验也应按场景拆：unseen query concept、reachable query、high sequence support、low direct history，而不是只看总体消融。

#### ISG-CD, KDD 2025

现实问题：图式 CDM 里的学生-题目边、题目-概念边、概念关系边存在异质性和不确定性，普通 GNN 会把不可靠边一起传播。

写作链条：作者先指出 graph CDM 中边语义不一致和不确定边会损害诊断，再提出 informative and stable graph。叙事重点是“可靠边”，不是泛泛的“加图”。

公式推理：基础 graph CDM 用 BCE 训练诊断函数。ISG-CD 引入 edge differentiation layer、semantic graph refinement 和 HSIC/information bottleneck 类正则，最终训练目标可概括为：

```text
L_all = L_BCE + beta * L_HSIC
```

其中 `L_HSIC` 让表示保留与预测有关的信息，同时减少对不稳定边的依赖。

实验设计：ASSIST、Junyi、MOOC-Radar；五折交叉验证；指标 ACC/AUC/DOA；表 2-4 主结果，表 5 消融，图 4 不确定边检测，图 5/6 超参敏感性，表 6 训练策略比较。

关键数据：ASSIST AUC 0.7604，Junyi AUC 0.8058；作者也明确写 Junyi 是一题一概念，消融和敏感性要按数据特点解释。

可模仿点：CRG 必须被写成 reliable route graph，而不是简单 concept graph。CRG 的小实验应包括 held-out route retrieval 和 support corruption。

#### LRCD, KDD 2025

现实问题：教育数据跨平台/跨学科时，target domain 没有响应日志，传统 CD 依赖 ID 和历史交互，无法 zero-shot。

写作链条：作者先把任务升级为 zero-shot cross-domain cognitive diagnosis，再指出教育数据“输入简单且敏感”，不能假设 target 有足够行为数据。语言表示被引入为跨域桥梁。

公式推理：对每个 domain 有响应日志和 Q 矩阵，LRCD 把学生、题目、概念转成 textual profiles，再映射到 language space，之后用 student/exercise/concept mapper 转到 cognitive space。训练目标是多 domain 加权 BCE：

```text
L = sum_m w_m * L_Rm
```

其中 `L_Rm` 是第 m 个源域的 BCE；zero-shot 推理时只用文本 profile 和映射器。

实验设计：SLP、EDM、MOOC；subject-level zero-shot、platform-level zero-shot、overlap-student；主表、ablation、scale-up、不同 text embedding model、t-SNE 和 case study。

结果写法：论文不与传统同 split CDM 直接比 SOTA，而是定义新场景，再证明 zero-shot 能工作。

可模仿点：如果我们采用 Concept Reachability，也要先证明数据中确实存在“当前概念无法直接观测但可从历史概念到达”的场景，再评价 CRG/LCRF。

### 附录或条件参考论文

| 论文 | 降级原因 | 仍可借鉴的点 |
|---|---|---|
| FACD, IJCAI 2025 | CCF 2026 为 B；若学校按旧目录可恢复 | 早期靠群体协同，后期靠个体序列，模块递进关系很适合“全局路线图 + 个性化辅导过滤”。 |
| KAN2CD, IJCAI 2025 | CCF 2026 为 B；若学校按旧目录可恢复 | 小幅性能提升也可用可解释函数曲线和概念贡献图支撑。 |
| OSCD, KDD 2026 | KDD 是 A，但正式录用/会议时间仍需核验 | 多噪声类型、support corruption、robust architecture search 可借鉴。 |
| LLM4CD, CIKM 2025 | CIKM 为 B | 文本与状态双模块消融可借鉴，但不能做主叙事依据。 |
| ESR-CD, FineCD, PromptCD, Generative CD, Transfer-Q, Exploratory DeepCDM | 需中科院一区证据 | 可作为写法和图表参考，不能支撑“顶会顶刊趋势”。 |
| LCST, DiaCDM | venue 不满足当前硬约束 | 只保留新数据形态或对话诊断的灵感。 |

### 数据集指标对照

这些数值只作为参考区间。不同清洗、split 和任务定义不能直接比较，也不能直接写“超过 SOTA”。

| dataset | paper | venue | setting | metric/result | 是否可直接和我们比较 |
|---|---|---|---|---|---|
| ASSIST09 | AD4CD | AAAI 2025 | 过滤学生少于 10 响应；不同 backbone + AD4CD | KSCD+ AUC 0.7823，KANCD+ AUC 0.7815，NCD+ AUC 0.7672 | 不完全可比；split 和是否用响应时间不同。 |
| ASSIST09 | DBCD | AAAI 2026 | Full/Random/Uniform test construction | NCD-DBCD full AUC 75.28%，MIRT-DBCD full AUC 74.68% | 不可直接比较；任务是 debiasing test construction。 |
| ASSIST/09-like | ISG-CD | KDD 2025 | 五折随机 log split | ISG-CD AUC 0.7604，ACC 0.7322，DOA 0.6582 | 不可直接比较；五折 log split 与我们原 split 不同。 |
| Junyi | AD4CD | AAAI 2025 | 使用响应时间；过滤学生少于 10 响应 | KANCD+ AUC 0.7646，NCD+ AUC 0.7557 | 不可直接比较；它的 Junyi concept 数和清洗与我们不同。 |
| Junyi | DMC-CDM | AAAI 2025 | 每学生 80/10/10 split；过滤交互少于 30 | DMC-CDM AUC 0.8834，ACC 0.8597，F1 0.9168 | 不可直接比较；清洗和 split 明显不同。 |
| Junyi | ISG-CD | KDD 2025 | 五折随机 log split | ISG-CD AUC 0.8058，ACC 0.7672，DOA 0.6728 | 不可直接比较；但可作为 graph-CD 参考区间。 |
| ASSIST17 | AD4CD | AAAI 2025 | 使用响应时间；过滤学生少于 10 响应 | KSCD+ AUC 0.7370，KANCD+ AUC 0.7340，MIRT+ AUC 0.7254 | 不完全可比；响应时间特征不同。 |
| Assist17 | NCDLA | AAAI 2026 | 7:1:2 split | 论文报告 ACC/AUC/F1/DOA，主张多 backbone 一致提升 | 需回表格 OCR 核验具体 AUC 后才可比较。 |
| NeurIPS2020/NIPS34 | NCDLA | AAAI 2026 | 7:1:2 split；鲁棒图诊断 | 用作 noise robustness benchmark | 与我们的 NIPS34 清洗不一定一致。 |
| NeurIPS2020 | DFCD | KDD 2025 | open student learning environment | open/unseen 场景 AUC/ACC | 任务不同，只能借鉴 open/冷启动实验。 |
| SLP/EDM/MOOC | LRCD | KDD 2025 | zero-shot cross-domain CD | AUC% 主表 | 任务不同，不能和我们主表比较。 |

### 近两年论文的共性写法

1. 先证明真实场景存在，再提出模块。NCDLA 先证明噪声主空间现象，DBCD 先定义 MNAR，DFCD 先定义 open environment，LRCD 先定义 zero-shot domain。
2. 主模块和副模块不强行等权。DMC-CDM 明确 multi-perspective 比 semantic extractor 更关键；DBCD 承认复杂 backbone 提升较小。
3. 全局 AUC 不是唯一证据。顶会论文普遍使用场景分组、噪声曲线、冷启动、反事实、可视化、case study 来补机制链。
4. 消融必须对应公式项。删模块不是随便关开关，而是删除公式里的某个变量、loss 项、输入源或图边。
5. 论文的好故事通常是递进式：数据现象 -> 科学问题 -> 公式定义 -> 模块 -> 主表 -> 消融 -> 机制图。

### 多数据集证据边界

本地深读论文支持一个明确结论：顶会 CD 论文通常不会要求每个模块在每个数据集、每个指标、每个机制实验上都同等强。更常见的写法是：主数据集负责证明核心问题，辅助数据集验证泛化或补充场景，机制实验只在模块应发挥作用的子场景中展开。

可直接模仿的写法包括：

- DBCD：在不同 test construction 和多个 backbone 上证明 debiasing 框架有效，但不要求每个子模块在每个 backbone 上贡献同等大；复杂 backbone 的提升较小也被解释为机制差异。
- DMC-CDM：明确 multi-perspective consolidation 是主贡献，semantic extractor 是辅助贡献；组件贡献有主次，不强行等权。
- FACD：协同诊断在早期更重要，个性化诊断在后期更重要；模块按学习阶段递进，而不是在所有阶段都要求同样强。
- AD4CD：用 ASSIST09、ASSIST17、Junyi 证明异常/因果场景，重点是多个 backbone 和异常机制一致，不是每个数据集每个模块都掉 1-2 点。

因此 ConceptSkillCDM 的核心主线应固定为 ASSIST09、Junyi、ASSIST17 三个数据集。NIPS34 可以保留在附录说明“多概念密集对照下 same-query posterior 不显著”，但不放进主结论。

### 对 CRG/LCRF 的最终决策

当前 ConceptSkillCDM 不应该再为了让每个数据集 no_CRG/no_LCRF 掉 1-2 点而继续累加残差。最稳的路线是保留已恢复的稳定代码内核，把论文故事改写为：

```text
现实问题：Concept Reachability under Sparse Response Evidence
CRG：Concept Reachability Graph，全局路线图，回答“当前概念能否由学生历史概念通过训练集学习路径到达”。
LCRF：Learner-Conditioned Reachability Filter，个性化辅导过滤器，回答“这条可达路径对当前学生是否可信”。
```

推荐主数据集：

| 用途 | 数据集 | 理由 |
|---|---|---|
| 主数据集 1 | assist_09 | full 达标，CRG/LCRF 消融都明显，且有 reachability 现象。 |
| 主数据集 2 | assist_17 | CRG 消融强，可证明 sequence reachability；LCRF 作为辅助，不夸大。 |
| 主数据集 3 | junyi | full 达标，最适合证明单概念下 sequence route；但 LCRF 消融弱，只做 CRG 强证据。 |

推荐实验包：

1. 数据现象统计：direct seen rate、topK bridgeable rate、item cooccurrence edges、sequence density、history length。
2. CRG 充分性：held-out concept reachability retrieval，指标 Hit@K、NDCG@K、MRR，对比 self-only、item-only、seq-only、random。
3. CRG 必要性：support corruption，不重训，按 25/50/75/100% 替换 evidence support，观察 reachable subgroup 掉分。
4. LCRF 必要性：actual/shuffle/mean/no_LCRF counterfactual，固定 checkpoint 和 CRG，只打乱或平均学生状态。
5. LCRF 充分性：same query different learners，展示相同 CRG support 被不同学生状态过滤成不同 posterior。

写作落点：

- 不能写“CRG/LCRF 在所有数据集都充分必要”。应写成“CRG 是主路线图，LCRF 是在学生状态有辨识度时的个性化过滤”。
- 对 Junyi：强调 CRG，不强夸 LCRF。
- 对 assist17：强调 sequence reachability 和 CRG 消融。
- 对 assist09/assist17：重点画 LCRF 的个体化 case；NIPS34 不作为核心 case。
- 若要达到顶会论文写法，必须把每个小实验对应到公式变量或 support 操作，而不是只画漂亮图。

## 结论先行

1. 近两年 CD 论文很少只靠“全数据集 full 比 ablation 大幅高 1-2 个点”讲故事。更常见的是：先定义一个真实教育场景，再证明模块在这个场景中有效。
2. 很多论文的消融并不是每个模块都大幅下降。有些模块只下降 0.2-0.8 个点，但配合噪声、稀疏、冷启动、跨域、可解释案例、热力图、鲁棒性曲线，仍然能形成完整证据链。
3. 对我们当前 CRG/LCRF 来说，最危险的是反复为了让 no_CRG/no_LCRF 掉分而累加残差。这样会让模块边界越来越脏，论文也不好写。
4. 更稳的方向是回到“效果好、边界清楚”的版本，保留主表 full/no_CRG/no_LCRF，再补论文式小实验：
   - CRG 讲“全局可靠学习地图”：held-out transition retrieval、support corruption、CRG-relevant subgroup、噪声/边扰动鲁棒性。
   - LCRF 讲“学生个体小地图”：actual/shuffle/mean LCRF 反事实、同一全局图下不同学生 posterior、历史邻居相关 case、早期/低历史/高 support 场景。
5. 如果要改代码，优先小范围修补模块边界，不要继续把 CRG/LCRF 变成大而杂的黑盒主干。

## 论文筛选表

| ID | 论文 | 场景/问题 | 参考价值 | 本地 PDF |
|---|---|---|---|---|
| NCDLA | AAAI 2026, Noise-Aware Graph-based CD Through Low-Rank Alignment | 图 CD 噪声鲁棒 | CRG 的可靠图/边噪声实验 | `docs/paper_review_2025_2026/pdfs/aaai26_ncdla.pdf` |
| DBCD | AAAI 2026, Debiased Cognitive Diagnosis | 选择性作答/MNAR/反事实 | 模块主次贡献写法 | `docs/paper_review_2025_2026/pdfs/aaai26_dbcd.pdf` |
| KCD | AAAI 2025, Knowledge is Power | LLM 先验、冷启动 | case、t-SNE、dropout 冷启动 | `docs/paper_review_2025_2026/pdfs/aaai25_kcd.pdf` |
| DMC-CDM | AAAI 2025, Multi-Perspective Consolidation | CD 逆问题/信息缺失 | 多视角证据链 | `docs/paper_review_2025_2026/pdfs/aaai25_dmccdm.pdf` |
| AD4CD | AAAI 2025, Causal-Guided Anomaly Detection | 猜测/失误/异常行为 | 因果故事 + 异常样本 | `docs/paper_review_2025_2026/pdfs/aaai25_ad4cd.pdf` |
| FACD | IJCAI 2025, Fast-Adaptive CD | CAT 早期诊断 | CRG/LCRF 递进关系最像 | `docs/paper_review_2025_2026/pdfs/ijcai25_facd.pdf` |
| KAN2CD | IJCAI 2025, KAN for Neural CD | 可解释神经 CD | 小收益也能靠解释性成立 | `docs/paper_review_2025_2026/pdfs/ijcai25_kan2cd.pdf` |
| OSCD | KDD 2026, One-Shot NAS for Robust CD | 噪声架构搜索 | 结构鲁棒性/扰动实验 | `docs/paper_review_2025_2026/pdfs/kdd26_oscd.pdf` |
| DFCD | KDD 2025, Dual-Fusion CD | 开放学习环境/未见实体 | 场景拆分而非全局均值 | `docs/paper_review_2025_2026/pdfs/kdd25_dfcd.pdf` |
| ISG-CD | KDD 2025, Informative and Stable Graph CD | 异质边/不确定边 | 可靠边与图结构证据 | `docs/paper_review_2025_2026/pdfs/kdd25_isgcd.pdf` |
| LRCD | KDD 2025, Language Representation CD | 零样本跨域 | 标准 AUC 不是唯一证据 | `docs/paper_review_2025_2026/pdfs/kdd25_lrcd.pdf` |
| LLM4CD | CIKM 2025 候选, LLM4CD | 文本语义 + 状态建模 | 状态/文本双模块消融 | `docs/paper_review_2025_2026/pdfs/cikm25_llm4cd.pdf` |
| ESR-CD | FCS 2025, Enhancing Student Representations | 学生-概念稀疏屏障 | 子群/弱覆盖实验 | `docs/paper_review_2025_2026/pdfs/fcs25_esrcd.pdf` |
| FineCD | FCS 2025, Foundation Model Enhanced Derivative-Free CD | 小样本个体诊断 | 小样本任务重定义 | `docs/paper_review_2025_2026/pdfs/fcs25_fdecd.pdf` |
| LCST | FDE 2025, LLM-Guided Cognitive State Transfer | 跨域状态迁移 | 概念关系推理图 | `docs/paper_review_2025_2026/pdfs/fde25_lcst.pdf` |
| PromptCD | IEEE TCSS 2025 | 双方面跨域 CD | 个性化 prompt 作为主模块 | `docs/paper_review_2025_2026/pdfs/tcss25_promptcd.pdf` |
| Generative CD | IEEE TLT 2026 | 生成式诊断范式 | 诊断可靠性/可识别性 | `docs/paper_review_2025_2026/pdfs/tlt26_generative_cd.pdf` |
| Transfer-Q | Acta Psychologica Sinica 2026 | 迁移学习 + Q 矩阵约束 | Q 约束与迁移故事 | `docs/paper_review_2025_2026/pdfs/psyacta26_transfer_qmatrix.pdf` |
| DiaCDM | ICASSP 2026 | 师生对话诊断 | 新数据形态和过程证据 | `docs/paper_review_2025_2026/pdfs/icassp26_diacdm.pdf` |
| Exploratory DeepCDM | Psychometrika 2026 | 可识别深层生成 CDM | 热力图/可识别性实验 | `docs/paper_review_2025_2026/pdfs/psychometrika25_exploratory_deepcdm.pdf` |

## 1. NCDLA, AAAI 2026

来源：[AAAI PDF](https://ojs.aaai.org/index.php/AAAI/article/download/38665/42627)

核心科学问题：图式 CDM 面对猜测、失误、错误日志等噪声时，为什么仍然能学到有效信息，如何在不破坏真实认知信号的前提下增强鲁棒性。

故事写法：作者不是直接说“我们加一个去噪模块”，而是先做经验观察：噪声主要累积在低奇异成分，主子空间相对稳定。然后自然引出低秩重构和 spectral anchor regularization。这个顺序很关键，先解释现象，再给模块。

模块设计：用原始交互矩阵和低秩重构矩阵分别构图，并区分正确/错误响应图。低秩对齐负责结构层面的噪声过滤，spectral anchor 负责表示层面的主子空间稳定。

实验关系：主实验比较 IRT/MIRT/NCD/KaNCD 等 CDM 与图增强版本；鲁棒实验人工注入不同强度噪声；消融去掉低秩对齐或 spectral anchor；还用奇异值图解释为什么方法有效。

结果写法：不是只报告 clean AUC，而是强调噪声越强，优势越能体现。消融里低秩对齐更重要，anchor 是细化约束。作者承认模块贡献有主次。

可模仿点：CRG 模块如果叫“全局学习地图”，不能只做 no_CRG。应该加 CRG support corruption/noisy edge 实验：逐步污染或删除 evidence edge，看 CRG 是否在高噪声和高 support 样本上更稳。

## 2. DBCD, AAAI 2026

来源：[AAAI PDF](https://ojs.aaai.org/index.php/AAAI/article/download/39981/43942)

核心科学问题：学生不是随机做题，作答缺失存在 MNAR 和混杂因素。普通 CDM 可能把“谁选择了哪些题”误当作真实能力。

故事写法：先把教育平台中的选择性作答建模成 missing-not-at-random，再用反事实回答“如果学生面对另一个选择机制，诊断还稳定吗”。叙事很像 causal ML，不靠模块名取巧。

模块设计：相似样本构造反事实，β-VAE 建潜在外生混杂，门控融合到现有 CDM。它是一个 framework，可以套在 IRT、DINA、MIRT、NCD 等 backbone 上。

实验关系：在 full/random/uniform test 不同缺失模式下比较；还在多个 backbone 上加 DBCD，证明不是只服务某一个模型；消融区分 counterfactual sampling 和 β-VAE。

结果写法：简单 backbone 获益更大，复杂 backbone 获益较小；β-VAE 往往比 counterfactual 部分更关键。作者没有硬说每个组件同等重要，而是把主次解释成符合机制。

可模仿点：我们的 LCRF 不一定要在所有数据集都掉 1-2 点。可以写成 CRG 是主路径，LCRF 是在“个体化偏差明显”的样本上修正。LCRF 的实验需要 actual/shuffle/mean personal state 反事实，而不是只看全局 no_LCRF。

## 3. KCD, AAAI 2025

来源：[AAAI PDF](https://ojs.aaai.org/index.php/AAAI/article/download/31992/34147)

核心科学问题：低频学生、低频题目和新题目缺少行为先验，LLM 具备学科语义知识，但语义空间和行为诊断空间不一致。

故事写法：先指出 CD 的低频和冷启动痛点，再说 LLM 不直接等于诊断能力，需要 cognitive level alignment。这个写法避免“我们用了 LLM 所以更强”的空泛叙事。

模块设计：LLM 生成学生/题目认知描述，之后用行为空间和语义空间的对齐损失、mask reconstruction 等让 LLM 知识进入 CD 任务。

实验关系：主表比较传统 CDM 和增强版本；冷启动实验通过删除部分学生/题目历史模拟低频；t-SNE 展示表示对齐；case 展示诊断报告的合理性。

结果写法：它强调 alignment 后的 representation 更接近真实行为结构。不是所有证据都来自 AUC，图和 casLCRF 是叙事重要部分。

可模仿点：我们对 CRG 的“地图”也可以先用 held-out concept transition 证明它捕捉学习路径，再进入预测任务。否则直接上 AUC 容易被 uniform/random support 反驳。

## 4. DMC-CDM, AAAI 2025

来源：[AAAI PDF](https://ojs.aaai.org/index.php/AAAI/article/download/32105/34260)

核心科学问题：CD 是从有限观测恢复认知状态的逆问题，观测不完整导致 ill-posed。

故事写法：先用逆问题解释为什么单一观察不足，再把“多视角观察”定义成恢复认知状态的必要信息。方法像是理论动机推出来的，而不是堆模块。

模块设计：single-perspective extractor 从单一视角抽取认知状态，conditional diffusion 做 multi-perspective consolidation，语义 extractor 是辅助。

实验关系：主实验三数据集；消融区分 w/o multi-perspective、w/o semantic extractor、w/o both；K 值敏感性说明多视角数量不是越多越好；稀疏训练集比例说明鲁棒性。

结果写法：作者明确写 multi-perspective consolidation 比 semantic extractor 更重要。这个主次关系使论文更可信。

可模仿点：CRG/LCRF 也应该有主次：CRG 是全局地图基底，LCRF 是局部个性化校准。不要为了让两个模块都大幅掉分而把 LCRF 写成另一个主干。

## 5. AD4CD, AAAI 2025

来源：[AAAI PDF](https://ojs.aaai.org/index.php/AAAI/article/download/33344/35499)

核心科学问题：学生会猜对、会失误，题目也可能有异常属性。普通 CDM 把这些异常直接当成能力证据，会污染诊断。

故事写法：先用因果图解释学生能力、题目属性、响应时间、异常行为之间的混杂路径，然后把 anomaly detection 变成去混杂的必要步骤。

模块设计：用学生和题目的响应时间分布检测左尾/右尾异常；用重构无偏真实能力的 loss 作为异常分数；最后把异常特征接入多个 backbone。

实验关系：ASSIST09、ASSIST17、Junyi 三个有响应时间的数据集；IRT/DINA/MIRT/NCD/KSCD/KANCD 都加上 AD4CD，表格中每个 “+” 版本都提升；还有 anomaly score 和因果解释。

结果写法：它把提升写成 framework 对多 backbone 都有效，而不是单模型偶然。ASSIST09 上 KSCD+、KANCD+ 的 AUC 提升并不巨大，但因为多个 backbone 一致提升，证据仍然充分。

可模仿点：如果我们的 CRG/LCRF 只在某一个 backbone/变体上强，故事弱；如果在 09、Junyi、17 上方向一致，即使部分数据集 drop 小，也可以通过场景实验补强。

## 6. FACD, IJCAI 2025

来源：[IJCAI PDF](https://www.ijcai.org/proceedings/2025/0648.pdf)

核心科学问题：计算机自适应测试中，学生刚开始作答很少，传统 CDM 早期诊断慢且不稳定。

故事写法：它把诊断过程分阶段：早期需要借助相似学生的协同信息，后期需要学生自己的序列信息。这个递进关系非常适合模仿。

模块设计：dynamic collaborative diagnosis 使用相似学生提供群体辅助，dynamic personalized diagnosis 使用学生自身序列建模，最后动态融合。

实验关系：FrcSub、EDMCup2023、NeurIPS2020；比较不同 CAT 选题策略下 5/10/15 步诊断；消融 FA-C 和 FA-P；t-SNE 展示 mastery 分布；还比较推理时间。

结果写法：协同模块在早期更重要，个性化模块在后期稳定诊断中更重要。两个模块不是并列堆叠，而是按学习过程递进。

可模仿点：这是我们 CRG/LCRF 最应该参考的写法。CRG 是“大地图”，解决历史不足时的方向引导；LCRF 是“小地图”，当学生有局部历史后做个体化路线修正。实验应按学生历史长度或作答阶段分组。

## 7. KAN2CD, IJCAI 2025

来源：[IJCAI PDF](https://www.ijcai.org/proceedings/2025/0878.pdf)

核心科学问题：神经 CDM 有预测效果，但 MLP interaction function 很难解释每个知识点如何影响答题概率。

故事写法：这篇的叙事不是“我大幅提升 AUC”，而是“在保留或小幅提升性能的同时提高可解释性”。它非常适合解释为什么 CD 论文不一定要求每个消融都有巨大差距。

模块设计：用 KAN 替换或重构神经诊断函数，让每个概念维度的函数形状可视化。还强调单调性和参数效率。

实验关系：ASSIST、SLP、Junyi、FrcSub；主表比较传统和神经方法；Wilcoxon 显著性；运行时间与参数量；函数曲线和概念重要性可视化；超参敏感性。

结果写法：性能提升有时很小，但解释图承担了核心贡献。论文用“可解释神经诊断”而不是“新 SOTA 性能”来定位。

可模仿点：我们的 LCRF 如果全局 no_LCRF 下降小，可以转成解释性贡献：同一 CRG support 下，不同学生的 posterior 如何因 history/recent mastery 改变。case 必须选机制强样本，不随机选。

## 8. OSCD, KDD 2026

来源：[arXiv PDF](https://arxiv.org/pdf/2601.04918)

核心科学问题：CDM 架构在不同噪声类型下鲁棒性不同，人工设计架构很难覆盖真实教育噪声。

故事写法：先定义四类噪声：log miss、exercise confusion、Q-matrix confusion、log flip。然后把架构搜索目标设成 clean AUC、noisy AUC 和一致性约束的多目标优化。

模块设计：one-shot supernet 覆盖不同 CDM 架构组合，通过 Pareto 选择鲁棒结构。

实验关系：ASSIST09、SLP-Math；clean/noisy 场景；多噪声强度；ranking fidelity；搜索时间；最终架构可视化。

结果写法：不是只问 clean test 是否更高，而是问不同噪声下谁更稳。KDD 论文很重视 scenario-specific robustness。

可模仿点：CRG support corruption 可以直接借鉴这种思路：固定模型，不重训，把 CRG support 逐步 corruption，比较 original 和 degree-matched random。若 CRG-relevant 样本掉分更大，说明 CRG 的边是必要的。

## 9. DFCD, KDD 2025

来源：[arXiv PDF](https://arxiv.org/pdf/2410.15054)

核心科学问题：开放学习环境中会遇到新学生、新题目、新概念，纯 ID embedding 无法泛化，纯文本语义又缺少行为证据。

故事写法：它把任务拆成 unseen student、unseen exercise、unseen concept，不把所有样本混在一起看平均 AUC。这种场景拆分是论文说服力来源。

模块设计：LLM refine exercise/concept text，response matrix 汇总行为，dual-fusion attention 把文本和行为融合，graph encoder 传播结构。

实验关系：NeurIPS2020、XES3G5M、MOOCRadar；分别跑三类 open scenario；消融 w/o text、w/o response、w/o attention；再看 DOA、versatility、t-SNE 和文本 refine case。

结果写法：某些场景下某个消融差距很小，但在 unseen exercise/concept 中 response removal 很关键。作者按模块适用场景解释，不要求全场景同幅度下降。

可模仿点：我们应该把 CRG/LCRF 的小实验定义为“它该发挥作用的样本”。例如 CRG 看 history concept 与 query concept 是否被图连接，LCRF 看学生历史邻居是否出现过且 mastery/recent 有差异。

## 10. ISG-CD, KDD 2025

来源：[公开 PDF](https://fi.ee.tsinghua.edu.cn/public/publications/9cdd9d04-f1d2-11f0-b382-2aaffa21d846.pdf)

核心科学问题：图式 CDM 往往把所有响应边看成同质边，但正确/错误响应的含义不同，猜测/失误又会引入不确定边。

故事写法：先指出“边的异质性”和“边的不确定性”两个图问题，再提出 informative and stable graph。问题和模块一一对应。

模块设计：semantic-aware GNN 区分不同边语义，information bottleneck 识别 informative edges，稳定图用于诊断。

实验关系：ASSIST、Junyi、MOOC-Radar；主实验 AUC/ACC/DOA；不确定边检测实验；消融；超参敏感性；训练策略比较。

结果写法：AUC 提升不是所有数据都非常大，但 DOA、uncertain edge detection 和 ablation 共同支撑“可靠图”的故事。

可模仿点：CRG 模块要从“图传播提升 AUC”升级成“可靠地图”。除了 no_CRG，还要给 edge retrieval、edge corruption、support survival、边证据热力图。

## 11. LRCD, KDD 2025

来源：[arXiv PDF](https://arxiv.org/pdf/2501.13943)

核心科学问题：传统 CDM 依赖 domain-specific ID 和 Q-matrix，很难跨学科、跨平台、零样本诊断。

故事写法：它用“语言认知画像”把不同域的学生、题目、概念放到统一空间。叙事中心是 zero-shot，而不是普通随机划分下多高。

模块设计：text cognitive profile 表达学生/题目/概念，language-cognitive mapper 将文本语义映射到认知诊断空间。

实验关系：SLP、MOOC、EDM；subject-level zero-shot、platform-level zero-shot、overlap-student；主表、消融、case、附录中的普通场景。

结果写法：跨域/零样本的提升很大，普通场景不是唯一重点。作者把数据拆分方式变成核心实验设计。

可模仿点：如果我们的模型在新数据集 full/no_CRG/no_LCRF 不稳定，不必强行全写成主结果。核心数据集固定为 09、Junyi、17，其他数据集只作为 generalization appendix。

## 12. LLM4CD, CIKM 2025 候选

来源：[arXiv PDF](https://arxiv.org/pdf/2504.05542)

状态风险：本地下载版本标为 CIKM25 候选，需要最终录用页面再核验。可以作为写法参考，但不建议作为“顶会已录用”硬引用。

核心科学问题：CDM 需要开放语义信息和学生状态信息同时参与，单纯 LLM 文本或单纯行为状态都不够。

故事写法：宏观 text semantic encoder 和微观 state encoder 分工明确，最后 MoE/GAT 融合。它把模型写成“文本理解 + 认知状态”两个互补通道。

模块设计：macro text encoder 处理题目/知识点语义，micro state encoder 处理响应行为，MoE adapter 或 graph attention 融合。

实验关系：整体性能、cold-start、文本构造、消融 State/Text/LLM/GAT/MoE、诊断报告 case。

结果写法：用 ID 替换 LLM 文本会显著变差，说明语义不是装饰。GAT/MoE 的贡献相对小，但通过 case 和冷启动补强。

可模仿点：LCRF 的“个体小地图”也可以用反事实替换：actual student state、shuffle state、mean state、no_LCRF。如果 actual 明显更好，LCRF 的个性化就比普通 no_LCRF 更好解释。

## 13. ESR-CD, Frontiers of Computer Science 2025

来源：[FCS PDF](https://journal.hep.com.cn/fcs/EN/PDF/10.1007/s11704-025-40591-2)

核心科学问题：学生可能做了很多题，但覆盖的概念很少，导致 student-concept mastery 稀疏。普通 CDM 对未练过概念的 mastery 很难估计。

故事写法：它先提出 student-concept sparsity barrier，然后区分 comprehension degree 和 application ability。这个拆分把“学生能力”从“具体概念掌握”中剥离出来。

模块设计：application ability 用稠密学生-概念条目估计；comprehension degree 用 matrix factorization 和 exercise-concept relation refinement 增强。

实验关系：随机划分和 concept weak coverage split 两种场景；主表；鲁棒性；特别强调弱覆盖场景下提升更大。

结果写法：随机划分平均提升约 0.5%，弱覆盖场景提升更明显。作者没有掩盖随机场景提升有限，而是把目标场景对准 sparsity barrier。

可模仿点：LCRF 模块应该考虑学生历史覆盖的邻居概念，而不是只在题目多概念上重排。Junyi 单概念并不意味着 LCRF 无法发挥，关键是“当前概念和学生历史概念是否通过 CRG support 相关”。

## 14. FineCD, Frontiers of Computer Science 2025

来源：[FCS PDF](https://journal.hep.com.cn/fcs/EN/PDF/10.1007/s11704-024-40029-1)

核心科学问题：少量题目响应下，传统参数优化式 CDM 难以为单个学生即时诊断，且题干文本信息没有被充分利用。

故事写法：它重定义任务场景：每个 quiz 留一题测试，训练题很少，要求模型根据题干和少量响应做个体预测。这样传统 CDM 不再占优势。

模块设计：foundation model/LLM 辅助的 derivative-free 诊断，不依赖传统参数反复优化。它强调推理逻辑，而不是在大规模行为矩阵里拟合。

实验关系：NeurIPS20 子集；DINA、IRT、NCDM、KANCD、LLM-Naive、Human；10 次重复；还用 human interview 支撑三步推理策略。

结果写法：传统方法接近随机，LLM-Naive 也不如 FineCD，说明直接把 LLM 丢进去不够，需要诊断流程。

可模仿点：如果要让 LCRF 更像“二次教学/小地图”，可以把实验设计成少历史或早期阶段，而不是全量 test 混合。

## 15. LCST, Frontiers of Digital Education 2025

来源：[FDE PDF](https://journal.hep.com.cn/fde/EN/PDF/10.1007/s44366-025-0054-y)

核心科学问题：目标学科没有学生作答日志时，如何进行跨域认知状态迁移。

故事写法：把 LLM 作为 educational expert，先推理概念间关系，再把源域认知状态迁移到目标域。它讲的是“没有目标日志时，教师如何借助概念关系迁移判断”。

模块设计：LLM-guided concept relationship reasoning + cognitive state transfer。核心不是让 LLM 直接预测，而是让 LLM 生成可解释的跨域概念桥。

实验关系：SLP 多学科；single-domain 和 multi-domain diagnosis；概念关系推理图；不同 LLM 对比。

结果写法：LCST 接近 oracle 或明显优于零样本 baseline，图上展示 LLM 推理出来的概念关系，让读者相信迁移来源。

可模仿点：CRG 的地图必须能画出来。建议保留 concept transition graph、局部 support map、held-out transition retrieval 排名图，而不是只给 AUC。

## 16. PromptCD, IEEE TCSS 2025

来源：[作者公开 PDF](https://le-wu.com/files/Publications/JOURNAL/PromptCD_TCSS.pdf)

核心科学问题：跨域 CD 同时存在学生差异和题目差异，仅迁移学生或题目一侧都不够。

故事写法：它把跨域 CD 拆成 dual-aspect：overlapping entities 用 personalized prompts，domain gap 用 shared domain-adaptive prompts。两个模块分工很清楚。

模块设计：个性化 prompt 捕捉学生/题目特定信息，共享 prompt 捕捉域差异，再映射到 CD 表示空间。

实验关系：SLP 多域；student-side 和 exercise-side 跨域；消融删除 personalized prompts 或 shared prompts；feature visualization；personalized learning guidance。

结果写法：个性化 prompt 往往是主贡献，shared prompt 有时只小幅提升。论文接受模块贡献不均衡，因为叙事上主副模块明确。

可模仿点：CRG 可以是主问题，LCRF 可以是副问题。但要把 LCRF 写成“在 CRG 已给方向后做个性化微调”，不要硬写成和 CRG 同等规模的第二主干。

## 17. Generative CD, IEEE TLT 2026

来源：[arXiv PDF](https://arxiv.org/pdf/2507.09831)

核心科学问题：传统 CDM 是 transductive prediction paradigm，需要重新优化参数才能诊断新学生，且诊断结果可识别性和可靠性不足。

故事写法：它直接提出范式转变：从 predictive cognitive diagnosis 到 generative cognitive diagnosis。叙事重点是诊断可靠性、可识别性、即时推理，不只是 AUC。

模块设计：用生成过程把 cognitive state inference 和 response prediction 解耦，提出 G-IRT 和 ID-CDM 两个实例化。

实验关系：真实 CD 数据集上的预测性能；新学生即时诊断；可靠性和单调性条件；理论性质与实验结合。

结果写法：它把“为什么诊断结果可信”放在与预测一样重要的位置。这类论文说明 CD 顶刊接受理论/可靠性/解释性作为核心贡献。

可模仿点：我们的 CRG/LCRF 不要只追求 AUC。若 CRG/LCRF 可解释，必须补“地图边是否合理”和“个性化 posterior 是否由学生状态驱动”的证据。

## 18. Transfer-Q, Acta Psychologica Sinica 2026

来源：[ScienceEngine PDF](https://www.sciengine.com/parse/pdf/0439-755X/BFD1500394A44D8586A299ABFA285F9C.pdf?attname=Cognitive+diagnosis+method+via+neural+networks+with+transfer+learning+and+Q-matrix+constraints.pdf)

核心科学问题：神经网络 CDM 的可解释性依赖 Q-matrix，跨场景迁移时还要避免破坏 Q 约束。

故事写法：更偏心理测量写法，强调 Q-matrix constraints 和 transfer learning 对诊断解释的约束意义。不是单纯深度模型性能竞赛。

模块设计：把迁移学习和 Q-matrix 约束结合，保证目标域诊断结果仍然贴合专家知识结构。

实验关系：通常包含仿真或真实数据迁移设置、约束前后比较、不同样本量/迁移强度下的表现。

结果写法：心理学期刊更关心模型约束是否合理、诊断参数是否可解释、迁移是否稳定。

可模仿点：CRG 的 train-only evidence 可以写成 Q-matrix 与学习序列共同形成的约束，不要写成随意学习一张图。LCRF 也不能读 student-id shortcut。

## 19. DiaCDM, ICASSP 2026

来源：[arXiv PDF](https://arxiv.org/pdf/2509.24821)

核心科学问题：真实教学过程不只有答题日志，师生对话中包含大量诊断线索。传统 CDM 无法利用 Initiation-Response-Evaluation 过程。

故事写法：把数据形态从 response log 扩展到 dialogue，把诊断线索拆成提问、回答、评价等过程结构。叙事中心是“过程证据”。

模块设计：面向 teacher-student dialogue 的诊断模型，抽取对话中的认知状态线索，再映射到概念掌握。

实验关系：对话诊断数据、与文本/日志 baseline 比较、过程结构消融、case 展示对话如何影响诊断。

结果写法：新场景论文通常不要求直接和所有传统数据集 SOTA 对齐，而是强调数据和证据来源不同。

可模仿点：我们的 LCRF 可以写成“学习过程中的局部路线证据”，特别是 recent mastery/history neighbor count，而不是题目静态属性。

## 20. Exploratory DeepCDM, Psychometrika 2026

来源：[Cambridge PDF](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/16E69BA84BFA7A0C93BFC06A59AAEDFF/S0033312325100653a.pdf/deep_generative_modeling_for_cognitive_diagnosis_via_exploratory_deepcdms.pdf)

核心科学问题：在未知或不完整 Q-matrix 下，如何做可识别、可解释的深层生成认知诊断。

故事写法：心理测量写法很重视 identifiability、estimation、simulation。它不是先堆模型，而是先证明模型在什么条件下能恢复可解释结构。

模块设计：深层生成模型、layer-wise EM、非线性 spectral initialization、L1 sparsity 等，用于探索性 Q/属性结构恢复。

实验关系：仿真实验验证参数恢复和可识别性；真实 TIMSS 等数据用热力图展示属性结构；模型比较更关注诊断结构是否合理。

结果写法：热力图、属性矩阵恢复、可识别性证明是核心，不是预测 AUC 唯一指标。

可模仿点：CRG 的 concept map 和 LCRF 的 local posterior 都应该有热力图/局部案例。只要可解释性是主卖点，实验图必须能让人看懂“模块到底学了什么”。

## 近期论文的共同叙事模板

### 模板 A：先发现现象，再设计模块

代表论文：NCDLA、ESR-CD、DMC-CDM。

结构：
1. 真实 CD 场景中存在一个现象，例如噪声集中在低奇异成分、学生-概念覆盖稀疏、观测不完整。
2. 用统计图或小实验证明现象存在。
3. 设计模块只解决这个现象。
4. 主实验 + 场景实验 + 消融 + 可视化闭环。

适合我们：先用数据诊断证明 CRG 该解决“概念关系缺失/学习路径缺失”，LCRF 该解决“同一全局地图下学生路径不同”，再上模块。

### 模板 B：主模块和副模块明确分工

代表论文：DMC-CDM、DBCD、PromptCD、FACD。

结构：
1. 主模块解决核心科学问题。
2. 副模块补一个明确的残余问题。
3. 消融允许主模块掉得多、副模块掉得少。
4. 用场景解释为什么副模块不是每个样本都强。

适合我们：CRG 是主问题“全局学习地图”，LCRF 是副问题“个体小地图”。LCRF 不必在所有数据集都掉 2 个点，但必须在个性化场景中被 actual/shuffle/mean 证明。

### 模板 C：性能提升有限，但解释性强

代表论文：KAN2CD、Exploratory DeepCDM、Transfer-Q。

结构：
1. 明确说现有神经 CDM 不可解释或约束不足。
2. 方法保持性能同时提供可解释结构。
3. 用曲线、热力图、case、显著性检验支撑。

适合我们：如果 CRG/LCRF 的全局 AUC 提升有限，就把小实验图做硬：CRG map retrieval/corruption，LCRF actual vs shuffled posterior。

### 模板 D：不要全 test 混合，按场景拆

代表论文：DFCD、LRCD、FACD、ESR-CD。

结构：
1. 按 cold-start、early-stage、weak coverage、open exercise、zero-shot 等场景拆数据。
2. 模块只在目标场景中必须强。
3. 总体平均只是补充。

适合我们：CRG-relevant subgroup 和 LCRF-personalization subgroup 比全局 no_CRG/no_LCRF 更重要。全局表保留，但故事重点应该是“地图发挥作用的样本”。

## 对 ConceptSkillCDM 的直接决策

### 是否继续当前不断累加残差的重构

不建议继续。原因是当前方向容易变成：为了让 no_CRG/no_LCRF 掉分，把 CRG/LCRF 都接到主诊断路径不同位置，导致模块语义越来越不清楚。近期论文确实会改主诊断路径，但前提是科学问题清楚，例如噪声、缺失、跨域、早期诊断。我们现在如果继续乱加，论文会更难写。

### 是否回溯到之前效果好的版本

建议回溯或至少冻结之前强消融版本作为主线。之前已经有一个很有价值的表：

| dataset | full | no_CRG | no_LCRF |
|---|---:|---:|---:|
| assist_09 | 0.7796 | 0.6817 | 0.6927 |
| junyi | 0.8291 | 0.6356 | 0.7986 |
| assist_17 | 0.7868 | 0.6807 | 0.7093 |

这比近期探测中 full 降到 0.73 左右的方向更适合作为论文主表。后续应该围绕这个版本补机制实验，而不是继续改到主表崩掉。

注意：这张表需要用服务器最终 logs/results 重新核验路径、run id、seed、checkpoint，不能只凭聊天记忆写论文。

### CRG 模块建议命名和科学问题

推荐名字：Reliable Global Learning Map。

科学问题：在真实学习平台中，单个学生的作答历史覆盖有限，题目可能只有单概念标注，直接诊断会缺少概念之间的学习路径关系。CRG 用 train-only item cooccurrence、sequence transition 和 self retention 构建全局学习地图，解决“学生当前题目应该参考哪些相邻概念”的方向问题。

建议实验：
1. Held-out concept transition retrieval：只用 train 构图，预测 valid/test 中学生后续概念 transition。指标用 Hit@K、Recall@K、NDCG、MRR。对比 A_fused、seq-only、item-only、self-only、degree-matched random。
2. CRG support corruption：不重训，在 inference 时把 CRG support 按 25/50/75/100% 替换成 degree-matched random support。看 CRG-relevant 样本 AUC/BCE 是否随 corruption 下降。
3. CRG-relevant subgroup：用 train-only 特征分组，例如 query concept 是否与学生历史 concept 被 CRG support 连接、support mass 高低、sequence evidence 强弱。只写“收益集中在高 support/强 sequence evidence 场景”，不要硬写严格单调。
4. 局部地图 case：画某个题目或 concept 的 CRG support，标注 item/seq/self evidence 来源，并展示 no_CRG 错、full 对的样本。

### LCRF 模块建议命名和科学问题

推荐名字：Personalized Local Route Map。

科学问题：同一张全局学习地图不能直接等价于每个学生的学习路线。学生近期做过哪些邻居概念、掌握得如何、是否刚刚出错，会影响当前题目应该参考哪些邻居。LCRF 在 CRG 给出的 support 上做学生条件化的局部重加权，解决“同一全局地图下，不同学生下一步该看哪条局部路线”的问题。

建议实验：
1. Actual/shuffle/mean/no_LCRF 反事实：不重训或少量固定 checkpoint inference，actual LCRF 使用真实学生状态，shuffle LCRF 打乱学生状态，mean LCRF 使用群体平均状态。若 actual 明显优于 shuffle/mean/no_LCRF，说明 LCRF 不是固定补丁。
2. 同一 CRG support，不同学生小地图：固定同一 query concept 或 exercise，展示不同学生 history/recent mastery，画 LCRF posterior 差异和 full/no_LCRF 概率差异。
3. LCRF-strong case 选择规则：不能随机选。必须满足 full 正确、no_LCRF 错误、LCRF gain 高、posterior KL/delta 高、top shifted support 在学生历史中出现过、query concept 多样。
4. 早期/局部历史分组：参考 FACD，把 LCRF 的作用写成“学生有一定局部历史后做个性化修正”，不是所有样本都同等强。

### 论文故事的推荐结构

1. 现实问题：CDM 需要诊断学生对概念的掌握，但现实平台里概念覆盖稀疏、单概念题多、学生路径不同。
2. 第一层问题 CRG：没有全局学习地图时，模型只看当前题目/学生局部历史，无法知道概念之间的经验学习路径。
3. CRG 模块：Reliable Global Learning Map，从 train-only 的 item evidence 和 sequence evidence 构建可解释概念地图。
4. 第二层问题 LCRF：全局地图对所有学生相同，但不同学生的局部学习路径不同。
5. LCRF 模块：Personalized Local Route Map，在 CRG support 上根据学生历史 mastery/recent state 做 posterior reweighting，不生成新边。
6. 主实验：full/no_CRG/no_LCRF，09、Junyi、17 三个核心数据集。
7. CRG 小实验：transition retrieval、support corruption、CRG-relevant subgroup、局部地图 case。
8. LCRF 小实验：actual/shuffle/mean/no_LCRF、同图不同学生、LCRF-strong case。
9. 讨论：CRG 是主贡献，LCRF 是个性化 refinement；LCRF 在单概念或低局部历史数据集上可能较弱，这是符合机制的。

## 下一步执行建议

1. 先停止继续当前 PLRC/残差累加方向的训练，不再新增结构。
2. 清点服务器 logs/results/checkpoints，找回强消融版本的 commit、配置、run id 和 checkpoint。
3. 在该版本上重跑或核验 09、Junyi、17 的 full/no_CRG/no_LCRF。
4. 不改模型先补 CRG/LCRF 机制实验脚本，优先不重训的 inference counterfactual。
5. 只有当机制实验发现明确缺口，再做最小代码改动。例如 LCRF actual 不强时，只改 LCRF 的 personal-state 使用方式；CRG support corruption 不敏感时，只改 CRG support 构造，不改预测头。

## 不建议的做法

1. 不建议为了消融掉分继续把 CRG/LCRF 接入更多 residual 路径。这样会让 no_CRG/no_LCRF 看起来好，但模块无法解释。
2. 不建议要求每个数据集都达到 1-2 个点的 no_LCRF drop。近期论文更强调目标场景和主副模块关系。
3. 不建议把 CRG 写成“学 prerequisite”。sequence transition 最多写 empirical learning-path relation，除非另有先修验证。
4. 不建议把 LCRF 写成“重新生成学生个性化图”。更稳的是：LCRF 不扩 support，只在 CRG 的 support 上做局部 posterior reweighting。
5. 不建议把所有新数据集都塞进主表。如果 FrcSub/EdNet/ASSIST12 清洗后消融不稳定，可以放 appendix 或 generalization table。




