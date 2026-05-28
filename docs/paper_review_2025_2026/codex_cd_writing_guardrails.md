# Codex 写作约束：通用 CD 论文写作规则（v4）

> 用途：后续让 Codex 修改、扩写或重构认知诊断（Cognitive Diagnosis, CD）论文时，将本文件作为项目写作约束。它是**通用规则**，不绑定某一个具体模型或数据集。不要把本文件内容直接并入论文正文。

---

## 0. 核心原则

CD 论文应按正常顶会/顶刊论文逻辑组织：

```text
具体教育场景 / 失败模式
→ 问题定义
→ 方法模块
→ 主性能结果
→ 消融与分析
→ 简洁讨论与边界
```

不要写成：

```text
内部实验审计报告
→ 机制证据链
→ 哪些数据集强、哪些数据集弱
→ 不能证明什么
→ 自我辩护式限制
```

写作目标是让读者看到：

1. 现有 CD 方法在什么现实教育场景下失效；
2. 本文提出的模块如何对应这个失败模式；
3. 实验如何支持模型有效性和模块作用；
4. 边界在哪里，但不防御、不自我削弱。

---

## 1. 教育学 / 心理学理论如何使用

### 1.1 理论必须服务于论文主线

教育学和心理学理论可以增强 CD 论文的说服力，但不能作为装饰性引用。理论要回答以下问题之一：

| 理论类型 | 适合支撑的问题 | 不应写成 |
|---|---|---|
| 心理测量 / Psychometrics | CD 输出为什么应对应 latent ability / mastery / attributes | 泛泛说“本方法有心理学基础” |
| IRT / MIRT / DINA / Q-matrix | 标准 CD 任务、属性掌握、题目-概念关系 | 把基础 CD 模型讲得过长 |
| Evidence-Centered Design / Assessment | 诊断输出应由观测响应支持，适合解释“诊断依据缺失” | 反复写“证据链”“可审计”，造成 AI 味 |
| Learning Progression | 学习过程可能沿可观察概念路径发展，适合解释序列路线 | 把 sequence transition 写成 prerequisite 或因果先修 |
| ZPD / Scaffolding | 学习支持应随学生当前状态调整，适合解释个性化过滤 / 支架选择 | 声称模型真正优化了教学支架或学习增益 |
| Mastery Learning | 学生掌握度和近期表现可作为个性化诊断信号 | 把 mastery learning 当成实验结论 |
| Causal / Counterfactual Theory | 选择偏差、异常行为、混杂、反事实数据 | 没有因果识别却声称因果结论 |
| Information Theory | 信息损失、不完全观测、多视角补全、瓶颈 | 无理论推导却堆 MI 符号 |
| Robustness / Noise Theory | 噪声、猜测、失误、Q-matrix 错误、log flip | 把普通噪声实验写成强理论证明 |

### 1.2 理论应出现在五个位置，但每处只承担一个功能

| 位置 | 理论的作用 | 写法 |
|---|---|---|
| Introduction | 用理论或教育场景定义失败模式 | “From the perspective of ..., this setting indicates ...” |
| Problem Definition | 将失败模式转成可操作变量 | “We characterize this condition by ...” |
| Method | 解释模块设计为什么合理 | “Motivated by ..., this module ...” |
| Experiments | 解释为什么做某类分析 | “We evaluate whether ... under this condition.” |
| Discussion | 回收教育意义与边界 | “These results suggest ..., but do not imply ...” |

不要把理论全部堆在 Related Work 的一个小节里；也不要在每节都重复同一个理论名。

### 1.3 教育理论的正确写法模板

#### Evidence-centered / measurement perspective

可以写：

```text
From an educational measurement perspective, a CD model should connect observed responses with latent mastery estimates. When the target concept is not directly covered by a learner's history, this connection becomes indirect and needs to be modeled explicitly.
```

避免写：

```text
We build an auditable evidence chain and provide operational mechanism evidence.
```

#### Learning progression / empirical route

可以写：

```text
Sequence transitions are used as empirical routes observed in learning logs. They are related to learning progressions in the sense of observable developmental paths, but they are not treated as deterministic prerequisite relations.
```

必须避免：

```text
Sequence transitions are prerequisites.
Sequence routes prove causal learning progressions.
```

#### ZPD / scaffolding / mastery learning

可以写：

```text
A globally available support concept may not be equally useful for every learner. Consistent with the idea of adaptive support, the filter conditions route weights on the learner's mastery, recent performance, and observation reliability.
```

避免写：

```text
The model implements ZPD.
The model provides real instructional scaffolding.
```

更稳的表述是：

```text
The design is inspired by learner-adaptive support, rather than claiming to optimize instructional intervention.
```

---

## 2. “证据 / 审计 / 可审计”类词的边界

### 2.1 可以使用的情况

“evidence / diagnostic evidence / evidence-centered” 在教育测量语境中可以出现，但必须满足：

1. 第一次出现时明确其教育测量含义；
2. 后文用更具体的词替代，如 route statistics、historical responses、concept relations；
3. 不把“evidence chain”变成全文高频口号。

推荐频率：

- Abstract：最多 0–1 次。
- Introduction：最多 1–2 次。
- Method：尽量用 concrete signals / training relations。
- Experiments：不用“evidence chain”，用具体实验名。
- Discussion：最多 1 次，用于回收教育测量含义。

### 2.2 慎用或禁用词

避免高频使用：

- auditable / audit / 可审计 / 审计
- evidence chain / 证据链
- operational evidence / 操作性证据
- mechanism evidence / 机制证据
- sufficient and necessary / 充分必要
- cannot prove / cannot exclude / 不能证明 / 无法排除
- weak evidence / strong evidence dataset / 主证据 / 弱证据

### 2.3 推荐替换

| 不推荐 | 推荐 |
|---|---|
| auditable route support | explicit concept route / traceable concept route |
| evidence chain | concept route / diagnostic link / route support |
| evidence-constrained | route-constrained / support-constrained |
| operational mechanism evidence | empirical analysis / component analysis |
| weak evidence | dataset-specific behavior / less pronounced effect |
| cannot prove necessity | the effect is stronger/weaker under this setting |
| LCRF strongly works on dataset X | learner-state conditioning produces larger/smaller changes on dataset X |

---

## 3. 标题规则

### 3.1 标题应直接表达问题与方法

标题通常包含：

```text
Method idea + Target scenario + CD
```

例子：

```text
Concept Reachability for Cognitive Diagnosis under Sparse Concept Coverage
Route-Constrained Cognitive Diagnosis under Sparse Concept Coverage
Learner-Conditioned Concept Reachability for Cognitive Diagnosis
```

### 3.2 标题避免过度抽象

避免：

```text
Evidence-Constrained Concept Reachability for Bridging Concept Evidence Gaps
Auditable Concept Route Evidence for Cognitive Diagnosis
Operational Mechanism Evidence for Cognitive Diagnosis
```

除非论文真的以教育测量证据链为主理论，并有清晰 formalization，否则不要让标题堆 “evidence / audit”。

---

## 4. 摘要规则

### 4.1 摘要四句结构

摘要只写四件事：

1. CD 任务背景；
2. 现有方法忽略的具体场景；
3. 本文方法的两个核心动作；
4. 实验总体结论。

推荐结构：

```text
Cognitive diagnosis estimates learners' mastery states from response logs and Q-matrices. Existing methods ... however ... . To address this issue, we propose ... . Experiments on ... show that ... .
```

### 4.2 摘要不要写成实验清单

避免在摘要里同时列出：

- route retrieval;
- support perturbation;
- counterfactual;
- same-query case;
- stress test;
- dataset-specific differences;
- limitation.

可以压缩为：

```text
Further analyses on route retrieval, support perturbation, and learner-state replacement indicate that the proposed modules behave consistently with the target scenario.
```

### 4.3 摘要不要写防御性边界

避免：

```text
This is not a sufficient and necessary proof.
Student-ID shortcut cannot be fully excluded.
The evidence is weak on some datasets.
```

边界放 Discussion 或 Appendix。

---

## 5. Introduction 写作规则

### 5.1 推荐段落结构

1. **CD task and educational value**：CD 是什么，用于什么。
2. **Concrete failure mode**：现有方法在哪个真实场景失败。
3. **Why existing methods are insufficient**：为什么已有 CD / graph CD / LLM CD 不能直接解决。
4. **Proposed idea**：本文方法的两个核心模块。
5. **Figure 1 and contributions**：图示直觉 + 贡献列表。

### 5.2 第一段不要过度理论化

可以从教育应用和 CD 任务出发：

```text
Cognitive diagnosis estimates learners' mastery over knowledge concepts from historical response logs and Q-matrices. It supports downstream applications such as adaptive practice, exercise recommendation, and learning path planning.
```

如果使用 ECD / educational measurement，应在第二段自然引入：

```text
From a measurement perspective, the prediction should be connected to observations relevant to the target concept. When the target concept is absent from learner history, this link becomes indirect.
```

不要第一段就长篇解释 ECD、claim-evidence-reasoning，否则会偏离 KDD/AAAI CD 写法。

### 5.3 Figure 1 的位置

Figure 1 应放在 Introduction 中失败场景解释之后、方法贡献之前。它应承担：

- 说明问题；
- 展示为什么现有方法不够；
- 给出方法直觉。

Figure 1 不应承担完整方法公式或完整实验结果。

### 5.4 贡献列表

贡献列表写 3–4 条即可：

1. Formulate a concrete CD scenario/problem;
2. Propose method module A;
3. Propose method module B;
4. Conduct experiments and analyses.

不要写：

```text
We prove sufficient and necessary conditions.
We construct an operational mechanism evidence chain.
```

---

## 6. Related Work 规则

### 6.1 小节数量

通常 3 个小节足够：

```text
2.1 Cognitive Diagnosis Models
2.2 Graph-based / Relation-aware / Text-enhanced CD
2.3 Sparse, Robust, or Learner-adaptive Diagnosis
```

如果论文有明确教育理论贡献，可以增加：

```text
2.4 Educational Measurement and Learner-adaptive Support
```

但这个小节必须短，并且要和方法模块直接对应。

### 6.2 不要把教育理论小节写成泛泛背景

不要写：

```text
ECD, ZPD, scaffolding, mastery learning are important in education.
```

要写成：

```text
ECD motivates why direct concept coverage matters; learning progression motivates empirical concept routes; scaffolding motivates learner-conditioned filtering. We use these ideas as design motivation, not as causal claims about instruction.
```

### 6.3 Related Work 需要方法名和边界

每一类至少点名 2–4 个代表方法，并说明它们和本文的差异。

模板：

```text
Compared with X and Y, which focus on ..., our work focuses on ... .
```

避免：

```text
Existing works have limitations.
Many methods cannot solve our problem.
```

---

## 7. Problem Definition 规则

### 7.1 只定义对方法和实验必要的符号

主文 Problem Definition 保留：

- learners / exercises / concepts;
- response log;
- Q-matrix;
- query concept;
- history set;
- core problem condition.

不要把基础 CD 符号展开成过多 display equations。

### 7.2 不要过度分类样本

避免主文放三分类 cases：

```latex
D=1: direct coverage
D=0,R>0: bridgeable
D=0,R=0: unreachable
```

可以文字写：

```text
We characterize a query by whether its target concept is directly covered by learner history and whether it admits non-empty route support. These diagnostics are used for analysis rather than restricting the training or evaluation instances.
```

### 7.3 如果有教育测量解释

可加一句：

```text
This condition indicates that the target mastery estimate must rely on indirect concept relations rather than direct historical responses.
```

不要写成 “evidence chain failure” 多次重复。

---

## 8. 公式写作规则

### 8.1 主文公式总量

主文公式应控制在 5–8 个关键公式。只保留直接服务于贡献的公式。

通常保留：

1. task / prediction objective，若有必要；
2. core representation or graph construction；
3. module-specific scoring function；
4. normalization / posterior / aggregation；
5. training objective，只有当 loss 是贡献的一部分时保留。

### 8.2 哪些公式应删除或放附录

通常不放主文：

- 标准 BCE loss；
- 泛化 MLP 预测头；
- 重复的 Q-matrix 定义；
- 与核心贡献无关的 feature normalization；
- 只用于实验分组的诊断变量；
- 长表格式 cases；
- 显而易见的 softmax / sigmoid 细节。

除非：

- loss 被理论推导使用；
- Q-matrix 变换是贡献；
- normalization 决定模型核心性质。

### 8.3 公式方向必须统一

如果定义关系矩阵或图：

```latex
A(c,k)
```

必须明确：

- row entity 是 query / source / target？
- column entity 是 support / neighbor / next concept？
- 方向是否表示时间顺序、支持关系、传播方向，还是只是索引方式？

示例写法：

```text
We store A(c,k) as the weight of support concept k for query concept c. The direction is used for support indexing and does not imply a prerequisite relation.
```

如果真实实现是 history-to-query：

```latex
A(h,c)
```

则不要在 LCRF 中又写成 query-to-support，除非明确定义转置或 incoming support。

### 8.4 避免循环定义

不要写：

```latex
A(a,b)=softmax_{b\in S_a}(s(a,b)),
S_A(c)=TopK A(c,k).
```

因为 support 在 softmax 前后被重复定义。

推荐：

```latex
\mathcal{N}(a)=\{b: \text{candidate relation exists}\},
```

```latex
A(a,b)=\operatorname{softmax}_{b\in\mathcal{N}(a)}(s(a,b)/\tau),
```

```latex
S_A(c)=\operatorname{TopK}_{k\in\mathcal{N}(c)} A(c,k).
```

### 8.5 避免重复边或重复特征

如果 item co-occurrence 和 self-retention 都存在，需要避免自环重复计入：

```latex
M^{item}_{a,b}=\sum_e Q_{e,a}Q_{e,b}\mathbb{I}[a\ne b].
```

### 8.6 教育学变量解释

公式变量应连接教育含义，但不要过度解释。

| 变量类型 | 教育含义 |
|---|---|
| mastery | 当前概念掌握状态 |
| recent performance | 近期表现或状态稳定性 |
| history count | 对该概念或路线的观测可靠性 |
| route neighbor | 与目标概念相关的历史概念 |
| posterior weight | 当前学生对候选支持概念的相对依赖 |

### 8.7 不要为了教育理论硬造公式

不要写没有实验或实现支持的：

- ZPD score;
- scaffolding utility;
- evidence chain completeness;
- learning progression probability;
- diagnostic validity score.

除非这些量真的被模型使用或实验评估。

---

## 9. 方法章节规则

### 9.1 方法结构

```text
4.1 Overview
4.2 Module A
4.3 Module B
4.4 Prediction and Optimization
```

每个模块按：

```text
Motivation → Input → Operation → Output → Constraint
```

### 9.2 教育解释要短

CRG / graph / route 模块前可以写：

```text
When direct historical coverage is missing, the model needs concept relations from training logs to connect the query concept with related historical concepts.
```

LCRF / adaptive / filtering 模块前可以写：

```text
A globally available route may not be equally useful for every learner, so the model conditions route weights on learner state.
```

不要把教育解释写成独立长段理论综述。

### 9.3 模块命名

模块名应简洁、功能明确：

- Graph / Roadmap / Route Constructor
- Learner-conditioned Filter
- Debiasing Module
- Anomaly Detector
- Alignment Module
- Fusion Module

避免：

- Evidence Auditor
- Mechanism Verifier
- Diagnostic Evidence Engine

---

## 10. 实验章节规则

### 10.1 推荐实验结构

```text
5.1 Experimental Settings
5.2 Overall Prediction Performance
5.3 Main Analysis for Target Scenario
5.4 Ablation Study
5.5 Robustness / Perturbation / Retrieval / Transfer Analysis
5.6 Case Study or Visualization
```

### 10.2 不要写成 RQ 审计报告

可以有研究问题，但不要全篇写成：

```text
RQ1 是否存在？
RQ2 是否充分？
RQ3 是否必要？
RQ4 是否不可替代？
```

更自然：

```text
We organize experiments into prediction performance, target-scenario analysis, component ablation, and further model behavior analysis.
```

### 10.3 主性能表必须优先

如果是方法论文，必须有：

- representative baselines;
- multiple datasets;
- AUC / ACC / RMSE or relevant metrics;
- mean ± std if possible;
- clear best / second-best marking.

不要让 ablation 表承担 overall performance 的作用。

### 10.4 子场景实验写法

当方法针对特定 failure mode 时，可以做 conditioned / subgroup experiments。

推荐：

```text
The gains are more pronounced in the target subgroup where ... .
```

避免：

```text
The method solves the problem in all cases.
```

必须报告：

- subgroup definition;
- sample size;
- metric;
- full model vs key baselines;
- confidence interval or repeated-run std if possible.

### 10.5 负向或压力测试

负向 stress test 不应放主文正结果。

放置位置：

- Appendix;
- Discussion 最后一小段。

写法：

```text
This stress test suggests that the proposed module supplements, rather than replaces, direct response information.
```

不要写：

```text
This experiment fails.
The model cannot ...
```

---

## 11. 图表规则

### 11.1 主文图表数量

8–10 页论文建议：

- 4–6 张主图；
- 2–3 张主表；
- 其余完整数值放 appendix。

### 11.2 图放置位置

| 图类型 | 推荐位置 | 作用 |
|---|---|---|
| Motivation / problem figure | Introduction | 建立失败模式和直觉 |
| Dataset phenomenon figure | Introduction 或 Experiment 前半 | 说明问题真实存在 |
| Framework figure | Method overview | 展示模块关系 |
| Route/retrieval/robustness figure | Experiments | 展示趋势 |
| Subgroup/conditioned result figure | Experiments target analysis | 展示目标场景收益 |
| Counterfactual / perturbation figure | Experiments analysis | 展示模块行为 |
| Case study figure | Experiments 末尾或 appendix | 解释具体样本 |

### 11.3 图表选择

| 实验类型 | 推荐展示 |
|---|---|
| overall performance | table |
| ablation | table |
| robustness under noise ratio | line plot |
| support corruption / perturbation | line plot |
| retrieval comparison | grouped bar |
| subgroup effect with CI | forest plot / dot-whisker |
| posterior / attention / route weights | heatmap + bar |
| t-SNE / representation | scatter plot |
| search process / NAS | flow figure + Pareto plot |
| case study | small diagram + appendix table |

### 11.4 合并图原则

相关图应合并成多面板图：

- history-to-query retrieval + held-out retrieval → route retrieval figure；
- support perturbation + learner-state counterfactual → component behavior figure；
- posterior heatmap + prediction change → same-query case figure；
- dataset statistics + motivation study → problem figure。

避免连续放多个小图，导致论文像实验记录。

### 11.5 图注规则

图注描述图中内容和趋势，不写“用于证明”。

推荐：

```text
Figure X reports AUC changes under different corruption ratios. Larger drops indicate stronger sensitivity to the perturbed relation.
```

避免：

```text
Figure X proves the mechanism validity and provides strong evidence.
```

---

## 12. 表格规则

### 12.1 表格只放数值

不要在表格里放：

- explanation column;
- claim status;
- strong/weak evidence;
- paper wording;
- dataset role。

这些写在正文中。

### 12.2 表格数字要精确

不要在正式表格中写：

- 约 0.014；
- around 0.01；
- slightly higher。

如果没有精确值，改用图或正文定性描述。

### 12.3 主文和附录分工

主文表格：

- dataset statistics;
- overall performance;
- ablation。

Appendix 表格：

- full subgroup numbers;
- all retrieval metrics;
- complete perturbation values;
- full counterfactual table;
- hyperparameters;
- more case studies。

---

## 13. 语言风格规则

### 13.1 使用具体动词

推荐：

- constructs;
- retrieves;
- aligns;
- filters;
- perturbs;
- compares;
- aggregates;
- calibrates;
- replaces;
- reconstructs。

避免过多抽象名词：

- mechanism;
- evidence;
- validity;
- audit;
- proof;
- sufficiency;
- necessity。

### 13.2 结果解释写法

推荐：

```text
The full model obtains the best AUC on all datasets, while the ACC gain is less consistent.
```

```text
The effect is more pronounced on the subgroup where ... .
```

```text
This pattern suggests that the module mainly improves ranking-oriented prediction.
```

避免：

```text
This proves the module is necessary.
```

```text
This is weak evidence.
```

```text
This cannot fully exclude ...
```

---

## 14. 会议模板规则

根据目标会议决定模板。

| 目标 | 模板 |
|---|---|
| AAAI | AAAI style；不要 Index Terms |
| KDD / CIKM / WWW / SIGIR | ACM style；需要 CCS Concepts / Keywords / ACM Reference Format |
| IEEE 会议 / 期刊 | IEEE style；可用 Index Terms |

不要用 IEEE 模板写 KDD/AAAI 稿。

---

## 15. 最新增补：CD 正文细节规则（v4.1）

### 15.1 数据集介绍必须完整

CD 论文的实验设置中必须有数据集规模信息。主文至少报告：

- students；
- exercises / items；
- concepts / knowledge components；
- interactions / response logs；
- split protocol；
- dataset-specific diagnostic rates（如果是本文贡献相关指标）。

如果提出新的数据现象指标，例如 direct-unseen rate 或 bridge-only rate，必须在表格前用一句话定义，不能只把缩写放进表头。推荐写法：

```text
Direct-unseen rate denotes the proportion of test queries whose target concept has not appeared in the learner history. Bridge-only rate denotes the proportion of direct-unseen queries whose target concept can still be connected to the learner history through the constructed concept route support.
```

### 15.2 避免二分式 AI 腔

少用高频二分句式：

- 不是……而是……
- 不仅……也……
- not only ... but also ...
- rather than ...

这类句式只在确实需要对比方法边界时使用。多数情况下改成正向句：

```text
The model estimates response probability and exposes concept-level route support for the target concept.
```

而不是：

```text
The model not only predicts accurately, but also provides clear concept-level support.
```

### 15.3 贡献列表不写实验清单

贡献列表应写“问题、方法、结果或资源”，不要把所有实验名称堆进去。避免：

```text
We evaluate the method through overall performance, ablation, coverage-conditioned prediction, support perturbation, and case studies.
```

推荐：

```text
We empirically show that the proposed route support improves prediction under sparse concept coverage and yields interpretable route-level behavior.
```

### 15.4 实验流程用常见词，不自造术语

不要把普通实验安排命名成“分层诊断协议”“机制证据链”等新术语。推荐：

- evaluation protocol；
- analysis protocol；
- experimental design；
- evaluation suite；
- empirical analysis；
- subgroup analysis；
- counterfactual analysis。

中文推荐用：

- 评估流程；
- 实验流程；
- 分组分析；
- 反事实分析；
- 个案分析。

### 15.5 消融命名

主模型在表格中默认写 `Full`。删除模块写 `w/o X`。只有在“模块有无”矩阵表中才写 `w/ X`。推荐：

```text
Full | w/o CRG | w/o LCRF
```

避免：

```text
w/ CRG+LCRF
```

因为主模型不应显得像某个额外插件组合。

### 15.6 公式排版

双栏论文中公式应按“核心操作分块 + 公式后解释变量”写，不要把 pipeline、score、softmax、posterior、loss 全部挤成长式。推荐：

1. 每个 display equation 只表达一个核心动作；
2. 长公式拆成短变量，例如先定义 \(z^{prior}\)、\(z^{state}\)，再定义 posterior；
3. 标准 BCE 可以保留一行或文字说明，但不要占据多行，除非 loss 本身是贡献；
4. 公式后用一句话解释每个变量，不要在公式里塞过多上标、下标和长文本；
5. 如果公式在单栏中明显过宽，优先拆成两个 display equations，而不是缩小字号。

### 15.7 模块边界不要反复强调

“LCRF 不新增边/不生成图”只在方法总览或 LCRF 小节出现一次即可。后文改写为：

```text
LCRF reweights the fixed CRG support.
```

中文写：

```text
LCRF 在固定 CRG 支持集内调整后验权重。
```

不要每个图注、摘要、贡献和结论都重复“不新增边”。

### 15.8 数据集替换分析

如果候选数据集不进入主线，需要给出数据机制理由，而不是只说“不适合”。例如：

```text
NIPS34 has dense multi-concept exercises and very low direct-unseen rate; therefore, it is more suitable as a dense-concept contrast than as the main sparse concept coverage dataset.
```

但这类替换分析通常不放在主文正文。若用户已经确定主数据集，不要在主文加入“我们也检查了某某数据集，但不替换某数据集”这类句子；可放在内部审稿包、附录说明或实验选择记录中。

### 15.9 图表与正文位置

ICDM/IEEE 双栏论文中，图表应尽量放在第一次引用附近，且图在解释段落之前。若两个相邻实验共同回答同一个研究问题，优先合并成一个单栏多部分图，而不是连续放两张小图。推荐：

- 主问题检索 + 覆盖条件预测：合并成一张单栏图；
- 支持集扰动的 AUC/BCE：可合并成一张紧凑图；
- LCRF 反事实 + same-query 个案：若篇幅允许可合并，若信息密度过高则分开。

如果图注已经解释变量，图内不要再放大标题；保持轴标签、图例和短缩写即可。

### 15.10 Discussion 的使用

短会议论文不一定需要单独 Discussion。若 Discussion 只是在重复结果，应删除该章节，将机制解释分别并入对应结果段、Limitations 和 Conclusion。只有当 Discussion 能提出新的教育含义、理论边界或跨数据集规律时才保留。

--- 

## 16. Codex 固定提示词

后续让 Codex 修改 CD 论文时，可直接使用：

```text
请根据以下规则修改论文。目标是写成正常 KDD/AAAI 风格的认知诊断论文，不要写成内部实验审计报告。

1. 论文主线必须是：具体教育场景中的失败模式 → 方法模块 → 实验结果 → 简洁讨论。
2. 教育学/心理学理论只能作为问题动机或模块解释，不要堆砌。ECD 可以用于说明诊断需要观测支持，learning progression 只能解释经验序列路线，不能写成 prerequisite；ZPD/scaffolding 只能解释 learner-conditioned support，不要声称模型真正优化教学支架。
3. 摘要只写任务、问题、方法和总体实验结论，不要列完整实验清单。
4. 删除或降频 auditable、evidence chain、operational evidence、sufficient/necessary proof、weak evidence 等表达。
5. 方法公式只保留核心模块公式。删除标准 BCE、泛化 MLP 预测头、过度 Q-matrix 定义和与贡献无关的分组变量公式。
6. 所有图矩阵方向必须清楚，例如 A(c,k) 是 query-to-support 还是 history-to-query。不得出现 support 集合先后循环定义。
7. 主文图表应控制数量：overall performance 和 ablation 用表；retrieval、robustness、counterfactual、subgroup analysis 用图；完整数值放 appendix。
8. 图注只描述图中内容和趋势，不写“证明/审计/机制有效性”。
9. 实验结论必须按数据集和场景准确表述，不能把部分数据集结果写成全局结论。
10. 讨论只写发现、教育含义和自然边界，不写防御性限制清单。
```

---

## 17. 投稿前自检清单

### 语言

- [ ] 摘要没有实验清单式堆砌。
- [ ] “evidence / audit / proof / weak evidence” 没有高频出现。
- [ ] 没有“不能证明 / 无法排除 / 不是充分必要”这类防御句。
- [ ] 教育理论没有变成装饰性引用。

### 结构

- [ ] Introduction 先讲具体问题，再讲方法。
- [ ] Related Work 有方法名和差异，不泛泛分类。
- [ ] Method 每个模块都有 input / operation / output / constraint。
- [ ] Experiments 有 overall performance，再有 ablation 和分析。
- [ ] Discussion 不像 limitation list。

### 公式

- [ ] 公式总量控制在核心贡献范围内。
- [ ] 图/矩阵方向已明确。
- [ ] support 定义无循环。
- [ ] 自环、共现、序列边没有重复计入。
- [ ] BCE 等标准公式没有占用主文空间，除非是贡献。

### 图表

- [ ] 主文图表没有过多。
- [ ] 相关实验已合并成多面板图。
- [ ] 表格没有解释列。
- [ ] 图注是描述式，不是证明式。
- [ ] 完整数值放 appendix。

### 实验

- [ ] baseline 数值已核验。
- [ ] 指标结论准确，例如 AUC 最好不等于所有指标最好。
- [ ] subgroup 样本量已报告。
- [ ] 单数据集结论没有被写成全局结论。
- [ ] 负向 stress test 不作为主文正向结果。
