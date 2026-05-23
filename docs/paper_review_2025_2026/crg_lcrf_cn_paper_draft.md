# 面向概念证据缺口的可审计概念可达性认知诊断

## 摘要

认知诊断旨在根据学生历史作答记录估计其对知识概念的掌握状态。现有神经认知诊断与图认知诊断模型通常依赖学生-题目响应日志和题目-概念矩阵进行预测，但在真实学习平台中，学生历史并不总是直接覆盖当前测试题涉及的目标概念。当目标概念缺少直接历史证据时，模型需要回答一个关键问题：**当前诊断能否获得可审计的概念支持？** 本文将这一现象定义为**概念证据缺口**。为缓解该问题，本文提出一种证据约束的概念可达性认知诊断框架，包括概念可达图（Concept Reachability Graph, CRG）和学习者条件化可达性过滤器（Learner-Conditioned Reachability Filter, LCRF）。CRG 从训练阶段可观测的题内概念共现、序列转移、自保持和可靠性统计中估计全局概念路线图；LCRF 在 CRG 给出的固定 support 内，根据学生掌握度、近期表现、路线邻居状态和历史计数重排后验路线权重。三组核心数据集 `assist_09`、`junyi` 和 `assist_17` 上的实验表明，CRG 能有效检索 held-out concept transitions，并在不同数据集上呈现不同的 support-dependence regime；LCRF 的学生状态反事实和 same-query case 进一步显示，同一 CRG support 可以被不同学生过滤成不同 posterior route。上述结果表明，CRG-LCRF 能够为稀疏概念证据场景提供可审计的路线支持与支持约束的个性化诊断。

**关键词**：认知诊断；概念证据缺口；概念可达图；训练阶段证据；学习者条件化过滤；机制可解释性

---

## 1. 引言

认知诊断（Cognitive Diagnosis, CD）是智能教育系统中的基础任务，其目标是根据学生历史作答记录推断学生对知识概念的掌握状态。诊断结果可以服务于个性化练习推荐、补救学习、学习路径规划和自适应测试。近年来，神经认知诊断和图认知诊断模型通过更复杂的学生-题目交互函数、概念图传播或学生-题目-概念异构图建模提升了预测性能。然而，这些方法通常默认学生历史记录能够为当前目标概念提供足够的直接证据。

在真实学习平台中，这一默认条件并不总是成立。学生不会完整练习所有知识点，当前测试题涉及的目标概念可能从未在该学生历史中出现，或者只在极少量历史记录中出现。此时，模型如果仅依赖学生已经作答过的概念维度，就会面临一个直接的诊断依据缺口：当前目标概念虽然需要被预测，但学生历史中没有足够直接证据。本文将这一现象称为**概念证据缺口**。

概念证据缺口并不等同于简单的数据稀疏。对于认知诊断而言，关键不只是学生历史长度是否短，而是当前目标概念是否能从学生历史概念中获得可审计连接。即使一道题只对应一个知识点，训练日志中仍可能存在从历史概念到目标概念的经验性学习路线，例如不同概念在学生训练序列中的转移关系。相反，如果只依赖题内多概念共现，那么在 `junyi` 这类 100% 单概念题数据集中，概念图将退化为近似空图。因此，本文不把题内概念共现作为唯一关系来源，而是将其与序列转移、自保持和可靠性统计共同作为可审计路线证据。

为解决这一问题，本文提出 CRG/LCRF 两阶段机制。CRG 首先估计全局概念可达 support，用于刻画当前目标概念能否从学生历史概念通过训练阶段路线被连接；LCRF 随后在该固定 support 上建模学习者条件化 posterior，用于判断同一条可达路线对当前学生的可信程度。CRG 的 support 由训练阶段证据估计，LCRF 的 posterior 在该 support 内完成学习者条件化重排。

图 \ref{fig:concept-gap} 给出了本文问题设定与方法直觉。与一般短历史或冷启动不同，概念证据缺口关注的是目标概念是否能从学生已有历史概念中获得可审计支持；因此，模型需要同时解决全局 support construction 和学习者条件化 filtering 两个问题。

![概念证据缺口与支持约束个性化。目标概念 \(c\) 可能没有被学生历史概念直接覆盖，形成 direct evidence gap。CRG 从训练阶段证据构造 reachable support；LCRF 在同一 support 内根据学习者状态重排 posterior weight。Sequence transition 表示训练日志中的经验路线，而非先修关系。](figures_candidate_pdf/alternate_fig1_problem_concept_evidence_gap.pdf){#fig:concept-gap width=100%}

如图 \ref{fig:concept-gap} 所示，概念证据缺口描述的是目标概念与学生历史概念之间缺少直接可审计连接的场景。左侧的学生历史包含已观测过的概念及有限正确、错误、未观测记录，目标概念 \(c\) 需要被预测，但并未出现在该学生的历史概念集合中。中间的 CRG 利用训练阶段经验路线把历史概念连接到目标概念，其中 sequence transition 表示训练日志中的经验学习路线，而非严格先修关系。右侧的 LCRF 进一步在 CRG 给出的固定 support 内进行个性化后验重排，使 posterior mass 随学生状态变化而变化。

本文贡献如下：

1. 提出“概念证据缺口”视角，将直接覆盖不足场景下的认知诊断转化为训练阶段概念路线的检索与过滤问题。
2. 设计 CRG，从题内共现、序列转移、自保持和可靠性统计中构造可审计全局概念路线图。
3. 设计 LCRF，根据学生 mastery、recent mastery、route-neighbor mastery、readiness gap 和历史计数对同一 CRG support 进行后验过滤。
4. 构建机制证据链，通过 route retrieval、support perturbation、learner-state counterfactual 和 same-query case 分别评估 CRG 的路线构造能力与 LCRF 的支持约束个性化行为。

---

## 2. 相关工作

### 2.1 认知诊断模型

传统认知诊断模型通常建立在心理测量理论之上，例如 IRT、MIRT 和 DINA。它们通过预设的交互函数描述学生能力、题目难度和作答结果之间的关系，具有较强解释性，但表达能力有限。神经认知诊断模型进一步使用神经网络建模复杂的学生-题目交互，使模型能够从大规模作答数据中学习非线性诊断函数。典型方法通常以学生嵌入、题目嵌入和概念嵌入作为输入，通过预测学生在题目上的作答结果来优化模型。

尽管这些方法提升了预测性能，但它们通常没有显式区分“目标概念是否有直接历史证据”和“目标概念是否能通过训练阶段概念路线被连接”。当学生历史中缺少某个目标概念时，模型可能只能依赖 ID embedding 或全局统计模式进行预测，难以给出可审计的概念层解释。

### 2.2 图认知诊断与关系感知建模

图认知诊断模型通过学生-题目图、题目-概念图或异构图传播建模高阶关系。相关工作表明，图结构可以提升认知诊断模型对学生和题目的表示能力，并有助于显式建模概念、题目和学生之间的依赖关系。与此同时，图结构也要求模型明确边的来源、方向和可靠性。若概念关系仅由题内共现或全局可训练参数隐式生成，当目标概念缺少学生历史中的直接证据时，模型仍难以说明当前预测依赖哪些可审计概念路线。本文从概念 support 来源出发，将概念关系建模为训练阶段可观测证据约束下的可达路线，并进一步在固定 support 内进行学习者条件化过滤。

### 2.3 认知诊断中的稀疏概念证据

现有研究已从冷启动学生/题目、非随机缺失响应、噪声交互和不可靠图边等角度研究认知诊断中的数据不足问题。这些工作主要关注实体缺失、观测偏差或交互噪声。与之不同，本文关注目标概念层面的直接证据缺失：学生可能拥有一定长度的历史作答记录，但这些记录并不直接覆盖当前题目所需的概念。因此，模型需要判断目标概念能否由历史概念通过可审计概念路线获得支持，并进一步判断该支持是否适合当前学生。CRG 和 LCRF 分别对应这两个层次：前者构造全局可达 support，后者在固定 support 内建模学习者条件化 posterior。

---

## 3. 问题定义：概念证据缺口

设学生、题目和概念集合分别为 \(\mathcal{U}\)、\(\mathcal{E}\) 和 \(\mathcal{C}\)。题目-概念矩阵 \(Q\) 描述每道题涉及的概念，学生作答日志 \(\mathcal{R}\) 由学生、题目和二值作答结果组成。给定学生 \(u\) 在时间 \(t\) 的 query exercise，记 \(H_{u,t}\) 为该学生此前作答历史中出现过的概念集合。

本文使用两个诊断量刻画 query 的概念证据条件。直接覆盖 \(D_{u,t}(c)\) 表示目标概念 \(c\) 是否已经出现在 \(H_{u,t}\) 中；CRG reachability 表示目标概念是否能在 CRG 给出的固定 support \(S_A(c)\) 中获得证据支持。这两个量用于数据现象分析和机制解释，而不是限制预测任务本身；模型仍在完整作答样本上训练与评估。概念可达图由训练阶段可观测的概念关系估计，并在验证与测试阶段保持固定。

---

## 4. 概念证据缺口现象

表 1 展示三个核心数据集的概念可达性画像。三组数据呈现不同的证据条件：有的数据包含少量题内共现，有的数据几乎完全依赖序列路线，有的数据在较长历史下更适合分析 support-dependence。

**表 1：三核心数据集的概念可达性画像**

| 数据集 | 单概念比例 | 多概念比例 | item edge density | sequence edge density | direct-unseen rate | bridge-only rate | 历史长度中位数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| assist_09 | 82.8% | 17.2% | 0.9% | 64.3% | 3.1% | 3.1% | 41 |
| junyi | 100.0% | 0.0% | 0.0% | 25.2% | 100.0% | 99.97% | 25 |
| assist_17 | 78.3% | 21.7% | 6.2% | 76.3% | 2.8% | 2.8% | 148 |

从表 1 可以看出，三组数据集均呈现不同形式的概念证据缺口。`junyi` 没有题内概念共现，但 direct-unseen rate 与 bridge-only rate 均很高，适合检验 sequence route 的可达支持。`assist_09` 和 `assist_17` 更接近常规 benchmark 场景，少量题内共现与较强序列路线共同存在。上述差异说明，概念可达性不能仅由题内多概念共现刻画，还需要训练日志中的经验学习路线提供补充支持。

为了进一步连接数据现象与 CRG 的路线构造能力，图 \ref{fig:data-crg-retrieval} 将数据集画像与 held-out transition retrieval 放在同一视图中。上半部分展示 direct-unseen 与 bridge-only 场景是否存在；下半部分检验训练阶段 CRG 路线是否能恢复 held-out concept transitions。

![数据证据缺口与 CRG 路线检索。上半部分展示三个数据集的概念证据条件，包括单概念比例、item/sequence edge density、direct-unseen rate、bridge-only rate 和历史长度。下半部分比较 self-only、random、degree-random 与 Best CRG 在 held-out transition retrieval 上的表现。CRG 在三个数据集上均优于 self/random controls，说明训练阶段经验路线能够为目标概念提供可达支持。](figures_main_pdf/fig2_nature_data_and_crg_retrieval.pdf){#fig:data-crg-retrieval width=100%}

图 \ref{fig:data-crg-retrieval} 支持两个结论。首先，概念证据缺口在三个数据集中以不同形式存在；其中 `junyi` 是 sequence-route-dominant 场景，`assist_09` 和 `assist_17` 是 item evidence 与 sequence evidence 共存的 benchmark 场景。其次，CRG 的 route retrieval 在三组数据上均优于 self-only 和 random controls，说明它捕获的不是简单自环或随机度数效应。

---

## 5. 方法

### 5.1 框架总览

本文框架包括两个递进组件：CRG 和 LCRF。CRG 负责全局路线构造，用于形成可审计、训练阶段约束的概念路线图；LCRF 负责在固定 CRG support 上进行学习者条件化后验过滤。整体流程为：

\[
\text{train-stage evidence}\rightarrow A_{\mathrm{CRG}}\rightarrow S_A(c)\rightarrow P_{u,t}(k|c)\rightarrow \hat{r}_{u,e}.
\]

其中 \(S_A(c)\) 是概念 \(c\) 的固定 CRG support，\(P_{u,t}(k|c)\) 是 LCRF 产生的学生条件化后验路线分布。图 \ref{fig:concept-gap} 已展示整体直觉；本节进一步给出 CRG 与 LCRF 的形式化定义。

### 5.2 概念可达图 CRG

CRG 为 query concept \(c\) 和候选 support concept \(k\) 估计全局路线分数。路线证据包括题内共现、经验序列转移、自保持，以及 source/receiver reliability。设归一化后的证据向量为：

\[
\phi_{c,k}=
\left[
\tilde M^{\mathrm{item}}_{c,k},
\tilde M^{\mathrm{seq}}_{c,k},
\mathbb{I}(c=k),
\mathrm{rel}^{src}_{c},
\mathrm{rel}^{rec}_{k}
\right],
\quad
s_{c,k}=\mathbf{w}^{\top}\phi_{c,k}.
\]

其中 \(\tilde M^{\mathrm{item}}\) 表示题内共现证据，\(\tilde M^{\mathrm{seq}}\) 表示训练日志中的经验学习路线；本文不将 sequence transition 解释为严格先修关系或因果依赖。CRG 首先收集具有训练阶段证据的候选集合 \(\mathcal{N}(c)\)，并在该候选集合上进行行归一化：

\[
A_{\mathrm{CRG}}(c,k)=
\operatorname{softmax}_{k\in \mathcal{N}(c)}
\left(\frac{s_{c,k}}{\tau}\right).
\]

最终，CRG 为每个 query concept 选择固定 support：

\[
S_A(c)=\operatorname{TopK}_{k\in \mathcal{N}(c)}A_{\mathrm{CRG}}(c,k).
\]

因此，\(A_{\mathrm{CRG}}\) 是由训练阶段证据估计得到的全局概念路线图，\(S_A(c)\) 是后续 LCRF 可使用的固定 support。

### 5.3 学习者条件化可达性过滤器 LCRF

CRG 给出全局 support，但同一条路线对不同学生不一定同等可信。LCRF 在 \(S_A(c)\) 内计算学生条件化后验：

\[
P_{u,t}(k|c)=
\operatorname{softmax}_{k\in S_A(c)}
\left[
\log A_{\mathrm{CRG}}(c,k)+
f_{\theta}(u,t,c,k)
\right].
\]

其中 \(f_{\theta}(u,t,c,k)\) 由 query mastery、recent mastery、route-neighbor mastery、readiness gap 和 support count 等学生状态信号计算。由于 softmax 被限制在 \(S_A(c)\) 内，LCRF 只改变 CRG 候选路线上的 posterior mass。预测时，posterior 可用于聚合 route-neighbor mastery：

\[
\tilde{m}_{u,t}(c)=
(1-\gamma)m_{u,t}(c)+
\gamma\sum_{k\in S_A(c)}P_{u,t}(k|c)m_{u,t}(k).
\]

最终，support-aware mastery representation 被送入诊断预测头得到作答概率 \(\hat r_{u,e}\)。模型使用训练作答上的二分类交叉熵进行优化。

---

## 6. 实验设计

本文实验围绕机制证据链展开，而不是只给出单一预测排行榜。图 \ref{fig:data-crg-retrieval}--\ref{fig:lcrf-same-query} 将实验组织为一条从数据现象到个案机制的证据链：首先确认概念证据缺口是否存在，其次检验 CRG 是否能基于训练阶段路线检索 held-out transitions，然后评估预测是否依赖 CRG support，最后通过 learner-state counterfactual 和 same-query posterior case 检验 LCRF 的支持约束个性化行为。

### 6.1 Experimental Settings

实验使用三个核心数据集：`assist_09`、`junyi` 和 `assist_17`。它们分别覆盖混合概念场景、单概念且强 bridge-only 场景，以及较长学生历史下的 support-dependence 场景。所有数据集均使用既定 train/validation/test split；CRG 由训练阶段可观测证据估计，并在验证和测试阶段保持固定。评价指标包括 AUC、ACC 和 RMSE，其中 AUC 作为主要预测指标，ACC 与 RMSE 用于补充衡量分类准确性和概率误差。

本文报告两类结果。第一类是预测性能与模块消融，用于比较完整模型、移除 CRG 的变体和移除 LCRF 的变体。第二类是机制实验，包括 CRG route retrieval、support perturbation、learner-state counterfactual 和 same-query posterior case。

> **待补实验**：正式投稿前需在相同数据划分、指标和调参协议下补齐 IRT、MIRT、DINA、NCDM、KaNCD、RCD/SCD/ORCDF 等代表性 CD baseline 的统一对比。

### 6.2 Overall Prediction Performance（待补）

表 2 预留主性能对比位置，用于报告 CRG-LCRF 与代表性认知诊断模型的整体预测性能。当前草稿不填入未运行的 baseline 数值，避免引入不可复现实验结果。

**表 2：Overall prediction performance（待补 baseline）**

| Dataset | Metric | IRT | MIRT | DINA | NCDM | KaNCD | RCD/SCD/ORCDF | CRG-LCRF |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| assist_09 | AUC | TBD | TBD | TBD | TBD | TBD | TBD | 0.7783 |
| assist_09 | ACC | TBD | TBD | TBD | TBD | TBD | TBD | 0.7407 |
| assist_09 | RMSE | TBD | TBD | TBD | TBD | TBD | TBD | 0.4178 |
| junyi | AUC | TBD | TBD | TBD | TBD | TBD | TBD | 0.8291 |
| junyi | ACC | TBD | TBD | TBD | TBD | TBD | TBD | 0.7688 |
| junyi | RMSE | TBD | TBD | TBD | TBD | TBD | TBD | 0.4008 |
| assist_17 | AUC | TBD | TBD | TBD | TBD | TBD | TBD | 0.7847 |
| assist_17 | ACC | TBD | TBD | TBD | TBD | TBD | TBD | 0.7151 |
| assist_17 | RMSE | TBD | TBD | TBD | TBD | TBD | TBD | 0.4321 |

### 6.3 Ablation Study

表 3 展示三核心数据集上的模块消融结果。`full` 表示 CRG+LCRF 完整模型；`no_CRG` 移除全局路线图；`no_LCRF` 保留 CRG 但移除学习者条件化过滤。

**表 3：三核心数据集模块消融结果**

| 数据集 | 模型变体 | AUC | ACC | RMSE | 相对 Full 的 AUC 下降 |
| --- | --- | ---: | ---: | ---: | ---: |
| assist_09 | full | 0.7783 | 0.7407 | 0.4178 | 0.0000 |
| assist_09 | no_CRG | 0.7671 | 0.7320 | 0.4222 | 0.0112 |
| assist_09 | no_LCRF | 0.7634 | 0.7309 | 0.4247 | 0.0149 |
| junyi | full | 0.8291 | 0.7688 | 0.4008 | 0.0000 |
| junyi | no_CRG | 0.8278 | 0.7691 | 0.4011 | 0.0013 |
| junyi | no_LCRF | 0.8286 | 0.7701 | 0.3994 | 0.0005 |
| assist_17 | full | 0.7847 | 0.7151 | 0.4321 | 0.0000 |
| assist_17 | no_CRG | 0.7647 | 0.6973 | 0.4413 | 0.0200 |
| assist_17 | no_LCRF | 0.7829 | 0.7132 | 0.4359 | 0.0018 |

表 3 显示，完整模型在 `assist_09` 和 `assist_17` 上均取得最高 AUC，相比 `no_CRG` 分别提升 1.12% 和 2.00%。移除 LCRF 在 `assist_09` 上带来更明显下降，而在 `junyi` 与 `assist_17` 的直接模块删除设置下影响较小。该结果表明，CRG 提供主要的 reachable-support 信号；LCRF 的作用进一步由第 6.6 节的学习者状态反事实分析刻画。

### 6.4 CRG Route Retrieval

本节检验 CRG 是否具备 route sufficiency evidence。我们使用训练集构造 CRG，并在 held-out concept transition 上评估检索。对比方法包括 self-only、random/uniform、degree-random 和 best CRG。

**表 4：CRG held-out transition retrieval 结果**

| 数据集 | Self Hit@10 | Random/Uniform Hit@10 | Degree-random Hit@10 | Best CRG Hit@10 | Best CRG NDCG@10 | Best CRG MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| assist_09 | 0.1124 | 0.1321 | 0.1358 | 0.3673 | 0.1964 | 0.1699 |
| junyi | 0.0020 | 0.0219 | 0.0171 | 0.1648 | 0.0782 | 0.0717 |
| assist_17 | 0.1320 | 0.1544 | 0.1618 | 0.4113 | 0.2293 | 0.1987 |

如图 \ref{fig:data-crg-retrieval} 下半部分所示，Best CRG 在三个数据集上均强于 self-only、random 和 degree-random controls。`junyi` 的 item edge density 为 0，但 CRG 仍能取得明显高于随机基线的 Hit@10，说明 sequence route 可以在缺少题内共现时提供主要可达支持。

### 6.5 CRG Support Perturbation

Route retrieval 说明 CRG 能恢复 held-out transitions，但还不能说明预测是否使用这些 support。为此，我们在不重训模型的情况下扰动 CRG support，并比较 evidence corruption、degree-random、sequence-shuffled 和 self-only fallback 对预测的影响。

**表 5：100% support corruption 下的预测损伤（all subgroup）**

| 数据集 | 破坏类型 | AUC drop | BCE increase |
| --- | --- | ---: | ---: |
| assist_09 | evidence corruption | 0.0148 | 0.0086 |
| assist_09 | degree-random | 约 0.0146 | 约 0.0071 |
| assist_09 | sequence-shuffled | 约 0.0043 | 约 -0.0007 |
| junyi | evidence corruption | 0.0019 | 0.0095 |
| assist_17 | evidence corruption | 0.0111 | 0.0223 |
| assist_17 | degree-random | 约 0.0032 | 约 0.0178 |
| assist_17 | sequence-shuffled | 约 0.0001 | 约 0.0001 |

![CRG support perturbation under different dataset regimes. The curves report prediction changes when the CRG support is progressively corrupted. `assist_17` provides the clearest prediction-level evidence: corrupting evidence-supported CRG support causes larger degradation than degree-matched random support. `assist_09` mainly shows support-substrate dependence because evidence corruption and degree-random are close. `junyi` is CRG-retrieval-dominant and shows weaker prediction-level corruption effects.](figures_main_pdf/fig3_nature_crg_support_corruption.pdf){#fig:crg-support-corruption width=100%}

图 \ref{fig:crg-support-corruption} 表明 support dependence 具有数据集差异。`assist_17` 是最清晰的 prediction-level support-dependence 场景；`assist_09` 表明模型依赖候选 support 空间，但 evidence corruption 与 degree-random 接近，因此更适合解释为 support-substrate dependence；`junyi` 的主要作用是支持数据现象和 CRG retrieval，而不是 prediction-level corruption 主证据。

### 6.6 Learner-State Counterfactual

本节检验 LCRF 的 learner-state dependence evidence。我们使用三种反事实变体分析 LCRF 对学习者状态的使用方式：`no_filter` 表示移除 LCRF；`mean_state` 用群体平均学生状态替代真实状态；`shuffle_state` 打乱学生状态。

**表 6：LCRF counterfactual 结果**

| 数据集 | 变体 | AUC | RMSE | 相对 Full 的 AUC 下降 |
| --- | --- | ---: | ---: | ---: |
| assist_09 | no_filter | 0.7634 | 0.4247 | 0.0149 |
| assist_09 | mean_state | 0.6065 | 0.4827 | 0.1718 |
| assist_09 | shuffle_state | 0.5751 | 0.5070 | 0.2032 |
| junyi | no_filter | 0.8286 | 0.3994 | 0.0005 |
| junyi | mean_state | 0.8259 | 0.4017 | 0.0033 |
| junyi | shuffle_state | 0.8188 | 0.4049 | 0.0103 |
| assist_17 | no_filter | 0.7829 | 0.4359 | 0.0018 |
| assist_17 | mean_state | 0.6441 | 0.4865 | 0.1406 |
| assist_17 | shuffle_state | 0.5966 | 0.5196 | 0.1881 |

![Learner-state counterfactual analysis for LCRF. The figure compares no-filter, mean-state, and shuffle-state variants against the full model. `assist_09` and `assist_17` show large drops under mean/shuffle replacements, indicating that LCRF posterior weights depend on learner-conditioned state signals. `junyi` shows a weaker LCRF response and is treated as a CRG-dominant regime.](figures_main_pdf/fig4_nature_lcrf_counterfactual_delta.pdf){#fig:lcrf-counterfactual width=100%}

表 6 与图 \ref{fig:lcrf-counterfactual} 区分了模块移除和状态替换两类干预。`no_filter` 反映移除 LCRF 后的整体预测变化；`mean_state` 和 `shuffle_state` 进一步检验真实学习者状态是否可以由群体平均状态或错配状态替代。`assist_09` 和 `assist_17` 在 mean/shuffle 干预下出现明显下降，说明 LCRF 的 posterior route 依赖学习者条件化状态。相比之下，`junyi` 上 LCRF counterfactual 变化较小，因此本文将其解释为 CRG 主导的数据集 regime。

### 6.7 Same-Query Posterior Case

为了展示 case-level mechanism evidence，本文固定同一个 query concept 和相同 CRG support，比较不同学生得到的 posterior route 分布。该实验用于展示 LCRF 的 case-level personalization 行为，而不是替代全样本统计检验。`assist_17` 的主 case 为 `assist_17_Q14_S25`。该 case 中所有学生共享相同 support，support size 为 25；候选统计显示 mean pairwise L1 最高可达到约 0.801，JS 约 0.129。two-student case 中，S1 与 S7 的 posterior route 差异明显。

![Same-query posterior case under fixed CRG support. All students share the same query concept and the same CRG support, while LCRF assigns different posterior route weights according to learner state. The case illustrates support-constrained personalization: LCRF changes posterior mass within \(S_A(c)\) without introducing new support concepts.](figures_main_pdf/fig5_nature_lcrf_same_query_posterior.pdf){#fig:lcrf-same-query width=100%}

**表 7：assist_17 two-student same-query posterior case**

| 学生 | true label | query mastery | recent mastery | pred_global | pred_full | top support | posterior prob | global prob | posterior-global |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| S1 | 1 | -1.065 | -1.405 | 0.268 | 0.241 | C7 | 0.600 | 0.243 | 0.356 |
| S1 | 1 | -1.065 | -1.405 | 0.268 | 0.241 | C12 | 0.088 | 0.149 | -0.062 |
| S1 | 1 | -1.065 | -1.405 | 0.268 | 0.241 | C33 | 0.075 | 0.037 | 0.038 |
| S7 | 0 | -0.568 | -0.409 | 0.427 | 0.324 | C12 | 0.348 | 0.149 | 0.199 |
| S7 | 0 | -0.568 | -0.409 | 0.427 | 0.324 | C7 | 0.287 | 0.243 | 0.044 |
| S7 | 0 | -0.568 | -0.409 | 0.427 | 0.324 | C4 | 0.169 | 0.096 | 0.073 |

图 \ref{fig:lcrf-same-query} 和表 7 展示了 LCRF 的个性化过滤机制：在同一 query、同一 CRG support 下，不同学生会得到不同 posterior route。S1 的 posterior mass 更集中于 C7，而 S7 更强调 C12；这说明 LCRF 并非为所有学生复用同一全局路线分布，而是在固定 support 内根据学习者状态调整路线权重。

---

## 7. 讨论

**Route evidence matters.** 当目标概念缺少直接历史证据时，模型需要从训练阶段日志中寻找可审计概念路线。实验表明，CRG 在三个数据集上均优于 self-only 与 random baselines，说明题内共现、序列转移和自保持证据可以共同形成有效的概念路线图。尤其在 `junyi` 中，题内共现完全缺失，sequence route 仍然能提供可达支持。

**Support-constrained personalization matters.** CRG 给出的 support 只回答“哪些概念可作为候选支持”，而 LCRF 进一步回答“哪些支持更适合当前学生”。Learner-state counterfactual 显示，`assist_09` 和 `assist_17` 中真实学生状态被均值化或错配替换后预测显著下降；same-query case 进一步展示了同一 support 在不同学生上的 posterior shift。

**Evidence regimes differ across datasets.** 三组数据集呈现不同 evidence regime。`junyi` 主要体现无题内共现条件下的 sequence-route 可达性；`assist_09` 同时体现模块消融与 support-substrate dependence；`assist_17` 在 support perturbation 与 same-query case 中给出更清晰的 route-dependence 与 personalization 现象。上述差异并不削弱方法主线，而是说明 CRG-LCRF 可以把数据现象拆解为 route construction、support dependence 与 learner-conditioned filtering 三个层次。

---

## 8. 结论

本文提出面向概念证据缺口的可审计概念可达性认知诊断框架。CRG 从训练阶段题内共现、序列转移和自保持证据中构造全局概念路线图，用于连接学生历史概念和当前目标概念；LCRF 在 CRG 固定 support 内，根据学生状态重排 posterior route，实现局部个性化过滤。

三核心数据集上的实验表明，CRG 能有效检索 held-out concept transitions，support perturbation 会影响预测表现；LCRF 在 `assist_09` 和 `assist_17` 上通过 mean/shuffle counterfactual 展示真实学生状态的重要性，并通过 same-query case 展示同一 support 被不同学生过滤成不同 posterior。该框架为认知诊断中的可审计概念路线建模和个性化路线过滤提供了一种有效方案。未来工作将进一步扩展 CRG 的高阶概念路线建模，并研究更细粒度的学习者状态来源分解。

---

## 参考文献占位

> 注：正式投稿时需要统一 BibTeX。以下只列本草稿中应引用的文献类型。

1. IRT、MIRT、DINA 等经典认知诊断模型。
2. Neural Cognitive Diagnosis / NCDM 原始论文。
3. RCD、SCD、KaNCD、ORCDF 等图式或神经认知诊断代表方法。
4. KCD：冷启动场景和 LLM prior alignment。
5. DBCD：MNAR / counterfactual debiasing。
6. NCDLA：噪声鲁棒认知诊断与谱结构分析。
7. ISG-CD：图边异质性与不确定性。
8. DFCD / LRCD：open / zero-shot 认知诊断场景。
