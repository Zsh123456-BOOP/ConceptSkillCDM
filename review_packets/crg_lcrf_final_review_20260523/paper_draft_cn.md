# 面向概念证据缺口的可审计概念可达性认知诊断

## 摘要

认知诊断旨在根据学生历史作答记录估计其对知识概念的掌握状态。现有神经认知诊断与图认知诊断模型通常依赖学生-题目响应日志和题目-概念矩阵进行预测，但在真实学习平台中，学生历史并不总是直接覆盖当前测试题涉及的目标概念。当目标概念缺少直接历史证据时，模型很难回答一个关键问题：**当前诊断是否有可审计的概念依据？** 本文将这一现象定义为**概念证据缺口**。为缓解该问题，本文提出一种证据约束的概念可达性认知诊断框架，包括概念可达图（Concept Reachability Graph, CRG）和学习者条件化可达性过滤器（Learner-Conditioned Reachability Filter, LCRF）。CRG 仅使用训练集中的题内概念共现、序列转移和自保持证据构造全局概念路线图；LCRF 不新增图边，而是在 CRG 给出的固定 support 内，根据学生掌握度、近期表现和历史计数重排后验路线权重。三组核心数据集 `assist_09`、`junyi` 和 `assist_17` 上的机制实验表明，CRG 在 held-out concept transition retrieval 上显著强于 self-only 和 random baselines；support corruption 进一步显示 `assist_17` 上模型明显依赖 evidence-supported CRG support，而 `assist_09` 主要体现 support-dependence，`junyi` 主要作为 CRG 数据现象与检索充分性证据。LCRF 的反事实实验显示，在 `assist_09` 和 `assist_17` 上，真实学生状态不能被打乱状态或群体平均状态替代；same-query case 进一步展示了同一 CRG support 会被不同学生过滤成不同 posterior route。本文强调，本文给出的是 CRG/LCRF 的**操作性机制证据**，而不是数学意义上的充分必要性证明。

**关键词**：认知诊断；概念证据缺口；概念可达图；训练集证据；学习者条件化过滤；机制可解释性

---

## 1. 引言

认知诊断（Cognitive Diagnosis, CD）是智能教育系统中的基础任务，其目标是根据学生历史作答记录推断学生对知识概念的掌握状态。诊断结果可以服务于个性化练习推荐、补救学习、学习路径规划和自适应测试。近年来，神经认知诊断和图认知诊断模型通过更复杂的学生-题目交互函数、概念图传播或学生-题目-概念异构图建模提升了预测性能。然而，这些方法通常默认学生历史记录能够为当前目标概念提供足够的直接证据。

在真实学习平台中，这一默认条件并不总是成立。学生不会完整练习所有知识点，当前测试题涉及的目标概念可能从未在该学生历史中出现，或者只在极少量历史记录中出现。此时，模型如果仅依赖学生已经作答过的概念维度，就会面临一个直接的诊断依据缺口：当前目标概念虽然需要被预测，但学生历史中没有足够直接证据。本文将这一现象称为**概念证据缺口**。

概念证据缺口并不等同于简单的数据稀疏。对于认知诊断而言，关键不只是学生历史长度是否短，而是当前目标概念是否能从学生历史概念中获得可审计连接。即使一道题只对应一个知识点，训练集日志中仍可能存在从历史概念到目标概念的经验性学习路线，例如不同概念在学生训练序列中的转移关系。相反，如果只依赖题内多概念共现，那么在 `junyi` 这类 100% 单概念题数据集中，概念图将退化为近似空图。因此，本文不把“题内概念共现”作为唯一关系来源，而是把它作为训练集经验路线的一种可用证据。

为解决这一问题，本文提出 CRG/LCRF 两阶段机制。CRG 回答“当前目标概念是否能从学生历史概念通过训练集路线被到达”；LCRF 回答“同一条可达路线是否适合当前学生”。二者不是并列堆叠的黑盒模块，而是全局路线发现与局部个性化过滤的递进关系。CRG 的 support 是可审计的、train-only 的；LCRF 的 posterior 只在该 support 内重排，不生成新图。

![概念证据缺口与支持约束个性化。当学生历史只覆盖少量历史概念 \(h_1,h_2,h_3\)，而当前目标概念 \(c\) 未被直接观测时，模型面临 direct evidence gap。CRG 使用训练集中的题内共现、序列转移、自保持和可靠性统计构造可审计的可达 support；LCRF 不新增 support，而是在相同 support 内根据学生状态重排 posterior weight。](figures/fig1_concept_gap.png){#fig:concept-gap width=100%}

如图 \ref{fig:concept-gap} 所示，本文关注的并不是一般意义上的短历史问题，而是目标概念与学生历史概念之间缺少直接可审计连接的情况。左侧的学生历史只包含已观测过的概念及有限正确、错误、未观测记录，目标概念 \(c\) 虽然需要被预测，却没有出现在该学生的历史概念集合中。中间的 CRG 利用训练集中的经验路线把历史概念连接到目标概念，其中 sequence transition 只表示训练日志中的经验学习路线，而不是严格先修关系。右侧的 LCRF 进一步在 CRG 给出的固定 support 内进行个性化后验重排，因此它改变的是 support 内部的 posterior mass，而不是引入新的概念节点或新边。

本文贡献如下：

1. 提出“概念证据缺口”视角，将弱直接覆盖场景下的认知诊断转化为训练集概念路线的检索与过滤问题。
2. 设计 CRG，只使用训练集题内共现、序列转移和自保持证据构造可审计全局概念路线图。
3. 设计 LCRF，在不新增 support 的前提下，根据学生 mastery、recent mastery 和历史计数对同一 CRG support 进行后验过滤。
4. 设计一组机制实验链：retrieval 验证 CRG 找路能力，support corruption 验证模型是否依赖 CRG support，state counterfactual 验证 LCRF 状态必要性，same-query case 验证同一 support 的个性化过滤。

---

## 2. 相关工作

### 2.1 认知诊断模型

传统认知诊断模型通常建立在心理测量理论之上，例如 IRT、MIRT 和 DINA。它们通过预设的交互函数描述学生能力、题目难度和作答结果之间的关系，具有较强解释性，但表达能力有限。神经认知诊断模型进一步使用神经网络建模复杂的学生-题目交互，使模型能够从大规模作答数据中学习非线性诊断函数。典型方法通常以学生嵌入、题目嵌入和概念嵌入作为输入，通过预测学生在题目上的作答结果来优化模型。

尽管这些方法提升了预测性能，但它们通常没有显式区分“目标概念是否有直接历史证据”和“目标概念是否能通过训练集概念路线被连接”。当学生历史中缺少某个目标概念时，模型可能只能依赖 ID embedding 或全局统计模式进行预测，难以给出可审计的概念层解释。

### 2.2 图认知诊断与概念关系建模

图认知诊断模型通过学生-题目图、题目-概念图或异构图传播建模高阶关系。相关工作表明，图结构可以提升认知诊断模型对学生和题目的表示能力。但图结构也引入两个问题：第一，图边的来源是否可审计；第二，图边对不同学生是否同等可信。近期 KDD 图认知诊断论文强调了图边异质性和不确定性，说明不是所有图边都应该被等价传播。本文与此类工作相似，都关注图边的语义和可靠性，但本文的关注点不同：我们不是从学生-题目边的不确定性出发，而是从目标概念缺少直接证据时的概念 support 来源出发。

### 2.3 稀疏、冷启动与证据缺口

近两年认知诊断论文常从具体失败模式切入。例如，冷启动论文强调新学生或新题目缺少先验；去偏论文强调作答记录存在非随机缺失；鲁棒图模型强调响应日志噪声；学生-概念稀疏论文强调学生只练习少量概念，导致部分概念维度无法被充分训练。这些工作共同说明，顶会论文的写法通常不是简单提出一个新模块，而是先明确现有 CDM 在什么现实场景下失效，再设计与该失败模式对应的机制实验。

本文的失败模式是：**当前目标概念缺少学生历史中的直接诊断证据**。与冷启动不同，本文不要求学生或题目完全未见；与一般数据稀疏不同，本文关注的是目标概念维度是否有可审计连接。CRG 和 LCRF 分别从全局路线图与个性化过滤两个层次缓解这一问题。

---

## 3. 问题定义：概念证据缺口

设学生集合为 \(\mathcal{U}\)，题目集合为 \(\mathcal{E}\)，概念集合为 \(\mathcal{C}\)。题目-概念矩阵为：

\[
Q\in\{0,1\}^{|\mathcal{E}|\times |\mathcal{C}|},
\]

其中 \(Q_{e,c}=1\) 表示题目 \(e\) 涉及概念 \(c\)。学生作答日志为：

\[
\mathcal{R}=\{(u,e_t,r_{u,t})\mid u\in\mathcal{U}, e_t\in\mathcal{E}, r_{u,t}\in\{0,1\}\}.
\]

学生 \(u\) 在时间 \(t\) 前的历史概念集合定义为：

\[
H_{u,t}=\bigcup_{\tau<t} \{c\in\mathcal{C}: Q_{e_{u,\tau},c}=1\}.
\]

对当前题目 \(e_t\) 的目标概念 \(c\)，定义直接历史证据指示：

\[
D_{u,t}(c)=\mathbb{I}[c\in H_{u,t}].
\]

当 \(D_{u,t}(c)=0\) 时，学生历史中没有目标概念 \(c\) 的直接证据。此时，如果训练集构造的概念路线图 \(A_{\mathrm{CRG}}\) 能够将历史概念连接到目标概念，则该目标概念仍具有可达支持：

\[
R_{u,t}(c)=\max_{h\in H_{u,t}} A_{\mathrm{CRG}}(h,c).
\]

据此，可以把样本分为三类：

\[
\begin{cases}
D_{u,t}(c)=1, & \text{直接覆盖；}\\
D_{u,t}(c)=0, R_{u,t}(c)>0, & \text{无直接证据但可桥接；}\\
D_{u,t}(c)=0, R_{u,t}(c)=0, & \text{无直接证据且不可达。}
\end{cases}
\]

本文关注第二类样本：目标概念缺少直接历史证据，但可以通过训练集路线从历史概念被连接。注意，\(A_{\mathrm{CRG}}\) 只由训练集构造，不能使用验证集或测试集中的概念转移。

---

## 4. 数据现象

表 1 展示三个核心数据集的概念可达性画像。它们并不是同一种数据分布，而是分别支撑论文主线的不同侧面。

**表 1：三核心数据集的概念可达性画像**

| 数据集    | 单概念比例 | 多概念比例 | item edge density | sequence edge density | direct-unseen rate | bridge-only rate | 历史长度中位数 | 论文角色                                                  |
| --------- | ---------: | ---------: | ----------------: | --------------------: | -----------------: | ---------------: | -------------: | --------------------------------------------------------- |
| assist_09 |      82.8% |      17.2% |              0.9% |                 64.3% |               3.1% |             3.1% |             41 | 平衡主例，CRG/LCRF 都可分析                               |
| junyi     |     100.0% |       0.0% |              0.0% |                 25.2% |             100.0% |           99.97% |             25 | 最强 CRG 数据现象，无题内共现但可通过 sequence route 桥接 |
| assist_17 |      78.3% |      21.7% |              6.2% |                 76.3% |               2.8% |             2.8% |            148 | CRG support corruption 和 LCRF case 最干净                |

从表 1 可以看出，本文不应写成“所有数据集都依赖多知识点题目共现”。相反，核心现象是：题内共现证据稀疏或缺失，但训练集中的 sequence route 仍然能够提供概念可达性。`junyi` 是最极端的例子，它没有题内概念共现，仍可以作为 CRG 检索充分性的关键证据。`assist_09` 和 `assist_17` 则更接近常规 benchmark 场景，少量多概念题与较强序列路线共同存在。

---

## 5. 方法

### 5.1 框架总览

本文框架包括两个递进组件：CRG 和 LCRF。CRG 是主模块，用于构造全局、可审计、训练集约束的概念路线图；LCRF 是副模块，用于在同一 CRG support 内根据学生状态重排 posterior route。整体流程为：

\[
\text{train-only evidence}\rightarrow A_{\mathrm{CRG}}\rightarrow S_c\rightarrow P_{u,t}(k|c)\rightarrow \hat{r}_{u,e}.
\]

其中 \(S_c\) 是概念 \(c\) 的固定 CRG support，\(P_{u,t}(k|c)\) 是 LCRF 产生的学生条件化后验路线分布。

![CRG/LCRF 机制架构。CRG 从 train-only evidence 构造全局可审计路线图 \(A\)，并为目标概念定义固定 support \(S_A(c)\)。LCRF 只接收学习者状态信号，并在 CRG 固定 support 内生成 support-constrained posterior。最终预测基于个性化 posterior 与诊断预测头得到 \(p(\mathrm{correct})\)。](figures/fig2_crg_lcrf_architecture.png){#fig:model-architecture width=100%}

图 \ref{fig:model-architecture} 展示了 CRG/LCRF 的整体机制路径。首先，CRG 只从训练集证据中构造概念路线图，证据包括题内共现、序列转移、自保持以及 source/receiver reliability；这些证据共同形成一个全局、可审计、行随机化的概念关系图 \(A_{\mathrm{CRG}}\)。随后，CRG 为当前 query concept 给出固定 support \(S_A(c)\)，该 support 决定了后续可被使用的候选概念范围。LCRF 不参与 support 构造，也不扩展 support，而是仅根据 query mastery、recent mastery、route-neighbor mastery、readiness gap 和 support count 等学习者状态信号，对同一 support 内的 posterior weight 进行个性化重排。该设计保证了 CRG 负责“给出可审计路线”，LCRF 负责“判断同一路线是否适合当前学生”。

### 5.2 概念可达图 CRG

CRG 只使用训练集中的可观察证据。对概念对 \((c,k)\)，定义三类证据：

**题内共现证据**：

\[
M^{\mathrm{item}}_{c,k}=\sum_{e\in\mathcal{E}^{tr}} Q_{e,c}Q_{e,k}.
\]

它表示同一题内两个概念共同出现的次数。若数据集中每道题只对应一个概念，则该项自然为 0。

**序列转移证据**：

\[
M^{\mathrm{seq}}_{c,k}=\sum_{u}\sum_t \mathbb{I}[c\in C(e_{u,t})]\mathbb{I}[k\in C(e_{u,t+1})].
\]

它表示训练集中概念 \(c\) 后接概念 \(k\) 的经验学习路线。本文不将其解释为严格先修关系，也不声称存在因果依赖。

**自保持证据**：

\[
M^{\mathrm{self}}_{c,k}=\mathbb{I}[c=k].
\]

三类证据归一化后进行融合：

\[
s(c,k)=\lambda_{item}z_{item}(c,k)+\lambda_{seq}z_{seq}(c,k)+\lambda_{self}\mathbb{I}[c=k]+b_k.
\]

最终得到行随机化的 CRG：

\[
A_{\mathrm{CRG}}(c,k)=\operatorname{softmax}_{k\in S_c}\left(\frac{s(c,k)}{\tau}\right),
\]

其中 \(S_c\) 是概念 \(c\) 的候选 support，\(\tau\) 是温度参数。所有 \(M^{item}\)、\(M^{seq}\) 和 \(M^{self}\) 均仅从训练集构造。

### 5.3 学习者条件化可达性过滤器 LCRF

CRG 给出全局 support，但同一条路线对不同学生不一定同等可信。LCRF 在 CRG support 内计算学生条件化后验：

\[
P_{u,t}(k|c)=\operatorname{softmax}_{k\in S_c}\left(\log A_{\mathrm{CRG}}(c,k)+\alpha_{u,t,c}\Delta_{u,t,c,k}\right).
\]

其中 \(\Delta_{u,t,c,k}\) 由学生状态构成：

\[
\Delta_{u,t,c,k}=f_{\theta}\left[m_{u,t}(c),\rho_{u,t}(c),n_{u,t}(c),m_{u,t}(k),\rho_{u,t}(k),n_{u,t}(k)\right].
\]

这里 \(m_{u,t}(c)\) 表示学生在 query concept 上的历史掌握估计，\(\rho_{u,t}(c)\) 表示近期掌握状态，\(n_{u,t}(c)\) 表示历史计数或可靠性。同理，\(k\) 对应 CRG support concept。

LCRF 的关键约束是：

\[
\operatorname{supp}(P_{u,t}(\cdot|c))\subseteq S_c.
\]

也就是说，LCRF 不新增 support concept，只在同一 support 内改变 posterior mass。预测时可将局部概念状态写成：

\[
\tilde{m}_{u,t}(c)=(1-\gamma)m_{u,t}(c)+\gamma\sum_{k\in S_c}P_{u,t}(k|c)m_{u,t}(k).
\]

最终预测头基于 \(\tilde{m}_{u,t}(c)\)、题目参数和概念集合输出学生对题目 \(e\) 的作答概率：

\[
\hat{r}_{u,e}=g_{\phi}\left(\{\tilde{m}_{u,t}(c):Q_{e,c}=1\},\;\text{item features}\right).
\]

训练目标采用标准二分类交叉熵：

\[
\mathcal{L}_{\mathrm{BCE}}=-\sum_{(u,e,r)\in\mathcal{R}^{tr}}\left[r\log \hat{r}_{u,e}+(1-r)\log(1-\hat{r}_{u,e})\right].
\]

需要强调的是，当前证据包不能完全排除 student-ID shortcut 的影响。因此论文中不应写“LCRF 完全不读学生 ID 捷径”，而应写为：LCRF 的路线重排被约束在 CRG support 内，学生状态来源审计作为限制分析报告。

---

## 6. 实验设计

本文实验围绕五个问题展开：

- **RQ1**：核心数据集是否存在概念证据缺口与可达路线现象？
- **RQ2**：CRG 是否能够作为 train-only roadmap 检索 held-out concept transition？
- **RQ3**：模型预测是否依赖 CRG support？
- **RQ4**：LCRF 中真实学生状态是否不可被均值化或打乱状态替代？
- **RQ5**：同一 CRG support 是否会被不同学生过滤成不同 posterior route？

![机制证据链。实验不把所有数据集都解释为同等充分或必要，而是按机制问题递进组织：先验证概念证据缺口是否存在，再验证 CRG 是否能用 train-only routes 检索 held-out transition，随后测试预测是否依赖 CRG support，最后通过 learner-state counterfactual 和 same-query posterior case 检验 LCRF 的个性化过滤作用。](figures/fig3_mechanism_evidence_chain.png){#fig:evidence-chain width=100%}

如图 \ref{fig:evidence-chain} 所示，本文的实验不是单一性能表驱动，而是按照机制证据链组织。第一步通过 dataset cards 说明核心数据集中确实存在 sparse history 与 target concept unseen 的现象；第二步通过 held-out transition retrieval 验证 CRG 是否能仅依赖训练集路线检索未来概念；第三步通过 support corruption 检查预测是否依赖 CRG 给出的 reachable support；第四步通过 learner-state counterfactual 检验真实学生状态是否可以被群体平均或错配状态替代；第五步通过 same-query posterior case 展示同一 query、同一 CRG support 下，不同学生会获得不同 posterior route。因此，图 \ref{fig:evidence-chain} 的作用是限定本文的实验解释边界：CRG 是主模块，LCRF 是固定 support 内的个性化过滤器，本文给出的是操作性机制证据，而不是所有数据集上的充分必要性证明。

### 6.1 主预测性能与模块消融

表 2 展示三核心数据集上的主消融结果。`full` 表示 CRG+LCRF 完整模型；`no_CRG` 移除全局路线图；`no_LCRF` 保留 CRG 但移除学习者条件化过滤。

**表 2：三核心数据集主消融结果**

| 数据集    | 模型变体 |    AUC |    ACC |   RMSE | 相对 Full 的 AUC 下降 | 解释                           |
| --------- | -------- | -----: | -----: | -----: | --------------------: | ------------------------------ |
| assist_09 | full     | 0.7783 | 0.7407 | 0.4178 |                0.0000 | 完整模型                       |
| assist_09 | no_CRG   | 0.7671 | 0.7320 | 0.4222 |                0.0112 | 移除路线图                     |
| assist_09 | no_LCRF  | 0.7634 | 0.7309 | 0.4247 |                0.0149 | 移除个性化过滤                 |
| junyi     | full     | 0.8291 | 0.7688 | 0.4008 |                0.0000 | 完整模型                       |
| junyi     | no_CRG   | 0.8278 | 0.7691 | 0.4011 |                0.0013 | 全局消融弱                     |
| junyi     | no_LCRF  | 0.8286 | 0.7701 | 0.3994 |                0.0005 | LCRF 弱                        |
| assist_17 | full     | 0.7847 | 0.7151 | 0.4321 |                0.0000 | 完整模型                       |
| assist_17 | no_CRG   | 0.7647 | 0.6973 | 0.4413 |                0.0200 | CRG 消融较强                   |
| assist_17 | no_LCRF  | 0.7829 | 0.7132 | 0.4359 |                0.0018 | no-filter 较弱，但状态反事实强 |

表 2 的作用是证明模型在三核心数据集上可用，并展示模块删除后的总体影响。它不能被解释为“两个模块在所有数据集上同等重要”。更准确的解释是：CRG 是主路线图贡献，`assist_17` 与 `assist_09` 均有明显信号；`junyi` 主要用于证明 CRG 的数据现象和检索充分性；LCRF 的强证据来自后续状态反事实和 same-query case。

### 6.2 CRG 充分性：held-out transition retrieval

为了验证 CRG 是否具备 train-only 找路能力，本文使用训练集构造 CRG，并在 held-out concept transition 上评估检索。对比方法包括 self-only、random/uniform、degree-random 和 best CRG。

**表 3：CRG held-out transition retrieval 结果**

| 数据集    | Self Hit@10 | Random/Uniform Hit@10 | Degree-random Hit@10 | Best CRG Hit@10 | Best CRG NDCG@10 | Best CRG MRR |
| --------- | ----------: | --------------------: | -------------------: | --------------: | ---------------: | -----------: |
| assist_09 |      0.1124 |                0.1321 |               0.1358 |          0.3673 |           0.1964 |       0.1699 |
| junyi     |      0.0020 |                0.0219 |               0.0171 |          0.1648 |           0.0782 |       0.0717 |
| assist_17 |      0.1320 |                0.1544 |               0.1618 |          0.4113 |           0.2293 |       0.1987 |

结果显示，Best CRG 在三个数据集上均显著强于 self-only 和 random baselines。`junyi` 尤其重要，因为其 item edge density 为 0，说明 CRG 的检索能力并不依赖题内多概念共现；sequence route 可以成为主要路线来源。

### 6.3 CRG 支持依赖：support corruption

Retrieval 只能证明 CRG 能找路，不能证明预测时模型依赖 CRG support。因此，本文进一步进行 support corruption：在不重训的情况下，替换或破坏 CRG support，并观察预测性能变化。对照包括 evidence support corruption、degree-matched random support、sequence-shuffled support 和 self-only fallback。

**表 4：100% support corruption 下的预测损伤（all subgroup）**

| 数据集    | 破坏类型            |  AUC drop | BCE increase | 解释                                        |
| --------- | ------------------- | --------: | -----------: | ------------------------------------------- |
| assist_09 | evidence corruption |    0.0148 |       0.0086 | support 被破坏后预测下降                    |
| assist_09 | degree-random       | 约 0.0146 |    约 0.0071 | 与 evidence 接近，只能写 support-dependence |
| assist_09 | sequence-shuffled   | 约 0.0043 |   约 -0.0007 | 效应较弱                                    |
| junyi     | evidence corruption |    0.0019 |       0.0095 | AUC 弱，BCE 有变化，谨慎报告                |
| assist_17 | evidence corruption |    0.0111 |       0.0223 | 最干净的 CRG necessity 证据                 |
| assist_17 | degree-random       | 约 0.0032 |    约 0.0178 | evidence 的 AUC 损伤明显更强                |
| assist_17 | sequence-shuffled   | 约 0.0001 |    约 0.0001 | 近似无损伤                                  |

表 4 表明，CRG 的 prediction-level necessity 是数据集依赖的。`assist_17` 是最强证据：evidence corruption 明显强于 degree-random，尤其 AUC drop gap 明显。`assist_09` 中 evidence corruption 与 degree-random 非常接近，因此只能说明模型依赖 support substrate，不能写成 evidence edge 独占有效。`junyi` 的 prediction-level corruption 较弱，不作为 CRG 必要性主证据。

### 6.4 LCRF 必要性：学生状态反事实

LCRF 的必要性通过三种反事实变体验证：`no_filter` 表示移除 LCRF；`mean_state` 用群体平均学生状态替代真实状态；`shuffle_state` 打乱学生状态。

**表 5：LCRF counterfactual 结果**

| 数据集    | 变体          |    AUC |   RMSE | 相对 Full 的 AUC 下降 | 解释                 |
| --------- | ------------- | -----: | -----: | --------------------: | -------------------- |
| assist_09 | no_filter     | 0.7634 | 0.4247 |                0.0149 | 移除 LCRF 有损伤     |
| assist_09 | mean_state    | 0.6065 | 0.4827 |                0.1718 | 平均状态不可替代     |
| assist_09 | shuffle_state | 0.5751 | 0.5070 |                0.2032 | 错配状态严重伤害预测 |
| junyi     | no_filter     | 0.8286 | 0.3994 |                0.0005 | LCRF 弱              |
| junyi     | mean_state    | 0.8259 | 0.4017 |                0.0033 | 弱                   |
| junyi     | shuffle_state | 0.8188 | 0.4049 |                0.0103 | 弱                   |
| assist_17 | no_filter     | 0.7829 | 0.4359 |                0.0018 | 去掉过滤器整体损伤小 |
| assist_17 | mean_state    | 0.6441 | 0.4865 |                0.1406 | 平均状态不可替代     |
| assist_17 | shuffle_state | 0.5966 | 0.5196 |                0.1881 | 错配状态严重伤害预测 |

表 5 需要分开解释。`no_filter` 反映模块移除后的整体增益；`mean_state` 和 `shuffle_state` 反映真实学习者状态是否不可替代。`assist_09` 和 `assist_17` 的 mean/shuffle 损伤很大，说明 LCRF 不是固定全局补丁；`junyi` 效应较弱，因此不作为 LCRF 主证据。

### 6.5 LCRF 充分性：same-query posterior case

为了验证 LCRF 是否真的在同一 CRG support 内做个性化过滤，本文固定同一个 query concept 和相同 CRG support，比较不同学生得到的 posterior route 分布。

`assist_17` 的主 case 为 `assist_17_Q14_S25`。该 case 中所有学生共享相同 support，support size 为 25；候选统计显示 mean pairwise L1 最高可达到约 0.801，JS 约 0.129。two-student case 中，S1 与 S7 的 posterior route 差异明显。

**表 6：assist_17 two-student same-query posterior case**

| 学生 | true label | query mastery | recent mastery | pred_global | pred_full | top support | posterior prob | global prob | posterior-global |
| ---- | ---------: | ------------: | -------------: | ----------: | --------: | ----------- | -------------: | ----------: | ---------------: |
| S1   |          1 |        -1.065 |         -1.405 |       0.268 |     0.241 | C7          |          0.600 |       0.243 |            0.356 |
| S1   |          1 |        -1.065 |         -1.405 |       0.268 |     0.241 | C12         |          0.088 |       0.149 |           -0.062 |
| S1   |          1 |        -1.065 |         -1.405 |       0.268 |     0.241 | C33         |          0.075 |       0.037 |            0.038 |
| S7   |          0 |        -0.568 |         -0.409 |       0.427 |     0.324 | C12         |          0.348 |       0.149 |            0.199 |
| S7   |          0 |        -0.568 |         -0.409 |       0.427 |     0.324 | C7          |          0.287 |       0.243 |            0.044 |
| S7   |          0 |        -0.568 |         -0.409 |       0.427 |     0.324 | C4          |          0.169 |       0.096 |            0.073 |

该 case 支持 LCRF 的机制解释：同一 query、同一 CRG support 下，不同学生会得到不同 posterior route。它不能被写成全体样本上的统计性证明，而应写成 mechanism case。更稳的说法是：Figure 5 展示了 LCRF 如何把同一全局 support 过滤成学生局部 posterior route。

### 6.6 学生状态来源审计与限制

表 7 展示 LCRF state-source audit。已有结果支持 `mean_state_keep_id` 和 `shuffle_state_keep_id` 会显著伤害 `assist_09` 和 `assist_17`，说明学习者状态确实重要。但当前 inference hook 未支持 `shuffle_id_keep_state` 和 `zero_id_keep_state`，因此不能声称 student-ID shortcut 已被完全排除。

**表 7：LCRF state-source audit**

| 数据集    | 变体                  |    AUC | AUC drop | 解释             |
| --------- | --------------------- | -----: | -------: | ---------------- |
| assist_09 | full                  | 0.7783 |   0.0000 | baseline         |
| assist_09 | shuffle_state_keep_id | 0.5751 |   0.2032 | 打乱状态严重损伤 |
| assist_09 | mean_state_keep_id    | 0.6065 |   0.1718 | 平均状态严重损伤 |
| assist_09 | global_only           | 0.7634 |   0.0149 | 等价 no-filter   |
| assist_17 | full                  | 0.7847 |   0.0000 | baseline         |
| assist_17 | shuffle_state_keep_id | 0.5966 |   0.1881 | 打乱状态严重损伤 |
| assist_17 | mean_state_keep_id    | 0.6441 |   0.1406 | 平均状态严重损伤 |
| assist_17 | global_only           | 0.7829 |   0.0018 | 等价 no-filter   |
| junyi     | full                  | 0.8291 |   0.0000 | baseline         |
| junyi     | shuffle_state_keep_id | 0.8188 |   0.0103 | 弱               |
| junyi     | mean_state_keep_id    | 0.8259 |   0.0033 | 弱               |

论文中应将此结果放入附录或限制分析，而不是主文强证据。推荐写法：当前实验支持真实学生状态在 `assist_09` 和 `assist_17` 中不可被均值或打乱状态替代，但由于缺少 ID ablation hook，本文不能完全排除 student-ID shortcut 的影响。

---

## 7. 讨论

本文的关键不是提出又一个图增强认知诊断模型，而是把概念诊断中的证据来源拆解为两个层次。第一层是全局可审计路线图：在学生历史缺少目标概念直接证据时，模型需要知道哪些历史概念可以通过训练集路线支持当前概念。第二层是局部个性化过滤：同一条路线对不同学生不一定同样可信，需要由学生状态决定 posterior route mass。

从结果看，CRG 的 retrieval sufficiency 是最稳定的证据。三核心数据集上，CRG 均显著强于 random/self。CRG 的 prediction-level support dependence 则具有数据集差异，`assist_17` 最强，`assist_09` 只能写 support-dependence，`junyi` 较弱。LCRF 的必要性主要体现在 `assist_09` 和 `assist_17` 的 mean/shuffle counterfactual 中。same-query case 则提供可解释机制个案，展示同一 support 如何被不同学生过滤。

这组结论与模块定位一致：CRG 是主贡献，解决全局概念证据缺口；LCRF 是副贡献，解决同一 support 下的学生个性化可信度判断。

---

## 8. 局限性

第一，CRG support corruption 没有在所有数据集上证明 evidence edge 都优于 degree-random support。`assist_17` 提供最清晰 evidence-control gap，但 `assist_09` 只能证明 support-dependence，`junyi` 在 prediction-level corruption 上较弱。

第二，`junyi` 虽然是最强 CRG 数据现象数据集，但不能用来强证 LCRF。它更适合作为“无 item co-occurrence 但存在 sequence route”的 CRG 充分性证据。

第三，当前 state-source audit 不能完全排除 student-ID shortcut。虽然 LCRF 被限制在固定 CRG support 内，但现有 inference hook 未完整支持 shuffle-id 或 zero-id 反事实，因此本文不能声称完全排除 student-ID shortcut。

第四，当前证据包主要是机制证据与模块消融。若作为完整投稿版本，还需要补充标准 baseline 主表，说明 full model 与 IRT、MIRT、DINA、NCD、KaNCD、RCD 等代表方法在相同 split 下的预测性能差异。

---

## 9. 结论

本文提出面向概念证据缺口的可审计概念可达性认知诊断框架。CRG 从训练集题内共现、序列转移和自保持证据中构造全局概念路线图，用于连接学生历史概念和当前目标概念；LCRF 在 CRG 固定 support 内，根据学生状态重排 posterior route，实现局部个性化过滤。

三核心数据集上的实验表明，CRG 能有效检索 held-out concept transition，并在 `assist_17` 上表现出较强 prediction-level support dependence；LCRF 在 `assist_09` 和 `assist_17` 上通过 mean/shuffle counterfactual 证明真实学生状态不可替代，并通过 same-query case 展示同一 support 被不同学生过滤成不同 posterior。总体而言，本文提供的是 CRG/LCRF 的操作性机制证据，而非数学意义上的充分必要性证明。该框架为认知诊断中的可审计概念路线建模和个性化路线过滤提供了一种可解释方案。

---

## 参考文献占位

> 注：正式投稿时需要统一 BibTeX。以下只列本草稿中应引用的文献类型。

1. Neural Cognitive Diagnosis / NCDM 原始论文。
2. RCD、SCD、KaNCD 等图式或神经认知诊断代表方法。
3. KCD：冷启动场景和 LLM prior alignment 写法参考。
4. DBCD：MNAR / counterfactual 写法参考。
5. NCDLA：先做现象分析，再提出机制和鲁棒性实验的写法参考。
6. ISG-CD：图边异质性与不确定性，support corruption/control 的写法参考。
7. ESR-CD：student-concept sparsity barrier 相关问题设定参考。
8. DFCD / LRCD：open / zero-shot 场景下从现实应用问题切入的写法参考。
