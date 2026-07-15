# 主线查新结论（2026-07-15，摘要级检索）

三个方向并行检索的汇总。总判断：**"组合新、部件不新"——完整主张未见先例，但每个部件都有邻居，写作时必须显式引用并划清边界。**

## 方向一：CD/KT 中的标签泄漏 / 统计先验捷径

**结论**："防泄漏响应证据 + 精确 LOO"完整表述在 CD 领域未见先例。

- KT 侧已有明确的 label-leakage 文献线（必须引用）：pyKT (NeurIPS 2022) 量化 KC 展开序列 one-by-one 评测泄漏（AUC 虚高 8–13%）；"Addressing Label Leakage in KT" (CSEDU 2025)；"Leakage-Free and Recency-Aware Embeddings" (2025)。但其机制是"同题多 KC 展开/序列未来信息互漏"，与我们的"训练标签导出的聚合统计先验回灌输入"不同。
- LOO 目标编码是表格 ML/CTR 成熟实践（target encoding 文献），不能宣称发明 LOO。差异点：(a) CD 前向内部多通道响应充分统计的精确逐样本扣除（label/count/residual 同扣）；(b) 训练/验证/测试不对称契约；(c) 固定消融变体量化捷径虚高。
- 批判靶标真实存在：ORCDF (KDD 2024) 把 response signals 编入输入图边；ICDM (WWW 2024)、ID-CDF 用响应聚合作输入。**指名批评前需查其开源代码确认是否扣除当前样本。**
- 定位建议：novelty 收窄为"首个在 CD 中识别并量化'训练标签统计先验回灌'捷径、并以 train-only 充分统计 + 精确 LOO 给出防泄漏响应证据设计的工作"。

## 方向二：概念图传播 × 证据稀疏概念校准

**结论**：完整主张（无标签双通道图先验 + 消息传递、目标指向证据稀疏概念的校准）未见先例，但问题本身已被命名。

- 问题已有名字：KaNCD 的 low knowledge coverage、KSCD 的 non-interactive concepts、DisKCD 的 untested knowledge、**ESR-CD (FCS 2025) 的 student-concept sparsity barrier（最接近）**。不能宣称"首次发现"，只能宣称新刻画（按证据量分桶的 concept-evidence gap 量化）与新解法。
- 解法先例但机制不同：KaNCD/KSCD 用低秩/嵌入隐式补全（MF 非图传播）；DisKCD/TechCD 靠侧信息/专家图；ESR-CD 稀疏掩码+MF。共现图在 KT (GKT/PEBG) 用于表示增强而非掌握度校准。
- "校准"框架（ReliCD、UCD）只做不确定性量化，不改估计、不用图传播。**"图传播作为校准手段 + 按证据稀疏度分层评估"是可辩护的空白。**
- ORCDF 的 oversmoothing 警告可反向引用，论证"定向传播"的必要性。
- 审稿风险：KaNCD、KSCD、ESR-CD、DisKCD 必须逐一区分"补全未接触概念"vs"校准低证据概念"，并给出证据分桶实验。

## 方向三：Graph + IRT 与单一 2PL 读出

**结论**："图编码 + IRT 读出"已有人做（不能当首创）；**"全部证据汇入唯一 Q-masked 标量 2PL、零神经残差/校准分支"未检出先例。**

- 已有组合：CD 综述记载 Gated-GNN→IRT；GCAKT (2025, KT)；ORCDF 可套 IRT 骨干。
- 最接近的 GEAR-CD (Sci Reports 2025)：宣称 IRT 对齐，但读出是 MLP 包裹的 a(θ−b)——正是我们禁掉的东西，是理想对照。
- Deep-IRT/DIRT 有纯标量 IRT 读出但无图编码（Deep-IRT 近似 1PL 无区分度）。
- "auditable" 措辞在 CD 文献几乎不出现；邻居是可解释性（NCDM 单调性）与可辨识性（ID-CDF）。
- 定位建议：贡献从"Graph+IRT 组合"下移到"证据汇聚方式 + 读出纯度"：所有图先验与 LOO 防泄漏证据在 θ 之前融合，读出是可断言的 a(θ−b) 恒等式；主要对照坐标 ORCDF、GEAR-CD、Deep-IRT/DIRT、ID-CDF。

## 汇总：论文定位一句话

首个在 CD 中把"训练标签统计回灌"识别为可量化捷径，并以 train-only 充分统计 + 精确逐行 LOO 构造防泄漏响应证据、经无标签概念图定向传播校准证据稀疏概念、最终汇入单一可断言 a(θ−b) 标量 2PL 读出的认知诊断框架。

（完整论文列表见各 agent 检索记录；指名批评 ORCDF/ICDM 前需核对其开源代码。）
