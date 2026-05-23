# 图像生成 / 可编辑图提示词（中文）

以下提示词用于后续使用 image2 / Figma / Illustrator / PPT 重画论文图。当前包中已同时提供对应 SVG 草图，可直接编辑。

## 图1：概念证据缺口问题图

**目标**：展示认知诊断中“目标概念没有被学生历史直接覆盖”的问题，并说明 CRG 如何用训练集路线桥接，LCRF 如何个性化过滤。

**提示词**：
> 生成一张顶会论文风格的机制问题图，白底、简洁、矢量风格。左上是“学生历史”，包含已作答概念 h1、h2、h3 和有限正确/错误记录；右上是“目标概念 c”，用虚线箭头表示学生历史到目标概念之间存在“直接证据缺口”。下方显示“训练集经验路线”，包括题内共现（可用时）、序列转移、自保持三种来源；再通过箭头连接到“CRG 可达 support”，然后连接到“LCRF 个性化过滤”。图中明确标注：sequence transition 是 empirical learning route，不是 prerequisite；LCRF 不新增 support，只在同一 support 内重排 posterior。整体风格类似 AAAI/KDD 论文小图，少颜色、少标题、短标签。

## 图2：CRG/LCRF 模型结构图

**目标**：展示从 train-only evidence 到 CRG roadmap，再到 fixed support、LCRF posterior 和 prediction 的流程。

**提示词**：
> 生成一张认知诊断模型结构图。流程从左到右依次为：Train-only evidence（item co-occurrence、sequence transition、self-retention），CRG roadmap，Fixed support，LCRF posterior，Prediction。下方从 Learner state（query mastery、recent mastery、support count）指向 LCRF posterior。强调 CRG 是全局可审计路线图，LCRF 是 support-constrained posterior filter。不要画成神经网络堆叠图，而要画成模块边界清晰的机制图。使用白底、蓝色表示 CRG，粉色表示 LCRF，黄色表示训练集证据，绿色表示学生状态。

## 图3：机制证据链图

**目标**：展示论文实验如何分别验证数据现象、CRG 充分性、CRG 支持依赖、LCRF 必要性、LCRF 个案充分性。

**提示词**：
> 生成一张横向机制证据链图，包含五个小面板：数据现象、CRG 充分性、CRG 支持依赖、LCRF 必要性、LCRF 机制个案。每个小面板只写一个可检验 claim 和对应实验名称：dataset cards、held-out transition retrieval、support corruption、learner-state counterfactual、same-query posterior case。图底部加一句写作原则：CRG 是主模块，LCRF 是固定 support 内的个性化过滤；避免写成所有数据集都充分必要。风格紧凑，适合论文 Figure 1 或 appendix overview。
