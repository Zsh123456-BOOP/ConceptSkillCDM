# 消融与小实验方案（草案，2026-07-15）

围绕主线（防泄漏响应证据 + 概念图定向传播校准 + 单一 2PL 读出）设计，不模仿 KnoField-CD 的实验矩阵。
待 v10b 结构冻结后执行；所有 train 先行、test 一次性解封。

## 一、主消融表（论文表 3 候选）

固定变体（`--model_variant`），每个都有单一可审计含义：

| 变体 | 检验的主张 |
|---|---|
| full | 完整模型 |
| no_response_evidence | 响应证据整体的贡献 |
| no_evidence_anchor | 证据锚定 θ（v10 机制）的贡献；退化为 v9 行为 |
| no_evidence_propagation | **图传播通道**的贡献（主线核心主张） |
| no_message_passing | GNN 状态传播的贡献 |
| item_only / exposure_only | 两类无标签图先验各自的贡献 |
| degree_random | 关系"身份"是否有效（度匹配随机图对照） |
| no_pairwise_loss | 训练目标消融 |

- 范围：6 数据集 × 9 变体 × seed 42（一轮）；full 变体另加 seed 43/44 报均值±方差。
- 预算：~54 run × 平均 30 分钟 ÷ 3 GPU ≈ 9 小时墙钟。

## 二、小实验（各自绑定一个主线主张）

### S1 证据缺口校准曲线（核心小实验）
按验证/测试行"目标概念上的同概念训练证据量"分桶（0 / 1–2 / 3–5 / ≥6 次），
报各桶 AUC：full vs no_evidence_propagation vs no_response_evidence。
预期：证据越稀疏的桶，传播通道的增益越大；junyi（100% 零证据）整集即为极端桶。
→ 直接量化"图传播补证据缺口"，呼应闭式探针的相关性发现。

### S2 闭式证据下限表（分析节/附录）
tools/evidence_only_probe.py 的 6 数据集表（item / 直接证据 / 传播证据 / 组合 / 无证据行占比），
并给出"传播增益 vs 无证据行占比"的相关性散点。零参数下限也为各数据集难度提供标尺。

### S3 泄漏捷径量化（主线卖点实验）
在闭式探针中加 `--include_self` 开关：把当前行标签计入其自身证据（模拟无 LOO 的统计先验回灌），
对比正常 LOO 版本的 AUC 虚高幅度（预期显著虚高）。
→ 把"为什么必须精确 LOO"变成一个数字，支撑防泄漏设计的必要性；纯离线，不动模型。

### S4 锚定权重的可解释性
读取各数据集 full 模型学到的三通道非负权重（直接率 / 难度残差 / 图传播），
对照各数据集证据密度：预期 nips34（证据密）传播权重低、moocradar/xes3g5m（缺口大）传播权重高。
→ 模型自己学出"何时信直接证据、何时信图传播"。

### S5 证据 dropout 的作用（训练动力学）
v10a（无 dropout）vs v10b（有 dropout）在 assist_17/nips34 的 train/val AUC 曲线对比：
无 dropout 时充分统计捷径与状态通路共适应、加速记忆化；dropout 恢复泛化。
→ 呼应"防捷径"叙事的训练层面证据。数据已在手（v10a 运行保留）。

## 三、执行顺序

1. v10b 结构冻结（assist_17/nips34 探针确认不再倒退，junyi/moocradar/xes/ednet 全量 v10b）。
2. 主消融表（一）。
3. S1–S4 脚本化（tools/ 下各一个独立脚本，输出 CSV）。
4. 全部 train 完成后按数据集一次性 `--run_mode test` 解封，汇总最终表。
