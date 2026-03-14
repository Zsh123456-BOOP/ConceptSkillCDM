# DisentangledCDM 项目说明与阅读指引

## 1. 项目简介

本仓库实现了基于 DisentangledCDM 的认知诊断模型，面向 `assist_09` / `assist_17` / `junyi` 等数据集。整体思路是结合图结构学习与解耦表示：由概念嵌入学习前置关系图，通过单调传播得到知识掌握度，再配合非补偿聚合与 DINA 风格的预测头，提高可解释性并利用概念图结构约束模型行为。项目主要解决两个问题：一是显式建模概念先修关系，二是让预测头具备非补偿与 guess/slip 的可解释参数。

## 2. 代码总体结构

```text
.
├── main.py                  # 入口脚本：解析参数、数据处理、训练/验证/测试与保存结果
├── run_all_datasets.py      # 一键跑多数据集脚本，含 GPU 负载均衡与默认配置
├── src/
│   ├── config.py            # 命令行参数与默认超参数
│   ├── dataset.py           # 数据预处理、清洗与 DataLoader 构建
│   ├── disentangled_cdm.py  # 主模型 DisentangledCDM
│   ├── graph_modules.py     # GraphStructureLearner 与 MonotonicPropagator
│   ├── cdm_loss.py          # CDMLoss：BCE + DAG + Sparse + HSIC
│   ├── trainer.py           # 训练与评估循环
│   └── utils.py             # 日志、种子与设备选择
└── data/                    # 各数据集文件夹（train/valid/test 等 CSV）
```

快速索引：想看训练流程看 `main.py` + `src/trainer.py`；想看模型看 `src/disentangled_cdm.py` 和 `src/graph_modules.py`；想了解损失与约束看 `src/cdm_loss.py`；数据读取与清洗看 `src/dataset.py`；批量跑实验看 `run_all_datasets.py`。

## 3. 数据格式与预处理

- 输入 CSV 约定列：`stu_id, exer_id, cpt_seq, label`；`cpt_seq` 是逗号分隔的概念 ID 列表（字符串形式）。
- `CognitiveDataProcessor` 主流程（`src/dataset.py`）：
  - 读取 train/valid/test；如存在分层测试（high/medium/low）也会一并读取。
  - 统一清洗：按 `min_stu_interactions`、`min_exer_interactions` 过滤冷启动学生/冷门题；按 `min_poison_count` 删除正确率为 0 或 1 的“毒题”。
  - 学生 ID 转字符串规范化；`junyi` 额外做 transductive 对齐，只保留训练集中出现过的学生，验证/测试中的冷启动学生会被剔除。
  - 构建 ID 映射：`stu2idx / exer2idx / cpt2idx`，并统计 `num_students / num_exercises / num_concepts`。
  - 生成三元组 `(stu_idx, exer_idx, label)`，对应 `CognitiveDataset`。
  - 构建 Q-matrix（形状 `num_exercises × num_concepts`）：遍历唯一题目的 `cpt_seq`，将涉及的概念置为 1。
  - 计算学生作答统计并基于训练集正确率分位数写回 `args.q_low / args.q_high`，用于后续分层。
- `get_loaders` 返回的 batch 结构：`(stu_ids, exer_ids, labels, q_mask)`，其中 `q_mask = Q_matrix[exer_ids]`，形状 `(B, num_concepts)`。

设计意图：清洗规则让不同数据集遵循统一阈值；Q-matrix 与 q_mask 在 collate 阶段注入，模型只需要关注 `student_ids` 与 `q_mask`。

## 4. 模型结构总览（DisentangledCDM）

- 输入：`student_ids`、`exercise_ids`（当前未用但保留接口）、`q_mask`（每道题的概念掩码）。
- 输出：预测概率 `pred_prob`，前置图 `adj_dag`，知识表示 `h_knowledge`，技巧表示 `z_skill`。
- 关键子模块：
  1. **GraphStructureLearner**（`src/graph_modules.py`）  
     从可学习的 `concept_emb` 生成关系图。Head1 使用注意力 `QK^T / sqrt(d)` 经 ReLU(t - τ) 得到有向前置图 `A_dag`（对角置零，τ≈0.05）；Head2（如启用）用高斯核距离构造对称相似图并同样阈值稀疏化。
  2. **MonotonicPropagator**（`src/graph_modules.py`）  
     输入初始知识状态 `h_init` 与 `adj_dag`，按入度归一化聚合邻居贡献，再以 `impact_factor=0.5` 的单调融合 `h_init + 0.5 * tanh(agg)`，并裁剪到 `[0,1]`，保证先修只会正向提升。
  3. **知识与技巧表示**（`src/disentangled_cdm.py`）  
     - `student_emb` → `knowledge_head`（MLP+Sigmoid）→ `h_init`（K 维掌握概率）；  
     - `student_emb` → `skill_head`（Linear+Tanh）→ `z_skill`；  
     - `guess_slip_generator` 线性映射 `z_skill`，经 Sigmoid 后乘 0.3 得到 `guess_prob` 与 `slip_prob`。
  4. **预测（诊断头）**  
     - 用 `q_mask` 对 `h_knowledge` 加上 `(1 - q_mask) * 1e9` 的大数惩罚，只保留相关概念；  
     - softmin（用 `alpha=10` 的负 log-sum-exp）近似短板效应，得到 `knowledge_score`；  
     - DINA 组合：`P(correct) = (1 - slip) * knowledge_score + guess * (1 - knowledge_score)`。

## 5. 损失函数与训练逻辑

- 损失（`src/cdm_loss.py`，`CDMLoss`）：
  - 预测损失：`BCE(pred, label)`。
  - DAG 约束：`dag_constraint_loss(adj_dag) = tr(exp(A)) - d`，鼓励无环前置图。
  - 稀疏约束：`L1` 正则在 `adj_dag` 上。
  - 总损失：`loss = pred + λ_dag * dag + λ_sparse * sparse + λ_hsic * hsic`，超参来自 config。
- 训练循环（`src/trainer.py`）：  
  - `train_epoch`：前向得到 `(pred, adj_dag, h_knowledge, z_skill)`，计算总损失，梯度裁剪 `max_norm=1.0`，Adam 更新。日志中分别打印 pred/dag/sparse/hsic 分项。  
  - `evaluate`：推理模式，无梯度，计算 AUC/ACC/RMSE（AUC 异常时回退 0.5）。
- 入口（`main.py`）：  
  - 构造数据与模型，早停基于验证 AUC；保存最佳模型到 `logs/best_{dataset}_bs{batch}_emb{dim_emb}_{tag}.pth`；最终加载最佳权重在主测试与分层测试（high/medium/low）上评估，并把指标与超参写入 `results/all_datasets_results.csv`。

## 6. 运行方法

- 单数据集示例（默认超参见 `src/config.py`）：  
  ```bash
  python main.py --dataset assist_09 --seed 888 --batch_size 512 --dim_emb 64 --dim_skill 4 \
    --lambda_dag 0.5 --lambda_sparse 0.01 --lambda_hsic 0.1 \
    --min_stu_interactions 10 --min_exer_interactions 10 --min_poison_count 10
  ```
  需要调整路径时可改 `--data_root ./data/assist_09`（默认自动拼接 dataset 子目录）。
- 一键多数据集：  
  - `run_all_datasets.py` 会按 `DATASET_CONFIGS` 依次启动 `assist_09 / assist_17 / junyi`，自动负载均衡到 `ALLOWED_GPUS`，并根据数据集需求设置显存阈值（Junyi 要求更高）。  
  - 每个子进程带有 `--tag {dataset}_disentangled_full`，训练完成的结果会被统一追加到 `results/all_datasets_results.csv`。

## 7. 重要超参数与调参建议

- 模型维度：`dim_emb`（学生/概念嵌入，默认 64，Junyi 128），`dim_skill`（技巧向量维度，默认 4）。
- 结构正则：`lambda_dag`、`lambda_sparse`、`lambda_hsic` 分别控制 DAG、稀疏、解耦强度。
- 数据清洗：`min_stu_interactions`、`min_exer_interactions`、`min_poison_count` 控制用户/题目过滤与毒题清洗阈值。
- 调参小贴士：
  1. 数据量大时可适当提高 `dim_emb`，但需同步调大学习率 warmup 或降低 `lr` 防止不收敛。  
  2. 如果学出的图过密，可提高 `lambda_sparse` 或增大 `GraphStructureLearner` 的阈值 `tau`（代码中默认 0.05/0.1）。  
  3. 若发现知识与技巧混淆，可调高 `lambda_hsic`，同时观察 AUC 是否稳定。

## 8. TODO / 未来工作

- 增强题目侧：可加入题目 embedding 或文本/标签特征，让 `exercise_ids` 在模型中起作用。  
- HSIC 计算加速：对大 batch 可考虑子采样或低秩近似，降低 O(B²) 开销。  
- 可视化与日志：增加图结构与知识分布的可视化（tensorboard 或保存邻接矩阵热力图）。  
- 训练策略：尝试学习式 `impact_factor` 或多步图传播，检验对单调性与性能的影响。  
- 评估丰富性：增加分组指标的可选输出（如按题目难度分层）并记录到结果 CSV。


1. 新建一个 tmux 会话
tmux new -s cd_ablation


进到这个会话后，做平时的操作：

cd /你的项目路径/ConceptSkillCDM
conda activate xph_env    # 如果你有环境的话
python run_ablation.py --datasets assist_09,junyi --gpus 0,1 --max_concurrent 2


这个命令跑着就行了。

2. 跑着的时候如何“挂起”？

按下：

Ctrl + B，然后松开，再按 D


终端会提示类似：

[detached from cd_ablation]
说明你已经从 tmux 会话里退出了，但里面的 Python 还在跑。
此时你就算 SSH 断了，这个 tmux 里的进程也不会停。

3. 重新连上去看进度

下次 SSH 连上机器后：
tmux ls                      # 看当前有哪些 tmux 会话
tmux attach -t cd_ablation   # 或 tmux a -t cd_ablation



现在可以看logs喝results了，已经运行完成现在继续debug


已处理 1m 24s
已完成你这轮结果的继续 debug，结论很明确。

结论

assist_09：模块1、模块3都有效。
full=0.782316，no_module1=0.757835，no_module3=0.755943。
对应差值：full-no_module1=+0.024481，full-no_module3=+0.026373。
junyi：模块1有效，模块3几乎无效且轻微负迁移。
full=0.827342，no_module1=0.820260，no_module3=0.827731。
对应差值：full-no_module1=+0.007081，full-no_module3=-0.000389。
结果文件：

m13_ablation_summary.csv
m13_ablation_diagnosis.csv
问题定位

模块1里的“图相关分支”依然没学起来。
assist_09/junyi 的 full 都是 graph_entropy_ratio=1.0，并持续触发 Graph entropy ratio too high。
日志见：
assist_09 baseline_full
junyi baseline_full
junyi 的 personal graph 确认塌缩。
alpha_std=0.0，warn_alpha_collapse_count=17，Module Activity: Personal✗。
assist_09 的模块3贡献很强，甚至偏“重”。
gate_mean=0.9945，delta_over_irt=0.5639，M3 高频告警（22）。
junyi 的模块3贡献很弱。
gate_mean=0.3478，delta_over_irt=0.0236，所以 no_module3 不降反升一点点是合理现象。
核心解释

你这版不是“模块1整体无效”，而是“模块1中的图结构分支（A/E）无效”；模块1整体仍有贡献（因为去掉模块1 AUC 会掉）。
junyi 上模块3不明显，主要是它对 IRT 是弱补丁而不是主路径（低 gate、低 delta/irt）。
下一步建议（最有效）

做一次“图分支定向消融”，别只看 no_module1：跑 no_concept_graph、no_personal_graph（你现有框架可直接做）。
在 junyi 上用 m3_rescue 档再跑一轮，确认模块3是“可被救活”还是“数据上确实不需要”。
结果稳定性再加 2 个 seed（例如 43、44），避免单 seed 偶然性。
如果你同意，我下一步直接给你一条最小命令集，专跑“图分支定向消融 + junyi 模块3救援”。


当前模块1和模块3是有两个叠加起来的是吗？我是否能将其中的有用的模块保留，然后没用的删除，保留两个组件来写论文？是否对A,B,C,E进行消融然后看看到底是那几块有用？


已处理 50s
是的，当前结构可以理解为两层增强叠加在主干上：

模块2(D) 是主干（IRT头）。
模块1(A+E) 先改变表示（knowledge_state / relation_used）。
模块3(B+C) 在 irt_logit 上做残差校正（total_logit = irt_logit + gate * mf_logit）。
所以它不是“模块1和模块3直接相加”，而是“模块1改特征，模块3改输出”，两者都会影响最终结果。

你这个问题的正确做法就是：对 A/B/C/E 做细粒度消融，不要只看“去掉模块1/3”。
从你现在结果看，assist_09 上 M1/M3 都有收益；junyi 上 M1小幅有效，M3接近无效甚至轻微负迁移。说明要拆到子模块看。

建议最小实验顺序（高效且可解释）：

D
D+A
D+A+E（判断 E 是否真有增益）
best(D/A/E)+B
best(D/A/E)+B+C（判断 C 是增益还是噪声）
判定标准建议用 3 个 seed 的均值：

跨数据集稳定提升（例如 AUC 提升 > 0.003 且方差可控）=> 保留进主模型。
只在单一数据集有效 => 作为“可选模块/扩展实验”写论文。
结论上：你完全可以保留“有用的两个组件”来写论文，但前提是先完成 A/B/C/E 消融证据链。
如果你要，我下一步可以直接给你一份 run_abce_ablation.py（按 best_configs.py 读取参数、自动汇总 CSV、带诊断日志）。

已完成，新增脚本：run_abce_ablation.py。

关键点：

A/B/C/E 单独消融已内置：full, no_A, no_B, no_C, no_E（定义在 run_abce_ablation.py:71）。
自动读取 best_configs.py，并按 GPU 并发启动（作业构建在 run_abce_ablation.py:711）。
自动解析训练日志诊断指标（在 run_abce_ablation.py:138）。
自动生成汇总和“保留/删除/继续调参”建议（在 run_abce_ablation.py:441）。
已通过语法检查和 --dry_run 自检。
你现在可直接跑：

python run_abce_ablation.py --datasets assist_09,junyi --gpus 0 --max_concurrent 1 --max_per_gpu 1
输出文件：

results/abce_ablation_diagnosis.csv
results/abce_ablation_summary.csv
results/abce_ablation_summary_mean.csv
说明：

脚本默认会在检测到 C 在 best 配置中“实际未激活”时，自动追加 c_probe（强制开启 C 再对比 no_C），避免“消融C但本来就没开”的假结论。
如果你只想跑纯 best 单组消融，加 --no_auto_c_probe。