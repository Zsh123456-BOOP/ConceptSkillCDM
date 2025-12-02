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
  - 解耦约束：`HSIC(h_knowledge, z_skill)`，鼓励知识掌握与技巧向量独立。
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
