# ConceptSkillCDM

本仓库当前实现一条单一、可审计的 **Response-Evidence Graph-IRT** 认知诊断主线。旧 CRG/LCRF 中的 personal posterior、roadmap/tutor 残差和多套预测分支已被移除；v9 使用 train-only 学生—概念响应证据，并在训练时对当前样本做精确 leave-one-out，避免旧统计先验把目标标签复制回输入。

此前已由真实输出证实有害的学生—题目协同图和逐概念题目难度偏移仍保持物理删除，不以关闭开关留在生产主线。v9 保留单一 Q-masked 2PL；响应证据由原始概念正确率与题目难度校正残差两个充分统计通道组成，不增加额外预测 logit 分支。

## 模型主线

模型只使用训练集构建两类证据：

- item co-occurrence：同一题目的概念共现；
- student co-exposure：同一学生在训练集中接触过的概念共现；该证据与 CSV 行顺序无关；
- self-loop：概念自身状态保留。
- response evidence：学生在各概念上的正确数/作答数，以及 `实际结果−该题在其他学生中的期望正确率`。训练样本精确减去自身标签、计数和残差；验证/测试只读取完整 train 统计。

前向路径只有一条：

```text
train-only Q/student-exposure metadata + two-channel leave-one-out response evidence
        -> row-stochastic concept graph
        -> evidence-initialized student + concept state
        -> graph message passing
        -> Q-masked scalar ability + scalar item difficulty/discrimination
        -> 2PL-IRT logit
```

关系强度、receiver bias、self-loop 和 temperature 通过联合训练目标的梯度学习。标签统计仅以原始 correct/count 充分统计保存；训练前向会减掉当前行，验证与测试不会读取自身 split 标签。代码级不变量仍是：

```python
details["logits"] == details["irt_logit"]
```

训练 batch 同时计算 BCE 与全体正负样本 logit 差的 pairwise logistic surrogate，固定等权组合；单类 batch 自动退回 BCE。没有 personal posterior、query correction、theta calibration、学生—题目协同图、逐概念题目难度或额外 logit residual。已经验证无效的 Hadamard、student-concept low-rank、ratio-cap 和 Q-item matching 也已删除。

## 代码入口

- `main.py`：训练和推理 CLI。
- `src/dataset.py`：train-only ID/Q/图先验构建与数据加载。
- `src/model.py` / `src/model_cdm.py`：公开模型和唯一 Graph-IRT 前向。
- `src/model_graph.py`：概念关系学习与图消息传递。
- `src/prediction_head.py`：2PL-IRT 题目参数和能力读出。
- `src/trainer.py`：训练、验证、测试、checkpoint 和简洁图诊断。
- `run_graph_ablation.py`：统一的结构消融 runner。

## 消融语义

只保留以下互不混淆的变体：

- `full`：概念图、证据锚定 θ（直接率 + 难度残差 + 图传播率三通道，非负权重），训练目标为纯 BCE。
- `no_response_evidence`：完整移除 train 响应证据 buffer 与投影（模块 A 消融）。
- `no_graph_calibration`：同时移除状态消息传递与锚定的图传播通道（模块 B 消融）。
- `no_evidence_anchor`：证据只进初始状态，θ 无锚定通道（v9 行为）。
- `no_evidence_propagation`：锚定保留直接率与残差通道，移除图传播通道。
- `pairwise_auc`：诊断变体；在纯 BCE 之上加回历史 pairwise-AUC 目标（test 证明其无增益）。
- `ema_bce`：训练结构诊断；目标同 full，另以固定 0.9 的逐 epoch 权重 EMA 做验证和 checkpoint 选择。EMA 不开放为 CLI 调参项，推理仍为单模型。
- `no_message_passing`：令 `graph_propagation_alpha=0`，输出状态严格等于初始 student+concept state。
- `item_only`：只使用题目概念共现。
- `exposure_only`：只使用学生概念共接触证据。
- `degree_random`：保持逐行邻居数量的随机支持图，检验具体关系身份是否有效。

运行 dry-run：

```bash
python run_graph_ablation.py \
  --datasets assist_09,assist_17,junyi,moocradar,xes3g5m \
  --ablations full,no_response_evidence,no_pairwise_loss,no_message_passing,item_only,exposure_only,degree_random \
  --seeds 42 \
  --run_mode train \
  --gpus 0 \
  --dry_run
```

## 验证

Smoke 文件是可直接执行的脚本：

```bash
python -m compileall -q main.py experiment_configs.py run_graph_ablation.py src tests tools
python tests/smoke_prediction_head.py
python tests/smoke_graph_propagation_alpha.py
python tests/smoke_graph_priors.py
python tests/smoke_label_isolation.py
python tests/smoke_ablation.py
python tests/smoke_runtime_regressions.py
python tests/smoke_graph_runner.py
python tests/smoke_ablation_flags.py
python tests/smoke_gpu_selector.py
python tests/smoke_concurrent_results.py
python tests/smoke_training_protocol.py
```

最小 CPU 训练闭环：

```bash
python main.py \
  --dataset_name assist_09 \
  --model_variant full \
  --epochs 1 \
  --batch_size 128 \
  --max_train_batches 2 \
  --max_val_batches 1 \
  --run_mode train \
  --num_workers 0 \
  --no_cuda \
  --save_dir checkpoints/local_graph_irt_smoke \
  --log_dir logs/local_graph_irt_smoke
```

正式实验先使用 `--run_mode train` 只按 validation 选择 checkpoint：这一阶段不会打开 `test.csv`，checkpoint 只保存 train/valid 指纹。结构与配置锁定后，再对已有 `--run_id` 显式运行 `--run_mode test`。测试数据路径、数据集名、模型变体和 seed 均由 checkpoint 绑定；首次打开测试集前会写入 `test_seal.json`，之后即使进程中断也拒绝二次测试，已封印目录也拒绝重新训练覆盖 checkpoint。

删除标签统计捷径后，指标下降应被记录为旧成绩依赖 shortcut 的证据，不应通过恢复旧残差来掩盖。

## 服务器规范

服务器 `10.154.22.11` 上的代码同步必须走 git：

```bash
cd /home/zsh/ConceptSkillCDM
git pull
```

禁止使用 Base64 或其他不可审计的命令包装。清理 `logs/`、`results/`、`checkpoints/` 前必须确认路径和 run id。
