# ConceptSkillCDM

本仓库当前实现一条单一、可审计的 **Graph-IRT** 认知诊断主线。此前的 CRG/LCRF reachability 假设、全训练集标签统计先验和多套预测残差已被移除：现有三元组数据接口无法提供严格的 query-time 历史前缀，继续保留这些分支会把实验性捷径误写成因果机制。

当前 v6 针对的是实体协同信息缺口。模型从过滤后的训练集构建一张无标签学生—题目二部图，对每个 query 精确移除自身边，再聚合其余邻居；因此训练样本不能通过当前边看到目标题身份，验证/测试也不会读取其交互。协同学生状态进入概念图，协同题目状态生成逐概念难度，最终仍只有一条 2PL 路径。v5 中已经饱和并过拟合的 Q-item matching 已完整删除。

## 模型主线

图的先验支持与初始强度只使用训练集中的无标签元数据证据：

- student-item support：训练集中学生和题目是否发生过交互，不使用答对/答错；
- item co-occurrence：同一题目的概念共现；
- student co-exposure：同一学生在训练集中接触过的概念共现；该证据与 CSV 行顺序无关；
- self-loop：概念自身状态保留。

前向路径只有一条：

```text
train-only student-item support + Q/student-exposure metadata
        -> query-edge-excluded collaborative student/item context
        -> row-stochastic concept graph
        -> collaborative student + concept state
        -> graph message passing
        -> per-concept theta and collaborative item difficulty
        -> Q-mask concept-logit aggregation
        -> 2PL-IRT logit
```

关系强度、receiver bias、self-loop 和 temperature 仍通过 BCE 梯度学习；但标签不会被预计算成学生/题目/概念特征或 buffer。代码级不变量是：

```python
details["logits"] == details["irt_logit"]
```

没有 personal posterior、统计 target encoding、query correction、theta calibration 或额外 logit residual。已经验证无效的 Hadamard、student-concept low-rank 和 ratio-cap 分支也已删除。这个模型应被描述为静态图结构、题目条件化的认知诊断模型，不应再声称解决 concept reachability 或 evidence-gap transfer。

## 代码入口

- `main.py`：训练和推理 CLI。
- `src/dataset.py`：train-only ID/Q/图先验构建与数据加载。
- `src/model.py` / `src/model_cdm.py`：公开模型和唯一 Graph-IRT 前向。
- `src/response_graph.py`：query-safe 学生—题目协同聚合。
- `src/model_graph.py`：概念关系学习与图消息传递。
- `src/prediction_head.py`：2PL-IRT 题目参数和能力读出。
- `src/trainer.py`：训练、验证、测试、checkpoint 和简洁图诊断。
- `run_graph_ablation.py`：统一的结构消融 runner。

## 消融语义

只保留以下互不混淆的变体：

- `full`：学生—题目协同图与概念图均启用。
- `no_response_graph`：精确旁路学生—题目协同聚合，保留相同参数和逐概念 2PL 头。
- `no_message_passing`：令 `graph_propagation_alpha=0`，输出状态严格等于初始 student+concept state。
- `item_only`：只使用题目概念共现。
- `exposure_only`：只使用学生概念共接触证据。
- `degree_random`：保持逐行邻居数量的随机支持图，检验具体关系身份是否有效。

运行 dry-run：

```bash
python run_graph_ablation.py \
  --datasets assist_09,assist_17,junyi \
  --ablations full,no_response_graph,no_message_passing,item_only,exposure_only,degree_random \
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
python tests/smoke_response_graph.py
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
