# ConceptSkillCDM

本仓库当前实现一条单一、可审计的 **Graph-IRT** 认知诊断主线。此前的 CRG/LCRF reachability 假设、全训练集标签统计先验和多套预测残差已被移除：现有三元组数据接口无法提供严格的 query-time 历史前缀，继续保留这些分支会把实验性捷径误写成因果机制。

## 模型主线

图的先验支持与初始强度只使用训练集中的无标签元数据证据：

- item co-occurrence：同一题目的概念共现；
- student co-exposure：同一学生在训练集中接触过的概念共现；该证据与 CSV 行顺序无关；
- self-loop：概念自身状态保留。

前向路径只有一条：

```text
train-only Q/student-exposure metadata
        -> row-stochastic concept graph
        -> student + concept state
        -> graph message passing
        -> concept ability theta
        -> Q-mask aggregation
        -> 2PL-IRT logit
```

关系强度、receiver bias、self-loop 和 temperature 仍通过 BCE 梯度学习；但标签不会被预计算成学生/题目/概念特征或 buffer。代码级不变量是：

```python
details["logits"] == details["irt_logit"]
```

没有 personal posterior、统计 target encoding、query correction、theta calibration 或额外 logit residual。这个模型应被描述为静态图结构认知诊断基线，不应再声称解决 concept reachability 或 evidence-gap transfer。

结构调优可显式启用 `--student_concept_interaction hadamard`，在知识状态内部加入
`scale * sqrt(d) * (student ⊙ concept)`；默认值仍为 `none`，旧 checkpoint 行为不变。
该交互仍只由 BCE 梯度学习，不读取标签统计，也不增加第二条预测或 logit 残差路径。

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

- `full`：item + student co-exposure 图证据，正常消息传递。
- `no_message_passing`：令 `graph_propagation_alpha=0`，输出状态严格等于初始 student+concept state。
- `item_only`：只使用题目概念共现。
- `exposure_only`：只使用学生概念共接触证据。
- `degree_random`：保持逐行邻居数量的随机支持图，检验具体关系身份是否有效。

运行 dry-run：

```bash
python run_graph_ablation.py \
  --datasets assist_09,assist_17,junyi \
  --ablations full,no_message_passing,item_only,exposure_only,degree_random \
  --seeds 42 \
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
python tests/smoke_student_concept_interaction.py
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
  --max_test_batches 1 \
  --num_workers 0 \
  --no_cuda \
  --save_dir checkpoints/local_graph_irt_smoke \
  --log_dir logs/local_graph_irt_smoke
```

删除标签统计捷径后，指标下降应被记录为旧成绩依赖 shortcut 的证据，不应通过恢复旧残差来掩盖。

## 服务器规范

服务器 `10.154.22.11` 上的代码同步必须走 git：

```bash
cd /home/zsh/ConceptSkillCDM
git pull
```

禁止使用 Base64 或其他不可审计的命令包装。清理 `logs/`、`results/`、`checkpoints/` 前必须确认路径和 run id。
