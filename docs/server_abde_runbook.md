# Server ABDE Runbook

推荐做法是直接用 `run_abce_ablation.py` 作为主入口，原因有两点：

1. 它现在已经原生支持 `A/B/D/E` 消融，且本地已验证 `no_D -> --ablate_module2`。
2. 它自带诊断与汇总 CSV，比 `run_ablation.py` 更适合后续让 GPT 统一分析。

备选方案是继续只用 `run_ablation.py`。优点是脚本更老、更朴素；缺点是它只适合基础调度，不负责诊断汇总，因此不推荐作为这轮主流程。

## 1. 修正服务器仓库分支状态

你服务器上的报错：

```bash
git pull
Your configuration specifies to merge with the ref 'refs/heads/codex/b-rescue-20260329'
from the remote, but no such ref was fetched.
```

说明当前本地分支的 upstream 还指向已经删掉的旧远端分支。既然现在只保留 `master`，建议直接把服务器仓库强制对齐到 `origin/master`。

在服务器执行：

```bash
cd ~/ConceptSkillCDM
git fetch origin --prune
git switch master
git branch --unset-upstream || true
git branch --set-upstream-to=origin/master master
git reset --hard origin/master
git status --short --branch
```

如果你不想直接 `reset --hard`，备选是先 `git branch backup/server-diverged-$(date +%Y%m%d-%H%M%S)` 再 reset。  
但从当前目标看，旧实验历史已经没有保留价值，所以推荐直接对齐。

## 2. 激活环境

```bash
conda activate xph_env
python -V
```

## 3. 一键清理并跑 ABDE

仓库里已经新增了脚本：

[`tools/run_abde_full.sh`](C:\Users\zsh\Desktop\test_xph\ConceptSkillCDM\tools\run_abde_full.sh)

默认行为：

- 清空 `logs/*`、`results/*`、`checkpoints/*`
- 运行两个数据集：`assist_09,junyi`
- 跑 `full,no_A,no_B,no_D,no_E,B_q_only,B_no_q`
- 默认使用 `ae_dominant` profile
- 自动扫描空闲 GPU，最多使用 2 张；如果没有空闲卡，直接跳过不运行
- 默认空闲阈值：`memory.used <= 256MiB` 且 `utilization.gpu <= 5%`
- 每张卡最多 1 个任务
- 生成诊断输出

服务器执行：

```bash
cd ~/ConceptSkillCDM
chmod +x tools/run_abde_full.sh
./tools/run_abde_full.sh
```

## 4. 常用变体

只跑 A/E 相关：

```bash
ABLATIONS=full,no_A,no_E ./tools/run_abde_full.sh
```

手动指定 GPU：

```bash
GPUS=1,3 ./tools/run_abde_full.sh
```

跑 3 个 seed：

```bash
SEEDS=42,43,44 ./tools/run_abde_full.sh
```

尝试 E rescue：

```bash
PROFILES=best,e_rescue,all_rescue ./tools/run_abde_full.sh
```

如果你要补充别的 `run_abce_ablation.py` 参数，可以通过 `EXTRA_ARGS` 传进去：

```bash
EXTRA_ARGS="--delta_threshold 0.002 --poll_interval 15" ./tools/run_abde_full.sh
```

## 5. 结果文件

跑完后重点看：

- `results/abce_ablation_diagnosis.csv`
- `results/abce_ablation_summary.csv`
- `results/abce_ablation_summary_mean.csv`

后续给 GPT 分析时，优先喂 `summary.csv` 和 `summary_mean.csv`，再用 `diagnosis.csv` 查 A/E 失败原因。
