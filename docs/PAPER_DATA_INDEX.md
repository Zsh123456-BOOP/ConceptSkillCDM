# 论文数据总索引（2026-07-19，实验数值更新后）

本文件把论文（`docs/paper_review_2025_2026/LEA_CD_paper_cn_v1.md`）每一张表/图映射到
其源数据与生成脚本，确保可复现、无遗漏。架构 `graph_irt_v10`，数据划分与
KnoField-CD 字节级一致。全部数字来自 sealed test。

## 一、正文表格

| 表 | 内容 | 源数据 |
|---|---|---|
| 表1 | 数据集统计 | 与 KnoField-CD 表1 一致（同划分） |
| 表2 | 主表 AUC/ACC（Full 最高 AUC；ACC 取同次运行） | `results/multiseed_auc_runs.xlsx`；逐运行 CSV 为 `results/multiseed_auc_runs.csv` |
| 表3 | 两模块消融（Full 最高 AUC；各消融最低 AUC） | 同上 |
| 表4 | 统计通道使用量与置零 AUC 降幅 | `results/anchor_contribution_v2.csv` |

baseline 数字（IRT/DINA/NCDM/KaNCD/ICDM/RCD/ORCDF/SCD/SVGCD）引自 KnoField-CD 表2，同划分可比。

## 二、正文图

| 图 | 内容 | 源数据 | 脚本 |
|---|---|---|---|
| 图1 fig_motivation | (a) 同概念作答覆盖分桶占比；(b) 三种构造的统计 AUC | `results/evidence_gain_curve_v3.csv`（rows 列）；泄漏数字内嵌自 `tools/evidence_only_probe.py --leak_mode` | `tools/make_paper_figures.py` |
| 图2 fig_analysis_probes | (a) 逐样本泄漏量 + 命题1 理论包络；(b) 邻居统计的预测增益按图距离分档 | `results/leakage_fan.csv`；`results/neighbor_information.csv` | `tools/leakage_fan.py` + `tools/neighbor_information_probe.py` → 同上 |
| 图3 fig_framework | LEA-CD 框架示意 | — | `make_paper_figures.py` 内嵌 |
| 图4 fig_gain_curve | 合并增益倒 U 曲线 + 95% bootstrap CI | `results/evidence_gain_curve_v3.csv` | `tools/evidence_gain_curve.py --bootstrap 1000` → 同上 |

图文件：`docs/paper_figures/*.pdf`（矢量，投稿用）与 `*.png`（预览）；
论文内嵌副本在 `docs/paper_review_2025_2026/figures/`。

## 三、分析节命题与验证

| 命题 | 内容 | 验证 |
|---|---|---|
| 命题1 泄漏偏差 | Δ=(y−m̂)/(n+2)；n=0 子集泄漏 AUC→1 | 图1(b) Junyi=1.0000 精确命中；图2 逐样本包络 |
| 标签无关性（4.2 性质一） | 翻转 y_i 不改变样本 i 的任何输入 | `tests/smoke_label_isolation.py`（bitwise 不变） |
| 单调性（4.2 性质二） | 预测对同概念正确率单调不减 | 非负权重构造 + `tests/smoke_prediction_head.py` |

早期命题草稿与推导：`docs/analysis_propositions_20260716.md`。

## 四、备询数据（不入正文）

| 内容 | 文件 | 用途 |
|---|---|---|
| 逐 run 原始值 | `results/multiseed_auc_runs.xlsx` / `results/multiseed_auc_runs.csv`（108 行） | 表2/表3 的原始记录 |
| 学到的计数门参数 | `results/gate_reliability.csv` + `tools/gate_reliability.py` | 审稿人问"门控学到了什么"时的答案（边际价值叙事） |
| MLP 头探针 | checkpoint `*_mlphead_s42_0718`（服务器） | 2PL vs NCD 头：六数据集 MLP 全败（−0.0008~−0.0065，EdNet 未收敛） |

## 五、生成与复现工具清单

| 工具 | 用途 |
|---|---|
| `tools/evidence_only_probe.py` | 不经训练的统计探针 / 三种泄漏构造 |
| `tools/leakage_fan.py` | 图2(a) 逐样本泄漏量（闭式） |
| `tools/neighbor_information_probe.py` | 图2(b) 邻居信息量分档（闭式） |
| `tools/evidence_gain_curve.py` | 图4 分桶增益 + bootstrap CI |
| `tools/anchor_contribution.py` | 表4 通道使用量与因果置零 |
| `tools/aggregate_multiseed.py` | 表2/表3 多 seed 汇总 |
| `tools/make_paper_figures.py` | 一键生成全部正文图 |
| `tools/adapt_public_benchmarks.py` 等适配器 | 六个公开数据集的再生（CSV git-ignore，仅清单入库） |
