# 论文数据总索引（2026-07-18，代码清理后）

本文件把论文（`docs/paper_review_2025_2026/LEA_CD_paper_cn_v1.md`）每一张表/图映射到
其源数据与生成脚本，确保可复现、无遗漏。架构 `graph_irt_v10`，数据划分与
KnoField-CD 字节级一致。全部数字来自 sealed test。

## 一、正文表格

| 表 | 内容 | 源数据 |
|---|---|---|
| 表1 | 数据集统计 | 与 KnoField-CD 表1 一致（同划分） |
| 表2 | 主表 AUC/ACC（mean±std，6 seed） | `docs/final_results_v3_multiseed_20260718.md` 表一；`results/multiseed_auc_summary.csv` |
| 表3 | 两模块消融（mean±std，6 seed） | 同上表二；checkpoint：`*_fin_0717`（s42）+ `*_ms6_0717`（其余 5 seed） |

baseline 数字（IRT/DINA/NCDM/KaNCD/ICDM/RCD/ORCDF/SCD/SVGCD）引自 KnoField-CD 表2，同划分可比。

## 二、正文图

| 图 | 内容 | 源数据 | 脚本 |
|---|---|---|---|
| 图1 fig_motivation | (a) 同概念作答覆盖分桶占比；(b) 三种构造的统计 AUC | `results/evidence_gain_curve_v3.csv`（rows 列）；泄漏数字内嵌自 `tools/evidence_only_probe.py --leak_mode` | `tools/make_paper_figures.py` |
| 图2 fig_leakage_fan | 逐样本泄漏量 vs n + 命题1 理论包络 | `results/leakage_fan.csv` | `tools/leakage_fan.py` → 同上 |
| 图3 fig_neighbor_decay | 邻居统计的预测增益按图距离分档 | `results/neighbor_information.csv` | `tools/neighbor_information_probe.py` → 同上 |
| 图4 fig_framework | LEA-CD 框架示意 | — | `make_paper_figures.py` 内嵌 |
| 图5 fig_ablation | 移除各模块的 AUC 降幅（6 seed 误差棒） | `results/multiseed_auc_summary.csv` | 同上 |
| 图6 fig_gain_curve | 合并增益倒 U 曲线 + 95% bootstrap CI | `results/evidence_gain_curve_v3.csv` | `tools/evidence_gain_curve.py --bootstrap 1000` → 同上 |
| 图7 fig_channel_scatter | 通道使用量 × 置零 AUC 降幅 | `results/anchor_contribution_v2.csv` | `tools/anchor_contribution.py --causal` → 同上 |

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
| 逐 run 多 seed 原始值 | `results/multiseed_auc_runs.csv`（108 行） | 表2/表3 的原始记录 |
| 学到的计数门参数 | `results/gate_reliability.csv` + `tools/gate_reliability.py` | 审稿人问"门控学到了什么"时的答案（边际价值叙事） |
| MLP 头探针 | checkpoint `*_mlphead_s42_0718`（服务器） | 2PL vs NCD 头：六数据集 MLP 全败（−0.0008~−0.0065，EdNet 未收敛） |

## 五、生成与复现工具清单

| 工具 | 用途 |
|---|---|
| `tools/evidence_only_probe.py` | 不经训练的统计探针 / 三种泄漏构造 |
| `tools/leakage_fan.py` | 图2 逐样本泄漏量（闭式） |
| `tools/neighbor_information_probe.py` | 图3 邻居信息量分档（闭式） |
| `tools/evidence_gain_curve.py` | 图6 分桶增益 + bootstrap CI |
| `tools/anchor_contribution.py` | 图7 通道使用量与因果置零 |
| `tools/aggregate_multiseed.py` | 表2/表3 多 seed 汇总 |
| `tools/make_paper_figures.py` | 一键生成全部正文图 |
| `tools/adapt_public_benchmarks.py` 等适配器 | 六个公开数据集的再生（CSV git-ignore，仅清单入库） |
