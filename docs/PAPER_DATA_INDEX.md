# 论文数据总索引（2026-07-18 multi-seed 定稿）

本文件把论文每一张表/图映射到其源数据、生成脚本与代码版本，确保可复现、无遗漏。
终版架构代码：commit `fbaeb1c`（多头搬运锚定为生产默认）。数据划分与 KnoField-CD 字节级一致。

## 一、正文表格

| 表 | 内容 | 源数据 | 生成 |
|---|---|---|---|
| 表1 | 数据集统计 | 与 KnoField-CD 表1 一致（同划分） | 直接引用 |
| 表2 | 主表 6 数据集 × 10 方法 test AUC（mean±std） | `docs/final_results_v3_multiseed_20260718.md` 表一 | 6 seed sealed test（42/43/44/2025/2026/2027） |
| 表3 | 两模块消融 test AUC（mean±std） | `docs/final_results_v3_multiseed_20260718.md` 表二 | 6 seed：`*_fin_0717`（s42）+ `*_ms6_0717`（其余） |

baseline 数字（IRT/DINA/NCDM/KaNCD/ICDM/RCD/ORCDF/SCD/SVGCD）引自 KnoField-CD 表2，同划分可比。
本方法 seed42 test：ASSIST17 0.7891 / Junyi 0.8305 / NIPS34 0.7902 / EdNet 0.7487 / MOOCRadar 0.9345 / XES3G5M 0.8009。

## 二、正文图

| 图 | 内容 | 源数据 | 脚本 |
|---|---|---|---|
| fig_evidence_gain_curve | 证据增益 vs 证据量（合并曲线 + bootstrap CI） | `results/evidence_gain_curve_v2.csv` | `tools/evidence_gain_curve.py --bootstrap 1000` → `tools/make_paper_figures.py` |
| fig_anchor_contribution | 各通道 Δθ 使用量 + 因果置零 AUC 降幅 | `results/anchor_contribution_v2.csv` | `tools/anchor_contribution.py --causal` → 同上 |
| fig_ablation | 两模块移除的 AUC 降幅（分组条形） | final_results_v2 表二 | `make_paper_figures.py` 内嵌 |
| fig_leakage | 零参数统计在 LOO / self-leak / corpus-leak 下的 AUC | `tools/evidence_only_probe.py --leak_mode` | `make_paper_figures.py` 内嵌 |
| fig_noise_degradation | 证据优势随标签噪声退化（附录/分析） | final_results_v2 表三 | `make_paper_figures.py` 内嵌 |

图文件：`docs/paper_figures/*.pdf`（矢量，投稿用）与 `*.png`（预览）。

## 三、分析节命题与验证

| 命题 | 内容 | 验证数据 |
|---|---|---|
| P1 标签翻转不变性 | 训练样本输入对自身标签偏导恒为零 | `tests/smoke_label_isolation.py`（bitwise 不变） |
| P2 证据单调性 | ∂p/∂证据 ≥ 0（非负权重构造保证） | 构造证明 + `smoke_prediction_head.py` |
| P3 泄漏偏差定理 | Δ=(y−m̂)/(n+2)；零证据占比→100% 则泄漏 AUC→1 | fig_leakage（junyi=1.0 精确命中） |
| P4 充分统计分解 | 性能 = 同概念充分统计下限 + 跨概念结构增量 | `results/evidence_gain_curve.csv` + S2 探针 |

命题陈述与证明骨架：`docs/analysis_propositions_20260716.md`。

## 四、小实验数据（终版架构）

| 实验 | 文件 | 状态 |
|---|---|---|
| S1 证据分桶（逐数据集） | `results/s1v2_{ds}_{full,woA,woB}.csv` ×18 | 终版 |
| S1 合并增益曲线 | `results/evidence_gain_curve_v2.csv`（含 CI 与 POOLED_COMPLETE 构成控制） | 终版（正文主图） |
| S2 闭式证据下限 | `tools/evidence_only_probe.py` 可复算 | 模型无关 |
| S3 泄漏量化 | `tools/evidence_only_probe.py --leak_mode {corpus,self}` | 模型无关（fig_leakage） |
| S4 通道贡献分解 | `results/anchor_contribution_v2.csv`（使用量 + 因果降幅） | 终版（替代裸权重表） |
| N4 标签噪声 | final_results_v2 表三（36 个 sealed test） | 终版（附录/分析） |

## 五、生成与复现工具清单

| 工具 | 用途 |
|---|---|
| `tools/evidence_only_probe.py` | S2 下限 / S3 泄漏量化（零参数统计） |
| `tools/evidence_gap_buckets.py` | S1 逐数据集分桶 |
| `tools/evidence_gain_curve.py` | S1 合并增益曲线 |
| `tools/anchor_contribution.py` | S4 通道 Δθ 贡献 |
| `tools/calibration_ece.py` | ECE 校准（附录，正文一句话） |
| `tools/spectral_gap.py` | λ₂ 谱隙（多跳负结果说明） |
| `tools/clamp_saturation_check.py` | 截断饱和诊断（说明截断非瓶颈） |
| `tools/make_paper_figures.py` | 一键生成全部正文图 |
| `tools/aggregate_multiseed.py` | 多 seed 汇总（`results/multiseed_auc_runs.csv` / `_summary.csv`） |

## 六、状态

- multi-seed（42/43/44/2025/2026/2027 × full/woA/woB × 6 数据集，108 sealed test）已完成，见 v3 文档。
- 待办：正文图按 v2 CSV 与 multi-seed 重出（S1 加 CI 带，消融图加误差棒）。
- ECE 与 N4 噪声实验已裁出正文，不再列为待补。
