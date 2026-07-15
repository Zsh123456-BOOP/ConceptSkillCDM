# 实验账本（2026-07-15 起）

数据划分与 KnoField-CD 仓库字节级一致（6 数据集 md5 已核对）。
目标区间：**超过 KnoField-CD Table 2 中最强非 KnoField baseline，不超过 KnoField-CD 本身**。

## 目标参照（KnoField-CD Table 2，test AUC）

| 数据集 | 最强 baseline | 其值 | KnoField-CD（上限） |
|---|---|---:|---:|
| assist_17 | ORCDF | 0.7875 | 0.8013 |
| junyi | RCD | 0.8320 | 0.8334 |
| nips34 | ORCDF | 0.7893 | 0.7898 |
| ednet_kt1 | ORCDF | 0.7480 | 0.7520 |
| moocradar | ORCDF | 0.9332 | 0.9344 |
| xes3g5m | RCD | 0.7944 | 0.7989 |

## 闭式证据下限（tools/evidence_only_probe.py，valid AUC，无任何学习参数）

| 数据集 | item | 直接证据 | 直接+item | 传播证据 | 传播+item | 无证据行占比 |
|---|---:|---:|---:|---:|---:|---:|
| assist_17 | 0.7077 | 0.6688 | 0.7419 | 0.6937 | **0.7622** | 6.9% |
| junyi | 0.7891 | 0.7891(失效) | 0.7891 | **0.7948** | 0.7923 | **100%** |

关键结论：
- junyi 是 concept-evidence gap 的极端案例：每条验证行的概念都是该学生从未接触的（1 题 1 概念且学生不重复概念），直接证据完全失效，传播证据仍 +0.006 —— 论文故事的天然案例。
- assist_17 上零参数"传播证据+item"= 0.7622，而 v9 训练模型仅 0.7646：学习部分在闭式统计之上几乎没有增值 → 架构必须让图直接为证据搬运负责（v10 anchor 动机）。

## 模型演进记录（val AUC，seed 42）

| 数据集 | v7 (8fb7a2f) | v8 (8b63ba3/4db22b5) | v9 (0930eae, run v9b2_0715) | v10 anchor (db9817b+) |
|---|---:|---:|---:|---:|
| assist_17 | 0.7694 (test 0.7718) | — | 0.7646 | |
| junyi | 0.8232 | — | | |
| nips34 | — | — | 0.7892 | |
| ednet_kt1 | — | — | 0.7476 (best@ep120, 被上限掐断) | |
| moocradar | — | 0.9311 | | |
| xes3g5m | 0.7890 (test 0.7958) | 0.7906 / no_pairwise 0.7927 (test 0.7973) | | |

## v10 变更（commit db9817b, 136cc0e）

1. **证据锚定 θ**：θ_c = 状态读出 + softplus(w)·[直接率证据, 难度残差证据, 图传播率证据]，非负权重保单调，anchor 在唯一 2PL 读出之前 → `logits == irt_logit` 不变量保持；传播通道 = 学习到的行随机概念图搬运 LOO 证据（图从此直接为证据校准负责）。
2. **训练协议修复**：assist_17/junyi/ednet/nips34 的 plateau patience < early stop，LR 衰减从未触发过 → 已修。
3. **ednet epochs 120→250**：v9 best@120 说明被上限掐断。

## 运行记录

- `v9b2_0715`：6 数据集 full 基线（v9 代码，0930eae），GPU 1/2/3。
- 计划 `v10a_0715`：同 6 数据集 full（v10 代码），对照 v9。
