# Junyi full vs no_module3 近似同分诊断报告（基于 C_diag）

## 1) 目标与范围
- 目标：定位 `de7469c`（"加入图对角线惩罚，抑制near-identity退化"）后，Junyi 上 `gpd_base_full` 与 `gpd_base_no_module3` 指标几乎一致的原因。
- 范围：仅做工程链路诊断（CSV + logs + 代码），不重跑实验。

## 2) Phase 0：锁定 commit 与记录

### 2.1 锁定 C_diag
- commit: `de7469c`
- 来源命令：`git log --oneline --decorate --grep "对角线惩罚" -n 20`

### 2.2 CSV 中对应记录（`results/experiment_results.csv`）
- 近似同分组（最关键的一组）：
  - `2026-02-11 18:48:50` | `junyi` | `gpd_base_full` | `seed=42` | `test_auc=0.829017`
  - `2026-02-11 18:50:21` | `junyi` | `gpd_base_no_module3` | `seed=42` | `test_auc=0.829179`
- 差值约 `0.000162`，属于“几乎一致”，但**非完全相同**。

### 2.3 日志文件映射
- full -> `logs/junyi_ablation_full_seed42/train_20260211_181557.log`
- no_module3 -> `logs/junyi_ablation_no_module3_seed42/train_20260211_182238.log`
- 这两条记录属于同一批次执行（时间接近、目录不同、日志均完整）。

## 3) Phase 1：是否工程伪像（复用目录/跳过训练/覆盖产物）

### 3.1 目录命名是否区分 variant
- `run_ablation.py:99-103`（修改前）使用：`{dataset}_ablation_{abl}_seed{seed}`，full/no3 目录本来就不同。

### 3.2 两条日志均显示“真实训练发生”
- full 日志：
  - `logs/junyi_ablation_full_seed42/train_20260211_181557.log:94` `Starting training...`
  - `...:95` `Epoch [001/100]`
  - `...:127` `Checkpoint saved: checkpoints/junyi_ablation_full_seed42/checkpoint_epoch_10.pth`
  - `...:188` `Test metrics - AUC: 0.8290`
- no3 日志：
  - `logs/junyi_ablation_no_module3_seed42/train_20260211_182238.log:94` `Starting training...`
  - `...:95` `Epoch [001/100]`
  - `...:127` `Checkpoint saved: checkpoints/junyi_ablation_no_module3_seed42/checkpoint_epoch_10.pth`
  - `...:182` `Test metrics - AUC: 0.8292`

### 3.3 未发现 skip/reuse/cache 关键字
- 在对应日志中检索 `skip/already/reuse/cache/resume` 未命中有效跳过语义。

### 3.4 结论
- **结论：B) 非工程伪像。**
- full 与 no3 在该轮对比中是独立训练、独立目录、独立 checkpoint 的结果，不是“同一产物被复用”。

## 4) Phase 2：full 是否“名义开启、实际等效关闭”

### 4.1 开关与物理存在性
- full：
  - `...full...log:89` `enable_module3=True`
  - `...full...log:90` `physical(has_mf_branch=True)`
- no3：
  - `...no_module3...log:89` `enable_module3=False`
  - `...no_module3...log:90` `physical(has_mf_branch=False)`

### 4.2 模块活跃度证据
- full：
  - `...full...log:160` `MF logit |mean|: 0.2478`
  - `...full...log:162` `Fusion gate mean: 0.530`
  - `...full...log:163` `Status: ✓ ACTIVE`
- no3：
  - `...no_module3...log:154` `MF logit |mean|: 0.0000`
  - `...no_module3...log:156` `Fusion gate mean: 0.000`
  - `...no_module3...log:157` `Status: ✗ INACTIVE`

### 4.3 结论
- **full 并非“等效关闭”**。module3 在 full 运行中处于活跃状态，但对最终 AUC 提升很小（因此出现 near-tie）。

## 5) Phase 3：新增防线（已提交）

### 防线 1：CSV 增加不可抵赖运行事实列
- commit: `0bfc7a9`
- message: `记录git_sha与run_dir等运行事实到实验结果CSV，便于复现实验对齐`
- 文件：`src/experiment_utils.py`
- 新增字段：
  - `git_sha`
  - `run_dir`
  - `log_path`
  - `config_hash`
  - `final_has_mf_branch`
- 关键位置：
  - `src/experiment_utils.py` 新增 `_get_git_sha/_get_log_path_from_logger/_build_config_hash`
  - `append_summary_csv` 写入上述字段并前置列顺序

### 防线 2：ablation 输出目录包含 timestamp/run_id，避免覆盖
- commit: `399e9dc`
- message: `修复ablation输出目录命名，加入时间戳避免不同运行互相覆盖`
- 文件：`run_ablation.py`
- 改动：
  - 新增 `--run_id` 参数（可选）
  - 默认使用当前时间戳作为 `run_session`
  - tag 改为：`{dataset}_ablation_{variant}_seed{seed}_{run_session}`

## 6) 最终判定
- 该轮 `junyi` 上 full/no3 几乎一致，不是目录复用/缓存跳过/旧结果覆盖导致。
- module3 在 full 中真实启用且活跃；近似同分更可能是“贡献已存在但边际很小”。
- 已加两道工程防线，后续每条 CSV 可追溯到 git 与具体日志/目录，且不会再被同名目录覆盖干扰。
