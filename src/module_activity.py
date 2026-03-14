# src/module_activity.py
"""
模块活跃度检测工具：判断关键子模块（Concept Graph, Personal Graph, Skill/MF）是否真正被使用。

用于：在 run_all_datasets.py 训练时输出检测报告，帮助调参确保模块生效。

检测逻辑：
1. Concept Graph: 检测邻接矩阵行熵 —— 非均匀分布表示学到了有意义的结构
2. Personal Graph: 检测个性化 gate alpha —— 均值和方差都要足够
3. Skill/MF: 检测 MF 残差贡献 —— 方差越大表示模块在起作用
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model import CognitiveDiagnosisModel


def _get_base_model(model) -> CognitiveDiagnosisModel:
    """获取基础模型（处理 DataParallel 包装）"""
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def compute_module_activity(
    model,
    data_loader: DataLoader,
    device: torch.device,
    num_samples: int = 500,
) -> Dict[str, Any]:
    """
    计算各模块的活跃度指标。

    Args:
        model: 训练中的模型（可能是 DataParallel 包装的）
        data_loader: 数据加载器（通常用验证集）
        device: 计算设备
        num_samples: 采样数量（足够统计即可，不需要太多）

    Returns:
        包含各模块活跃度指标的字典
    """
    base_model = _get_base_model(model)
    model.eval()
    results: Dict[str, Any] = {}

    # 收集的数据
    gate_alphas: List[float] = []
    alpha_biases: List[float] = []
    mf_logits: List[float] = []
    irt_logits: List[float] = []
    fusion_gates: List[float] = []
    personal_matrix_deltas: List[float] = []
    personal_matrix_student_stds: List[float] = []

    sample_count = 0

    with torch.no_grad():
        for batch in data_loader:
            if sample_count >= num_samples:
                break

            student_ids, exercise_ids, labels = batch
            student_ids = student_ids.to(device)
            exercise_ids = exercise_ids.to(device)

            # 获取详细输出
            _, details = model(
                student_ids,
                exercise_ids,
                return_details=True,
                return_logits=True,
            )

            # 1) Gate alpha（个性化图混合系数）
            if details.get("alpha") is not None:
                gate_alphas.extend(details["alpha"].squeeze().cpu().numpy().tolist())
            if details.get("alpha_student_bias") is not None:
                alpha_biases.extend(details["alpha_student_bias"].squeeze().cpu().numpy().tolist())

            personal_matrices = details.get("personal_matrices")
            relation_matrices = details.get("relation_matrices")
            if personal_matrices is not None and relation_matrices is not None:
                pm = personal_matrices.detach()
                gm = relation_matrices.detach().mean(dim=0, keepdim=True)
                delta = (pm - gm).abs().mean(dim=(-1, -2))
                personal_matrix_deltas.extend(delta.cpu().numpy().tolist())
                personal_matrix_student_stds.append(float(pm.std(dim=0, unbiased=False).mean().item()))

            # 2) MF/IRT logits（用于计算贡献）
            if details.get("mf_logit") is not None:
                mf_logits.extend(details["mf_logit"].cpu().numpy().tolist())
            if details.get("irt_logit") is not None:
                irt_logits.extend(details["irt_logit"].cpu().numpy().tolist())
            if details.get("gate") is not None:
                fusion_gates.extend(details["gate"].cpu().numpy().tolist())

            sample_count += len(student_ids)

    # ==================== 计算活跃度指标 ====================

    # 1) Concept Graph 活跃度（通过全局邻接矩阵行熵判断）
    if base_model.structure_module.relation_learning is not None:
        with torch.no_grad():
            relation_matrices, _ = base_model.structure_module.relation_learning()  # (H, C, C)
            # 计算平均行熵
            A = relation_matrices.cpu().numpy()  # (H, C, C)
            # 对每个头、每行计算熵
            eps = 1e-12
            row_entropies = -np.sum(A * np.log(A + eps), axis=-1)  # (H, C)
            mean_row_entropy = row_entropies.mean()
            max_row_entropy = np.log(A.shape[-1])  # log(C)

            # 如果接近均匀分布（熵接近最大），说明没学到有意义的结构
            entropy_ratio = mean_row_entropy / max_row_entropy if max_row_entropy > 0 else 0.0

        results["graph_enabled"] = True
        results["graph_mean_row_entropy"] = float(mean_row_entropy)
        results["graph_max_row_entropy"] = float(max_row_entropy)
        results["graph_entropy_ratio"] = float(entropy_ratio)
        # 判断：熵比接近1（均匀）= 没学到；熵比接近0（过度稀疏）= 也没意义
        results["graph_trivial"] = entropy_ratio > 0.95  # 均匀分布
        results["graph_over_sparse"] = entropy_ratio < 0.05  # 过度稀疏
        results["graph_active"] = 0.05 < entropy_ratio < 0.95
    else:
        results["graph_enabled"] = False
        results["graph_active"] = False

    # 2) Personal Graph 活跃度（通过 gate_alpha 判断）
    if gate_alphas:
        alpha_arr = np.array(gate_alphas)
        results["personal_graph_enabled"] = True
        results["personal_gate_mean"] = float(alpha_arr.mean())
        results["personal_gate_std"] = float(alpha_arr.std())
        bias_std = float(np.array(alpha_biases).std()) if alpha_biases else 0.0
        matrix_delta = float(np.mean(personal_matrix_deltas)) if personal_matrix_deltas else 0.0
        matrix_student_std = float(np.mean(personal_matrix_student_stds)) if personal_matrix_student_stds else 0.0
        results["personal_alpha_bias_std"] = bias_std
        results["personal_matrix_delta"] = matrix_delta
        results["personal_matrix_student_std"] = matrix_student_std
        # 判断：alpha 或 personal graph 几乎不偏离全局图时，都视为 trivial
        results["personal_graph_trivial"] = alpha_arr.mean() < 0.05 or matrix_delta < 0.005
        # 只有 alpha 和 personal graph 本身都表现出差异时，才视为真正活跃
        results["personal_graph_active"] = (
            alpha_arr.mean() > 0.1
            and alpha_arr.std() > 0.01
            and matrix_delta > 0.01
            and (bias_std > 0.01 or matrix_student_std > 0.001)
        )
    else:
        results["personal_graph_enabled"] = False
        results["personal_graph_active"] = False

    # 3) Skill/MF 活跃度
    if mf_logits and irt_logits:
        mf_arr = np.array(mf_logits)
        irt_arr = np.array(irt_logits)

        # MF 贡献的绝对值和方差
        mf_abs_mean = np.abs(mf_arr).mean()
        mf_std = mf_arr.std()
        irt_abs_mean = np.abs(irt_arr).mean()

        # 如果有 fusion gate，计算平均 gate 值
        if fusion_gates:
            gate_arr = np.array(fusion_gates)
            fusion_gate_mean = gate_arr.mean()
        else:
            fusion_gate_mean = 0.5  # 默认

        results["mf_enabled"] = True
        results["mf_abs_mean"] = float(mf_abs_mean)
        results["mf_std"] = float(mf_std)
        results["irt_abs_mean"] = float(irt_abs_mean)
        results["fusion_gate_mean"] = float(fusion_gate_mean)
        # 判断：如果 MF 的贡献接近 0 或方差太小，说明没用
        results["mf_trivial"] = mf_abs_mean < 0.01 and mf_std < 0.01
        results["mf_active"] = mf_abs_mean > 0.05 or mf_std > 0.05
    elif base_model.skill_encoder is not None:
        results["mf_enabled"] = True
        results["mf_active"] = True  # 存在就假设活跃（无法从details获取）
    else:
        results["mf_enabled"] = False
        results["mf_active"] = False

    model.train()  # 恢复训练模式

    # 确保所有值都是 JSON 可序列化的（转换 numpy 类型）
    def to_serializable(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [to_serializable(x) for x in obj]
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        return obj

    results = to_serializable(results)
    return results


def format_activity_brief(activity: Dict[str, Any]) -> str:
    """
    格式化为简短的一行报告（用于训练过程中输出）。

    Example: "Graph[OK] Personal[X] MF[OK]"
    """
    parts = []

    if activity.get("graph_enabled"):
        status = "[OK]" if activity.get("graph_active") else "[X]"
        parts.append(f"Graph{status}")

    if activity.get("personal_graph_enabled"):
        status = "[OK]" if activity.get("personal_graph_active") else "[X]"
        parts.append(f"Personal{status}")

    if activity.get("mf_enabled"):
        status = "[OK]" if activity.get("mf_active") else "[X]"
        parts.append(f"MF{status}")

    if not parts:
        return "No modules enabled"

    return " ".join(parts)


def format_activity_report(
    activity: Dict[str, Any],
    dataset_name: str = "unknown",
    seed: int = 42,
    epoch: int = 0,
) -> str:
    """
    格式化为完整的报告（训练结束后输出）。
    """
    lines = [
        "=" * 60,
        "         MODULE ACTIVITY REPORT",
        "=" * 60,
        f"Dataset: {dataset_name} | Seed: {seed} | Epoch: {epoch}",
        "",
    ]

    # 1. Concept Graph
    lines.append("1. Concept Graph Module:")
    if activity.get("graph_enabled"):
        entropy_ratio = activity.get("graph_entropy_ratio", 0.0)

        if activity.get("graph_active"):
            status = "ACTIVE (learning meaningful structure)"
            advice = ""
        elif activity.get("graph_over_sparse"):
            status = "OVER-SPARSE (degenerated to near-identity)"
            advice = "   -> Consider: decrease lambda_sparse"
        elif activity.get("graph_trivial"):
            status = "TRIVIAL (uniform distribution, not learning)"
            advice = "   -> Consider: increase lambda_sparse or training longer"
        else:
            status = "INACTIVE"
            advice = ""

        lines.append(f"   - Mean row entropy: {activity.get('graph_mean_row_entropy', 0):.3f} / {activity.get('graph_max_row_entropy', 0):.3f}")
        lines.append(f"   - Entropy ratio: {entropy_ratio:.1%}")
        lines.append(f"   - Status: {status}")
        if advice:
            lines.append(advice)
    else:
        lines.append("   - Status: DISABLED (use_concept_graph=False or ablate_concept_graph=True)")

    lines.append("")

    # 2. Personal Graph
    lines.append("2. Personal Graph Module:")
    if activity.get("personal_graph_enabled"):
        gate_mean = activity.get("personal_gate_mean", 0.0)
        gate_std = activity.get("personal_gate_std", 0.0)
        alpha_bias_std = activity.get("personal_alpha_bias_std", 0.0)
        matrix_delta = activity.get("personal_matrix_delta", 0.0)
        matrix_student_std = activity.get("personal_matrix_student_std", 0.0)

        if activity.get("personal_graph_active"):
            status = "ACTIVE (using personalization)"
            advice = ""
        elif activity.get("personal_graph_trivial"):
            status = "INACTIVE (alpha or personal graph barely deviates from global)"
            advice = "   -> Consider: strengthen direct student-specific path or personal graph signal"
        else:
            status = "MARGINAL (some gate movement, but personalization is weak)"
            advice = ""

        lines.append(f"   - Gate alpha mean: {gate_mean:.3f}")
        lines.append(f"   - Gate alpha std: {gate_std:.3f}")
        lines.append(f"   - Alpha bias std: {alpha_bias_std:.3f}")
        lines.append(f"   - Personal/global delta: {matrix_delta:.4f}")
        lines.append(f"   - Inter-student matrix std: {matrix_student_std:.4f}")
        lines.append(f"   - Status: {status}")
        if activity.get("personal_graph_trivial"):
            lines.append(advice)
    else:
        lines.append("   - Status: DISABLED (use_personal_graph=False)")

    lines.append("")

    # 3. Skill/MF
    lines.append("3. Skill/MF Module:")
    if activity.get("mf_enabled"):
        mf_abs = activity.get("mf_abs_mean", 0.0)
        mf_std = activity.get("mf_std", 0.0)
        fusion_gate = activity.get("fusion_gate_mean", 0.0)

        if activity.get("mf_active"):
            status = "ACTIVE (contributing to predictions)"
            advice = ""
        elif activity.get("mf_trivial"):
            status = "INACTIVE (near-zero contribution)"
            advice = "   -> Consider: check skill_dim or learning_rate"
        else:
            status = "MARGINAL"
            advice = ""

        lines.append(f"   - MF logit |mean|: {mf_abs:.4f}")
        lines.append(f"   - MF logit std: {mf_std:.4f}")
        lines.append(f"   - Fusion gate mean: {fusion_gate:.3f}")
        lines.append(f"   - Status: {status}")
        if activity.get("mf_trivial"):
            lines.append(advice)
    else:
        lines.append("   - Status: DISABLED (use_mf_branch=False or ablate_skill_encoder=True)")

    lines.append("")

    # Overall summary
    all_active = []
    all_inactive = []

    for mod, key in [
        ("Concept Graph", "graph"),
        ("Personal Graph", "personal_graph"),
        ("Skill/MF", "mf"),
    ]:
        if activity.get(f"{key}_enabled"):
            if activity.get(f"{key}_active"):
                all_active.append(mod)
            else:
                all_inactive.append(mod)

    lines.append("-" * 60)
    lines.append("SUMMARY:")
    if all_active:
        lines.append(f"   Active modules: {', '.join(all_active)}")
    if all_inactive:
        lines.append(f"   Inactive modules: {', '.join(all_inactive)}")
    if not all_inactive:
        lines.append("   All enabled modules are functioning properly!")
    lines.append("=" * 60)

    return "\n".join(lines)
