"""
模块活跃度检测工具：只评估论文主线中的 A/E 两部分是否真正工作。

用于：
1. 训练中输出简报，快速判断全局概念图 A 是否真的进入 queried concept 读出
2. 训练结束输出完整报告，判断个性化概念图 E 是否在 query stage 产生有效且不过度的修正
"""

from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
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
    计算 A/E 模块活跃度指标。

    Returns:
        包含 Concept Graph 与 Personal Graph 活跃度指标的字典
    """
    base_model = _get_base_model(model)
    was_training = model.training
    model.eval()
    results: Dict[str, Any] = {}

    gate_alphas: List[float] = []
    alpha_biases: List[float] = []
    personal_matrix_deltas: List[float] = []
    personal_matrix_student_stds: List[float] = []
    query_row_global_readout_deltas: List[float] = []
    query_row_personal_message_deltas: List[float] = []
    query_row_posterior_kls: List[float] = []
    personal_query_row_stds: List[float] = []
    personal_to_graph_query_ratios: List[float] = []

    sample_count = 0

    with torch.no_grad():
        for batch in data_loader:
            if sample_count >= num_samples:
                break

            student_ids, exercise_ids, _ = batch
            student_ids = student_ids.to(device)
            exercise_ids = exercise_ids.to(device)

            _, details = model(
                student_ids,
                exercise_ids,
                return_details=True,
                return_logits=True,
            )

            alpha = details.get("alpha_effective", details.get("alpha"))
            if alpha is not None:
                gate_alphas.extend(alpha.reshape(-1).detach().cpu().numpy().tolist())

            alpha_bias = details.get("alpha_student_bias")
            if alpha_bias is not None:
                alpha_biases.extend(alpha_bias.reshape(-1).detach().cpu().numpy().tolist())

            pm_delta = details.get("personal_matrix_delta")
            if pm_delta is not None:
                personal_matrix_deltas.extend(pm_delta.detach().reshape(-1).cpu().numpy().tolist())
            pm_student_std = details.get("personal_matrix_student_std")
            if pm_student_std is not None:
                personal_matrix_student_stds.append(float(pm_student_std.detach().item()))
            q_graph = details.get("query_row_global_readout_delta")
            if q_graph is not None:
                query_row_global_readout_deltas.extend(q_graph.reshape(-1).detach().cpu().numpy().tolist())
            q_personal = details.get("query_row_personal_message_delta")
            if q_personal is not None:
                query_row_personal_message_deltas.extend(q_personal.reshape(-1).detach().cpu().numpy().tolist())
            q_kl = details.get("query_row_posterior_kl")
            if q_kl is not None:
                query_row_posterior_kls.extend(q_kl.reshape(-1).detach().cpu().numpy().tolist())
            q_std = details.get("personal_query_row_std")
            if q_std is not None:
                personal_query_row_stds.extend(q_std.reshape(-1).detach().cpu().numpy().tolist())
            q_ratio = details.get("personal_to_graph_query_ratio")
            if q_ratio is not None:
                personal_to_graph_query_ratios.extend(q_ratio.reshape(-1).detach().cpu().numpy().tolist())

            sample_count += len(student_ids)

    relation_learning = getattr(base_model.structure_module, "relation_learning", None)
    if relation_learning is not None:
        with torch.no_grad():
            relation_matrices, _ = relation_learning()
            A = relation_matrices.cpu().numpy()
            eps = 1e-12
            row_entropies = -np.sum(A * np.log(A + eps), axis=-1)
            mean_row_entropy = float(row_entropies.mean())
            max_row_entropy = float(np.log(A.shape[-1])) if A.shape[-1] > 1 else 0.0
            entropy_ratio = mean_row_entropy / max_row_entropy if max_row_entropy > 0 else 0.0

        results["graph_enabled"] = True
        results["graph_mean_row_entropy"] = mean_row_entropy
        results["graph_max_row_entropy"] = max_row_entropy
        results["graph_entropy_ratio"] = float(entropy_ratio)
        results["graph_trivial"] = bool(entropy_ratio > 0.95)
        results["graph_over_sparse"] = bool(entropy_ratio < 0.05)
        query_graph_delta = float(np.mean(query_row_global_readout_deltas)) if query_row_global_readout_deltas else 0.0
        results["query_row_global_readout_delta"] = query_graph_delta
        results["graph_active"] = bool(0.05 < entropy_ratio < 0.95 and query_graph_delta > 1e-3)
    else:
        results["graph_enabled"] = False
        results["graph_active"] = False

    if gate_alphas:
        alpha_arr = np.asarray(gate_alphas, dtype=np.float64)
        bias_std = float(np.asarray(alpha_biases, dtype=np.float64).std()) if alpha_biases else 0.0
        matrix_delta = float(np.mean(personal_matrix_deltas)) if personal_matrix_deltas else 0.0
        matrix_student_std = float(np.mean(personal_matrix_student_stds)) if personal_matrix_student_stds else 0.0
        query_personal_delta = float(np.mean(query_row_personal_message_deltas)) if query_row_personal_message_deltas else 0.0
        query_posterior_kl = float(np.mean(query_row_posterior_kls)) if query_row_posterior_kls else 0.0
        query_personal_std = float(np.mean(personal_query_row_stds)) if personal_query_row_stds else 0.0
        query_ratio = float(np.mean(personal_to_graph_query_ratios)) if personal_to_graph_query_ratios else 0.0

        results["personal_graph_enabled"] = True
        results["personal_gate_mean"] = float(alpha_arr.mean())
        results["personal_gate_std"] = float(alpha_arr.std())
        results["personal_alpha_bias_std"] = bias_std
        results["personal_matrix_delta"] = matrix_delta
        results["personal_matrix_student_std"] = matrix_student_std
        results["query_row_personal_message_delta"] = query_personal_delta
        results["query_row_posterior_kl"] = query_posterior_kl
        results["personal_query_row_std"] = query_personal_std
        results["personal_to_graph_query_ratio"] = query_ratio
        results["personal_graph_trivial"] = bool(query_personal_delta < 0.002 and query_posterior_kl < 0.002)
        results["personal_graph_active"] = bool(
            query_personal_delta > 0.002
            and query_posterior_kl > 1e-4
            and query_personal_std > 1e-4
        )
        results["personal_graph_risk"] = bool(
            results["personal_graph_active"]
            and (
                query_ratio > 1.0
                or (
                    results.get("query_row_global_readout_delta", 0.0) > 1e-6
                    and query_personal_delta > 1.2 * results.get("query_row_global_readout_delta", 0.0)
                )
            )
        )
    else:
        results["personal_graph_enabled"] = False
        results["personal_graph_active"] = False
        results["personal_graph_risk"] = False

    if was_training:
        model.train()

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

    return to_serializable(results)


def format_activity_brief(activity: Dict[str, Any]) -> str:
    """格式化为训练中的一行简报。"""
    parts = []

    if activity.get("graph_enabled"):
        status = "[LIVE]" if activity.get("graph_active") else "[X]"
        parts.append(f"Graph{status}")

    if activity.get("personal_graph_enabled"):
        if activity.get("personal_graph_risk"):
            status = "[RISK]"
        else:
            status = "[LIVE]" if activity.get("personal_graph_active") else "[X]"
        parts.append(f"Personal{status}")

    if not parts:
        return "No graph modules enabled"

    return " ".join(parts)


def format_activity_report(
    activity: Dict[str, Any],
    dataset_name: str = "unknown",
    seed: int = 42,
    epoch: int = 0,
) -> str:
    """格式化为训练结束时的完整 A/E 活跃度报告。"""
    lines = [
        "=" * 60,
        "         MODULE ACTIVITY REPORT",
        "=" * 60,
        f"Dataset: {dataset_name} | Seed: {seed} | Epoch: {epoch}",
        "",
    ]

    lines.append("1. Concept Graph Module (A):")
    if activity.get("graph_enabled"):
        entropy_ratio = activity.get("graph_entropy_ratio", 0.0)
        query_graph_delta = activity.get("query_row_global_readout_delta", 0.0)

        if activity.get("graph_active"):
            status = "LIVE (global graph is entering queried concept readout)"
            advice = ""
        elif activity.get("graph_over_sparse"):
            status = "OVER-SPARSE (degenerated to near-identity)"
            advice = "   -> Consider: decrease lambda_sparse"
        elif activity.get("graph_trivial"):
            status = "TRIVIAL (uniform distribution, not learning)"
            advice = "   -> Consider: increase graph regularization pressure or train longer"
        else:
            status = "INACTIVE"
            advice = ""

        lines.append(
            f"   - Mean row entropy: {activity.get('graph_mean_row_entropy', 0):.3f} / "
            f"{activity.get('graph_max_row_entropy', 0):.3f}"
        )
        lines.append(f"   - Entropy ratio: {entropy_ratio:.1%}")
        lines.append(f"   - Query readout delta: {query_graph_delta:.4f}")
        lines.append(f"   - Status: {status}")
        if advice:
            lines.append(advice)
    else:
        lines.append("   - Status: DISABLED (use_concept_graph=False or no_A)")

    lines.append("")

    lines.append("2. Personal Graph Module (E):")
    if activity.get("personal_graph_enabled"):
        gate_mean = activity.get("personal_gate_mean", 0.0)
        gate_std = activity.get("personal_gate_std", 0.0)
        alpha_bias_std = activity.get("personal_alpha_bias_std", 0.0)
        matrix_delta = activity.get("personal_matrix_delta", 0.0)
        matrix_student_std = activity.get("personal_matrix_student_std", 0.0)
        query_personal_delta = activity.get("query_row_personal_message_delta", 0.0)
        query_posterior_kl = activity.get("query_row_posterior_kl", 0.0)
        personal_query_row_std = activity.get("personal_query_row_std", 0.0)
        query_ratio = activity.get("personal_to_graph_query_ratio", 0.0)

        if activity.get("personal_graph_risk"):
            status = "RISK (personal correction is active but may be overriding global query readout)"
            advice = "   -> Consider: increase lambda_personal_kl / lambda_personal_query_residual or reduce personal_query_correction_scale"
        elif activity.get("personal_graph_active"):
            status = "LIVE (state-driven personalization is visible at query stage)"
            advice = ""
        elif activity.get("personal_graph_trivial"):
            status = "INACTIVE (personal query correction is effectively flat)"
            advice = "   -> Consider: improve query-conditioned posterior signal instead of simply enlarging alpha"
        else:
            status = "MARGINAL (some movement, but personalization is weak)"
            advice = ""

        lines.append(f"   - Alpha mean: {gate_mean:.3f}")
        lines.append(f"   - Alpha std: {gate_std:.3f}")
        lines.append(f"   - Alpha bias std: {alpha_bias_std:.3f}")
        lines.append(f"   - Personal/global delta: {matrix_delta:.4f}")
        lines.append(f"   - Inter-student matrix std: {matrix_student_std:.4f}")
        lines.append(f"   - Query personal message delta: {query_personal_delta:.4f}")
        lines.append(f"   - Query posterior KL: {query_posterior_kl:.4f}")
        lines.append(f"   - Query personal std: {personal_query_row_std:.4f}")
        lines.append(f"   - Personal/global query ratio: {query_ratio:.4f}")
        lines.append(f"   - Status: {status}")
        if advice:
            lines.append(advice)
    else:
        lines.append("   - Status: DISABLED (use_personal_graph=False or no_E)")

    lines.append("")

    active_modules = []
    inactive_modules = []
    for mod, key in (("Concept Graph (A)", "graph"), ("Personal Graph (E)", "personal_graph")):
        if activity.get(f"{key}_enabled"):
            if activity.get(f"{key}_active"):
                active_modules.append(mod)
            else:
                inactive_modules.append(mod)

    lines.append("-" * 60)
    lines.append("SUMMARY:")
    if active_modules:
        lines.append(f"   Active modules: {', '.join(active_modules)}")
    if inactive_modules:
        lines.append(f"   Inactive modules: {', '.join(inactive_modules)}")
    if not inactive_modules:
        lines.append("   All enabled graph modules are functioning properly.")
    lines.append("=" * 60)

    return "\n".join(lines)
