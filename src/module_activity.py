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
    query_row_message_projection_gains: List[float] = []
    query_row_message_alignments: List[float] = []
    query_row_self_support_masses: List[float] = []
    query_row_graph_support_masses: List[float] = []
    query_row_global_head_vars: List[float] = []
    personal_query_writeback_deltas: List[float] = []
    personal_query_row_stds: List[float] = []
    personal_item_support_added_rates: List[float] = []
    personal_item_support_added_masses: List[float] = []
    personal_to_graph_query_ratios: List[float] = []
    personal_bad_row_rate_active_vals: List[float] = []
    personal_query_trust_scale_vals: List[float] = []

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
            q_gain = details.get("query_row_message_projection_gain")
            if q_gain is not None:
                query_row_message_projection_gains.extend(q_gain.reshape(-1).detach().cpu().numpy().tolist())
            q_align = details.get("query_row_message_alignment")
            if q_align is not None:
                query_row_message_alignments.extend(q_align.reshape(-1).detach().cpu().numpy().tolist())
            q_self_support = details.get("query_row_self_support_mass")
            if q_self_support is not None:
                query_row_self_support_masses.extend(q_self_support.reshape(-1).detach().cpu().numpy().tolist())
            q_graph_support = details.get("query_row_graph_support_mass")
            if q_graph_support is not None:
                query_row_graph_support_masses.extend(q_graph_support.reshape(-1).detach().cpu().numpy().tolist())
            q_head_var = details.get("query_row_global_head_var")
            if q_head_var is not None:
                query_row_global_head_vars.extend(q_head_var.reshape(-1).detach().cpu().numpy().tolist())
            q_writeback = details.get("personal_query_writeback_delta")
            if q_writeback is not None:
                personal_query_writeback_deltas.extend(q_writeback.reshape(-1).detach().cpu().numpy().tolist())
            q_std = details.get("personal_query_row_std")
            if q_std is not None:
                personal_query_row_stds.extend(q_std.reshape(-1).detach().cpu().numpy().tolist())
            item_support_rate = details.get("personal_item_support_added_rate")
            if item_support_rate is not None:
                personal_item_support_added_rates.extend(
                    item_support_rate.reshape(-1).detach().cpu().numpy().tolist()
                )
            item_support_mass = details.get("personal_item_support_added_mass")
            if item_support_mass is not None:
                personal_item_support_added_masses.extend(
                    item_support_mass.reshape(-1).detach().cpu().numpy().tolist()
                )
            q_ratio = details.get("personal_to_graph_query_ratio_effective")
            if q_ratio is not None:
                personal_to_graph_query_ratios.extend(q_ratio.reshape(-1).detach().cpu().numpy().tolist())
            bad_row_rate_active = details.get("personal_bad_row_rate_active")
            if bad_row_rate_active is not None:
                personal_bad_row_rate_active_vals.extend(
                    bad_row_rate_active.reshape(-1).detach().cpu().numpy().tolist()
                )
            trust_scale = details.get("personal_query_trust_scale_mean")
            if trust_scale is not None:
                personal_query_trust_scale_vals.extend(
                    trust_scale.reshape(-1).detach().cpu().numpy().tolist()
                )

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
        query_head_var = float(np.mean(query_row_global_head_vars)) if query_row_global_head_vars else 0.0
        results["query_row_global_readout_delta"] = query_graph_delta
        results["query_row_global_head_var"] = query_head_var
        results["graph_active"] = bool(0.05 < entropy_ratio < 0.95 and query_graph_delta > 1e-3)
        if results["graph_active"] and query_head_var < 1e-3:
            results["graph_mode"] = "LIVE_BUT_COLLAPSED"
        elif results["graph_active"]:
            results["graph_mode"] = "LIVE"
        elif results["graph_over_sparse"]:
            results["graph_mode"] = "OVER_SPARSE"
        elif results["graph_trivial"]:
            results["graph_mode"] = "TRIVIAL"
        else:
            results["graph_mode"] = "INACTIVE"
    else:
        results["graph_enabled"] = False
        results["graph_active"] = False
        results["graph_mode"] = "DISABLED"

    if gate_alphas:
        alpha_arr = np.asarray(gate_alphas, dtype=np.float64)
        bias_std = float(np.asarray(alpha_biases, dtype=np.float64).std()) if alpha_biases else 0.0
        matrix_delta = float(np.mean(personal_matrix_deltas)) if personal_matrix_deltas else 0.0
        matrix_student_std = float(np.mean(personal_matrix_student_stds)) if personal_matrix_student_stds else 0.0
        query_personal_delta = float(np.mean(query_row_personal_message_deltas)) if query_row_personal_message_deltas else 0.0
        query_posterior_kl = float(np.mean(query_row_posterior_kls)) if query_row_posterior_kls else 0.0
        query_message_gain = float(np.mean(query_row_message_projection_gains)) if query_row_message_projection_gains else 0.0
        query_alignment = float(np.mean(query_row_message_alignments)) if query_row_message_alignments else 0.0
        query_self_support_mass = float(np.mean(query_row_self_support_masses)) if query_row_self_support_masses else 0.0
        query_graph_support_mass = float(np.mean(query_row_graph_support_masses)) if query_row_graph_support_masses else 0.0
        query_writeback = float(np.mean(personal_query_writeback_deltas)) if personal_query_writeback_deltas else query_personal_delta
        query_personal_std = float(np.mean(personal_query_row_stds)) if personal_query_row_stds else 0.0
        item_support_added_rate = (
            float(np.mean(personal_item_support_added_rates)) if personal_item_support_added_rates else 0.0
        )
        item_support_added_mass = (
            float(np.mean(personal_item_support_added_masses)) if personal_item_support_added_masses else 0.0
        )
        query_ratio = float(np.mean(personal_to_graph_query_ratios)) if personal_to_graph_query_ratios else 0.0
        bad_row_rate_active = (
            float(np.mean(personal_bad_row_rate_active_vals)) if personal_bad_row_rate_active_vals else 0.0
        )
        trust_scale_mean = (
            float(np.mean(personal_query_trust_scale_vals)) if personal_query_trust_scale_vals else 1.0
        )

        results["personal_graph_enabled"] = True
        results["personal_gate_mean"] = float(alpha_arr.mean())
        results["personal_gate_std"] = float(alpha_arr.std())
        results["personal_alpha_bias_std"] = bias_std
        results["personal_matrix_delta"] = matrix_delta
        results["personal_matrix_student_std"] = matrix_student_std
        results["query_row_personal_message_delta"] = query_personal_delta
        results["query_row_posterior_kl"] = query_posterior_kl
        results["query_row_message_projection_gain"] = query_message_gain
        results["query_row_message_alignment"] = query_alignment
        results["query_row_self_support_mass"] = query_self_support_mass
        results["query_row_graph_support_mass"] = query_graph_support_mass
        results["personal_query_writeback_delta"] = query_writeback
        results["personal_query_row_std"] = query_personal_std
        results["personal_item_support_added_rate"] = item_support_added_rate
        results["personal_item_support_added_mass"] = item_support_added_mass
        results["personal_to_graph_query_ratio"] = query_ratio
        results["personal_bad_row_rate_active"] = bad_row_rate_active
        results["personal_query_trust_scale_mean"] = trust_scale_mean
        results["personal_graph_trivial"] = bool(query_personal_delta < 0.002 and query_posterior_kl < 0.002)
        results["personal_graph_weak"] = bool(
            query_posterior_kl > 1e-4
            and (query_personal_delta <= 0.002 or query_message_gain < 0.05)
        )
        results["personal_graph_active"] = bool(
            query_personal_delta > 0.002
            and query_posterior_kl > 1e-4
            and query_personal_std > 1e-4
            and query_message_gain >= 0.05
        )
        results["personal_graph_risk"] = bool(
            results["personal_graph_active"]
            and (
                bad_row_rate_active > 0.10
                or trust_scale_mean < 0.98
                or query_ratio > 1.0
                or (
                    results.get("query_row_global_readout_delta", 0.0) > 1e-6
                    and query_personal_delta > 1.2 * results.get("query_row_global_readout_delta", 0.0)
                )
            )
        )
        if query_self_support_mass <= 1e-8 and query_graph_support_mass <= 1e-8:
            results["personal_graph_mode"] = "FLAT_SUPPORT"
        elif query_posterior_kl > 1e-4 and query_message_gain < 0.05:
            results["personal_graph_mode"] = "PROJ_COLLAPSE"
        elif query_writeback > 0.002 and query_alignment < 0.0:
            results["personal_graph_mode"] = "MISALIGNED"
        elif query_writeback > 0.002 and trust_scale_mean < 0.98:
            results["personal_graph_mode"] = "TRUST_CLIPPED"
        elif results["personal_graph_active"]:
            results["personal_graph_mode"] = "LIVE"
        elif results["personal_graph_weak"]:
            results["personal_graph_mode"] = "WEAK"
        else:
            results["personal_graph_mode"] = "INACTIVE"
    else:
        results["personal_graph_enabled"] = False
        results["personal_graph_active"] = False
        results["personal_graph_risk"] = False
        results["personal_graph_mode"] = "DISABLED"

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
        mode = str(activity.get("graph_mode", "LIVE" if activity.get("graph_active") else "X"))
        status = f"[{mode}]"
        parts.append(f"Graph{status}")

    if activity.get("personal_graph_enabled"):
        mode = str(activity.get("personal_graph_mode", "LIVE" if activity.get("personal_graph_active") else "X"))
        status = f"[{mode}]"
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
        graph_mode = str(activity.get("graph_mode", "INACTIVE"))

        if graph_mode == "LIVE":
            status = "LIVE (global graph is entering queried concept readout)"
            advice = ""
        elif graph_mode == "LIVE_BUT_COLLAPSED":
            status = "LIVE_BUT_COLLAPSED (graph enters query rows but head diversity is washed out)"
            advice = "   -> Consider: strengthen head-wise query gating or graph query adapter leverage"
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
        lines.append(f"   - Query head var: {activity.get('query_row_global_head_var', 0.0):.4f}")
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
        query_alignment = activity.get("query_row_message_alignment", 0.0)
        personal_query_row_std = activity.get("personal_query_row_std", 0.0)
        item_support_rate = activity.get("personal_item_support_added_rate", 0.0)
        item_support_mass = activity.get("personal_item_support_added_mass", 0.0)
        query_ratio = activity.get("personal_to_graph_query_ratio", 0.0)
        mode = str(activity.get("personal_graph_mode", "INACTIVE"))
        if mode == "LIVE":
            status = "LIVE (state-driven personalization is visible at query stage)"
            advice = ""
        elif mode == "PROJ_COLLAPSE":
            status = "PROJ_COLLAPSE (posterior moves, but message projection is collapsing)"
            advice = "   -> Consider: widen message basis or improve value-basis writer before enlarging posterior amplitude"
        elif mode == "MISALIGNED":
            status = "MISALIGNED (personalized writeback is anti-aligned with graph query message)"
            advice = "   -> Consider: tighten alignment gate or improve personal value basis semantics"
        elif mode == "TRUST_CLIPPED":
            status = "TRUST_CLIPPED (personal writeback exists but is repeatedly capped for safety)"
            advice = "   -> Consider: improve alignment before relaxing trust-region"
        elif mode == "FLAT_SUPPORT":
            status = "FLAT_SUPPORT (personal support is effectively empty at query stage)"
            advice = "   -> Consider: keep query-self support available even when A is ablated"
        elif mode == "WEAK":
            status = "WEAK (posterior is moving, but effective query correction is still too small)"
            advice = "   -> Consider: widen message basis or improve message projection leverage"
        else:
            status = "INACTIVE"
            advice = ""

        lines.append(f"   - Alpha mean: {gate_mean:.3f}")
        lines.append(f"   - Alpha std: {gate_std:.3f}")
        lines.append(f"   - Alpha bias std: {alpha_bias_std:.3f}")
        lines.append(f"   - Personal/global delta: {matrix_delta:.4f}")
        lines.append(f"   - Inter-student matrix std: {matrix_student_std:.4f}")
        lines.append(f"   - Query personal message delta: {query_personal_delta:.4f}")
        lines.append(f"   - Query posterior KL: {query_posterior_kl:.4f}")
        lines.append(f"   - Query message projection gain: {activity.get('query_row_message_projection_gain', 0.0):.4f}")
        lines.append(f"   - Query message alignment: {query_alignment:.4f}")
        lines.append(f"   - Query self support mass: {activity.get('query_row_self_support_mass', 0.0):.4f}")
        lines.append(f"   - Query graph support mass: {activity.get('query_row_graph_support_mass', 0.0):.4f}")
        lines.append(f"   - Query personal std: {personal_query_row_std:.4f}")
        lines.append(f"   - Item-local support added rate: {item_support_rate:.4f}")
        lines.append(f"   - Item-local support added mass: {item_support_mass:.4f}")
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
