"""Regularization helpers for CognitiveDiagnosisModel."""

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from src.model_ops import _masked_sparse_row_entropy


def _get_personal_reweightable_support(
    details: Dict[str, torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[bool, torch.Tensor]:
    """Return whether E has at least one query row with 2+ support choices."""
    support_valid_mask = details.get("support_valid_mask")
    query_row_active_mask = details.get("query_row_active_mask")
    if support_valid_mask is None or query_row_active_mask is None:
        return True, torch.tensor(1.0, device=device, dtype=dtype)

    support_count = support_valid_mask.bool().sum(dim=-1)
    query_rows = query_row_active_mask.bool().unsqueeze(1).expand_as(support_count)
    active_count = query_rows.float().sum().clamp(min=1.0)
    reweightable = query_rows & (support_count > 1)
    rate = reweightable.float().sum() / active_count
    return bool(reweightable.any().item()), rate.to(device=device, dtype=dtype)


def get_regularization_components(
    model,
    relation_matrices: torch.Tensor,
    details: Optional[Dict[str, torch.Tensor]] = None,
    base_loss: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Return decomposed regularization terms without changing optimization semantics."""
    device = relation_matrices.device
    terms: Dict[str, torch.Tensor] = {
        "graph_entropy": torch.tensor(0.0, device=device),
        "graph_diag": torch.tensor(0.0, device=device),
        "graph_uniform": torch.tensor(0.0, device=device),
        "graph_reg_scale": torch.tensor(1.0, device=device),
        "prediction_l2": torch.tensor(0.0, device=device),
        "personal_sparse": torch.tensor(0.0, device=device),
        "personal_kl": torch.tensor(0.0, device=device),
        "personal_query_residual": torch.tensor(0.0, device=device),
        "alpha_var": torch.tensor(0.0, device=device),
        "alpha_collapse": torch.tensor(0.0, device=device),
    }
    graph_reg_ramp_t = relation_matrices.new_tensor(model._get_graph_reg_ramp())
    personal_reg_ramp_t = relation_matrices.new_tensor(model._get_linear_warmup(model.personal_reg_warmup_epochs))
    if details is not None:
        details["graph_reg_ramp"] = graph_reg_ramp_t.detach()
        details["personal_reg_ramp"] = personal_reg_ramp_t.detach()

    if model.enable_module1 and model.use_concept_graph and model.lambda_graph_entropy > 0:
        if model.structure_module.relation_learning is not None:
            entropy = model.structure_module.relation_learning.get_entropy_sparsity(relation_matrices)
            num_nodes = max(2, int(relation_matrices.size(-1)))
            h_norm = entropy / (math.log(float(num_nodes)) + 1e-8)

            h_min = torch.tensor(model.graph_entropy_min, device=device, dtype=entropy.dtype)
            h_max = torch.tensor(model.graph_entropy_max, device=device, dtype=entropy.dtype)
            pen = F.relu(h_min - h_norm) + F.relu(h_norm - h_max)
            terms["graph_entropy"] = model.lambda_graph_entropy * pen * graph_reg_ramp_t

            if details is not None:
                details["graph_entropy_raw"] = entropy.detach()
                details["graph_entropy_norm"] = h_norm.detach()
                details["graph_entropy_pen"] = pen.detach()

            if model.lambda_graph_diag > 0:
                diag_mass = torch.diagonal(relation_matrices, dim1=-2, dim2=-1).mean()
                terms["graph_diag"] = model.lambda_graph_diag * diag_mass * graph_reg_ramp_t
                if details is not None:
                    details["graph_diag_mass"] = diag_mass.detach()

            num_nodes = max(2, int(relation_matrices.size(-1)))
            uniform_val = 1.0 / float(num_nodes)
            uniform_dist = torch.sqrt(
                torch.clamp((relation_matrices - uniform_val).pow(2).mean(), min=1e-12)
            )
            identity = torch.eye(
                num_nodes, device=device, dtype=relation_matrices.dtype
            ).unsqueeze(0).expand_as(relation_matrices)
            identity_dist = torch.sqrt(
                torch.clamp((relation_matrices - identity).pow(2).mean(), min=1e-12)
            )

            if model.lambda_graph_uniform > 0:
                uniform_margin = torch.tensor(
                    model.graph_uniform_margin, device=device, dtype=uniform_dist.dtype
                )
                uniform_pen = F.relu(uniform_margin - uniform_dist)
                terms["graph_uniform"] = model.lambda_graph_uniform * uniform_pen * graph_reg_ramp_t
                if details is not None:
                    details["graph_uniform_pen"] = uniform_pen.detach()

            if details is not None:
                details["graph_to_uniform_l2"] = uniform_dist.detach()
                details["graph_to_identity_l2"] = identity_dist.detach()

            tau_raw = getattr(
                model.structure_module.relation_learning,
                "temperature_raw",
                getattr(model.structure_module.relation_learning, "tau_raw", None),
            )
            tau = F.softplus(tau_raw) + 1e-6 if tau_raw is not None else relation_matrices.new_ones((1,))
            if details is not None:
                details["graph_tau_mean"] = tau.mean().detach()
                details["graph_tau_std"] = tau.std(unbiased=False).detach()

    if base_loss is not None and model.graph_reg_cap_ratio > 0:
        graph_reg_raw = terms["graph_entropy"] + terms["graph_diag"] + terms["graph_uniform"]
        cap = model.graph_reg_cap_ratio * base_loss.detach().abs()
        denom = graph_reg_raw.detach().abs() + 1e-8
        scale = torch.clamp(cap / denom, max=1.0)
        scale = torch.where(torch.isfinite(scale), scale, torch.ones_like(scale))
        terms["graph_reg_scale"] = scale.detach()
        terms["graph_entropy"] = terms["graph_entropy"] * scale
        terms["graph_diag"] = terms["graph_diag"] * scale
        terms["graph_uniform"] = terms["graph_uniform"] * scale
        if details is not None:
            details["graph_reg_raw"] = graph_reg_raw.detach()
            details["graph_reg_cap"] = cap.detach()
            details["graph_reg_scale"] = scale.detach()

    if model.prediction_l2_lambda > 0:
        reg_terms = []
        if model.exercise_encoder.b is not None and model.exercise_encoder.a_raw is not None:
            reg_terms.extend(
                [
                    model.exercise_encoder.b.weight.pow(2).mean(),
                    model.exercise_encoder.a_raw.weight.pow(2).mean(),
                ]
            )
        if reg_terms:
            terms["prediction_l2"] = model.prediction_l2_lambda * sum(reg_terms)

    if model.enable_module1 and model.use_personal_graph and details is not None:
        has_reweightable_support, reweightable_row_rate = _get_personal_reweightable_support(
            details,
            device=device,
            dtype=relation_matrices.dtype,
        )
        details["personal_reweightable_row_rate"] = reweightable_row_rate.detach()
        details["alpha_collapse_skipped_no_reweightable_support"] = relation_matrices.new_tensor(
            int(not has_reweightable_support), dtype=torch.long
        )

        posterior_prob = details.get("posterior_prob")
        support_valid_mask = details.get("support_valid_mask")
        if posterior_prob is not None and support_valid_mask is not None and model.lambda_sparse_personal > 0:
            terms["personal_sparse"] = (
                model.lambda_sparse_personal
                * _masked_sparse_row_entropy(posterior_prob, support_valid_mask.bool())
                * personal_reg_ramp_t
            )
        elif (
            "personal_matrices" in details
            and details["personal_matrices"] is not None
            and model.lambda_sparse_personal > 0
        ):
            pm = details["personal_matrices"]
            terms["personal_sparse"] = model.lambda_sparse_personal * model._row_entropy(pm) * personal_reg_ramp_t

        alpha_for_reg = details.get("alpha_effective", details.get("alpha"))
        if alpha_for_reg is not None and model.lambda_alpha > 0 and has_reweightable_support:
            alpha_flat = alpha_for_reg.view(-1)
            alpha_var = alpha_flat.var() + 1e-6
            if "alpha_student_bias" in details and details["alpha_student_bias"] is not None:
                alpha_bias_flat = details["alpha_student_bias"].view(-1)
                alpha_var = alpha_var + 0.5 * (alpha_bias_flat.var() + 1e-6)
                details["alpha_bias_std_runtime"] = alpha_bias_flat.std(unbiased=False).detach()
            terms["alpha_var"] = -model.lambda_alpha * alpha_var * personal_reg_ramp_t

        if alpha_for_reg is not None and model.lambda_alpha_min > 0 and has_reweightable_support:
            alpha_flat = alpha_for_reg.view(-1)
            alpha_std = alpha_flat.std(unbiased=False)
            alpha_target = torch.tensor(model.alpha_min_target, device=device, dtype=alpha_std.dtype)
            alpha_pen = F.relu(alpha_target - alpha_std)
            if "alpha_student_bias" in details and details["alpha_student_bias"] is not None:
                alpha_bias_flat = details["alpha_student_bias"].view(-1)
                alpha_bias_std = alpha_bias_flat.std(unbiased=False)
                alpha_pen = alpha_pen + 0.5 * F.relu(alpha_target - alpha_bias_std)
                details["alpha_bias_std_runtime"] = alpha_bias_std.detach()
            if details.get("personal_delta_pre_softmax_norm") is not None:
                delta_norm = details["personal_delta_pre_softmax_norm"]
                delta_target = alpha_target
                alpha_pen = alpha_pen + 0.5 * F.relu(delta_target - delta_norm)
                details["personal_delta_pre_softmax_norm_runtime"] = delta_norm.detach()
            if details.get("personal_delta_student_std") is not None:
                delta_student_std = details["personal_delta_student_std"]
                delta_student_target = 0.5 * alpha_target
                alpha_pen = alpha_pen + 0.5 * F.relu(delta_student_target - delta_student_std)
                details["personal_delta_student_std_runtime"] = delta_student_std.detach()
            terms["alpha_collapse"] = model.lambda_alpha_min * alpha_pen * personal_reg_ramp_t
            details["alpha_std_runtime"] = alpha_std.detach()
            details["alpha_collapse_pen"] = alpha_pen.detach()
        elif alpha_for_reg is not None and model.lambda_alpha_min > 0:
            alpha_flat = alpha_for_reg.view(-1)
            details["alpha_std_runtime"] = alpha_flat.std(unbiased=False).detach()
            details["alpha_collapse_pen"] = relation_matrices.new_tensor(0.0)

        posterior_prob = details.get("posterior_prob")
        global_support_prob = details.get("global_support_prob")
        support_valid_mask = details.get("support_valid_mask")
        query_row_active_mask = details.get("query_row_active_mask")
        if (
            posterior_prob is not None
            and global_support_prob is not None
            and support_valid_mask is not None
            and query_row_active_mask is not None
            and model.lambda_personal_kl > 0
        ):
            query_mask_sparse = (
                query_row_active_mask.float().unsqueeze(1).unsqueeze(-1)
                * support_valid_mask.bool().float()
            )
            query_count = query_mask_sparse.sum(dim=(1, 2, 3)).clamp(min=1.0)
            posterior_kl = (
                (
                    posterior_prob.clamp(min=1e-8)
                    * (
                        posterior_prob.clamp(min=1e-8).log()
                        - global_support_prob.clamp(min=1e-8).log()
                    )
                    * query_mask_sparse
                ).sum(dim=(1, 2, 3)) / query_count
            ).mean()
            terms["personal_kl"] = model.lambda_personal_kl * posterior_kl * personal_reg_ramp_t
            details["personal_kl_runtime"] = posterior_kl.detach()

        if model.lambda_personal_query_residual > 0 and details.get("personal_query_writeback_delta") is not None:
            residual = details["personal_query_writeback_delta"]
            residual_margin = relation_matrices.new_tensor(model.personal_query_residual_margin)
            residual_pen = F.relu(residual - residual_margin)
            terms["personal_query_residual"] = (
                model.lambda_personal_query_residual * residual_pen * personal_reg_ramp_t
            )
            details["personal_query_residual_runtime"] = residual.detach()
            details["personal_query_residual_pen"] = residual_pen.detach()

    total = (
        terms["graph_entropy"]
        + terms["graph_diag"]
        + terms["graph_uniform"]
        + terms["prediction_l2"]
        + terms["personal_sparse"]
        + terms["personal_kl"]
        + terms["personal_query_residual"]
        + terms["alpha_var"]
        + terms["alpha_collapse"]
    )
    terms["total"] = total
    return terms
