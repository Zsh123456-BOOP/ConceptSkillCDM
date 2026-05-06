"""Runtime forward path for CognitiveDiagnosisModel."""

from typing import Dict, Optional, Tuple, Union

import torch


def run_cdm_forward(
    model,
    student_ids: torch.Tensor,
    exercise_ids: torch.Tensor,
    concept_vector: Optional[torch.Tensor] = None,
    return_details: bool = False,
    return_logits: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    """
    Execute the main forward pass while keeping `model_cdm.py` focused on module
    construction and public API surface.
    """
    device = student_ids.device
    q_vector = concept_vector if concept_vector is not None else model.q_matrix[exercise_ids]

    s_out = model.structure_module(
        student_ids,
        identity_relations=model.identity_relations,
        concept_mask=q_vector,
    )
    relation_matrices = s_out["relation_matrices"]
    relation_used = s_out["relation_used"]
    personal_relation_spec = s_out.get("personal_relation_spec")
    knowledge_state = s_out["knowledge_state"]
    student_repr = s_out["student_repr"]
    gate_alpha = s_out["alpha"]
    personal_matrices = s_out["personal_matrices"]
    (
        global_query_context,
        query_row_global_readout_pre_gate_delta,
        query_row_global_readout_gate_mean,
        query_row_global_readout_delta,
        query_row_global_support_mass,
        query_row_global_head_var,
        query_row_global_head_margin,
    ) = model._build_global_query_readout(
        knowledge_state=knowledge_state,
        relation_matrices=relation_matrices,
        concept_mask=q_vector,
    )
    (
        personal_query_correction_raw,
        query_row_personal_message_delta_raw,
        query_row_posterior_delta_abs,
        query_row_posterior_kl,
        query_row_personal_support_mass,
        query_row_self_support_mass,
        query_row_graph_support_mass,
        query_row_global_local_rms,
        query_row_post_local_rms,
        query_row_delta_local_rms_raw,
        query_row_message_projection_gain,
    ) = model._build_personal_query_correction(
        knowledge_state=knowledge_state,
        relation_spec=personal_relation_spec,
        concept_mask=q_vector,
    )
    scaled_personal_query_correction = model.personal_query_correction_scale * personal_query_correction_raw
    aligned_personal_query_correction, personal_alignment_gate, query_row_message_alignment = model._apply_personal_alignment_gate(
        knowledge_state=knowledge_state,
        global_query_context=global_query_context,
        personal_query_correction=scaled_personal_query_correction,
        concept_mask=q_vector,
        query_row_posterior_kl=query_row_posterior_kl,
        query_row_self_support_mass=query_row_self_support_mass,
        query_row_graph_support_mass=query_row_graph_support_mass,
    )
    personal_query_writeback_delta = model._masked_query_rms(aligned_personal_query_correction, q_vector)
    personal_query_correction, personal_query_trust_scale, _, _ = model._apply_personal_query_trust_region(
        global_query_context=global_query_context,
        personal_query_correction=aligned_personal_query_correction,
        concept_mask=q_vector,
    )
    ae_query_state_residual, ae_query_state_residual_delta = model._build_ae_query_state_residual(
        knowledge_state=knowledge_state,
        global_query_context=global_query_context,
        personal_query_correction=personal_query_correction,
        concept_mask=q_vector,
    )
    query_rows = q_vector.float().unsqueeze(-1).bool()
    total_query_correction = global_query_context + personal_query_correction + ae_query_state_residual
    prediction_state = torch.where(query_rows, knowledge_state + total_query_correction, knowledge_state)
    readout_query_delta = model._masked_query_rms(total_query_correction, q_vector)
    query_row_personal_message_delta_raw = model._masked_query_rms(scaled_personal_query_correction, q_vector)
    query_row_personal_message_delta = model._masked_query_rms(personal_query_correction, q_vector)
    personal_message_projection_gap = query_row_posterior_delta_abs / query_row_personal_message_delta.clamp(min=1e-8)
    has_graph_query_signal_bool = bool(query_row_global_readout_delta.detach().item() >= 1e-8)
    has_graph_query_signal = torch.tensor(int(has_graph_query_signal_bool), device=device, dtype=torch.long)
    if has_graph_query_signal_bool:
        personal_to_graph_query_ratio = query_row_personal_message_delta / query_row_global_readout_delta.clamp(min=1e-8)
    else:
        personal_to_graph_query_ratio = query_row_personal_message_delta.new_tensor(0.0)
    personal_to_graph_query_ratio_effective = personal_to_graph_query_ratio.detach()
    personal_query_trust_scale_mean = personal_query_trust_scale.mean()
    personal_query_trust_scale_min = personal_query_trust_scale.min()
    personal_alignment_gate_mean = (
        (personal_alignment_gate.squeeze(-1) * q_vector.float()).sum()
        / q_vector.float().sum().clamp(min=1.0)
    )
    graph_query_adapter_gain = query_row_global_readout_delta / query_row_global_readout_pre_gate_delta.clamp(min=1e-8)

    b, a = model.exercise_encoder(exercise_ids)
    if return_details:
        irt_logit, diag_details = model.diagnosis_head(
            knowledge_state=prediction_state,
            concept_mask=q_vector,
            b=b,
            a=a,
            return_details=True,
        )
    else:
        irt_logit = model.diagnosis_head(
            knowledge_state=prediction_state,
            concept_mask=q_vector,
            b=b,
            a=a,
            return_details=False,
        )
        diag_details = None

    ae_logit_residual, ae_logit_residual_abs_mean = model._build_ae_logit_residual(
        student_ids=student_ids,
        knowledge_state=knowledge_state,
        relation_matrices=relation_matrices,
        global_query_context=global_query_context,
        personal_query_correction=personal_query_correction,
        concept_mask=q_vector,
    )
    total_logit = irt_logit + ae_logit_residual
    out_main = total_logit if return_logits else torch.sigmoid(total_logit)

    if not return_details:
        return out_main

    details: Dict[str, torch.Tensor] = {
        "enable_module1": torch.tensor(int(model.enable_module1), device=device),
        "use_concept_graph": torch.tensor(int(model.use_concept_graph), device=device),
        "use_personal_graph": torch.tensor(int(model.use_personal_graph), device=device),
        "share_concept_embeddings": torch.tensor(int(model.share_concept_embeddings), device=device),
        "graph_prior_logit_scale": torch.tensor(float(model.graph_prior_logit_scale), device=device),
        "ae_query_residual_scale": torch.tensor(float(model.ae_query_residual_scale), device=device),
        "ae_logit_residual_scale": torch.tensor(float(model.ae_logit_residual_scale), device=device),
        "ae_logit_residual_clip": torch.tensor(float(model.ae_logit_residual_clip), device=device),
        "ae_logit_dim": torch.tensor(float(model.ae_logit_dim), device=device),
        "relation_matrices": relation_matrices,
        "relation_used": relation_used,
        "personal_relation_spec": personal_relation_spec,
        "knowledge_state": knowledge_state,
        "prediction_state": prediction_state,
        "student_repr": student_repr,
        "q_vector": q_vector,
        "irt_b": b.detach(),
        "irt_a": a.detach(),
        "irt_logit": irt_logit.detach(),
        "ae_logit_residual": ae_logit_residual.detach(),
        "ae_logit_residual_abs_mean": ae_logit_residual_abs_mean.detach(),
        "logits": total_logit.detach(),
        "relation_identity_delta": s_out["relation_identity_delta"].detach(),
        "knowledge_state_backbone_delta": s_out["knowledge_state_backbone_delta"].detach(),
        "knowledge_state_graph_delta": s_out["knowledge_state_graph_delta"].detach(),
        "knowledge_state_personal_delta": s_out["knowledge_state_personal_delta"].detach(),
        "a_diag_semantic_ok": s_out["a_diag_semantic_ok"].detach(),
        "readout_query_delta": readout_query_delta.detach(),
        "query_row_graph_delta": query_row_global_readout_delta.detach(),
        "query_row_personal_delta": query_row_posterior_delta_abs.detach(),
        "readout_query_support_mass": query_row_global_support_mass.detach(),
        "query_row_global_readout_delta_raw": query_row_global_readout_pre_gate_delta.detach(),
        "query_row_global_readout_pre_gate_delta": query_row_global_readout_pre_gate_delta.detach(),
        "query_row_global_readout_gate_mean": query_row_global_readout_gate_mean.detach(),
        "query_row_global_readout_post_gate_delta": query_row_global_readout_delta.detach(),
        "query_row_global_readout_delta": query_row_global_readout_delta.detach(),
        "query_row_personal_message_delta": query_row_personal_message_delta,
        "query_row_personal_message_delta_raw": query_row_personal_message_delta_raw.detach(),
        "query_row_personal_message_delta_capped": query_row_personal_message_delta.detach(),
        "personal_message_delta_pre_trust": query_row_personal_message_delta_raw.detach(),
        "personal_message_delta_post_trust": query_row_personal_message_delta.detach(),
        "personal_message_projection_gap": personal_message_projection_gap.detach(),
        "query_row_posterior_delta_abs": query_row_posterior_delta_abs.detach(),
        "query_row_posterior_kl": query_row_posterior_kl.detach(),
        "query_row_global_local_rms": query_row_global_local_rms.detach(),
        "query_row_post_local_rms": query_row_post_local_rms.detach(),
        "query_row_delta_local_rms_raw": query_row_delta_local_rms_raw.detach(),
        "query_row_message_projection_gain": query_row_message_projection_gain.detach(),
        "query_row_message_alignment": query_row_message_alignment.detach(),
        "query_row_self_support_mass": query_row_self_support_mass.detach(),
        "query_row_graph_support_mass": query_row_graph_support_mass.detach(),
        "query_row_global_head_var": query_row_global_head_var.detach(),
        "query_row_global_head_margin": query_row_global_head_margin.detach(),
        "personal_query_writeback_delta": personal_query_writeback_delta.detach(),
        "ae_query_state_residual_delta": ae_query_state_residual_delta.detach(),
        "personal_alignment_gate_mean": personal_alignment_gate_mean.detach(),
        "graph_query_adapter_gain": graph_query_adapter_gain.detach(),
        "personal_to_graph_query_ratio": personal_to_graph_query_ratio.detach(),
        "personal_to_graph_query_ratio_effective": personal_to_graph_query_ratio_effective,
        "has_graph_query_signal": has_graph_query_signal,
        "personal_query_trust_scale_mean": personal_query_trust_scale_mean.detach(),
        "personal_query_trust_scale_min": personal_query_trust_scale_min.detach(),
        "query_row_global_support_mass": query_row_global_support_mass.detach(),
        "query_row_personal_support_mass": query_row_personal_support_mass.detach(),
        "personal_query_support_hops": torch.tensor(model.personal_query_support_hops, device=device, dtype=torch.long),
        "personal_query_message_gain": torch.tensor(model.personal_query_message_gain, device=device, dtype=knowledge_state.dtype),
    }
    details["irt_logit_for_reg"] = irt_logit
    details["readout_local_row_ratio"] = q_vector.float().mean().detach()

    if diag_details is not None:
        details.update(diag_details)

    if model.use_personal_graph:
        if gate_alpha is not None:
            details["alpha"] = gate_alpha
            details["alpha_detached"] = gate_alpha.detach()
        if s_out.get("alpha_logit") is not None:
            details["alpha_logit"] = s_out["alpha_logit"]
            details["alpha_logit_detached"] = s_out["alpha_logit"].detach()
        if s_out.get("alpha_student_bias") is not None:
            details["alpha_student_bias"] = s_out["alpha_student_bias"]
            details["alpha_student_bias_detached"] = s_out["alpha_student_bias"].detach()
        if s_out.get("alpha_effective") is not None:
            details["alpha_effective"] = s_out["alpha_effective"]
            details["alpha_effective_detached"] = s_out["alpha_effective"].detach()
        for key in (
            "alpha_base_mean",
            "alpha_delta_absmean",
            "alpha_saturation_ratio",
            "alpha_state_path_absmean",
            "alpha_id_path_absmean",
            "alpha_bias_path_absmean",
            "head_bias_path_absmean",
            "alpha_id_adapter_scale",
            "state_embedding_nonfinite_count",
            "context_repr_nonfinite_count",
            "state_logit_nonfinite_count",
            "id_logit_nonfinite_count",
            "alpha_base_nonfinite_count",
            "alpha_preclamp_nonfinite_count",
            "personal_state_mix",
            "personal_student_mix",
            "personal_student_adapter_scale",
            "personal_context_adapter_scale",
            "personal_delta_nonfinite_count",
            "personal_logits_nonfinite_count",
            "personal_matrix_nonfinite_count",
            "personal_bad_row_count",
            "personal_fallback_row_count",
            "personal_padded_row_count",
            "personal_bad_row_count_active",
            "personal_fallback_row_count_active",
            "personal_bad_row_rate_active",
            "personal_logits_absmax",
            "personal_logits_support_absmax",
            "personal_logits_masked_sentinel_absmax",
            "personal_delta_absmax",
            "local_row_ratio",
            "personal_support_density",
            "query_state_norm",
            "query_row_posterior_logit_delta_abs",
            "neighbor_row_posterior_logit_delta_abs",
            "personal_row_budget_mean",
            "personal_query_row_std",
            "used_no_a_support_fallback_mode",
            "local_row_mask",
            "active_row_index",
            "active_row_valid_mask",
            "query_row_active_mask",
            "neighbor_row_active_mask",
            "support_col_index",
            "support_valid_mask",
            "global_support_prob",
            "posterior_prob",
        ):
            if s_out.get(key) is not None:
                details[key] = s_out[key].detach()
        details["query_row_personal_delta"] = details["query_row_posterior_delta_abs"]
        if s_out.get("personal_gate_id_scale") is not None:
            details["personal_gate_id_scale"] = s_out["personal_gate_id_scale"]
        if s_out.get("personal_generator_id_scale") is not None:
            details["personal_generator_id_scale"] = s_out["personal_generator_id_scale"]
        if s_out.get("personal_alpha_bias_scale") is not None:
            details["personal_alpha_bias_scale"] = s_out["personal_alpha_bias_scale"]
        details["personal_matrices"] = personal_matrices
        if s_out.get("personal_matrix_delta") is not None:
            details["personal_matrix_delta"] = s_out["personal_matrix_delta"]
            details["personal_matrix_delta_detached"] = s_out["personal_matrix_delta"].detach()
        if s_out.get("personal_matrix_student_std") is not None:
            details["personal_matrix_student_std"] = s_out["personal_matrix_student_std"]
            details["personal_matrix_student_std_detached"] = s_out["personal_matrix_student_std"].detach()
        if s_out.get("personal_delta_pre_softmax_norm") is not None:
            details["personal_delta_pre_softmax_norm"] = s_out["personal_delta_pre_softmax_norm"]
            details["personal_delta_pre_softmax_norm_detached"] = s_out["personal_delta_pre_softmax_norm"].detach()
        if s_out.get("personal_delta_student_std") is not None:
            details["personal_delta_student_std"] = s_out["personal_delta_student_std"]
            details["personal_delta_student_std_detached"] = s_out["personal_delta_student_std"].detach()
        if s_out.get("alpha_head_std") is not None:
            details["alpha_head_std"] = s_out["alpha_head_std"]
            details["alpha_head_std_detached"] = s_out["alpha_head_std"].detach()
        details["personal_warmup_scale"] = s_out["personal_warmup_scale"].detach()
        details["personal_reg_warmup_scale"] = s_out["personal_reg_warmup_scale"].detach()

    return out_main, details
