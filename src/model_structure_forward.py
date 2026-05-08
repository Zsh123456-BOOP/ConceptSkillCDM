"""Runtime forward path for ConceptStructureModeling."""

from typing import Any, Dict, Optional

import torch

from src.model_ops import (
    _augment_support_cache_with_item_concepts,
    _build_no_a_query_support_cache,
    _build_support_cache,
    _gather_row_support,
    _masked_absmax_or_zero,
    _masked_support_softmax,
    _pack_active_row_index,
    _safe_zero_preserving_sqrt,
)


def run_structure_forward(
    module,
    student_ids: torch.Tensor,
    identity_relations: torch.Tensor,
    concept_mask: Optional[torch.Tensor] = None,
    student_concept_prior: Optional[torch.Tensor] = None,
    student_concept_recent_prior: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Execute the A+E structure forward pass.

    Kept as a helper so `model_structure.py` stays focused on module construction
    instead of carrying a 400+ line runtime implementation.
    """
    device = student_ids.device
    dtype = identity_relations.dtype
    batch_size = student_ids.size(0)

    if not module.enable_module:
        knowledge_state = torch.zeros((batch_size, module.num_concepts, module.knowledge_dim), device=device, dtype=dtype)
        student_repr = torch.zeros((batch_size, module.knowledge_dim), device=device, dtype=dtype)
        return {
            "relation_matrices": identity_relations,
            "relation_used": identity_relations,
            "knowledge_state": knowledge_state,
            "student_repr": student_repr,
            "alpha": None,
            "personal_matrices": None,
        }

    support_diagnostics = {}
    if module.use_concept_graph and module.relation_learning is not None:
        relation_matrices, _ = module.relation_learning()
        support_diagnostics = getattr(module.relation_learning, "_last_support_diagnostics", {}) or {}
    else:
        relation_matrices = identity_relations

    knowledge_state = module.knowledge_encoder(student_ids, relation_matrices)
    student_repr = knowledge_state.mean(dim=1)

    gate_alpha = None
    gate_alpha_logit = None
    gate_alpha_bias = None
    gate_alpha_effective = None
    personal_matrices = None
    personal_matrix_delta = None
    personal_matrix_student_std = None
    personal_delta_pre_softmax_norm = None
    personal_delta_student_std = None
    alpha_head_std = None
    alpha_state_path_absmean = None
    alpha_id_path_absmean = None
    alpha_bias_path_absmean = None
    head_bias_path_absmean = None
    alpha_id_adapter_scale = None
    personal_state_mix = None
    personal_student_mix = None
    personal_mastery_prior_scale = None
    personal_mastery_scores_absmean = None
    personal_recent_mastery_prior_scale = None
    personal_recent_mastery_scores_absmean = None
    personal_student_adapter_scale = None
    personal_context_adapter_scale = None
    personal_delta_nonfinite_count = None
    personal_logits_nonfinite_count = None
    personal_matrix_nonfinite_count = None
    personal_bad_row_count = None
    personal_fallback_row_count = None
    personal_logits_absmax = None
    personal_padded_row_count = None
    personal_bad_row_count_active = None
    personal_fallback_row_count_active = None
    personal_bad_row_rate_active = None
    personal_logits_support_absmax = None
    personal_logits_masked_sentinel_absmax = None
    personal_delta_absmax = None
    local_row_ratio = None
    personal_support_density = None
    query_state_norm = None
    alpha_base_mean = None
    alpha_delta_absmean = None
    alpha_saturation_ratio = None
    state_embedding_nonfinite_count = None
    context_repr_nonfinite_count = None
    state_logit_nonfinite_count = None
    id_logit_nonfinite_count = None
    alpha_base_nonfinite_count = None
    alpha_preclamp_nonfinite_count = None
    query_row_posterior_logit_delta_abs = None
    neighbor_row_posterior_logit_delta_abs = None
    query_row_posterior_delta_abs = None
    query_row_posterior_kl = None
    personal_row_budget_mean = None
    personal_query_row_std = None
    personal_item_support_added_rate = None
    personal_item_support_added_mass = None
    used_no_a_support_fallback_mode = None
    active_row_index = None
    active_row_valid_mask = None
    support_col_index = None
    support_valid_mask = None
    global_support_prob = None
    posterior_prob = None
    query_row_active_mask = None
    neighbor_row_active_mask = None
    relation_used = relation_matrices
    personal_relation_spec = None
    initial_state = module.knowledge_encoder.compose_initial_state(student_ids)
    knowledge_state_pre_personal = knowledge_state
    knowledge_state_backbone_delta = (knowledge_state_pre_personal - initial_state).pow(2).mean().sqrt()
    if module.use_concept_graph and module.relation_learning is not None:
        relation_identity_delta = (relation_matrices - identity_relations).pow(2).mean().sqrt()
        knowledge_state_graph_delta = knowledge_state_backbone_delta
    else:
        relation_identity_delta = relation_matrices.new_tensor(0.0)
        knowledge_state_graph_delta = relation_matrices.new_tensor(0.0)
    a_diag_semantic_ok = relation_matrices.new_tensor(1.0)
    knowledge_state_personal_delta = relation_matrices.new_tensor(0.0)
    local_row_mask = module._build_local_row_mask(
        concept_mask,
        relation_matrices if (module.use_concept_graph and module.relation_learning is not None) else None,
    )
    if local_row_mask is not None:
        local_row_ratio = local_row_mask.mean()

    if module.use_personal_graph and module.adaptive_gate is not None and module.personal_generator is not None:
        student_global_repr = module.knowledge_encoder.student_global(student_ids)
        query_mask = concept_mask.float() if concept_mask is not None else None
        local_mask = local_row_mask if local_row_mask is not None else query_mask
        query_state = module._masked_state_pool(knowledge_state, query_mask)
        local_state = module._masked_state_pool(knowledge_state, local_mask)
        local_dispersion = module._masked_state_std(knowledge_state, local_mask, local_state)
        global_state = knowledge_state.mean(dim=1)
        query_contrast = query_state - global_state
        local_contrast = local_state - global_state
        query_state_norm = query_state.norm(dim=-1).mean()
        if module.personal_disable_student_global_context:
            context_repr = torch.cat(
                [
                    query_state,
                    local_state,
                    local_dispersion,
                    local_state - query_state,
                    query_contrast,
                    local_contrast,
                ],
                dim=-1,
            )
        else:
            context_repr = torch.cat(
                [
                    student_global_repr,
                    query_state,
                    local_state,
                    local_dispersion,
                    local_state - query_state,
                    query_contrast,
                    local_contrast,
                ],
                dim=-1,
            )
        gate_state_repr = query_state + 0.5 * query_contrast
        generator_state_repr = query_state + 0.5 * local_contrast
        gate_id_scale = None
        generator_id_scale = None
        gate_id_embedding = None
        generator_id_embedding = None
        row_budget_mask, query_row_mask, neighbor_row_mask = module._build_personal_row_budget_mask(
            concept_mask,
            local_row_mask,
        )
        active_source_mask = None if row_budget_mask is None else (row_budget_mask > 0)
        if active_source_mask is None and local_row_mask is not None:
            active_source_mask = local_row_mask.bool()
        active_row_index, active_row_valid_mask = _pack_active_row_index(active_source_mask)
        if row_budget_mask is not None and active_row_index is not None:
            row_budget_values = torch.gather(row_budget_mask, 1, active_row_index.clamp(min=0))
            row_budget_values = row_budget_values * active_row_valid_mask.to(dtype=row_budget_values.dtype)
            personal_row_budget_mean = row_budget_values.sum() / active_row_valid_mask.float().sum().clamp(min=1.0)
        else:
            row_budget_values = None
        if active_row_index is not None and active_row_valid_mask is not None:
            if query_row_mask is not None:
                query_row_active_mask = (
                    torch.gather(query_row_mask.bool(), 1, active_row_index.clamp(min=0))
                    & active_row_valid_mask
                )
            if neighbor_row_mask is not None:
                neighbor_row_active_mask = (
                    torch.gather(neighbor_row_mask.bool(), 1, active_row_index.clamp(min=0))
                    & active_row_valid_mask
                )
        support_row_cache = None
        if active_row_index is not None and active_row_valid_mask is not None:
            if not module.use_concept_graph:
                if module.personal_support_include_neighbors and local_row_mask is not None:
                    query_support_mask = local_row_mask.bool()
                elif query_row_mask is not None:
                    query_support_mask = query_row_mask.bool()
                elif active_source_mask is not None:
                    query_support_mask = active_source_mask.bool()
                else:
                    query_support_mask = torch.zeros(
                        (knowledge_state.size(0), module.num_concepts),
                        device=knowledge_state.device,
                        dtype=torch.bool,
                    )
                support_row_cache = _build_no_a_query_support_cache(
                    query_support_mask,
                    module.num_concepts,
                    include_neighbor_rows=module.personal_include_neighbor_rows,
                    local_hops=module.personal_query_support_hops,
                    num_heads=module.num_relation_heads,
                    active_row_index=active_row_index,
                    active_row_valid_mask=active_row_valid_mask,
                    dtype=knowledge_state.dtype,
                )
                used_no_a_support_fallback_mode = relation_matrices.new_tensor(1, dtype=torch.long)
            else:
                if module.personal_support_include_graph:
                    support_cache = _build_support_cache(
                        relation_matrices,
                        allow_full_support=not module.personal_support_only,
                        force_self_support=module.personal_support_include_query_self,
                    )
                    support_row_cache = _gather_row_support(support_cache, active_row_index, active_row_valid_mask)
                    support_row_cache = _augment_support_cache_with_item_concepts(
                        support_row_cache,
                        concept_mask,
                        active_row_index,
                        active_row_valid_mask,
                        item_support_mass=module.personal_item_support_mass,
                    )
                else:
                    if module.personal_support_include_neighbors and local_row_mask is not None:
                        query_support_mask = local_row_mask.bool()
                    elif query_row_mask is not None:
                        query_support_mask = query_row_mask.bool()
                    else:
                        query_support_mask = active_source_mask.bool() if active_source_mask is not None else None
                    support_row_cache = _build_no_a_query_support_cache(
                        query_support_mask,
                        module.num_concepts,
                        include_neighbor_rows=module.personal_support_include_neighbors,
                        local_hops=module.personal_query_support_hops,
                        num_heads=module.num_relation_heads,
                        active_row_index=active_row_index,
                        active_row_valid_mask=active_row_valid_mask,
                        dtype=knowledge_state.dtype,
                    )
                used_no_a_support_fallback_mode = relation_matrices.new_tensor(0, dtype=torch.long)
        if support_row_cache is not None:
            support_col_index = support_row_cache["support_col_index"]
            support_valid_mask = support_row_cache["support_valid_mask"]
            global_support_prob = support_row_cache["global_support_prob"]
            personal_item_support_added_rate = support_row_cache.get(
                "item_support_added_rate",
                relation_matrices.new_tensor(0.0),
            )
            personal_item_support_added_mass = support_row_cache.get(
                "item_support_added_mass",
                relation_matrices.new_tensor(0.0),
            )
        has_personal_active_rows = (
            support_row_cache is not None
            and active_row_valid_mask is not None
            and bool(active_row_valid_mask.any())
        )
        if has_personal_active_rows:
            gate_alpha_bias = None
            personal_warmup_scale = module._get_personal_warmup_scale()
            gate_alpha, gate_alpha_logit, gate_diag = module.adaptive_gate(
                gate_state_repr,
                context_repr,
                gate_id_embedding,
                id_adapter_scale=gate_id_scale,
                extra_bias=gate_alpha_bias,
                warmup_scale=personal_warmup_scale,
                return_diagnostics=True,
            )
            alpha_state_path_absmean = gate_diag["state_path_absmean"]
            alpha_id_path_absmean = gate_diag["id_path_absmean"]
            alpha_bias_path_absmean = gate_diag["bias_path_absmean"]
            head_bias_path_absmean = gate_diag["head_bias_path_absmean"]
            alpha_base_mean = gate_diag["alpha_base_mean"]
            alpha_delta_absmean = gate_diag["alpha_delta_absmean"]
            alpha_saturation_ratio = gate_diag["alpha_saturation_ratio"]
            alpha_id_adapter_scale = gate_diag["effective_id_scale"]
            state_embedding_nonfinite_count = gate_diag["state_embedding_nonfinite_count"]
            context_repr_nonfinite_count = gate_diag["context_repr_nonfinite_count"]
            state_logit_nonfinite_count = gate_diag["state_logit_nonfinite_count"]
            id_logit_nonfinite_count = gate_diag["id_logit_nonfinite_count"]
            alpha_base_nonfinite_count = gate_diag["alpha_base_nonfinite_count"]
            alpha_preclamp_nonfinite_count = gate_diag["alpha_preclamp_nonfinite_count"]
            gate_alpha_effective = gate_alpha
            personal_delta, personal_diag = module.personal_generator(
                generator_state_repr,
                context_repr,
                knowledge_state,
                student_id_embedding=generator_id_embedding,
                id_adapter_scale=generator_id_scale,
                active_row_index=active_row_index,
                active_row_valid_mask=active_row_valid_mask,
                support_row_cache=support_row_cache,
                row_budget_values=row_budget_values,
                student_concept_prior=student_concept_prior,
                student_concept_recent_prior=student_concept_recent_prior,
                return_diagnostics=True,
            )
            personal_state_mix = personal_diag["state_mix"]
            personal_student_mix = personal_diag["student_mix"]
            personal_mastery_prior_scale = personal_diag["mastery_prior_scale"]
            personal_mastery_scores_absmean = personal_diag["mastery_scores_absmean"]
            personal_recent_mastery_prior_scale = personal_diag["recent_mastery_prior_scale"]
            personal_recent_mastery_scores_absmean = personal_diag["recent_mastery_scores_absmean"]
            personal_student_adapter_scale = personal_diag["student_adapter_scale"]
            personal_context_adapter_scale = personal_diag["context_adapter_scale"]
            personal_delta_nonfinite_count = personal_diag["scores_nonfinite_count"]
            personal_delta_absmax = personal_diag["scores_absmax"]
            personal_delta = torch.nan_to_num(personal_delta, nan=0.0, posinf=1.0, neginf=-1.0)
            personal_support_density = (
                support_valid_mask.float().mean() if support_valid_mask is not None else relation_matrices.new_tensor(0.0)
            )
            posterior_delta = module.personal_delta_scale * personal_delta
            global_logprob = support_row_cache["global_support_logprob"] if support_row_cache is not None else None
            if global_logprob is not None:
                posterior_logits = torch.where(
                    support_valid_mask,
                    global_logprob + posterior_delta,
                    torch.full_like(posterior_delta, -30.0),
                )
                posterior_prob = _masked_support_softmax(
                    posterior_logits,
                    support_valid_mask,
                    fallback_prob=global_support_prob,
                    active_row_valid_mask=active_row_valid_mask,
                )
                personal_logits_nonfinite_count = posterior_logits.new_tensor(
                    int((~torch.isfinite(posterior_logits)).sum().item()), dtype=torch.long
                )
                personal_logits_absmax = torch.nan_to_num(
                    posterior_logits, nan=0.0, posinf=20.0, neginf=-20.0
                ).abs().max()
                personal_matrix_nonfinite_count = posterior_prob.new_tensor(
                    int((~torch.isfinite(posterior_prob)).sum().item()), dtype=torch.long
                )
                row_has_support = support_valid_mask.any(dim=-1)
                row_is_valid = active_row_valid_mask.unsqueeze(1).expand(-1, support_valid_mask.size(1), -1)
                bad_rows_all = row_is_valid & (~row_has_support)
                bad_rows_active = row_is_valid & (~row_has_support)
                padded_rows = ~row_is_valid
                active_row_count = row_is_valid.float().sum().clamp(min=1.0)
                personal_bad_row_count = posterior_prob.new_tensor(int(bad_rows_all.sum().item()), dtype=torch.long)
                personal_fallback_row_count = personal_bad_row_count.clone()
                personal_padded_row_count = posterior_prob.new_tensor(int(padded_rows.sum().item()), dtype=torch.long)
                personal_bad_row_count_active = posterior_prob.new_tensor(int(bad_rows_active.sum().item()), dtype=torch.long)
                personal_fallback_row_count_active = personal_bad_row_count_active.clone()
                personal_bad_row_rate_active = bad_rows_active.float().sum() / active_row_count
                personal_logits_support_absmax = _masked_absmax_or_zero(posterior_logits, support_valid_mask)
                personal_logits_masked_sentinel_absmax = _masked_absmax_or_zero(posterior_logits, ~support_valid_mask)

                valid_float = support_valid_mask.float()
                delta_denom = valid_float.sum(dim=(1, 2, 3)).clamp(min=1.0)
                personal_matrix_delta = (
                    (posterior_prob - global_support_prob).abs() * valid_float
                ).sum(dim=(1, 2, 3)) / delta_denom
                personal_matrix_student_std = posterior_prob.std(dim=0, unbiased=False).mean()
                personal_delta_pre_softmax_norm = _safe_zero_preserving_sqrt(
                    (posterior_delta.pow(2) * valid_float).sum() / valid_float.sum().clamp(min=1.0)
                )
                personal_delta_student_std = (posterior_delta * valid_float).std(dim=0, unbiased=False).mean()
                alpha_head_std = gate_alpha_effective.squeeze(-1).squeeze(-1).std(dim=1, unbiased=False).mean()

                if query_row_active_mask is not None:
                    query_mask_sparse = query_row_active_mask.float().unsqueeze(1).unsqueeze(-1) * valid_float
                    query_count = query_mask_sparse.sum(dim=(1, 2, 3)).clamp(min=1.0)
                    query_delta_abs_per_sample = (
                        posterior_delta.abs() * query_mask_sparse
                    ).sum(dim=(1, 2, 3)) / query_count
                    query_row_posterior_logit_delta_abs = query_delta_abs_per_sample.mean()
                    personal_query_row_std = query_delta_abs_per_sample.std(unbiased=False)
                    posterior_kl = posterior_prob.clamp(min=1e-8) * (
                        posterior_prob.clamp(min=1e-8).log()
                        - global_support_prob.clamp(min=1e-8).log()
                    )
                    query_row_posterior_kl = (posterior_kl * query_mask_sparse).sum(dim=(1, 2, 3)) / query_count
                    query_row_posterior_kl = query_row_posterior_kl.mean()
                if neighbor_row_active_mask is not None:
                    neighbor_mask_sparse = neighbor_row_active_mask.float().unsqueeze(1).unsqueeze(-1) * valid_float
                    neighbor_count = neighbor_mask_sparse.sum(dim=(1, 2, 3)).clamp(min=1.0)
                    neighbor_delta_per_sample = (
                        posterior_delta.abs() * neighbor_mask_sparse
                    ).sum(dim=(1, 2, 3)) / neighbor_count
                    neighbor_row_posterior_logit_delta_abs = neighbor_delta_per_sample.mean()

                relation_used = {
                    "global_matrices": relation_matrices,
                    "active_row_index": active_row_index,
                    "active_row_valid_mask": active_row_valid_mask,
                    "query_row_active_mask": query_row_active_mask,
                    "neighbor_row_active_mask": neighbor_row_active_mask,
                    "support_col_index": support_col_index,
                    "support_valid_mask": support_valid_mask,
                    "global_support_prob": global_support_prob,
                    "posterior_prob": posterior_prob,
                    "gate_alpha": gate_alpha_effective.squeeze(-1).squeeze(-1),
                }
                personal_relation_spec = relation_used
                personal_matrices = None

    def _support_diag_value(key: str) -> torch.Tensor:
        value = support_diagnostics.get(key)
        if value is None:
            return relation_matrices.new_tensor(0.0)
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=dtype)
        return relation_matrices.new_tensor(float(value))

    return {
        "relation_matrices": relation_matrices,
        "relation_used": relation_used,
        "personal_relation_spec": personal_relation_spec,
        "knowledge_state": knowledge_state,
        "student_repr": student_repr,
        "relation_identity_delta": relation_identity_delta,
        "knowledge_state_backbone_delta": knowledge_state_backbone_delta,
        "knowledge_state_graph_delta": knowledge_state_graph_delta,
        "knowledge_state_personal_delta": knowledge_state_personal_delta,
        "a_diag_semantic_ok": a_diag_semantic_ok,
        "support_candidate_size_mean": _support_diag_value("support_candidate_size_mean"),
        "support_final_size_mean": _support_diag_value("support_final_size_mean"),
        "support_item_survival_rate": _support_diag_value("support_item_survival_rate"),
        "support_seq_survival_rate": _support_diag_value("support_seq_survival_rate"),
        "support_item_seq_overlap": _support_diag_value("support_item_seq_overlap"),
        "support_self_retention_rate": _support_diag_value("support_self_retention_rate"),
        "alpha": gate_alpha,
        "alpha_logit": gate_alpha_logit,
        "alpha_student_bias": gate_alpha_bias,
        "alpha_effective": gate_alpha_effective,
        "alpha_state_path_absmean": alpha_state_path_absmean,
        "alpha_id_path_absmean": alpha_id_path_absmean,
        "alpha_bias_path_absmean": alpha_bias_path_absmean,
        "head_bias_path_absmean": head_bias_path_absmean,
        "alpha_base_mean": alpha_base_mean,
        "alpha_delta_absmean": alpha_delta_absmean,
        "alpha_saturation_ratio": alpha_saturation_ratio,
        "alpha_id_adapter_scale": alpha_id_adapter_scale,
        "state_embedding_nonfinite_count": state_embedding_nonfinite_count,
        "context_repr_nonfinite_count": context_repr_nonfinite_count,
        "state_logit_nonfinite_count": state_logit_nonfinite_count,
        "id_logit_nonfinite_count": id_logit_nonfinite_count,
        "alpha_base_nonfinite_count": alpha_base_nonfinite_count,
        "alpha_preclamp_nonfinite_count": alpha_preclamp_nonfinite_count,
        "personal_gate_id_scale": None,
        "personal_generator_id_scale": None,
        "personal_matrices": personal_matrices,
        "personal_matrix_delta": personal_matrix_delta,
        "personal_matrix_student_std": personal_matrix_student_std,
        "personal_delta_pre_softmax_norm": personal_delta_pre_softmax_norm,
        "personal_delta_student_std": personal_delta_student_std,
        "alpha_head_std": alpha_head_std,
        "query_row_posterior_logit_delta_abs": query_row_posterior_logit_delta_abs,
        "neighbor_row_posterior_logit_delta_abs": neighbor_row_posterior_logit_delta_abs,
        "query_row_posterior_delta_abs": query_row_posterior_delta_abs,
        "query_row_posterior_kl": query_row_posterior_kl,
        "personal_row_budget_mean": personal_row_budget_mean,
        "personal_query_row_std": personal_query_row_std,
        "personal_item_support_added_rate": personal_item_support_added_rate,
        "personal_item_support_added_mass": personal_item_support_added_mass,
        "used_no_a_support_fallback_mode": used_no_a_support_fallback_mode,
        "active_row_index": active_row_index,
        "active_row_valid_mask": active_row_valid_mask,
        "query_row_active_mask": query_row_active_mask,
        "neighbor_row_active_mask": neighbor_row_active_mask,
        "support_col_index": support_col_index,
        "support_valid_mask": support_valid_mask,
        "global_support_prob": global_support_prob,
        "posterior_prob": posterior_prob,
        "personal_state_mix": personal_state_mix,
        "personal_student_mix": personal_student_mix,
        "personal_mastery_prior_scale": personal_mastery_prior_scale,
        "personal_mastery_scores_absmean": personal_mastery_scores_absmean,
        "personal_recent_mastery_prior_scale": personal_recent_mastery_prior_scale,
        "personal_recent_mastery_scores_absmean": personal_recent_mastery_scores_absmean,
        "personal_student_adapter_scale": personal_student_adapter_scale,
        "personal_context_adapter_scale": personal_context_adapter_scale,
        "personal_delta_nonfinite_count": personal_delta_nonfinite_count,
        "personal_logits_nonfinite_count": personal_logits_nonfinite_count,
        "personal_matrix_nonfinite_count": personal_matrix_nonfinite_count,
        "personal_bad_row_count": personal_bad_row_count,
        "personal_fallback_row_count": personal_fallback_row_count,
        "personal_logits_absmax": personal_logits_absmax,
        "personal_padded_row_count": personal_padded_row_count,
        "personal_bad_row_count_active": personal_bad_row_count_active,
        "personal_fallback_row_count_active": personal_fallback_row_count_active,
        "personal_bad_row_rate_active": personal_bad_row_rate_active,
        "personal_logits_support_absmax": personal_logits_support_absmax,
        "personal_logits_masked_sentinel_absmax": personal_logits_masked_sentinel_absmax,
        "personal_delta_absmax": personal_delta_absmax,
        "local_row_ratio": local_row_ratio,
        "personal_support_density": personal_support_density,
        "query_state_norm": query_state_norm,
        "local_row_mask": local_row_mask,
        "personal_warmup_scale": torch.tensor(
            module._get_personal_warmup_scale(), device=device, dtype=dtype
        ),
        "personal_reg_warmup_scale": torch.tensor(
            module._get_personal_reg_warmup_scale(), device=device, dtype=dtype
        ),
        "personal_alpha_bias_scale": torch.tensor(
            module.personal_alpha_bias_scale, device=device, dtype=dtype
        ),
    }
