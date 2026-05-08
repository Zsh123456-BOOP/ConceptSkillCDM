
"""Sparse/local graph helper ops extracted from model.py."""

from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

def _pack_active_row_index(row_is_active: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if row_is_active is None:
        return None, None
    row_is_active = row_is_active.bool()
    B, C = row_is_active.shape
    counts = row_is_active.sum(dim=1)
    max_rows = int(counts.max().item()) if counts.numel() > 0 else 0
    device = row_is_active.device
    if max_rows <= 0:
        empty_index = torch.empty((B, 0), device=device, dtype=torch.long)
        empty_mask = torch.empty((B, 0), device=device, dtype=torch.bool)
        return empty_index, empty_mask

    active_row_index = torch.full((B, max_rows), -1, device=device, dtype=torch.long)
    active_row_valid_mask = torch.zeros((B, max_rows), device=device, dtype=torch.bool)
    for b in range(B):
        idx = torch.nonzero(row_is_active[b], as_tuple=False).reshape(-1)
        if idx.numel() == 0:
            continue
        active_row_index[b, : idx.numel()] = idx
        active_row_valid_mask[b, : idx.numel()] = True
    return active_row_index, active_row_valid_mask


def _build_support_cache(
    relation_matrices: torch.Tensor,
    *,
    allow_full_support: bool = False,
    force_self_support: bool = False,
) -> Dict[str, torch.Tensor]:
    H, C, _ = relation_matrices.shape
    device = relation_matrices.device
    dtype = relation_matrices.dtype

    if allow_full_support:
        support_col_index = torch.arange(C, device=device, dtype=torch.long).view(1, 1, C).expand(H, C, C)
        support_valid_mask = torch.ones((H, C, C), device=device, dtype=torch.bool)
        global_support_prob = torch.full((H, C, C), 1.0 / float(max(1, C)), device=device, dtype=dtype)
        return {
            "support_col_index": support_col_index,
            "support_valid_mask": support_valid_mask,
            "global_support_prob": global_support_prob,
            "global_support_logprob": global_support_prob.clamp(min=1e-8).log(),
        }

    support_counts = (relation_matrices > 0).sum(dim=-1)
    k_max = int(max(1, support_counts.max().item()))
    global_support_prob, support_col_index = torch.topk(relation_matrices, k=k_max, dim=-1)
    support_valid_mask = global_support_prob > 0

    if force_self_support:
        self_index = torch.arange(C, device=device, dtype=torch.long).view(1, C, 1).expand(H, -1, 1)
        self_prob = torch.diagonal(relation_matrices, dim1=-2, dim2=-1).unsqueeze(-1)
        self_prob = torch.maximum(self_prob, torch.full_like(self_prob, 0.05))
        has_self = (support_col_index == self_index).any(dim=-1)
        if not bool(has_self.all()):
            replace_slot = support_valid_mask.sum(dim=-1).clamp(min=1, max=k_max).long() - 1
            missing = ~has_self
            h_idx, c_idx = torch.nonzero(missing, as_tuple=True)
            slot_idx = replace_slot[h_idx, c_idx]
            support_col_index = support_col_index.clone()
            global_support_prob = global_support_prob.clone()
            support_valid_mask = support_valid_mask.clone()
            support_col_index[h_idx, c_idx, slot_idx] = self_index[h_idx, c_idx, 0]
            global_support_prob[h_idx, c_idx, slot_idx] = self_prob[h_idx, c_idx, 0]
            support_valid_mask[h_idx, c_idx, slot_idx] = True

    # Fallback: if a row becomes empty numerically, keep a self-loop support entry.
    row_has_support = support_valid_mask.any(dim=-1)
    if not bool(row_has_support.all()):
        missing = ~row_has_support
        missing_4d = missing.unsqueeze(-1)
        replacement_cols = torch.zeros_like(support_col_index)
        replacement_cols[..., 0] = torch.arange(C, device=device, dtype=torch.long).view(1, C).expand(H, -1)
        replacement_prob = torch.zeros_like(global_support_prob)
        replacement_prob[..., 0] = 1.0
        replacement_valid = torch.zeros_like(support_valid_mask)
        replacement_valid[..., 0] = True
        support_col_index = torch.where(missing_4d, replacement_cols, support_col_index)
        global_support_prob = torch.where(missing_4d, replacement_prob, global_support_prob)
        support_valid_mask = torch.where(missing_4d, replacement_valid, support_valid_mask)

    denom = global_support_prob.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    global_support_prob = global_support_prob / denom
    return {
        "support_col_index": support_col_index,
        "support_valid_mask": support_valid_mask,
        "global_support_prob": global_support_prob,
        "global_support_logprob": global_support_prob.clamp(min=1e-8).log(),
    }


def _build_no_a_query_support_cache(
    query_row_mask: torch.Tensor,
    num_concepts: int,
    include_neighbor_rows: bool = False,
    local_hops: int = 0,
    *,
    num_heads: int,
    active_row_index: Optional[torch.Tensor],
    active_row_valid_mask: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    del include_neighbor_rows  # no_A 下仅保留 query-local support，不再默默扩成 full-support
    device = query_row_mask.device
    query_row_mask = query_row_mask.bool()
    if active_row_index is None or active_row_valid_mask is None:
        active_row_index, active_row_valid_mask = _pack_active_row_index(query_row_mask)
    if active_row_index is None or active_row_valid_mask is None:
        empty_index = torch.empty((query_row_mask.size(0), num_heads, 0, 0), device=device, dtype=torch.long)
        empty_mask = torch.empty((query_row_mask.size(0), num_heads, 0, 0), device=device, dtype=torch.bool)
        empty_prob = torch.empty((query_row_mask.size(0), num_heads, 0, 0), device=device, dtype=dtype)
        return {
            "support_col_index": empty_index,
            "support_valid_mask": empty_mask,
            "global_support_prob": empty_prob,
            "global_support_logprob": empty_prob,
        }

    B = query_row_mask.size(0)
    R = active_row_index.size(1)
    if query_row_mask.numel() == 0:
        k_max = 1
    else:
        k_max = int(max(1, query_row_mask.sum(dim=1).max().item()))

    support_col_index = torch.zeros((B, num_heads, R, k_max), device=device, dtype=torch.long)
    support_valid_mask = torch.zeros((B, num_heads, R, k_max), device=device, dtype=torch.bool)
    global_support_prob = torch.zeros((B, num_heads, R, k_max), device=device, dtype=dtype)

    for b in range(B):
        query_cols = torch.nonzero(query_row_mask[b], as_tuple=False).reshape(-1)
        for r in range(R):
            if not bool(active_row_valid_mask[b, r]):
                continue
            row = int(active_row_index[b, r].item())
            if query_cols.numel() > 0:
                cols = query_cols
            else:
                cols = torch.tensor([row], device=device, dtype=torch.long)
            k = int(cols.numel())
            support_col_index[b, :, r, :k] = cols.view(1, -1).expand(num_heads, -1)
            support_valid_mask[b, :, r, :k] = True
            global_support_prob[b, :, r, :k] = 1.0 / float(max(1, k))

    global_support_logprob = torch.where(
        support_valid_mask,
        global_support_prob.clamp(min=1e-8).log(),
        torch.full_like(global_support_prob, -30.0),
    )
    return {
        "support_col_index": support_col_index,
        "support_valid_mask": support_valid_mask,
        "global_support_prob": global_support_prob,
        "global_support_logprob": global_support_logprob,
    }


def _gather_row_support(
    support_cache: Dict[str, torch.Tensor],
    active_row_index: torch.Tensor,
    active_row_valid_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    row_index = active_row_index.clamp(min=0)
    support_col_parts = []
    support_valid_parts = []
    global_prob_parts = []
    global_logprob_parts = []
    for h in range(support_cache["support_col_index"].size(0)):
        support_col_parts.append(support_cache["support_col_index"][h][row_index])
        support_valid_parts.append(support_cache["support_valid_mask"][h][row_index])
        global_prob_parts.append(support_cache["global_support_prob"][h][row_index])
        global_logprob_parts.append(support_cache["global_support_logprob"][h][row_index])

    support_col_index = torch.stack(support_col_parts, dim=1)
    support_valid_mask = torch.stack(support_valid_parts, dim=1)
    global_support_prob = torch.stack(global_prob_parts, dim=1)
    global_support_logprob = torch.stack(global_logprob_parts, dim=1)
    support_valid_mask = support_valid_mask & active_row_valid_mask.unsqueeze(1).unsqueeze(-1)
    global_support_prob = global_support_prob * support_valid_mask.to(dtype=global_support_prob.dtype)
    denom = global_support_prob.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    global_support_prob = torch.where(
        support_valid_mask.any(dim=-1, keepdim=True),
        global_support_prob / denom,
        global_support_prob,
    )
    return {
        "support_col_index": support_col_index,
        "support_valid_mask": support_valid_mask,
        "global_support_prob": global_support_prob,
        "global_support_logprob": torch.where(
            support_valid_mask,
            global_support_prob.clamp(min=1e-8).log(),
            torch.full_like(global_support_prob, -30.0),
        ),
    }


def _augment_support_cache_with_item_concepts(
    support_row_cache: Dict[str, torch.Tensor],
    concept_mask: Optional[torch.Tensor],
    active_row_index: Optional[torch.Tensor],
    active_row_valid_mask: Optional[torch.Tensor],
    *,
    item_support_mass: float,
) -> Dict[str, torch.Tensor]:
    """Add current-exercise concept columns to E's sparse support.

    This keeps E interpretable for multi-concept exercises: the posterior can
    reweight concepts that are jointly required by the same item, while
    single-concept items naturally reduce to the original self/global support.
    """
    mass = max(0.0, float(item_support_mass))
    if mass <= 0.0 or concept_mask is None:
        return support_row_cache
    if active_row_index is None or active_row_valid_mask is None or active_row_index.numel() == 0:
        return support_row_cache

    support_col_index = support_row_cache["support_col_index"]
    support_valid_mask = support_row_cache["support_valid_mask"].bool()
    global_support_prob = support_row_cache["global_support_prob"]
    if support_col_index.dim() != 4:
        return support_row_cache

    device = support_col_index.device
    dtype = global_support_prob.dtype
    concept_mask = concept_mask.to(device=device).bool()
    item_col_index, item_col_valid_mask = _pack_active_row_index(concept_mask)
    if item_col_index is None or item_col_valid_mask is None or item_col_index.numel() == 0:
        return support_row_cache

    B, H, R, _ = support_col_index.shape
    if item_col_index.size(0) != B:
        return support_row_cache

    Q = item_col_index.size(1)
    item_cols = item_col_index.clamp(min=0).view(B, 1, 1, Q).expand(B, H, R, Q)
    item_valid = item_col_valid_mask.view(B, 1, 1, Q).expand(B, H, R, Q)
    row_valid = active_row_valid_mask.to(device=device).bool().view(B, 1, R, 1)

    already_present = (
        (support_col_index.unsqueeze(-1) == item_cols.unsqueeze(-2))
        & support_valid_mask.unsqueeze(-1)
    ).any(dim=-2)
    append_valid = item_valid & row_valid & (~already_present)
    append_cols = torch.where(append_valid, item_cols, torch.zeros_like(item_cols))

    item_count = item_col_valid_mask.sum(dim=1).clamp(min=1).to(device=device, dtype=dtype).view(B, 1, 1, 1)
    append_prob = torch.where(
        append_valid,
        torch.full_like(append_cols, mass, dtype=dtype) / item_count,
        torch.zeros_like(append_cols, dtype=dtype),
    )

    old_prob = global_support_prob.to(dtype=dtype) * support_valid_mask.to(dtype=dtype)
    augmented_cols = torch.cat([support_col_index, append_cols], dim=-1)
    augmented_valid = torch.cat([support_valid_mask, append_valid], dim=-1)
    augmented_prob = torch.cat([old_prob, append_prob], dim=-1)
    augmented_prob = augmented_prob * augmented_valid.to(dtype=dtype)

    denom = augmented_prob.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    row_has_support = augmented_valid.any(dim=-1, keepdim=True)
    augmented_prob = torch.where(row_has_support, augmented_prob / denom, augmented_prob)
    augmented_logprob = torch.where(
        augmented_valid,
        augmented_prob.clamp(min=1e-8).log(),
        torch.full_like(augmented_prob, -30.0),
    )
    added_slots = torch.cat([torch.zeros_like(support_valid_mask), append_valid], dim=-1)
    active_denom = row_valid.expand(B, H, R, Q).float().sum().clamp(min=1.0)
    added_rate = append_valid.float().sum() / active_denom
    added_mass = (augmented_prob * added_slots.to(dtype=dtype)).sum(dim=-1)
    added_mass = added_mass.sum() / row_has_support.float().sum().clamp(min=1.0)

    out = dict(support_row_cache)
    out.update(
        {
            "support_col_index": augmented_cols,
            "support_valid_mask": augmented_valid,
            "global_support_prob": augmented_prob,
            "global_support_logprob": augmented_logprob,
            "item_support_added_mask": added_slots,
            "item_support_added_rate": added_rate.detach(),
            "item_support_added_mass": added_mass.detach(),
        }
    )
    return out


def _gather_head_rows(x: torch.Tensor, row_index: torch.Tensor, row_valid_mask: torch.Tensor) -> torch.Tensor:
    if row_index is None:
        return x.new_zeros((x.size(0), x.size(1), 0, x.size(-1)))
    if row_index.numel() == 0:
        return x.new_zeros((x.size(0), x.size(1), 0, x.size(-1)))
    idx = row_index.clamp(min=0).unsqueeze(1).unsqueeze(-1).expand(-1, x.size(1), -1, x.size(-1))
    gathered = torch.gather(x, 2, idx)
    return gathered * row_valid_mask.unsqueeze(1).unsqueeze(-1).to(dtype=gathered.dtype)


def _gather_head_support_features(x: torch.Tensor, support_col_index: torch.Tensor, support_valid_mask: torch.Tensor) -> torch.Tensor:
    B, H, _, D = x.shape
    if support_col_index.numel() == 0:
        return x.new_zeros((*support_col_index.shape, D))
    batch_idx = torch.arange(B, device=x.device, dtype=torch.long).view(B, 1, 1)
    outs = []
    for h in range(H):
        cols = support_col_index[:, h]
        gathered = x[:, h][batch_idx.expand_as(cols), cols]
        outs.append(gathered)
    out = torch.stack(outs, dim=1)
    return out * support_valid_mask.unsqueeze(-1).to(dtype=out.dtype)


def _normalize_sparse_scores(
    scores: torch.Tensor,
    support_valid_mask: torch.Tensor,
    row_budget: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mask = support_valid_mask.to(dtype=scores.dtype)
    scores = torch.nan_to_num(scores, nan=0.0, posinf=6.0, neginf=-6.0) * mask
    denom = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    mean = (scores * mask).sum(dim=-1, keepdim=True) / denom
    centered = (scores - mean) * mask
    row_var = ((centered.pow(2) * mask).sum(dim=-1, keepdim=True) / denom).clamp_min(1e-8)
    row_rms = row_var.sqrt()
    centered = centered / row_rms
    centered = torch.tanh(centered) * mask
    mean2 = (centered * mask).sum(dim=-1, keepdim=True) / denom
    centered = (centered - mean2) * mask
    if row_budget is not None:
        centered = centered * row_budget.unsqueeze(1).unsqueeze(-1).to(dtype=centered.dtype, device=centered.device)
    centered = torch.tanh(centered) * mask
    return torch.nan_to_num(centered, nan=0.0, posinf=1.0, neginf=-1.0)


def _masked_support_softmax(
    logits: torch.Tensor,
    support_valid_mask: torch.Tensor,
    fallback_prob: Optional[torch.Tensor] = None,
    active_row_valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    masked_logits = torch.where(
        support_valid_mask,
        logits,
        torch.full_like(logits, -30.0),
    )
    probs = F.softmax(masked_logits, dim=-1)
    probs = probs * support_valid_mask.to(dtype=probs.dtype)
    denom = probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    row_has_support = support_valid_mask.any(dim=-1)
    normalized = torch.where(row_has_support.unsqueeze(-1), probs / denom, torch.zeros_like(probs))
    if fallback_prob is None:
        fallback = support_valid_mask.to(dtype=probs.dtype)
        fallback = fallback / fallback.sum(dim=-1, keepdim=True).clamp(min=1.0)
    else:
        fallback = torch.nan_to_num(fallback_prob.to(dtype=probs.dtype), nan=0.0, posinf=0.0, neginf=0.0)
        fallback = fallback / fallback.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    if active_row_valid_mask is None:
        active_rows = torch.ones_like(row_has_support, dtype=torch.bool)
    else:
        active_rows = active_row_valid_mask.bool()
        if active_rows.dim() == 2:
            active_rows = active_rows.unsqueeze(1).expand(-1, logits.size(1), -1)
    missing_active = active_rows & (~row_has_support)
    out = torch.where(missing_active.unsqueeze(-1), fallback, normalized)
    return out * active_rows.unsqueeze(-1).to(dtype=out.dtype)


def _masked_sparse_row_entropy(probs: torch.Tensor, support_valid_mask: torch.Tensor) -> torch.Tensor:
    mask = support_valid_mask.to(dtype=probs.dtype)
    probs = probs.clamp(min=1e-12) * mask
    denom = mask.sum(dim=-1).clamp(min=1.0)
    row_entropy = -(probs * probs.clamp(min=1e-12).log()).sum(dim=-1)
    return (row_entropy * (denom > 0).to(dtype=probs.dtype)).sum() / (denom > 0).to(dtype=probs.dtype).sum().clamp(min=1.0)


def _masked_absmax_or_zero(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if tensor.numel() == 0 or mask.numel() == 0:
        return tensor.new_tensor(0.0)
    valid = mask.bool()
    if not bool(valid.any()):
        return tensor.new_tensor(0.0)
    values = torch.nan_to_num(
        tensor.masked_select(valid),
        nan=0.0,
        posinf=30.0,
        neginf=-30.0,
    )
    return values.abs().max() if values.numel() > 0 else tensor.new_tensor(0.0)


def _safe_zero_preserving_sqrt(tensor: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    safe = torch.sqrt(tensor.clamp(min=eps))
    return torch.where(tensor > 0, safe, torch.zeros_like(safe))


def _apply_sparse_local_posterior(
    states: torch.Tensor,
    relation_spec: Dict[str, torch.Tensor],
    *,
    reduce_heads: bool,
) -> torch.Tensor:
    global_matrices = relation_spec["global_matrices"]
    if states.dim() != 3:
        raise ValueError(f"Expected states shape (B,C,D), got {tuple(states.shape)}")
    B, C, D = states.shape
    H = global_matrices.size(0)
    expanded_states = states.unsqueeze(1).expand(-1, H, -1, -1)

    global_outs = []
    for h in range(H):
        A = global_matrices[h].to(dtype=states.dtype)
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-12)
        global_outs.append(torch.matmul(A, states))
    global_out = torch.stack(global_outs, dim=1)  # (B,H,C,D)

    local_messages = _compute_sparse_local_messages(expanded_states, relation_spec)
    active_row_index = local_messages["active_row_index"]
    active_row_valid_mask = local_messages["active_row_valid_mask"]
    if active_row_index is None or active_row_valid_mask is None or active_row_index.numel() == 0:
        return global_out.mean(dim=1) if reduce_heads else global_out

    mixed_local = local_messages["mixed_local"]
    out = global_out.clone()
    batch_idx = torch.arange(B, device=states.device, dtype=torch.long).unsqueeze(1).expand_as(active_row_index)
    valid = active_row_valid_mask
    row_index = active_row_index.clamp(min=0)
    for h in range(H):
        out[:, h][batch_idx[valid], row_index[valid]] = mixed_local[:, h][valid]
    return out.mean(dim=1) if reduce_heads else out


def _compute_sparse_local_messages(
    expanded_states: torch.Tensor,
    relation_spec: Dict[str, torch.Tensor],
    support_value_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    global_matrices = relation_spec["global_matrices"]
    active_row_index = relation_spec.get("active_row_index")
    active_row_valid_mask = relation_spec.get("active_row_valid_mask")
    if active_row_index is None or active_row_valid_mask is None or active_row_index.numel() == 0:
        empty_shape = (expanded_states.size(0), expanded_states.size(1), 0, expanded_states.size(-1))
        return {
            "active_row_index": active_row_index,
            "active_row_valid_mask": active_row_valid_mask,
            "query_row_active_mask": relation_spec.get("query_row_active_mask"),
            "neighbor_row_active_mask": relation_spec.get("neighbor_row_active_mask"),
            "global_local": expanded_states.new_zeros(empty_shape),
            "post_local": expanded_states.new_zeros(empty_shape),
            "delta_local_raw": expanded_states.new_zeros(empty_shape),
            "message_delta": expanded_states.new_zeros(empty_shape),
            "mixed_local": expanded_states.new_zeros(empty_shape),
        }

    support_col_index = relation_spec["support_col_index"]
    support_valid_mask = relation_spec["support_valid_mask"]
    global_support_prob = relation_spec["global_support_prob"]
    posterior_prob = relation_spec["posterior_prob"]
    gate_alpha = relation_spec["gate_alpha"]

    support_features = _gather_head_support_features(expanded_states, support_col_index, support_valid_mask)
    query_rows = _gather_head_rows(expanded_states, active_row_index, active_row_valid_mask)
    support_values = (
        support_value_fn(support_features, query_rows)
        if support_value_fn is not None
        else support_features
    )
    global_local = (global_support_prob.unsqueeze(-1) * support_values).sum(dim=-2)
    post_local = (posterior_prob.unsqueeze(-1) * support_values).sum(dim=-2)
    delta_local_raw = post_local - global_local
    message_delta = gate_alpha.unsqueeze(-1).unsqueeze(-1) * delta_local_raw
    mixed_local = global_local + message_delta
    return {
        "active_row_index": active_row_index,
        "active_row_valid_mask": active_row_valid_mask,
        "query_row_active_mask": relation_spec.get("query_row_active_mask"),
        "neighbor_row_active_mask": relation_spec.get("neighbor_row_active_mask"),
        "query_rows": query_rows,
        "support_values": support_values,
        "global_local": global_local,
        "post_local": post_local,
        "delta_local_raw": delta_local_raw,
        "message_delta": message_delta,
        "mixed_local": mixed_local,
    }
