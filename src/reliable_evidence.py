"""Train-only reliable evidence builders for the A/E redesign probes.

The helpers in this module deliberately avoid model parameters.  They construct
interpretable statistics from the training split only:

- response transition priors split by correct vs. wrong destination response;
- student ability quantile groups;
- group-concept relative mastery logits for sparse student fallback.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from src.dataset import _parse_concept_seq
from src.trainer import _safe_logit


def _ordered_train_df(train_df: pd.DataFrame) -> pd.DataFrame:
    ordered = train_df.copy()
    ordered["_source_order"] = np.arange(len(ordered))
    order_cols = [
        col
        for col in ("timestamp", "time", "order_id", "original_row_id")
        if col in ordered.columns
    ]
    return ordered.sort_values(["stu_id", *order_cols, "_source_order"], kind="mergesort")


def _row_normalize_observed(counts: torch.Tensor) -> torch.Tensor:
    counts = counts.detach().float().clamp(min=0.0)
    if counts.dim() != 2 or counts.size(0) != counts.size(1):
        raise ValueError(f"counts must be square, got {tuple(counts.shape)}")
    c = int(counts.size(0))
    if c == 1:
        return torch.ones(1, 1, dtype=torch.float32)
    eye = torch.eye(c, dtype=counts.dtype, device=counts.device)
    counts = counts * (1.0 - eye)
    row_sum = counts.sum(dim=1, keepdim=True)
    return torch.where(row_sum > 0, counts / row_sum.clamp(min=1e-12), torch.zeros_like(counts))


def build_response_transition_priors(
    train_df: pd.DataFrame,
    cpt_id_map: Dict[int, int],
    *,
    max_hops: int = 3,
    decay: float = 0.70,
    student_reliability_lambda: float = 5.0,
) -> Dict[str, torch.Tensor | Dict[str, float]]:
    """Build incoming concept transition priors split by destination label.

    Row semantics match the existing graph encoder: ``prior[c, k]`` means
    concept ``c`` aggregates evidence from support concept ``k``.  For a
    student sequence ``k -> c``, evidence is therefore written to row ``c``.

    The split is based on the destination response label.  Correct destination
    interactions contribute to ``right_prior``; wrong destinations contribute to
    ``wrong_prior``.  Each student's transition counts are normalized before
    contributing to the global matrix, with a reliability weight so extremely
    short histories do not dominate.
    """
    c = len(cpt_id_map)
    if c <= 0:
        raise ValueError("cpt_id_map must not be empty")
    if c == 1:
        one = torch.ones(1, 1, dtype=torch.float32)
        return {
            "right_prior": one,
            "wrong_prior": one,
            "right_count": torch.zeros(1, 1, dtype=torch.float32),
            "wrong_count": torch.zeros(1, 1, dtype=torch.float32),
            "row_reliability": torch.ones(1, dtype=torch.float32),
            "stats": {
                "response_raw_transition_mass": 0.0,
                "response_weighted_mass": 0.0,
                "response_student_count": 0.0,
                "response_right_edge_count": 0.0,
                "response_wrong_edge_count": 0.0,
            },
        }

    max_hops = max(1, int(max_hops))
    decay = max(0.0, min(1.0, float(decay)))
    reliability_lambda = max(0.0, float(student_reliability_lambda))
    right_count = torch.zeros(c, c, dtype=torch.float32)
    wrong_count = torch.zeros(c, c, dtype=torch.float32)
    raw_mass = 0.0
    weighted_mass = 0.0
    student_count = 0.0

    required = {"stu_id", "cpt_seq", "label"}
    if not required.issubset(set(train_df.columns)):
        missing = sorted(required - set(train_df.columns))
        raise ValueError(f"train_df missing required columns: {missing}")

    ordered = _ordered_train_df(train_df)
    for _, stu_df in ordered.groupby("stu_id", sort=False):
        history: List[tuple[List[int], float]] = []
        for seq, label_value in zip(stu_df["cpt_seq"].values, stu_df["label"].values):
            concepts = sorted({cpt_id_map[cid] for cid in _parse_concept_seq(seq) if cid in cpt_id_map})
            if not concepts:
                continue
            history.append((concepts, float(label_value)))
        if len(history) < 2:
            continue

        stu_right = torch.zeros_like(right_count)
        stu_wrong = torch.zeros_like(wrong_count)
        stu_total = 0.0
        for idx, (src_concepts, _) in enumerate(history):
            for hop in range(1, max_hops + 1):
                dst_idx = idx + hop
                if dst_idx >= len(history):
                    break
                dst_concepts, dst_label = history[dst_idx]
                weight = (decay ** (hop - 1)) / float(len(src_concepts) * len(dst_concepts))
                target = stu_right if dst_label >= 0.5 else stu_wrong
                for src_c in src_concepts:
                    for dst_c in dst_concepts:
                        if src_c == dst_c:
                            continue
                        target[dst_c, src_c] += weight
                        stu_total += weight
        if stu_total <= 0.0:
            continue
        reliability = (
            stu_total / (stu_total + reliability_lambda)
            if reliability_lambda > 0.0
            else 1.0
        )
        right_count += reliability * (stu_right / float(stu_total))
        wrong_count += reliability * (stu_wrong / float(stu_total))
        raw_mass += stu_total
        weighted_mass += reliability
        student_count += 1.0

    total_count = right_count + wrong_count
    row_mass = total_count.sum(dim=1)
    row_reliability = row_mass / (row_mass + reliability_lambda) if reliability_lambda > 0.0 else (row_mass > 0).float()
    return {
        "right_prior": _row_normalize_observed(right_count),
        "wrong_prior": _row_normalize_observed(wrong_count),
        "right_count": right_count,
        "wrong_count": wrong_count,
        "row_reliability": row_reliability.clamp(min=0.0, max=1.0).to(dtype=torch.float32),
        "stats": {
            "response_raw_transition_mass": float(raw_mass),
            "response_weighted_mass": float(weighted_mass),
            "response_student_count": float(student_count),
            "response_right_edge_count": float((right_count > 0).float().sum().item()),
            "response_wrong_edge_count": float((wrong_count > 0).float().sum().item()),
        },
    }


def assign_student_quantile_groups(student_logits: torch.Tensor, num_groups: int = 5) -> torch.Tensor:
    """Assign students to deterministic ability quantile groups."""
    logits = student_logits.detach().float().cpu().reshape(-1)
    n = int(logits.numel())
    groups = max(1, int(num_groups))
    if n == 0:
        return torch.zeros(0, dtype=torch.long)
    order = torch.argsort(logits, stable=True)
    ranks = torch.empty(n, dtype=torch.long)
    ranks[order] = torch.arange(n, dtype=torch.long)
    group_ids = torch.div(ranks * groups, max(n, 1), rounding_mode="floor").clamp(max=groups - 1)
    return group_ids.to(dtype=torch.long)


def build_group_concept_logits(
    *,
    student_ids: torch.Tensor,
    exercise_ids: torch.Tensor,
    labels: torch.Tensor,
    q_matrix: torch.Tensor,
    student_group_ids: torch.Tensor,
    num_groups: int,
    concept_rate: torch.Tensor,
    smoothing: float = 4.0,
) -> torch.Tensor:
    """Build group-concept mastery logits relative to train concept base rates."""
    groups = max(1, int(num_groups))
    q = q_matrix.detach().float().cpu()
    c = int(q.size(1))
    group_ids = student_group_ids.detach().long().cpu()
    concept_rate = concept_rate.detach().float().cpu().clamp(min=1e-4, max=1.0 - 1e-4)
    counts = torch.zeros(groups, c, dtype=torch.float32)
    correct = torch.zeros(groups, c, dtype=torch.float32)

    for sid_t, eid_t, label_t in zip(student_ids.long().cpu(), exercise_ids.long().cpu(), labels.float().cpu()):
        sid = int(sid_t.item())
        eid = int(eid_t.item())
        if sid < 0 or sid >= group_ids.numel() or eid < 0 or eid >= q.size(0):
            continue
        gid = int(group_ids[sid].item())
        if gid < 0 or gid >= groups:
            continue
        active = q[eid]
        if active.sum().item() <= 0.0:
            continue
        counts[gid] += active
        correct[gid] += active * float(label_t.item())

    smooth = max(0.0, float(smoothing))
    rate = (correct + smooth * concept_rate.view(1, -1)) / (counts + smooth).clamp(min=1e-6)
    logits = (_safe_logit(rate) - _safe_logit(concept_rate).view(1, -1)).clamp(min=-2.5, max=2.5)
    return torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=torch.float32)
