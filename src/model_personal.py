"""Interpretable E module: query-local posterior reweighting."""

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model_ops import (
    _gather_head_rows,
    _gather_head_support_features,
    _normalize_sparse_scores,
)


class AdaptiveGate(nn.Module):
    """可解释的 E 混合系数 alpha。

    alpha 只由三项组成：
    - head_bias: 每个关系头的固定基线；
    - query_norm_weight: 当前题目相关概念状态强度；
    - context_norm_weight: 局部子图上下文强度。

    不再读取 student-id embedding，也不通过 MLP 产生 gate。
    """

    def __init__(
        self,
        student_dim: int,
        context_dim: int,
        num_heads: int,
        max_alpha: float = 0.35,
        hidden_dim: Optional[int] = None,
        max_id_adapter_scale: float = 0.0,
        alpha_temperature: float = 2.0,
        alpha_budget: float = 0.10,
        alpha_base_init: float = 0.08,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.max_alpha = float(max_alpha)
        self.alpha_temperature = max(1e-4, float(alpha_temperature))
        self.alpha_budget = max(0.0, float(alpha_budget))
        safe_base = min(max(float(alpha_base_init), 1e-4), max(self.max_alpha - 1e-4, 1e-4))
        base_ratio = min(max(safe_base / max(self.max_alpha, 1e-4), 1e-4), 1.0 - 1e-4)
        head_bias_init = math.log(base_ratio / (1.0 - base_ratio))
        self.head_bias = nn.Parameter(torch.full((self.num_heads,), head_bias_init))
        self.query_norm_weight = nn.Parameter(torch.zeros(self.num_heads))
        self.context_norm_weight = nn.Parameter(torch.zeros(self.num_heads))

    @staticmethod
    def _standardize_feature(value: torch.Tensor) -> torch.Tensor:
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        if value.numel() <= 1:
            return torch.zeros_like(value)
        return (value - value.mean()).div(value.std(unbiased=False).clamp(min=1e-6))

    def forward(
        self,
        state_embedding: torch.Tensor,
        context_repr: torch.Tensor,
        student_id_embedding: Optional[torch.Tensor] = None,
        id_adapter_scale: Optional[torch.Tensor] = None,
        extra_bias: Optional[torch.Tensor] = None,
        warmup_scale: float = 1.0,
        return_diagnostics: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        dtype = state_embedding.dtype
        device = state_embedding.device
        query_norm = self._standardize_feature(state_embedding.pow(2).mean(dim=-1).sqrt()).view(-1, 1)
        context_norm = self._standardize_feature(context_repr.pow(2).mean(dim=-1).sqrt()).view(-1, 1)
        base_logit = self.head_bias.to(device=device, dtype=dtype).view(1, self.num_heads)
        alpha_base = self.max_alpha * torch.sigmoid(base_logit)
        delta_logit = (
            query_norm * self.query_norm_weight.to(device=device, dtype=dtype).view(1, self.num_heads)
            + context_norm * self.context_norm_weight.to(device=device, dtype=dtype).view(1, self.num_heads)
        )
        if extra_bias is not None:
            delta_logit = delta_logit + extra_bias.to(device=device, dtype=dtype)
        warmup = float(max(0.0, min(1.0, warmup_scale)))
        alpha_delta = self.alpha_budget * torch.tanh(delta_logit / self.alpha_temperature)
        alpha_preclamp = alpha_base + warmup * alpha_delta
        alpha = alpha_preclamp.clamp(min=0.0, max=self.max_alpha)
        alpha4 = alpha.unsqueeze(-1).unsqueeze(-1)
        pre4 = alpha_preclamp.unsqueeze(-1).unsqueeze(-1)
        if not return_diagnostics:
            return alpha4, pre4

        diagnostics = {
            "state_logit": query_norm.expand(-1, self.num_heads),
            "id_logit": torch.zeros_like(alpha),
            "head_bias_logit": base_logit.expand_as(alpha),
            "alpha_bias_logit": torch.zeros_like(alpha) if extra_bias is None else extra_bias.to(device=device, dtype=dtype),
            "alpha_base": alpha_base.expand_as(alpha),
            "alpha_delta": alpha_delta,
            "state_path_absmean": (query_norm.abs().mean() * self.query_norm_weight.abs().mean()).detach(),
            "id_path_absmean": alpha.new_tensor(0.0),
            "bias_path_absmean": alpha.new_tensor(0.0) if extra_bias is None else extra_bias.abs().mean(),
            "head_bias_path_absmean": base_logit.abs().mean(),
            "alpha_base_mean": alpha_base.mean(),
            "alpha_delta_absmean": alpha_delta.abs().mean(),
            "alpha_saturation_ratio": (alpha >= max(self.max_alpha - 1e-4, 0.0)).float().mean(),
            "effective_id_scale": alpha.new_tensor(0.0),
            "state_embedding_nonfinite_count": state_embedding.new_tensor(
                int((~torch.isfinite(state_embedding)).sum().item()), dtype=torch.long
            ),
            "context_repr_nonfinite_count": context_repr.new_tensor(
                int((~torch.isfinite(context_repr)).sum().item()), dtype=torch.long
            ),
            "state_logit_nonfinite_count": alpha.new_tensor(int((~torch.isfinite(delta_logit)).sum().item()), dtype=torch.long),
            "id_logit_nonfinite_count": alpha.new_tensor(0, dtype=torch.long),
            "alpha_base_nonfinite_count": alpha.new_tensor(int((~torch.isfinite(alpha_base)).sum().item()), dtype=torch.long),
            "alpha_preclamp_nonfinite_count": alpha.new_tensor(int((~torch.isfinite(alpha_preclamp)).sum().item()), dtype=torch.long),
            "alpha_logit_nonfinite_count": alpha.new_tensor(int((~torch.isfinite(alpha_preclamp)).sum().item()), dtype=torch.long),
        }
        return alpha4, pre4, diagnostics


class PersonalRelationGenerator(nn.Module):
    """可解释的 E 后验生成器。

    对每个 query concept row，仅在 A 提供的 support 上重加权：
    residual(c -> k) = normalized_contrast(1 - cosine(state_c, state_k))
        + gamma * normalized_contrast(train_mastery_u,k - train_mastery_u,c)
        + eta * normalized_contrast(train_recent_mastery_u,k - train_recent_mastery_u,c)

    因此 E 的含义是“根据当前学生在题目局部概念上的状态差异，以及 train-only
    学生-概念掌握度对比和近期掌握度对比，调整 A 的邻接支持”。它不生成新边，
    也不读取 student-id embedding 作为 shortcut。
    """

    def __init__(
        self,
        student_dim: int,
        context_dim: int,
        knowledge_dim: int,
        num_concepts: int,
        num_heads: int,
        rank: int = 4,
        hidden_dim: Optional[int] = None,
        max_state_adapter_scale: float = 0.0,
        max_context_adapter_scale: float = 0.0,
        max_id_adapter_scale: float = 0.0,
        mastery_prior_scale: float = 0.0,
        recent_mastery_prior_scale: float = 0.0,
    ):
        super().__init__()
        self.num_concepts = int(num_concepts)
        self.num_heads = int(num_heads)
        self.rank = int(rank)
        self.mastery_prior_scale = max(0.0, float(mastery_prior_scale))
        self.recent_mastery_prior_scale = max(0.0, float(recent_mastery_prior_scale))
        self.max_state_mix = 1.20
        self.max_student_mix = 0.0
        self.state_mix_logit = nn.Parameter(torch.tensor(1.0986123))
        self.student_mix_logit = nn.Parameter(torch.tensor(-20.0), requires_grad=False)

    def _build_student_concept_prior_scores(
        self,
        *,
        prior: torch.Tensor,
        knowledge_state: torch.Tensor,
        active_row_index: torch.Tensor,
        support_row_cache: Dict[str, torch.Tensor],
        row_budget_values: Optional[torch.Tensor],
    ) -> torch.Tensor:
        mastery = torch.tanh(prior.to(device=knowledge_state.device, dtype=knowledge_state.dtype) / 2.0)
        B, H, R, K = support_row_cache["support_col_index"].shape
        row_prior = torch.gather(mastery, 1, active_row_index.clamp(min=0))
        row_prior = row_prior.view(B, 1, R, 1).expand(B, H, R, K)
        support_cols = support_row_cache["support_col_index"].clamp(min=0)
        mastery_heads = mastery.unsqueeze(1).expand(B, H, self.num_concepts)
        support_prior = torch.gather(
            mastery_heads,
            2,
            support_cols.reshape(B, H, R * K),
        ).reshape(B, H, R, K)
        mastery_advantage = support_prior - row_prior
        return _normalize_sparse_scores(
            mastery_advantage,
            support_row_cache["support_valid_mask"],
            row_budget=row_budget_values,
        )

    def forward(
        self,
        student_state_embedding: torch.Tensor,
        context_repr: torch.Tensor,
        knowledge_state: torch.Tensor,
        student_id_embedding: Optional[torch.Tensor] = None,
        id_adapter_scale: Optional[torch.Tensor] = None,
        active_row_index: Optional[torch.Tensor] = None,
        active_row_valid_mask: Optional[torch.Tensor] = None,
        support_row_cache: Optional[Dict[str, torch.Tensor]] = None,
        row_budget_values: Optional[torch.Tensor] = None,
        student_concept_prior: Optional[torch.Tensor] = None,
        student_concept_recent_prior: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if support_row_cache is None:
            raise ValueError("support_row_cache is required for PersonalRelationGenerator.forward")
        if active_row_index is None or active_row_valid_mask is None:
            raise ValueError("active rows are required for interpretable E")

        centered = knowledge_state - knowledge_state.mean(dim=1, keepdim=True)
        state = F.normalize(centered, dim=-1, eps=1e-6)
        state_heads = state.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        row_state = _gather_head_rows(state_heads, active_row_index, active_row_valid_mask)
        support_state = _gather_head_support_features(
            state_heads,
            support_row_cache["support_col_index"],
            support_row_cache["support_valid_mask"],
        )
        cosine = (row_state.unsqueeze(-2) * support_state).sum(dim=-1)
        cosine = torch.nan_to_num(cosine, nan=0.0, posinf=1.0, neginf=-1.0).clamp(min=-1.0, max=1.0)
        contrast = 1.0 - cosine
        residual = _normalize_sparse_scores(
            contrast,
            support_row_cache["support_valid_mask"],
            row_budget=row_budget_values,
        )
        mastery_scores = torch.zeros_like(residual)
        if (
            self.mastery_prior_scale > 0.0
            and student_concept_prior is not None
            and student_concept_prior.dim() == 2
            and student_concept_prior.size(1) == self.num_concepts
        ):
            mastery_scores = self._build_student_concept_prior_scores(
                prior=student_concept_prior,
                knowledge_state=knowledge_state,
                active_row_index=active_row_index,
                support_row_cache=support_row_cache,
                row_budget_values=row_budget_values,
            )
            residual = residual + self.mastery_prior_scale * mastery_scores
        recent_mastery_scores = torch.zeros_like(residual)
        if (
            self.recent_mastery_prior_scale > 0.0
            and student_concept_recent_prior is not None
            and student_concept_recent_prior.dim() == 2
            and student_concept_recent_prior.size(1) == self.num_concepts
        ):
            recent_mastery_scores = self._build_student_concept_prior_scores(
                prior=student_concept_recent_prior,
                knowledge_state=knowledge_state,
                active_row_index=active_row_index,
                support_row_cache=support_row_cache,
                row_budget_values=row_budget_values,
            )
            residual = residual + self.recent_mastery_prior_scale * recent_mastery_scores
        state_mix = self.max_state_mix * torch.sigmoid(self.state_mix_logit)
        scores = state_mix * residual
        scores = torch.tanh(scores) * support_row_cache["support_valid_mask"].to(dtype=scores.dtype)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=-1.0)
        if not return_diagnostics:
            return scores

        valid = support_row_cache["support_valid_mask"].to(dtype=scores.dtype)
        active_count = active_row_valid_mask.float().sum(dim=1).mean()
        diagnostics = {
            "state_mix": scores.new_tensor(float(state_mix.item())),
            "student_mix": scores.new_tensor(0.0),
            "student_adapter_scale": scores.new_tensor(0.0),
            "state_adapter_scale": scores.new_tensor(0.0),
            "id_adapter_scale": scores.new_tensor(0.0),
            "context_adapter_scale": scores.new_tensor(0.0),
            "state_scores_absmean": (scores.abs() * valid).sum() / valid.sum().clamp(min=1.0),
            "mastery_prior_scale": scores.new_tensor(self.mastery_prior_scale),
            "mastery_scores_absmean": (mastery_scores.abs() * valid).sum() / valid.sum().clamp(min=1.0),
            "recent_mastery_prior_scale": scores.new_tensor(self.recent_mastery_prior_scale),
            "recent_mastery_scores_absmean": (recent_mastery_scores.abs() * valid).sum() / valid.sum().clamp(min=1.0),
            "student_scores_absmean": scores.new_tensor(0.0),
            "scores_absmax": scores.abs().max(),
            "active_row_count_mean": active_count,
            "scores_nonfinite_count": scores.new_tensor(int((~torch.isfinite(scores)).sum().item()), dtype=torch.long),
        }
        return scores, diagnostics
