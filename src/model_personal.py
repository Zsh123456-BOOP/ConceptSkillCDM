
"""Personal graph gate and posterior generator modules."""

import math
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model_ops import (
    _gather_head_rows,
    _gather_head_support_features,
    _normalize_sparse_scores,
    _pack_active_row_index,
)

# 模块 1（E）：个性化图相关组件 - AdaptiveGate / PersonalRelationGenerator
# ======================================================

class AdaptiveGate(nn.Module):
    """个性化图混合系数 alpha（B,H,1,1）。

    设计原则：
    - 主信号来自 state/context，不再保留 raw student bias -> alpha 的短路；
    - student-id 只保留极小 adapter；
    - 通过固定 head bias 控制 alpha 基线，而不是让 student embedding 直接主导。
    """

    def __init__(
        self,
        student_dim: int,
        context_dim: int,
        num_heads: int,
        max_alpha: float = 0.35,
        hidden_dim: Optional[int] = None,
        max_id_adapter_scale: float = 0.03,
        alpha_temperature: float = 2.0,
        alpha_budget: float = 0.10,
        alpha_base_init: float = 0.08,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        hid = max(1, max(student_dim, context_dim) // 2 if hidden_dim is None else int(hidden_dim))
        self.state_norm = nn.LayerNorm(student_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.state_proj = nn.Linear(student_dim, hid, bias=False)
        self.context_proj = nn.Linear(context_dim, hid)
        self.fusion_proj = nn.Linear(hid, hid)
        self.state_to_logit = nn.Linear(student_dim, self.num_heads, bias=False)
        self.context_to_logit = nn.Linear(context_dim, self.num_heads)
        self.id_to_logit = nn.Linear(student_dim, self.num_heads, bias=False)
        self.out = nn.Linear(hid, self.num_heads)
        self.max_alpha = float(max_alpha)
        self.max_id_adapter_scale = max(0.0, float(max_id_adapter_scale))
        self.alpha_temperature = max(1e-4, float(alpha_temperature))
        self.alpha_budget = max(0.0, float(alpha_budget))
        self.id_adapter_logit = nn.Parameter(torch.tensor(-2.9444390))
        safe_base = min(max(float(alpha_base_init), 1e-4), max(self.max_alpha - 1e-4, 1e-4))
        base_ratio = safe_base / max(self.max_alpha, 1e-4)
        base_ratio = min(max(base_ratio, 1e-4), 1.0 - 1e-4)
        head_bias_init = math.log(base_ratio / (1.0 - base_ratio))
        self.head_bias = nn.Parameter(torch.full((self.num_heads,), head_bias_init))

        nn.init.xavier_normal_(self.state_proj.weight)
        nn.init.xavier_normal_(self.context_proj.weight)
        nn.init.zeros_(self.context_proj.bias)
        nn.init.xavier_normal_(self.fusion_proj.weight)
        nn.init.zeros_(self.fusion_proj.bias)
        nn.init.xavier_normal_(self.state_to_logit.weight, gain=0.7)
        nn.init.xavier_normal_(self.context_to_logit.weight, gain=0.35)
        nn.init.zeros_(self.context_to_logit.bias)
        nn.init.xavier_normal_(self.id_to_logit.weight, gain=0.15)
        nn.init.xavier_normal_(self.out.weight, gain=0.5)
        nn.init.constant_(self.out.bias, -1.0)

    def forward(
        self,
        state_embedding: torch.Tensor,
        context_repr: torch.Tensor,
        student_id_embedding: torch.Tensor,
        id_adapter_scale: Optional[torch.Tensor] = None,
        extra_bias: Optional[torch.Tensor] = None,
        warmup_scale: float = 1.0,
        return_diagnostics: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        state_norm = self.state_norm(state_embedding)
        context_norm = self.context_norm(context_repr)
        context_hidden = self.context_proj(context_norm)
        state_hidden = self.state_proj(state_norm)
        hidden = F.silu(context_hidden + state_hidden)
        hidden = F.silu(self.fusion_proj(hidden))

        state_logit = (
            self.state_to_logit(state_norm)
            + self.context_to_logit(context_norm)
            + self.out(hidden)
        )
        effective_id_scale = self.max_id_adapter_scale * torch.sigmoid(self.id_adapter_logit)
        if id_adapter_scale is not None:
            effective_id_scale = effective_id_scale * id_adapter_scale.to(
                device=state_embedding.device,
                dtype=state_embedding.dtype,
            )
        id_logit = effective_id_scale * torch.tanh(self.id_to_logit(student_id_embedding))
        head_bias_logit = self.head_bias.view(1, self.num_heads).to(
            device=state_embedding.device,
            dtype=state_embedding.dtype,
        )
        if extra_bias is None:
            alpha_bias_logit = torch.zeros_like(state_logit)
        else:
            alpha_bias_logit = extra_bias.to(
                device=state_embedding.device,
                dtype=state_embedding.dtype,
            )
        warmup = float(max(0.0, min(1.0, warmup_scale)))
        alpha_base = self.max_alpha * torch.sigmoid(head_bias_logit + alpha_bias_logit)
        delta_input = (state_logit + id_logit) / self.alpha_temperature
        alpha_delta = self.alpha_budget * torch.tanh(delta_input)
        alpha = torch.clamp(
            alpha_base + warmup * alpha_delta,
            min=0.0,
            max=self.max_alpha,
        )
        alpha_preclamp = alpha_base + warmup * alpha_delta
        if not return_diagnostics:
            return alpha.unsqueeze(-1).unsqueeze(-1), alpha_preclamp.unsqueeze(-1).unsqueeze(-1)

        saturation_margin = max(1e-4, 0.05 * max(self.max_alpha, 1e-4))
        saturation_ratio = (alpha >= max(self.max_alpha - saturation_margin, saturation_margin)).float().mean()
        diagnostics = {
            "state_logit": state_logit,
            "id_logit": id_logit,
            "head_bias_logit": head_bias_logit,
            "alpha_bias_logit": alpha_bias_logit,
            "alpha_base": alpha_base,
            "alpha_delta": alpha_delta,
            "state_path_absmean": state_logit.abs().mean(),
            "id_path_absmean": id_logit.abs().mean(),
            "bias_path_absmean": alpha_bias_logit.abs().mean(),
            "head_bias_path_absmean": head_bias_logit.abs().mean(),
            "alpha_base_mean": alpha_base.mean(),
            "alpha_delta_absmean": alpha_delta.abs().mean(),
            "alpha_saturation_ratio": saturation_ratio,
            "effective_id_scale": alpha.new_tensor(float(torch.as_tensor(effective_id_scale).item())),
            "state_embedding_nonfinite_count": state_embedding.new_tensor(
                int((~torch.isfinite(state_embedding)).sum().item()), dtype=torch.long
            ),
            "context_repr_nonfinite_count": context_repr.new_tensor(
                int((~torch.isfinite(context_repr)).sum().item()), dtype=torch.long
            ),
            "state_logit_nonfinite_count": state_logit.new_tensor(
                int((~torch.isfinite(state_logit)).sum().item()), dtype=torch.long
            ),
            "id_logit_nonfinite_count": id_logit.new_tensor(
                int((~torch.isfinite(id_logit)).sum().item()), dtype=torch.long
            ),
            "alpha_base_nonfinite_count": alpha_base.new_tensor(
                int((~torch.isfinite(alpha_base)).sum().item()), dtype=torch.long
            ),
            "alpha_preclamp_nonfinite_count": alpha_preclamp.new_tensor(
                int((~torch.isfinite(alpha_preclamp)).sum().item()), dtype=torch.long
            ),
            "alpha_logit_nonfinite_count": alpha_preclamp.new_tensor(
                int((~torch.isfinite(alpha_preclamp)).sum().item()), dtype=torch.long
            ),
        }
        return (
            alpha.unsqueeze(-1).unsqueeze(-1),
            alpha_preclamp.unsqueeze(-1).unsqueeze(-1),
            diagnostics,
        )


class PersonalRelationGenerator(nn.Module):
    """state-primary 的 per-head 个性化邻接 residual 生成器。

    设计目标：
    - 主信号来自 knowledge_state 的逐概念差异，而不是 student-id 的全局 shortcut；
    - student/context 仅作为低秩 adapter 注入到 state query/key；
    - 输出的是有界 residual logits，后续在 A 的 support 上做 posterior reweighting。
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
        max_state_adapter_scale: float = 0.08,
        max_context_adapter_scale: float = 0.12,
        max_id_adapter_scale: float = 0.04,
    ):
        super().__init__()
        self.student_norm = nn.LayerNorm(student_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.state_norm = nn.LayerNorm(knowledge_dim)
        self.num_concepts = int(num_concepts)
        self.num_heads = int(num_heads)
        self.rank = int(rank)
        self.max_state_mix = 1.20
        self.max_student_mix = 0.10
        self.max_state_adapter_scale = max(0.0, float(max_state_adapter_scale))
        self.max_context_adapter_scale = max(0.0, float(max_context_adapter_scale))
        self.max_id_adapter_scale = max(0.0, float(max_id_adapter_scale))
        hidden = max(1, max(student_dim, context_dim) // 2 if hidden_dim is None else int(hidden_dim))

        self.context_proj = nn.Linear(context_dim, hidden)
        self.hidden_proj = nn.Linear(hidden, hidden)
        self.state_query_proj = nn.Linear(knowledge_dim, self.num_heads * self.rank, bias=False)
        self.state_key_proj = nn.Linear(knowledge_dim, self.num_heads * self.rank, bias=False)
        self.context_to_u = nn.Linear(hidden, self.num_heads * num_concepts * rank, bias=False)
        self.context_to_v = nn.Linear(hidden, self.num_heads * num_concepts * rank, bias=False)
        self.state_to_u = nn.Linear(student_dim, self.num_heads * num_concepts * rank, bias=False)
        self.state_to_v = nn.Linear(student_dim, self.num_heads * num_concepts * rank, bias=False)
        self.id_to_u = nn.Linear(student_dim, self.num_heads * num_concepts * rank, bias=False)
        self.id_to_v = nn.Linear(student_dim, self.num_heads * num_concepts * rank, bias=False)
        self.state_adapter_logit = nn.Parameter(torch.tensor(-1.7346010))
        self.context_adapter_logit = nn.Parameter(torch.tensor(-2.1972246))
        self.id_adapter_logit = nn.Parameter(torch.tensor(-2.9444390))
        self.state_mix_logit = nn.Parameter(torch.tensor(1.0986123))
        self.student_mix_logit = nn.Parameter(torch.tensor(-2.9444390))

        nn.init.xavier_normal_(self.context_proj.weight)
        nn.init.zeros_(self.context_proj.bias)
        nn.init.xavier_normal_(self.hidden_proj.weight)
        nn.init.zeros_(self.hidden_proj.bias)
        nn.init.xavier_normal_(self.state_query_proj.weight, gain=0.8)
        nn.init.xavier_normal_(self.state_key_proj.weight, gain=0.8)
        nn.init.xavier_normal_(self.context_to_u.weight)
        nn.init.xavier_normal_(self.context_to_v.weight)
        nn.init.xavier_normal_(self.state_to_u.weight, gain=0.3)
        nn.init.xavier_normal_(self.state_to_v.weight, gain=0.3)
        nn.init.xavier_normal_(self.id_to_u.weight, gain=0.15)
        nn.init.xavier_normal_(self.id_to_v.weight, gain=0.15)

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
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        student_code = torch.tanh(self.student_norm(student_state_embedding))
        if student_id_embedding is None:
            student_id_code = torch.zeros_like(student_code)
        else:
            student_id_code = torch.tanh(self.student_norm(student_id_embedding))
        state_residual = knowledge_state - knowledge_state.mean(dim=1, keepdim=True)
        state_code = torch.tanh(self.state_norm(state_residual))
        hidden = self.context_proj(self.context_norm(context_repr))
        hidden = F.silu(hidden)
        hidden = F.silu(self.hidden_proj(hidden))
        B = hidden.size(0)

        state_q = torch.tanh(self.state_query_proj(state_code))
        state_k = torch.tanh(self.state_key_proj(state_code))
        state_q = state_q.view(B, self.num_concepts, self.num_heads, self.rank).permute(0, 2, 1, 3)
        state_k = state_k.view(B, self.num_concepts, self.num_heads, self.rank).permute(0, 2, 1, 3)

        context_u = self.context_to_u(hidden).view(B, self.num_heads, self.num_concepts, self.rank)
        context_v = self.context_to_v(hidden).view(B, self.num_heads, self.num_concepts, self.rank)
        state_u = self.state_to_u(student_code).view(B, self.num_heads, self.num_concepts, self.rank)
        state_v = self.state_to_v(student_code).view(B, self.num_heads, self.num_concepts, self.rank)
        id_u = self.id_to_u(student_id_code).view(B, self.num_heads, self.num_concepts, self.rank)
        id_v = self.id_to_v(student_id_code).view(B, self.num_heads, self.num_concepts, self.rank)

        state_adapter_scale = self.max_state_adapter_scale * torch.sigmoid(self.state_adapter_logit)
        context_adapter_scale = self.max_context_adapter_scale * torch.sigmoid(self.context_adapter_logit)
        id_adapter_scale_effective = self.max_id_adapter_scale * torch.sigmoid(self.id_adapter_logit)
        if id_adapter_scale is not None:
            id_adapter_scale_effective = id_adapter_scale_effective * id_adapter_scale.to(
                device=knowledge_state.device,
                dtype=knowledge_state.dtype,
            )
        adapter_q = (
            context_adapter_scale * torch.tanh(context_u)
            + state_adapter_scale * torch.tanh(state_u)
            + id_adapter_scale_effective * torch.tanh(id_u)
        )
        adapter_k = (
            context_adapter_scale * torch.tanh(context_v)
            + state_adapter_scale * torch.tanh(state_v)
            + id_adapter_scale_effective * torch.tanh(id_v)
        )

        if active_row_index is None or active_row_valid_mask is None:
            full_row_mask = torch.ones(
                (B, self.num_concepts),
                device=knowledge_state.device,
                dtype=torch.bool,
            )
            active_row_index, active_row_valid_mask = _pack_active_row_index(full_row_mask)
        if support_row_cache is None:
            raise ValueError("support_row_cache is required for PersonalRelationGenerator.forward")

        q_rows = _gather_head_rows(state_q, active_row_index, active_row_valid_mask)
        q_rows_adapter = _gather_head_rows(state_q + adapter_q, active_row_index, active_row_valid_mask)
        k_support = _gather_head_support_features(state_k, support_row_cache["support_col_index"], support_row_cache["support_valid_mask"])
        k_support_adapter = _gather_head_support_features(
            state_k + adapter_k,
            support_row_cache["support_col_index"],
            support_row_cache["support_valid_mask"],
        )

        state_scores = (q_rows.unsqueeze(-2) * k_support).sum(dim=-1) / math.sqrt(self.rank)
        state_scores = _normalize_sparse_scores(state_scores, support_row_cache["support_valid_mask"])
        adapter_scores = (q_rows_adapter.unsqueeze(-2) * k_support_adapter).sum(dim=-1) / math.sqrt(self.rank)
        adapter_scores = _normalize_sparse_scores(
            adapter_scores - state_scores,
            support_row_cache["support_valid_mask"],
            row_budget=row_budget_values,
        )

        state_mix = self.max_state_mix * torch.sigmoid(self.state_mix_logit)
        student_mix = self.max_student_mix * torch.sigmoid(self.student_mix_logit)
        scores = state_mix * state_scores + student_mix * adapter_scores
        scores = _normalize_sparse_scores(scores, support_row_cache["support_valid_mask"], row_budget=row_budget_values)
        if not return_diagnostics:
            return scores

        diagnostics = {
            "state_mix": scores.new_tensor(float(state_mix.item())),
            "student_mix": scores.new_tensor(float(student_mix.item())),
            "student_adapter_scale": scores.new_tensor(float(torch.as_tensor(state_adapter_scale).item())),
            "state_adapter_scale": scores.new_tensor(float(torch.as_tensor(state_adapter_scale).item())),
            "id_adapter_scale": scores.new_tensor(float(torch.as_tensor(id_adapter_scale_effective).item())),
            "context_adapter_scale": scores.new_tensor(float(context_adapter_scale.item())),
            "state_scores_absmean": state_scores.abs().mean(),
            "student_scores_absmean": adapter_scores.abs().mean(),
            "scores_absmax": scores.abs().max(),
            "active_row_count_mean": active_row_valid_mask.float().sum(dim=1).mean(),
            "scores_nonfinite_count": scores.new_tensor(int((~torch.isfinite(scores)).sum().item()), dtype=torch.long),
        }
        return scores, diagnostics
