
"""A/E structure modeling assembly.

The final paper-facing names for A and E are intentionally left open.
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model_graph import MultiHeadRelationLearning, StudentKnowledgeEncoder
from src.model_personal import AdaptiveGate, PersonalRelationGenerator
from src.model_structure_forward import run_structure_forward

# 模块 1（A+E）组合：ConceptStructureModeling（支持“完全消融”）
# ======================================================

class ConceptStructureModeling(nn.Module):
    """
    模块 1：概念结构建模（A + E）

    关键：enable_module=False 时，模块1“完全消融”：
      - 不实例化 relation_learning / knowledge_encoder / personal_* 任何参数
      - forward 直接输出：
          relation_matrices = identity_relations
          relation_used     = identity_relations
          knowledge_state   = 全0 (B,C,D)
          student_repr      = 全0 (B,D)
    这样可确保：
      - A/E/跨概念传播/概念embedding 全部不存在
      - 不会出现“虽然不用，但参数还在训练”的不彻底消融
    """

    def __init__(
        self,
        num_students: int,
        num_concepts: int,
        knowledge_dim: int,
        num_relation_heads: int,
        num_gnn_layers: int,
        dropout: float,
        graph_dropout: Optional[float],
        graph_tau_init: float,
        gnn_residual_weight: float,
        use_concept_graph: bool,
        graph_topk: Optional[int],
        allow_self_loop: bool,
        graph_identity_residual: float,
        graph_propagation_alpha: float,
        student_global_scale: float,
        student_global_mode: str,
        # personal graph
        use_personal_graph: bool,
        personal_rank: int,
        personal_max_alpha: float,
        personal_delta_scale: float,
        personal_warmup_epochs: int,
        personal_reg_warmup_epochs: Optional[int],
        personal_student_dim: int,
        personal_alpha_temperature: float,
        personal_alpha_budget: float,
        personal_alpha_base_init: float,
        personal_alpha_bias_scale: float,
        personal_freeze_alpha_gate: bool,
        personal_disable_student_global_context: bool,
        personal_local_hops: int,
        personal_include_neighbor_rows: bool,
        personal_query_row_budget: float,
        personal_neighbor_row_budget: float,
        personal_query_support_hops: int,
        personal_support_only: bool,
        personal_support_include_query_self: bool,
        personal_support_include_graph: bool,
        personal_support_include_neighbors: bool,
        personal_item_support_mass: float,
        personal_mastery_prior_scale: float,
        personal_recent_mastery_prior_scale: float,
        personal_mastery_count_smoothing: float,
        enable_personal_support_value_proj: bool,
        graph_edge_bias_rank: int,
        graph_prior_matrix: Optional[torch.Tensor] = None,
        graph_sequence_prior_matrix: Optional[torch.Tensor] = None,
        graph_prior_logit_scale: float = 0.0,
        # 完全消融开关
        enable_module: bool = True,
    ):
        super().__init__()
        self.enable_module = bool(enable_module)

        # 保存形状信息：用于完全消融时构造全0张量
        self.num_concepts = int(num_concepts)
        self.knowledge_dim = int(knowledge_dim)
        self.num_relation_heads = int(num_relation_heads)
        self.personal_max_alpha = float(personal_max_alpha)
        self.personal_delta_scale = max(0.0, float(personal_delta_scale))
        self.personal_warmup_epochs = max(0, int(personal_warmup_epochs))
        self.personal_reg_warmup_epochs = (
            self.personal_warmup_epochs
            if personal_reg_warmup_epochs is None
            else max(0, int(personal_reg_warmup_epochs))
        )
        self.personal_student_dim = max(1, int(personal_student_dim))
        self.personal_alpha_temperature = max(1e-4, float(personal_alpha_temperature))
        self.personal_alpha_budget = max(0.0, float(personal_alpha_budget))
        self.personal_alpha_base_init = max(0.0, float(personal_alpha_base_init))
        self.personal_alpha_bias_scale = max(0.0, float(personal_alpha_bias_scale))
        self.personal_freeze_alpha_gate = bool(personal_freeze_alpha_gate)
        self.personal_disable_student_global_context = bool(personal_disable_student_global_context)
        self.personal_local_hops = max(0, int(personal_local_hops))
        self.personal_include_neighbor_rows = bool(personal_include_neighbor_rows)
        self.personal_query_row_budget = max(0.0, float(personal_query_row_budget))
        self.personal_neighbor_row_budget = max(0.0, float(personal_neighbor_row_budget))
        self.personal_query_support_hops = max(0, int(personal_query_support_hops))
        self.personal_support_only = bool(personal_support_only)
        self.personal_support_include_query_self = bool(personal_support_include_query_self)
        self.personal_support_include_graph = bool(personal_support_include_graph)
        self.personal_support_include_neighbors = bool(personal_support_include_neighbors)
        self.personal_item_support_mass = max(0.0, float(personal_item_support_mass))
        self.personal_mastery_prior_scale = max(0.0, float(personal_mastery_prior_scale))
        self.personal_recent_mastery_prior_scale = max(0.0, float(personal_recent_mastery_prior_scale))
        self.personal_mastery_count_smoothing = max(0.0, float(personal_mastery_count_smoothing))
        self.enable_personal_support_value_proj = bool(enable_personal_support_value_proj)
        self.student_global_scale = max(0.0, float(student_global_scale))
        self.student_global_mode = str(student_global_mode or "vector").strip().lower()
        self._current_epoch = 1

        # -------- 完全消融：不创建任何可训练参数 --------
        if not self.enable_module:
            self.use_concept_graph = False
            self.use_personal_graph = False
            self.relation_learning = None
            self.knowledge_encoder = None
            self.adaptive_gate = None
            self.personal_generator = None
            return

        # -------- 正常启用：A/E 可选 --------
        self.use_concept_graph = bool(use_concept_graph)
        # E is a student-conditioned reweighting of A support. Without A there
        # is no interpretable support substrate to personalize.
        self.use_personal_graph = bool(use_personal_graph and self.use_concept_graph)

        # A) 全局概念图学习
        if self.use_concept_graph:
            graph_dropout_val = dropout if graph_dropout is None else float(graph_dropout)
            self.relation_learning = MultiHeadRelationLearning(
                num_concepts=num_concepts,
                concept_dim=knowledge_dim,
                num_heads=num_relation_heads,
                dropout=graph_dropout_val,
                tau_init=float(graph_tau_init),
                topk=graph_topk,
                allow_self_loop=allow_self_loop,
                identity_residual=graph_identity_residual,
                edge_bias_rank=graph_edge_bias_rank,
                prior_matrix=graph_prior_matrix,
                sequence_prior_matrix=graph_sequence_prior_matrix,
                prior_logit_scale=graph_prior_logit_scale,
            )
        else:
            self.relation_learning = None

        # 知识状态编码器（模块1启用时存在）
        self.knowledge_encoder = StudentKnowledgeEncoder(
            num_students=num_students,
            num_concepts=num_concepts,
            knowledge_dim=knowledge_dim,
            num_gnn_layers=num_gnn_layers,
            num_relation_heads=num_relation_heads,
            dropout=dropout,
            gnn_residual_weight=gnn_residual_weight,
            propagation_alpha=graph_propagation_alpha,
            student_global_scale=self.student_global_scale,
            student_global_mode=self.student_global_mode,
        )

        # E) 个性化图（可选）
        if self.use_personal_graph:
            context_dim = knowledge_dim * (6 if self.personal_disable_student_global_context else 7)
            self.adaptive_gate = AdaptiveGate(
                knowledge_dim,
                context_dim,
                num_heads=num_relation_heads,
                max_alpha=self.personal_max_alpha,
                hidden_dim=None,
                alpha_temperature=self.personal_alpha_temperature,
                alpha_budget=self.personal_alpha_budget,
                alpha_base_init=self.personal_alpha_base_init,
            )
            if self.personal_freeze_alpha_gate:
                for param in self.adaptive_gate.parameters():
                    param.requires_grad_(False)
            self.personal_generator = PersonalRelationGenerator(
                knowledge_dim,
                context_dim,
                knowledge_dim,
                num_concepts,
                num_relation_heads,
                personal_rank,
                hidden_dim=None,
                mastery_prior_scale=self.personal_mastery_prior_scale,
                recent_mastery_prior_scale=self.personal_recent_mastery_prior_scale,
                mastery_count_smoothing=self.personal_mastery_count_smoothing,
            )
        else:
            self.adaptive_gate = None
            self.personal_generator = None

    def set_epoch(self, epoch: int) -> None:
        self._current_epoch = max(1, int(epoch))

    def _get_personal_warmup_scale(self) -> float:
        if self.personal_warmup_epochs <= 0:
            return 1.0
        return min(1.0, float(self._current_epoch) / float(self.personal_warmup_epochs))

    def _get_personal_reg_warmup_scale(self) -> float:
        if self.personal_reg_warmup_epochs <= 0:
            return 1.0
        return min(1.0, float(self._current_epoch) / float(self.personal_reg_warmup_epochs))

    @staticmethod
    def _masked_state_pool(states: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return states.mean(dim=1)
        weights = mask.float()
        denom = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (states * weights.unsqueeze(-1)).sum(dim=1) / denom

    @staticmethod
    def _masked_state_std(states: torch.Tensor, mask: Optional[torch.Tensor], pooled: torch.Tensor) -> torch.Tensor:
        if mask is None:
            return states.std(dim=1, unbiased=False)
        weights = mask.float()
        denom = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        centered = (states - pooled.unsqueeze(1)).pow(2)
        var = (centered * weights.unsqueeze(-1)).sum(dim=1) / denom
        return torch.sqrt(var + 1e-12)

    def _build_local_row_mask(
        self,
        concept_mask: Optional[torch.Tensor],
        relation_matrices: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if concept_mask is None:
            return None
        local = (concept_mask > 0).float()
        if relation_matrices is None or self.personal_local_hops <= 0:
            return local

        if relation_matrices.dim() == 3:
            support = (relation_matrices.mean(dim=0) > 0).to(dtype=local.dtype)
            frontier = local
            for _ in range(self.personal_local_hops):
                frontier = (torch.matmul(frontier, support) > 0).to(local.dtype)
                local = torch.maximum(local, frontier)
            return local

        support = (relation_matrices.mean(dim=1) > 0).to(dtype=local.dtype)
        frontier = local.unsqueeze(1)
        for _ in range(self.personal_local_hops):
            frontier = (torch.bmm(frontier, support) > 0).to(local.dtype)
            local = torch.maximum(local, frontier.squeeze(1))
        return local

    def _build_personal_row_budget_mask(
        self,
        concept_mask: Optional[torch.Tensor],
        local_row_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if concept_mask is None and local_row_mask is None:
            return None, None, None

        if concept_mask is None:
            query_rows = torch.zeros_like(local_row_mask)
        else:
            query_rows = (concept_mask > 0).float()
        local_rows = query_rows if local_row_mask is None else (local_row_mask > 0).float()
        neighbor_rows = torch.clamp(local_rows - query_rows, min=0.0)
        if not self.personal_include_neighbor_rows:
            neighbor_rows = torch.zeros_like(neighbor_rows)
        row_budget_mask = self.personal_query_row_budget * query_rows
        if self.personal_include_neighbor_rows:
            row_budget_mask = row_budget_mask + self.personal_neighbor_row_budget * neighbor_rows
        return row_budget_mask, query_rows, neighbor_rows
    def forward(
        self,
        student_ids: torch.Tensor,
        identity_relations: torch.Tensor,   # (H,C,C)
        concept_mask: Optional[torch.Tensor] = None,
        student_concept_prior: Optional[torch.Tensor] = None,
        student_concept_recent_prior: Optional[torch.Tensor] = None,
        student_concept_count: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """输出统一字典，便于主模型组装 details 与正则项。"""
        return run_structure_forward(
            self,
            student_ids,
            identity_relations,
            concept_mask,
            student_concept_prior=student_concept_prior,
            student_concept_recent_prior=student_concept_recent_prior,
            student_concept_count=student_concept_count,
        )
