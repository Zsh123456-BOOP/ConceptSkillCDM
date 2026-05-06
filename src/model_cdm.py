
"""Top-level cognitive diagnosis model built on structure modeling + fixed prediction head."""

from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.model_ops import (
    _apply_sparse_local_posterior,
    _compute_sparse_local_messages,
    _safe_zero_preserving_sqrt,
)
from src.model_cdm_forward import run_cdm_forward
from src.model_regularization import get_regularization_components as _get_regularization_components
from src.model_structure import ConceptStructureModeling
from src.prediction_head import CognitiveDiagnosisHead, ExerciseDifficultyEncoder

class CognitiveDiagnosisModel(nn.Module):
    """
    主模型只保留两部分：
    - Module 1: ConceptStructureModeling（A + E）
    - Fixed Prediction Head: CognitiveDiagnosisHead（D）

    仅支持 ablate_module1。
    D 固定存在，不再提供 no_D；B 已物理移除。
    """

    def __init__(
        self,
        num_students: int,
        num_exercises: int,
        num_concepts: int,
        q_matrix: torch.Tensor,
        knowledge_dim: int = 32,
        num_relation_heads: int = 4,
        num_gnn_layers: int = 2,
        dropout: float = 0.3,
        use_concept_graph: bool = True,
        graph_topk: Optional[int] = None,
        allow_self_loop: bool = True,
        graph_identity_residual: float = 0.0,
        use_personal_graph: bool = False,
        personal_rank: int = 4,
        ablate_module1: bool = False,
        lambda_sparse_personal: float = 0.0,
        lambda_alpha: float = 0.0,
        lambda_graph_entropy: float = 0.01,
        graph_entropy_min: float = 0.15,
        graph_entropy_max: float = 0.85,
        lambda_graph_diag: float = 0.10,
        lambda_graph_uniform: float = 0.04,
        graph_uniform_margin: float = 0.10,
        graph_reg_warmup_epochs: int = 1,
        graph_reg_cap_ratio: float = 6.0,
        graph_dropout: Optional[float] = None,
        graph_tau_init: float = 1.0,
        graph_propagation_alpha: float = 0.20,
        graph_query_readout_scale: float = 0.35,
        graph_query_readout_2hop_scale: float = 0.15,
        prediction_l2_lambda: float = 5e-5,
        gnn_residual_weight: float = 0.5,
        personal_max_alpha: float = 0.35,
        personal_delta_scale: float = 1.0,
        personal_warmup_epochs: int = 0,
        personal_reg_warmup_epochs: Optional[int] = None,
        personal_student_dim: Optional[int] = None,
        lambda_alpha_min: float = 0.0,
        alpha_min_target: float = 0.0,
        personal_alpha_temperature: float = 2.0,
        personal_alpha_budget: float = 0.10,
        personal_alpha_base_init: float = 0.08,
        personal_alpha_bias_scale: float = 1.0,
        personal_disable_student_global_context: bool = False,
        personal_local_hops: int = 1,
        personal_include_neighbor_rows: bool = False,
        personal_query_row_budget: float = 1.0,
        personal_neighbor_row_budget: float = 0.30,
        personal_query_support_hops: int = 0,
        personal_support_only: bool = True,
        personal_query_correction_scale: float = 0.15,
        personal_query_correction_max_ratio: float = 0.20,
        personal_query_correction_min_graph_anchor: float = 0.01,
        personal_query_message_gain: float = 1.0,
        lambda_personal_kl: float = 0.0,
        lambda_personal_query_residual: float = 0.0,
        personal_query_residual_margin: float = 0.0,
        enable_personal_support_value_proj: bool = True,
        graph_query_gate_init_bias: float = 2.0,
        personal_support_include_query_self: bool = True,
        personal_support_include_graph: bool = True,
        personal_support_include_neighbors: bool = False,
        personal_value_use_global_basis: bool = True,
        personal_message_alignment_gate: bool = True,
        personal_projection_hidden_factor: int = 2,
        graph_headwise_query_gate: bool = True,
        graph_edge_bias_rank: int = 8,
        graph_query_adapter_enable: bool = True,
        graph_prior_logit_scale: float = 0.0,
        ae_query_residual_scale: float = 0.0,
        ae_logit_residual_scale: float = 0.0,
        ae_logit_residual_clip: float = 1.0,
        ae_irt_logit_scale: float = 1.0,
        ae_logit_dim: int = 32,
        share_concept_embeddings: bool = False,
    ):
        super().__init__()
        self.num_students = int(num_students)
        self.num_exercises = int(num_exercises)
        self.num_concepts = int(num_concepts)
        self.knowledge_dim = int(knowledge_dim)
        self.num_relation_heads = int(num_relation_heads)

        self.enable_module1 = not bool(ablate_module1)
        if not self.enable_module1:
            use_concept_graph = False
            use_personal_graph = False

        self.use_concept_graph = bool(use_concept_graph)
        self.use_personal_graph = bool(use_personal_graph)

        self.lambda_graph_entropy = float(lambda_graph_entropy)
        self.graph_entropy_min = float(graph_entropy_min)
        self.graph_entropy_max = float(graph_entropy_max)
        if self.graph_entropy_min > self.graph_entropy_max:
            self.graph_entropy_min, self.graph_entropy_max = self.graph_entropy_max, self.graph_entropy_min
        self.lambda_graph_diag = float(lambda_graph_diag)
        self.lambda_graph_uniform = float(lambda_graph_uniform)
        self.graph_uniform_margin = max(0.0, float(graph_uniform_margin))
        self.graph_reg_warmup_epochs = max(0, int(graph_reg_warmup_epochs))
        self.graph_reg_cap_ratio = max(0.0, float(graph_reg_cap_ratio))
        self.graph_dropout = graph_dropout
        self.graph_tau_init = float(graph_tau_init)
        self.graph_identity_residual = max(0.0, min(1.0, float(graph_identity_residual)))
        self.graph_propagation_alpha = max(0.0, min(1.0, float(graph_propagation_alpha)))
        self.graph_query_readout_scale = max(0.0, float(graph_query_readout_scale))
        self.graph_query_readout_2hop_scale = max(0.0, float(graph_query_readout_2hop_scale))
        self._current_epoch = 1
        self.lambda_sparse_personal = float(lambda_sparse_personal)
        self.lambda_alpha = float(lambda_alpha)
        self.prediction_l2_lambda = float(prediction_l2_lambda)
        self.personal_max_alpha = max(0.0, float(personal_max_alpha))
        self.personal_delta_scale = max(0.0, float(personal_delta_scale))
        self.personal_warmup_epochs = max(0, int(personal_warmup_epochs))
        self.personal_reg_warmup_epochs = (
            self.personal_warmup_epochs
            if personal_reg_warmup_epochs is None
            else max(0, int(personal_reg_warmup_epochs))
        )
        self.personal_student_dim = int(knowledge_dim if personal_student_dim is None else personal_student_dim)
        self.lambda_alpha_min = max(0.0, float(lambda_alpha_min))
        self.alpha_min_target = max(0.0, float(alpha_min_target))
        self.personal_alpha_temperature = max(1e-4, float(personal_alpha_temperature))
        self.personal_alpha_budget = max(0.0, float(personal_alpha_budget))
        self.personal_alpha_base_init = max(0.0, float(personal_alpha_base_init))
        self.personal_alpha_bias_scale = max(0.0, float(personal_alpha_bias_scale))
        self.personal_disable_student_global_context = bool(personal_disable_student_global_context)
        self.personal_local_hops = max(0, int(personal_local_hops))
        self.personal_include_neighbor_rows = bool(personal_include_neighbor_rows)
        self.personal_query_row_budget = max(0.0, float(personal_query_row_budget))
        self.personal_neighbor_row_budget = max(0.0, float(personal_neighbor_row_budget))
        self.personal_query_support_hops = max(0, int(personal_query_support_hops))
        self.personal_support_only = bool(personal_support_only)
        self.personal_query_correction_scale = max(0.0, float(personal_query_correction_scale))
        self.personal_query_correction_max_ratio = max(0.0, float(personal_query_correction_max_ratio))
        self.personal_query_correction_min_graph_anchor = max(0.0, float(personal_query_correction_min_graph_anchor))
        self.personal_query_message_gain = max(0.0, float(personal_query_message_gain))
        self.enable_personal_support_value_proj = bool(enable_personal_support_value_proj)
        self.graph_query_gate_init_bias = float(graph_query_gate_init_bias)
        self.lambda_personal_kl = max(0.0, float(lambda_personal_kl))
        self.lambda_personal_query_residual = max(0.0, float(lambda_personal_query_residual))
        self.personal_query_residual_margin = max(0.0, float(personal_query_residual_margin))
        self.personal_support_include_query_self = bool(personal_support_include_query_self)
        self.personal_support_include_graph = bool(personal_support_include_graph)
        self.personal_support_include_neighbors = bool(personal_support_include_neighbors)
        self.personal_value_use_global_basis = bool(personal_value_use_global_basis)
        self.personal_message_alignment_gate = bool(personal_message_alignment_gate)
        self.personal_projection_hidden_factor = max(1, int(personal_projection_hidden_factor))
        self.graph_headwise_query_gate = bool(graph_headwise_query_gate)
        self.graph_edge_bias_rank = max(1, int(graph_edge_bias_rank))
        self.graph_query_adapter_enable = bool(graph_query_adapter_enable)
        self.graph_prior_logit_scale = max(0.0, float(graph_prior_logit_scale))
        self.ae_query_residual_scale = max(0.0, float(ae_query_residual_scale))
        self.ae_logit_residual_scale = max(0.0, float(ae_logit_residual_scale))
        self.ae_logit_residual_clip = max(0.0, float(ae_logit_residual_clip))
        self.ae_irt_logit_scale = max(0.0, float(ae_irt_logit_scale))
        self.ae_logit_dim = max(1, int(ae_logit_dim))
        self.share_concept_embeddings = bool(share_concept_embeddings)

        self.register_buffer("q_matrix", q_matrix)
        graph_prior_matrix = self._build_graph_prior_matrix(q_matrix, num_concepts)
        self.register_buffer("graph_prior_matrix", graph_prior_matrix, persistent=False)
        self.register_buffer("ae_student_prior_logit", torch.zeros(num_students, dtype=torch.float32))
        self.register_buffer("ae_exercise_prior_logit", torch.zeros(num_exercises, dtype=torch.float32))
        self.register_buffer("ae_concept_prior_logit", torch.zeros(num_concepts, dtype=torch.float32))

        identity = torch.eye(num_concepts, dtype=torch.float32).unsqueeze(0).repeat(self.num_relation_heads, 1, 1)
        self.register_buffer("identity_relations", identity)

        self.structure_module = ConceptStructureModeling(
            num_students=num_students,
            num_concepts=num_concepts,
            knowledge_dim=knowledge_dim,
            num_relation_heads=num_relation_heads,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout,
            graph_dropout=self.graph_dropout,
            graph_tau_init=self.graph_tau_init,
            gnn_residual_weight=gnn_residual_weight,
            use_concept_graph=self.use_concept_graph,
            graph_topk=graph_topk,
            allow_self_loop=allow_self_loop,
            graph_identity_residual=self.graph_identity_residual,
            graph_propagation_alpha=self.graph_propagation_alpha,
            use_personal_graph=self.use_personal_graph,
            personal_rank=personal_rank,
            personal_max_alpha=self.personal_max_alpha,
            personal_delta_scale=self.personal_delta_scale,
            personal_warmup_epochs=self.personal_warmup_epochs,
            personal_reg_warmup_epochs=self.personal_reg_warmup_epochs,
            personal_student_dim=self.personal_student_dim,
            personal_alpha_temperature=self.personal_alpha_temperature,
            personal_alpha_budget=self.personal_alpha_budget,
            personal_alpha_base_init=self.personal_alpha_base_init,
            personal_alpha_bias_scale=self.personal_alpha_bias_scale,
            personal_disable_student_global_context=self.personal_disable_student_global_context,
            personal_local_hops=self.personal_local_hops,
            personal_include_neighbor_rows=self.personal_include_neighbor_rows,
            personal_query_row_budget=self.personal_query_row_budget,
            personal_neighbor_row_budget=self.personal_neighbor_row_budget,
            personal_query_support_hops=self.personal_query_support_hops,
            personal_support_only=self.personal_support_only,
            personal_support_include_query_self=self.personal_support_include_query_self,
            personal_support_include_graph=self.personal_support_include_graph,
            personal_support_include_neighbors=self.personal_support_include_neighbors,
            enable_personal_support_value_proj=self.enable_personal_support_value_proj,
            graph_edge_bias_rank=self.graph_edge_bias_rank,
            graph_prior_matrix=self.graph_prior_matrix,
            graph_prior_logit_scale=self.graph_prior_logit_scale,
            enable_module=self.enable_module1,
        )
        self.graph_query_gate = nn.Linear(knowledge_dim, 1)
        nn.init.zeros_(self.graph_query_gate.weight)
        nn.init.constant_(self.graph_query_gate.bias, self.graph_query_gate_init_bias)
        if self.graph_headwise_query_gate:
            self.graph_query_head_gate = nn.Linear(knowledge_dim, self.num_relation_heads)
            nn.init.zeros_(self.graph_query_head_gate.weight)
            nn.init.zeros_(self.graph_query_head_gate.bias)
        else:
            self.graph_query_head_gate = None
        if self.graph_query_adapter_enable:
            graph_hidden = max(knowledge_dim, knowledge_dim * self.personal_projection_hidden_factor)
            self.graph_query_adapter = nn.Sequential(
                nn.Linear(knowledge_dim * 3, graph_hidden),
                nn.GELU(),
                nn.Linear(graph_hidden, knowledge_dim),
            )
            nn.init.zeros_(self.graph_query_adapter[-1].weight)
            nn.init.zeros_(self.graph_query_adapter[-1].bias)
        else:
            self.graph_query_adapter = None
        ae_joint_enabled = self.enable_module1 and self.use_concept_graph and self.use_personal_graph
        ae_partial_enabled = self.enable_module1 and (self.use_concept_graph or self.use_personal_graph)
        if ae_joint_enabled and self.ae_query_residual_scale > 0.0:
            ae_hidden = max(knowledge_dim, knowledge_dim * self.personal_projection_hidden_factor)
            self.ae_query_state_adapter = nn.Sequential(
                nn.Linear(knowledge_dim * 4, ae_hidden),
                nn.GELU(),
                nn.Linear(ae_hidden, knowledge_dim),
            )
            nn.init.normal_(self.ae_query_state_adapter[-1].weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.ae_query_state_adapter[-1].bias)
        else:
            self.ae_query_state_adapter = None
        if ae_partial_enabled and self.ae_logit_residual_scale > 0.0:
            ae_logit_hidden = max(self.ae_logit_dim * 2, knowledge_dim)
            ae_logit_input_dim = self.ae_logit_dim * 6 + 4
            self.ae_student_logit_emb = nn.Embedding(num_students, self.ae_logit_dim)
            self.ae_concept_logit_emb = nn.Embedding(num_concepts, self.ae_logit_dim)
            self.ae_student_logit_bias = nn.Embedding(num_students, 1)
            self.ae_exercise_logit_bias = nn.Embedding(num_exercises, 1)
            self.ae_concept_logit_bias = nn.Embedding(num_concepts, 1)
            self.ae_graph_state_proj = nn.Linear(knowledge_dim, self.ae_logit_dim, bias=False)
            self.ae_personal_state_proj = nn.Linear(knowledge_dim, self.ae_logit_dim, bias=False)
            self.ae_logit_adapter = nn.Sequential(
                nn.Linear(ae_logit_input_dim, ae_logit_hidden),
                nn.GELU(),
                nn.Linear(ae_logit_hidden, 1),
            )
            nn.init.normal_(self.ae_student_logit_emb.weight, mean=0.0, std=0.05)
            nn.init.xavier_normal_(self.ae_concept_logit_emb.weight)
            nn.init.zeros_(self.ae_student_logit_bias.weight)
            nn.init.zeros_(self.ae_exercise_logit_bias.weight)
            nn.init.zeros_(self.ae_concept_logit_bias.weight)
            nn.init.xavier_normal_(self.ae_graph_state_proj.weight)
            nn.init.xavier_normal_(self.ae_personal_state_proj.weight)
            nn.init.normal_(self.ae_logit_adapter[-1].weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.ae_logit_adapter[-1].bias)
        else:
            self.ae_student_logit_emb = None
            self.ae_concept_logit_emb = None
            self.ae_student_logit_bias = None
            self.ae_exercise_logit_bias = None
            self.ae_concept_logit_bias = None
            self.ae_graph_state_proj = None
            self.ae_personal_state_proj = None
            self.ae_logit_adapter = None

        personal_hidden = max(knowledge_dim, knowledge_dim * self.personal_projection_hidden_factor)
        self.personal_value_proj_local = nn.Linear(knowledge_dim, knowledge_dim, bias=False)
        self.personal_value_proj_global = nn.Linear(knowledge_dim, knowledge_dim, bias=False)
        self.personal_query_writer = nn.Sequential(
            nn.Linear(knowledge_dim * 3, personal_hidden),
            nn.GELU(),
            nn.Linear(personal_hidden, knowledge_dim),
        )
        self.personal_align_gate = nn.Sequential(
            nn.Linear(knowledge_dim * 3 + 4, max(1, knowledge_dim // 2)),
            nn.GELU(),
            nn.Linear(max(1, knowledge_dim // 2), 1),
        )
        nn.init.eye_(self.personal_value_proj_local.weight)
        nn.init.eye_(self.personal_value_proj_global.weight)
        nn.init.zeros_(self.personal_query_writer[-1].weight)
        nn.init.zeros_(self.personal_query_writer[-1].bias)
        nn.init.zeros_(self.personal_align_gate[-1].weight)
        nn.init.constant_(self.personal_align_gate[-1].bias, 2.0)

        self._initialize_graph_prior_concept_embeddings()
        if self.share_concept_embeddings:
            self._tie_concept_embeddings()

        self.diagnosis_head = CognitiveDiagnosisHead(
            knowledge_dim=knowledge_dim,
            use_weight_norm=self.enable_module1,
        )
        self.exercise_encoder = ExerciseDifficultyEncoder(num_exercises=num_exercises)

    @staticmethod
    def _build_graph_prior_matrix(q_matrix: torch.Tensor, num_concepts: int) -> torch.Tensor:
        C = int(num_concepts)
        if C <= 0:
            raise ValueError(f"num_concepts must be positive, got {num_concepts}")
        if C == 1:
            return torch.ones(1, 1, dtype=torch.float32)

        q = q_matrix.detach().float()
        if q.dim() != 2 or q.size(1) != C:
            raise ValueError(f"q_matrix must have shape (num_exercises, {C}), got {tuple(q.shape)}")

        concept_occurs = (q > 0).float()
        cooccurrence = concept_occurs.t().matmul(concept_occurs)
        eye = torch.eye(C, dtype=torch.float32, device=cooccurrence.device)
        offdiag = 1.0 - eye
        cooccurrence = cooccurrence * offdiag

        uniform_offdiag = offdiag / float(C - 1)
        row_sum = cooccurrence.sum(dim=-1, keepdim=True)
        data_prior = cooccurrence / row_sum.clamp(min=1.0)
        data_prior = torch.where(row_sum > 0, data_prior, uniform_offdiag)

        smooth = 0.10
        prior = (1.0 - smooth) * data_prior + smooth * uniform_offdiag
        prior = prior * offdiag
        prior = prior / prior.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return prior.to(dtype=torch.float32)

    def _initialize_graph_prior_concept_embeddings(self) -> None:
        if not (self.enable_module1 and self.use_concept_graph and self.graph_prior_logit_scale > 0.0):
            return
        structure_module = getattr(self, "structure_module", None)
        if structure_module is None:
            return
        knowledge_encoder = getattr(structure_module, "knowledge_encoder", None)
        if knowledge_encoder is None or not hasattr(knowledge_encoder, "concept_emb"):
            return

        prior = self.graph_prior_matrix.detach().float()
        C = int(prior.size(0))
        if C <= 1:
            return

        sim = 0.5 * (prior + prior.t())
        sim = sim + torch.eye(C, device=sim.device, dtype=sim.dtype)
        try:
            eigvals, eigvecs = torch.linalg.eigh(sim.cpu())
        except RuntimeError:
            return

        width = min(int(knowledge_encoder.concept_emb.embedding_dim), C)
        eigvals = eigvals[-width:].clamp(min=1e-6).sqrt()
        basis = eigvecs[:, -width:] * eigvals.unsqueeze(0)
        basis = basis - basis.mean(dim=0, keepdim=True)
        basis = basis / basis.std(dim=0, keepdim=True).clamp(min=1e-6)
        basis = basis.to(device=knowledge_encoder.concept_emb.weight.device, dtype=knowledge_encoder.concept_emb.weight.dtype)
        scale = min(0.20, max(0.02, 0.20 * self.graph_prior_logit_scale))

        with torch.no_grad():
            knowledge_encoder.concept_emb.weight[:, :width].mul_(0.5).add_(basis * scale)
            relation_learning = getattr(structure_module, "relation_learning", None)
            if relation_learning is not None and relation_learning.concept_embeddings is not knowledge_encoder.concept_emb.weight:
                relation_learning.concept_embeddings[:, :width].mul_(0.5).add_(basis * scale)

    def _tie_concept_embeddings(self) -> None:
        if not self.enable_module1:
            return
        structure_module = getattr(self, "structure_module", None)
        if structure_module is None:
            return
        relation_learning = getattr(structure_module, "relation_learning", None)
        knowledge_encoder = getattr(structure_module, "knowledge_encoder", None)
        if relation_learning is None or knowledge_encoder is None:
            return
        relation_learning.concept_embeddings = knowledge_encoder.concept_emb.weight

    @staticmethod
    def _aggregate_with_relation(states: torch.Tensor, relation_matrices: torch.Tensor) -> torch.Tensor:
        if isinstance(relation_matrices, dict):
            return _apply_sparse_local_posterior(states, relation_matrices, reduce_heads=True)
        if relation_matrices.dim() == 3:
            A = relation_matrices.mean(dim=0).to(dtype=states.dtype)
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-12)
            return torch.matmul(A, states)
        if relation_matrices.dim() == 4:
            A = relation_matrices.mean(dim=1).to(dtype=states.dtype)
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-12)
            return torch.bmm(A, states)
        raise ValueError(f"Unsupported relation_matrices shape for aggregation: {tuple(relation_matrices.shape)}")

    @staticmethod
    def _aggregate_with_relation_heads(states: torch.Tensor, relation_matrices: torch.Tensor) -> torch.Tensor:
        if isinstance(relation_matrices, dict):
            return _apply_sparse_local_posterior(states, relation_matrices, reduce_heads=False)
        if relation_matrices.dim() == 3:
            outs = []
            for h in range(relation_matrices.size(0)):
                A = relation_matrices[h].to(dtype=states.dtype)
                A = A / (A.sum(dim=-1, keepdim=True) + 1e-12)
                outs.append(torch.matmul(A, states))
            return torch.stack(outs, dim=1)
        if relation_matrices.dim() == 4:
            outs = []
            for h in range(relation_matrices.size(1)):
                A = relation_matrices[:, h].to(dtype=states.dtype)
                A = A / (A.sum(dim=-1, keepdim=True) + 1e-12)
                outs.append(torch.bmm(A, states))
            return torch.stack(outs, dim=1)
        raise ValueError(f"Unsupported relation_matrices shape for head aggregation: {tuple(relation_matrices.shape)}")

    @staticmethod
    def _masked_query_rms(tensor: torch.Tensor, concept_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if concept_mask is None:
            return tensor.new_tensor(0.0)
        query_weight = concept_mask.float().unsqueeze(-1)
        denom = (query_weight.sum() * float(tensor.size(-1))).clamp(min=1.0)
        mean_sq = (tensor.pow(2) * query_weight).sum() / denom
        return _safe_zero_preserving_sqrt(mean_sq)

    @staticmethod
    def _masked_query_rms_per_sample(tensor: torch.Tensor, concept_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if concept_mask is None:
            return tensor.new_zeros((tensor.size(0),))
        query_weight = concept_mask.float().unsqueeze(-1)
        denom = (query_weight.sum(dim=(1, 2)) * float(tensor.size(-1))).clamp(min=1.0)
        mean_sq = (tensor.pow(2) * query_weight).sum(dim=(1, 2)) / denom
        return _safe_zero_preserving_sqrt(mean_sq)

    @staticmethod
    def _masked_query_pool(tensor: torch.Tensor, concept_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if concept_mask is None:
            return tensor.mean(dim=1)
        query_weight = concept_mask.float().unsqueeze(-1)
        denom = query_weight.sum(dim=1).clamp(min=1.0)
        return (tensor * query_weight).sum(dim=1) / denom

    def _build_ae_query_state_residual(
        self,
        *,
        knowledge_state: torch.Tensor,
        global_query_context: torch.Tensor,
        personal_query_correction: torch.Tensor,
        concept_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            self.ae_query_state_adapter is None
            or self.ae_query_residual_scale <= 0.0
            or not self.enable_module1
            or not self.use_concept_graph
            or not self.use_personal_graph
        ):
            zero = knowledge_state.new_tensor(0.0)
            return torch.zeros_like(knowledge_state), zero

        query_rows = concept_mask.float().unsqueeze(-1).bool()
        graph_active = global_query_context.pow(2).sum(dim=-1, keepdim=True) > 1e-12
        personal_active = personal_query_correction.pow(2).sum(dim=-1, keepdim=True) > 1e-12
        features = torch.cat(
            [
                knowledge_state,
                global_query_context,
                personal_query_correction,
                global_query_context * personal_query_correction,
            ],
            dim=-1,
        )
        residual = self.ae_query_state_adapter(features) * self.ae_query_residual_scale
        residual = torch.where(query_rows & graph_active & personal_active, residual, torch.zeros_like(residual))
        return residual, self._masked_query_rms(residual, concept_mask)

    def _build_ae_logit_residual(
        self,
        *,
        student_ids: torch.Tensor,
        exercise_ids: torch.Tensor,
        knowledge_state: torch.Tensor,
        relation_matrices: torch.Tensor,
        global_query_context: torch.Tensor,
        personal_query_correction: torch.Tensor,
        concept_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        zero_vec = knowledge_state.new_zeros((knowledge_state.size(0),))
        zero_scalar = knowledge_state.new_tensor(0.0)
        if (
            self.ae_logit_adapter is None
            or self.ae_student_logit_emb is None
            or self.ae_concept_logit_emb is None
            or self.ae_student_logit_bias is None
            or self.ae_exercise_logit_bias is None
            or self.ae_concept_logit_bias is None
            or self.ae_graph_state_proj is None
            or self.ae_personal_state_proj is None
            or self.ae_logit_residual_scale <= 0.0
            or not self.enable_module1
            or not (self.use_concept_graph or self.use_personal_graph)
        ):
            return zero_vec, zero_scalar

        dtype = knowledge_state.dtype
        device = knowledge_state.device
        a_gate = knowledge_state.new_tensor(1.0 if self.use_concept_graph else 0.0)
        e_gate = knowledge_state.new_tensor(1.0 if self.use_personal_graph else 0.0)
        joint_gate = a_gate * e_gate

        graph_rms = self._masked_query_rms_per_sample(global_query_context, concept_mask) * a_gate
        personal_rms = self._masked_query_rms_per_sample(personal_query_correction, concept_mask) * e_gate

        query_mask = concept_mask.float()
        query_weight = query_mask / query_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        concept_emb = self.ae_concept_logit_emb.weight
        query_self_context = query_weight.matmul(concept_emb) * a_gate
        if isinstance(relation_matrices, dict):
            global_relation = relation_matrices["global_matrices"]
        else:
            global_relation = relation_matrices
        A = global_relation.mean(dim=0).to(dtype=query_weight.dtype)
        A = A / A.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        graph_query_weight = query_weight.matmul(A)
        query_graph_context = graph_query_weight.matmul(concept_emb) * a_gate
        student_context = self.ae_student_logit_emb(student_ids) * e_gate
        mixed_concept_context = 0.65 * query_self_context + 0.35 * query_graph_context
        bilinear = (student_context * mixed_concept_context).sum(dim=-1) / math.sqrt(float(self.ae_logit_dim))
        student_prior = self.ae_student_prior_logit[student_ids].to(dtype=dtype, device=device)
        exercise_prior = self.ae_exercise_prior_logit[exercise_ids].to(dtype=dtype, device=device)
        concept_prior = query_weight.matmul(
            self.ae_concept_prior_logit.to(dtype=query_weight.dtype, device=query_weight.device).unsqueeze(-1)
        ).squeeze(-1)
        stat_prior = e_gate * student_prior + a_gate * (0.75 * exercise_prior + 0.75 * concept_prior)
        student_bias = e_gate * self.ae_student_logit_bias(student_ids).squeeze(-1)
        exercise_bias = a_gate * self.ae_exercise_logit_bias(exercise_ids).squeeze(-1)
        query_concept_bias = query_weight.matmul(self.ae_concept_logit_bias.weight).squeeze(-1)
        graph_concept_bias = graph_query_weight.matmul(self.ae_concept_logit_bias.weight).squeeze(-1)
        concept_bias = a_gate * (0.65 * query_concept_bias + 0.35 * graph_concept_bias)
        pooled_graph_state = self.ae_graph_state_proj(
            self._masked_query_pool(global_query_context, concept_mask)
        ) * a_gate
        pooled_personal_state = self.ae_personal_state_proj(
            self._masked_query_pool(personal_query_correction, concept_mask)
        ) * e_gate
        query_count = concept_mask.float().sum(dim=1, keepdim=True)
        query_density = query_count / float(max(1, self.num_concepts))
        graph_neighbor_mass = a_gate * (graph_query_weight * (1.0 - query_mask)).sum(dim=1, keepdim=True)
        scalar_feats = torch.cat(
            [
                graph_rms.unsqueeze(-1),
                personal_rms.unsqueeze(-1),
                (graph_rms * personal_rms).unsqueeze(-1),
                a_gate * query_density + graph_neighbor_mass,
            ],
            dim=-1,
        )
        features = torch.cat(
            [
                student_context,
                query_self_context,
                query_graph_context,
                student_context * query_graph_context,
                pooled_graph_state,
                pooled_personal_state,
                scalar_feats,
            ],
            dim=-1,
        )
        learned_raw = (
            bilinear
            + 0.20 * (student_bias + 0.75 * exercise_bias + 0.75 * concept_bias)
            + self.ae_logit_adapter(features).squeeze(-1)
        ) * joint_gate
        raw = stat_prior + 0.15 * learned_raw
        if self.ae_logit_residual_clip > 0.0:
            clip = self.ae_logit_residual_clip
            raw = clip * torch.tanh(raw / clip)
        residual = self.ae_logit_residual_scale * raw
        active = query_mask.sum(dim=1) > 0
        residual = torch.where(active, residual, torch.zeros_like(residual))
        return residual, residual.detach().abs().mean()

    def initialize_ae_logit_priors(
        self,
        *,
        student_logits: torch.Tensor,
        exercise_logits: torch.Tensor,
        concept_logits: torch.Tensor,
        scale: float,
    ) -> None:
        if (
            self.ae_student_logit_bias is None
            or self.ae_exercise_logit_bias is None
            or self.ae_concept_logit_bias is None
            or not self.enable_module1
            or not (self.use_concept_graph or self.use_personal_graph)
        ):
            return
        scale = max(0.0, float(scale))
        with torch.no_grad():
            student_target = torch.zeros_like(self.ae_student_prior_logit)
            exercise_target = torch.zeros_like(self.ae_exercise_prior_logit)
            concept_target = torch.zeros_like(self.ae_concept_prior_logit)
            s = student_logits.detach().to(
                device=student_target.device,
                dtype=student_target.dtype,
            ).view(-1)
            e = exercise_logits.detach().to(
                device=exercise_target.device,
                dtype=exercise_target.dtype,
            ).view(-1)
            c = concept_logits.detach().to(
                device=concept_target.device,
                dtype=concept_target.dtype,
            ).view(-1)
            student_target[: min(student_target.size(0), s.size(0))].copy_(
                s[: student_target.size(0)] * scale
            )
            exercise_target[: min(exercise_target.size(0), e.size(0))].copy_(
                e[: exercise_target.size(0)] * scale
            )
            concept_target[: min(concept_target.size(0), c.size(0))].copy_(
                c[: concept_target.size(0)] * scale
            )
            self.ae_student_prior_logit.copy_(student_target.clamp(min=-3.0, max=3.0))
            self.ae_exercise_prior_logit.copy_(exercise_target.clamp(min=-3.0, max=3.0))
            self.ae_concept_prior_logit.copy_(concept_target.clamp(min=-3.0, max=3.0))
            self.ae_student_logit_bias.weight.zero_()
            self.ae_exercise_logit_bias.weight.zero_()
            self.ae_concept_logit_bias.weight.zero_()

    def _apply_personal_query_trust_region(
        self,
        *,
        global_query_context: torch.Tensor,
        personal_query_correction: torch.Tensor,
        concept_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        global_query_rms = self._masked_query_rms_per_sample(global_query_context, concept_mask)
        personal_query_rms = self._masked_query_rms_per_sample(personal_query_correction, concept_mask)
        min_anchor = personal_query_correction.new_tensor(self.personal_query_correction_min_graph_anchor)
        max_ratio = personal_query_correction.new_tensor(self.personal_query_correction_max_ratio)
        max_allowed = max_ratio * torch.maximum(global_query_rms, min_anchor)
        trust_scale = torch.minimum(
            torch.ones_like(max_allowed),
            max_allowed / personal_query_rms.clamp(min=1e-8),
        )
        capped_correction = personal_query_correction * trust_scale.view(-1, 1, 1)
        return capped_correction, trust_scale, global_query_rms, personal_query_rms

    def _build_global_query_readout(
        self,
        knowledge_state: torch.Tensor,
        relation_matrices: torch.Tensor,
        concept_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            (self.graph_query_readout_scale <= 0 and self.graph_query_readout_2hop_scale <= 0)
            or not self.enable_module1
            or not self.use_concept_graph
        ):
            zero = knowledge_state.new_tensor(0.0)
            return torch.zeros_like(knowledge_state), zero, zero, zero, zero, zero, zero

        head_hop1 = self._aggregate_with_relation_heads(knowledge_state, relation_matrices)
        if isinstance(relation_matrices, dict):
            head_hop2 = self._aggregate_with_relation_heads(head_hop1.mean(dim=1), relation_matrices)
        else:
            hop2_parts = []
            for h in range(head_hop1.size(1)):
                if relation_matrices.dim() == 3:
                    A_h = relation_matrices[h].to(dtype=knowledge_state.dtype)
                    A_h = A_h / (A_h.sum(dim=-1, keepdim=True) + 1e-12)
                    hop2_parts.append(torch.matmul(A_h, head_hop1[:, h]))
                else:
                    A_h = relation_matrices[:, h].to(dtype=knowledge_state.dtype)
                    A_h = A_h / (A_h.sum(dim=-1, keepdim=True) + 1e-12)
                    hop2_parts.append(torch.bmm(A_h, head_hop1[:, h]))
            head_hop2 = torch.stack(hop2_parts, dim=1)

        query_mask = concept_mask.float()
        query_rows = query_mask.unsqueeze(-1).bool()
        query_head_logits = (
            self.graph_query_head_gate(knowledge_state)
            if self.graph_query_head_gate is not None
            else knowledge_state.new_zeros((*knowledge_state.shape[:2], self.num_relation_heads))
        )
        query_head_weight = F.softmax(query_head_logits, dim=-1)
        query_head_weight = torch.where(
            query_rows.expand_as(query_head_weight),
            query_head_weight,
            torch.zeros_like(query_head_weight),
        )
        head_weight = query_head_weight.permute(0, 2, 1).unsqueeze(-1)
        hop1 = (head_weight * head_hop1).sum(dim=1)
        hop2 = (head_weight * head_hop2).sum(dim=1)
        global_msg = (
            self.graph_query_readout_scale * (hop1 - knowledge_state)
            + self.graph_query_readout_2hop_scale * (hop2 - hop1)
        )
        global_msg = torch.where(query_rows, global_msg, torch.zeros_like(global_msg))
        if self.graph_query_adapter is not None:
            graph_write = 0.15 * global_msg + self.graph_query_adapter(
                torch.cat([knowledge_state, global_msg, global_msg - knowledge_state], dim=-1)
            )
        else:
            graph_write = global_msg
        query_state_gate = torch.sigmoid(self.graph_query_gate(knowledge_state))
        query_state_gate = torch.where(query_rows, query_state_gate, torch.zeros_like(query_state_gate))
        global_context = graph_write * query_state_gate
        query_row_graph_delta_pre = self._masked_query_rms(global_msg, concept_mask)
        query_row_graph_delta_post = self._masked_query_rms(global_context, concept_mask)
        gate_mean = (query_state_gate.squeeze(-1) * query_mask).sum() / query_mask.sum().clamp(min=1.0)

        head_strength = head_hop1.pow(2).mean(dim=-1).sqrt().permute(0, 2, 1)
        query_count = query_mask.sum(dim=1).clamp(min=1.0)
        query_head_var = ((head_strength.var(dim=-1, unbiased=False) * query_mask).sum(dim=1) / query_count).mean()
        top2 = torch.topk(query_head_weight, k=min(2, self.num_relation_heads), dim=-1).values
        if top2.size(-1) >= 2:
            head_margin = top2[..., 0] - top2[..., 1]
        else:
            head_margin = top2[..., 0]
        query_head_margin = ((head_margin * query_mask).sum(dim=1) / query_count).mean()
        graph_query_adapter_gain = query_row_graph_delta_post / query_row_graph_delta_pre.clamp(min=1e-8)

        query_seed = query_mask / query_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        query_support_heads = self._aggregate_with_relation_heads(query_seed.unsqueeze(-1), relation_matrices).squeeze(-1)
        query_support = (query_head_weight.permute(0, 2, 1) * query_support_heads).sum(dim=1)
        query_row_global_support_mass = (query_support * (1.0 - query_mask)).sum(dim=1).mean()
        return (
            global_context,
            query_row_graph_delta_pre,
            gate_mean,
            query_row_graph_delta_post,
            query_row_global_support_mass,
            query_head_var,
            query_head_margin,
        )

    def _build_personal_message_basis(
        self,
        knowledge_state: torch.Tensor,
        relation_spec: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        if not isinstance(relation_spec, dict) or self.personal_query_support_hops <= 0:
            return knowledge_state
        global_relation = relation_spec.get("global_matrices")
        if global_relation is None:
            return knowledge_state
        agg = knowledge_state
        accum = knowledge_state
        for _ in range(self.personal_query_support_hops):
            agg = self._aggregate_with_relation(agg, global_relation)
            accum = accum + agg
        return accum / float(self.personal_query_support_hops + 1)

    def _build_personal_query_correction(
        self,
        knowledge_state: torch.Tensor,
        relation_spec: Optional[Dict[str, torch.Tensor]],
        concept_mask: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        zero_scalar = knowledge_state.new_tensor(0.0)
        zero_tensor = torch.zeros_like(knowledge_state)
        if not isinstance(relation_spec, dict):
            return (
                zero_tensor,
                zero_scalar,
                zero_scalar,
                zero_scalar,
                zero_scalar,
                zero_scalar,
                zero_scalar,
                zero_scalar,
                zero_scalar,
                zero_scalar,
                zero_scalar,
            )

        active_row_index = relation_spec.get("active_row_index")
        active_row_valid_mask = relation_spec.get("active_row_valid_mask")
        if active_row_index is None or active_row_valid_mask is None or active_row_index.numel() == 0:
            return (
                zero_tensor,
                zero_scalar,
                zero_scalar,
                zero_scalar,
                zero_scalar,
                zero_scalar,
                zero_scalar,
                zero_scalar,
                zero_scalar,
                zero_scalar,
                zero_scalar,
            )

        local_basis = self._build_personal_message_basis(knowledge_state, relation_spec)
        if self.personal_value_use_global_basis:
            global_basis_seed = self._aggregate_with_relation(knowledge_state, relation_spec["global_matrices"])
            global_basis = self._build_personal_message_basis(global_basis_seed, relation_spec)
        else:
            global_basis = knowledge_state
        value_local = self.personal_value_proj_local(local_basis)
        value_global = self.personal_value_proj_global(global_basis)
        value_basis = self.personal_query_writer(
            torch.cat([value_local, value_global, value_local - value_global], dim=-1)
        )
        value_basis = value_basis + value_local

        expanded_values = value_basis.unsqueeze(1).expand(
            -1,
            relation_spec["global_matrices"].size(0),
            -1,
            -1,
        )
        local_messages = _compute_sparse_local_messages(
            expanded_values,
            relation_spec,
        )
        query_row_active_mask = relation_spec.get("query_row_active_mask")
        if query_row_active_mask is None:
            query_row_active_mask = active_row_valid_mask
        valid_query_rows = active_row_valid_mask & query_row_active_mask
        if not bool(valid_query_rows.any()):
            return (
                zero_tensor,
                zero_scalar,
                zero_scalar,
                zero_scalar,
                zero_scalar,
                zero_scalar,
                zero_scalar,
                zero_scalar,
                zero_scalar,
                zero_scalar,
                zero_scalar,
            )

        global_local = local_messages["global_local"].mean(dim=1)
        post_local = local_messages["post_local"].mean(dim=1)
        delta_local_raw = local_messages["delta_local_raw"]
        delta_local_raw_mean = delta_local_raw.mean(dim=1)
        delta_rms = _safe_zero_preserving_sqrt(
            delta_local_raw.pow(2).mean(dim=-1, keepdim=True)
        ).clamp(min=1e-4)
        effective_delta_local = (
            delta_local_raw / delta_rms
        ) * delta_rms.detach() * self.personal_query_message_gain
        gate_alpha = relation_spec["gate_alpha"].unsqueeze(-1).unsqueeze(-1)
        query_gate = gate_alpha / gate_alpha.detach().mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-4)
        query_gate = query_gate.clamp(min=0.35, max=2.50)
        message_delta_effective = query_gate * effective_delta_local

        correction = torch.zeros_like(knowledge_state)
        global_local_full = torch.zeros_like(knowledge_state)
        post_local_full = torch.zeros_like(knowledge_state)
        delta_local_raw_full = torch.zeros_like(knowledge_state)
        row_index = active_row_index.clamp(min=0)
        batch_idx = torch.arange(knowledge_state.size(0), device=knowledge_state.device, dtype=torch.long).unsqueeze(1).expand_as(row_index)
        global_local_full[batch_idx[valid_query_rows], row_index[valid_query_rows]] = global_local[valid_query_rows]
        post_local_full[batch_idx[valid_query_rows], row_index[valid_query_rows]] = post_local[valid_query_rows]
        delta_local_raw_full[batch_idx[valid_query_rows], row_index[valid_query_rows]] = delta_local_raw_mean[valid_query_rows]
        correction[batch_idx[valid_query_rows], row_index[valid_query_rows]] = message_delta_effective.mean(dim=1)[valid_query_rows]
        query_row_personal_message_delta = self._masked_query_rms(correction, concept_mask)
        query_row_global_local_rms = self._masked_query_rms(global_local_full, concept_mask)
        query_row_post_local_rms = self._masked_query_rms(post_local_full, concept_mask)
        query_row_delta_local_rms_raw = self._masked_query_rms(delta_local_raw_full, concept_mask)

        support_valid_mask = relation_spec["support_valid_mask"].bool()
        global_support_prob = relation_spec["global_support_prob"]
        posterior_prob = relation_spec["posterior_prob"]
        query_mask_sparse = valid_query_rows.float().unsqueeze(1).unsqueeze(-1) * support_valid_mask.float()
        query_count = query_mask_sparse.sum(dim=(1, 2, 3)).clamp(min=1.0)
        posterior_delta_abs = (
            ((posterior_prob - global_support_prob).abs() * query_mask_sparse).sum(dim=(1, 2, 3)) / query_count
        ).mean()
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
        query_row_message_projection_gain = query_row_delta_local_rms_raw / posterior_delta_abs.clamp(min=1e-6)

        query_mask = concept_mask.float()
        query_seed = query_mask / query_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        personal_support = self._aggregate_with_relation(query_seed.unsqueeze(-1), relation_spec).squeeze(-1)
        query_row_personal_support_mass = (personal_support * (1.0 - query_mask)).sum(dim=1).mean()
        row_index = active_row_index.clamp(min=0).unsqueeze(1).unsqueeze(-1)
        self_support_mask = relation_spec["support_col_index"] == row_index
        query_row_self_support_mass = (
            (posterior_prob * query_mask_sparse * self_support_mask.float()).sum(dim=(1, 2, 3)) / query_count
        ).mean()
        query_row_graph_support_mass = (
            (posterior_prob * query_mask_sparse * (~self_support_mask).float()).sum(dim=(1, 2, 3)) / query_count
        ).mean()
        return (
            correction,
            query_row_personal_message_delta,
            posterior_delta_abs,
            posterior_kl,
            query_row_personal_support_mass,
            query_row_self_support_mass,
            query_row_graph_support_mass,
            query_row_global_local_rms,
            query_row_post_local_rms,
            query_row_delta_local_rms_raw,
            query_row_message_projection_gain,
        )

    def _apply_personal_alignment_gate(
        self,
        *,
        knowledge_state: torch.Tensor,
        global_query_context: torch.Tensor,
        personal_query_correction: torch.Tensor,
        concept_mask: torch.Tensor,
        query_row_posterior_kl: torch.Tensor,
        query_row_self_support_mass: torch.Tensor,
        query_row_graph_support_mass: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_rows = concept_mask.float().unsqueeze(-1).bool()
        if not self.personal_message_alignment_gate:
            one = personal_query_correction.new_ones((*personal_query_correction.shape[:2], 1))
            zero = personal_query_correction.new_tensor(0.0)
            return personal_query_correction, one, zero

        cos_align = F.cosine_similarity(
            personal_query_correction,
            global_query_context,
            dim=-1,
            eps=1e-6,
        ).unsqueeze(-1)
        scalar_feats = torch.cat(
            [
                cos_align,
                torch.full_like(cos_align, float(query_row_posterior_kl.detach().item())),
                torch.full_like(cos_align, float(query_row_self_support_mass.detach().item())),
                torch.full_like(cos_align, float(query_row_graph_support_mass.detach().item())),
            ],
            dim=-1,
        )
        align_input = torch.cat(
            [
                knowledge_state,
                global_query_context,
                personal_query_correction,
                scalar_feats,
            ],
            dim=-1,
        )
        trust = torch.sigmoid(self.personal_align_gate(align_input))
        trust = torch.where(query_rows, trust, torch.zeros_like(trust))
        aligned = trust * personal_query_correction
        alignment = (
            (cos_align.squeeze(-1) * concept_mask.float()).sum()
            / concept_mask.float().sum().clamp(min=1.0)
        )
        return aligned, trust, alignment

    # ------------------------------
    # Fix #1：行熵稀疏度（用于 personal graph 正则）
    # ------------------------------
    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for graph-regularizer warmup (1-based)."""
        self._current_epoch = max(1, int(epoch))
        if self.structure_module is not None and hasattr(self.structure_module, "set_epoch"):
            self.structure_module.set_epoch(epoch)

    def _get_graph_reg_ramp(self) -> float:
        """Linear warmup factor for graph-related regularization terms."""
        if self.graph_reg_warmup_epochs <= 0:
            return 1.0
        return min(1.0, float(self._current_epoch) / float(self.graph_reg_warmup_epochs))

    def _get_linear_warmup(self, warmup_epochs: int) -> float:
        if warmup_epochs <= 0:
            return 1.0
        return min(1.0, float(self._current_epoch) / float(warmup_epochs))

    @staticmethod
    def _row_entropy(A: torch.Tensor) -> torch.Tensor:
        """Row-Entropy：对 row-stochastic 矩阵的稀疏性更有意义。"""
        A = A.clamp(min=1e-12)
        return -(A * A.log()).sum(dim=-1).mean()

    def forward(
        self,
        student_ids: torch.Tensor,
        exercise_ids: torch.Tensor,
        concept_vector: Optional[torch.Tensor] = None,
        return_details: bool = False,
        return_logits: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Fix #2：严格尊重 return_logits：
          - return_logits=False：返回 prob（sigmoid(logits)）
          - return_logits=True ：返回 logits（供 BCEWithLogitsLoss）

        return_details=True 时：
          - 返回 (logits/prob, details)
        """
        return run_cdm_forward(
            self,
            student_ids=student_ids,
            exercise_ids=exercise_ids,
            concept_vector=concept_vector,
            return_details=return_details,
            return_logits=return_logits,
        )

    def get_regularization_components(
        self,
        relation_matrices: torch.Tensor,
        details: Optional[Dict[str, torch.Tensor]] = None,
        base_loss: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Return decomposed regularization terms.
        This does not change optimization objective; it is used for logging/diagnostics.
        """
        return _get_regularization_components(
            self,
            relation_matrices=relation_matrices,
            details=details,
            base_loss=base_loss,
        )

    def get_regularization_loss(
        self,
        relation_matrices: torch.Tensor,
        details: Optional[Dict[str, torch.Tensor]] = None,
        base_loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        正则项汇总：
        (1) 全局概念图行熵（lambda_graph_entropy）—— 仅 module1 启用且 use_concept_graph=True 时有效
        (2) 固定预测头参数 L2（prediction_l2_lambda）
        (3) 个性化图稀疏 + alpha 惩罚 —— 仅 personal graph 存在时计入
        """
        terms = self.get_regularization_components(
            relation_matrices=relation_matrices,
            details=details,
            base_loss=base_loss,
        )
        return terms["total"]

    def get_student_diagnosis(self, student_id: int) -> Dict[str, torch.Tensor]:
        """
        诊断输出（用于 demo/可解释可视化）：
        - knowledge_mastery = sigmoid(theta_c)
        - student_repr = 模块1输出的学生表示（若 A/E 消融则为全 0）
        """
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            sid = torch.tensor([student_id], device=device, dtype=torch.long)

            s_out = self.structure_module(sid, identity_relations=self.identity_relations)
            ks = s_out["knowledge_state"].squeeze(0)  # (C,D)
            mastery = torch.sigmoid(self.diagnosis_head.theta_proj(ks).squeeze(-1))

            return {
                "knowledge_mastery": mastery,
                "student_repr": s_out["student_repr"].squeeze(0),
                "relation_matrices": s_out["relation_matrices"],
            }
