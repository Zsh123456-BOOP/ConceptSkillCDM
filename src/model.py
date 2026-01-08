# src/model.py
"""
Cognitive Diagnosis Model - Reviewer-friendly Dual-Branch Version

Fixes included (as requested):
1) Personal graph sparsity regularizer: replaced ineffective abs(mean) with row-entropy (meaningful under row-stochastic).
2) return_logits semantics: now respected end-to-end (model.forward no longer hard-codes return_logits=False in head).
3) NaN guard for degenerate concept size (C==1) when self-loop is disabled: safe fallback.
4) AMP / mixed-precision stability: adjacency is cast to Wh.dtype before matmul in ConceptGraphConv.
5) Optional weight sharing: concept embeddings used for relation learning and knowledge encoding are tied when concept graph is enabled
   (removes the “two unrelated concept embedding spaces” ambiguity).
"""

import math
from typing import Tuple, Optional, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrizations  # new API (no deprecation warning)


# ======================================================
# 1. Graph / Relation Learning
# ======================================================

class MultiHeadRelationLearning(nn.Module):
    """
    Multi-head concept adjacency A_h:
    - Row-stochastic (softmax)
    - Learnable positive temperature per head
    - Optional hard top-k sparsification per row
    - Sparsity regularizer: row entropy (principled under row-stochastic constraint)
    """

    def __init__(
        self,
        num_concepts: int,
        concept_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        tau_init: float = 1.0,
        topk: Optional[int] = None,
        allow_self_loop: bool = True,
    ):
        super().__init__()
        self.num_concepts = int(num_concepts)
        self.concept_dim = int(concept_dim)
        self.num_heads = int(num_heads)
        self.topk = topk
        self.allow_self_loop = bool(allow_self_loop)

        # Will be optionally tied to knowledge_encoder.concept_emb.weight in the main model
        self.concept_embeddings = nn.Parameter(torch.randn(num_concepts, concept_dim) * 0.02)

        self.Wq = nn.ModuleList([nn.Linear(concept_dim, concept_dim, bias=False) for _ in range(num_heads)])
        self.Wk = nn.ModuleList([nn.Linear(concept_dim, concept_dim, bias=False) for _ in range(num_heads)])

        # temperature should be positive -> softplus
        self.tau_raw = nn.Parameter(torch.ones(num_heads) * float(tau_init))

        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.xavier_normal_(self.concept_embeddings)
        for m in list(self.Wq) + list(self.Wk):
            nn.init.xavier_normal_(m.weight)

    @staticmethod
    def _apply_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
        """Hard keep top-k per row. Use -inf for masked entries so softmax yields exact 0."""
        vals, idx = torch.topk(scores, k=k, dim=-1)
        masked = torch.full_like(scores, float("-inf"))
        masked.scatter_(dim=-1, index=idx, src=vals)
        return masked

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            relation_matrices: (H, C, C) row-stochastic adjacency
            concept_embeddings: (C, D)
        """
        C, D = self.num_concepts, self.concept_dim
        x = self.concept_embeddings  # (C, D)

        # ---- Degenerate guard: C==1 ----
        # If C==1 and self-loop is disabled, masking would create all -inf => softmax NaN.
        # Safe fallback: always return identity adjacency for C==1.
        if C == 1:
            A = torch.ones((self.num_heads, 1, 1), device=x.device, dtype=x.dtype)
            return A, x

        tau = F.softplus(self.tau_raw) + 1e-6  # (H,)
        rels = []

        for h in range(self.num_heads):
            q = self.Wq[h](x)  # (C, D)
            k = self.Wk[h](x)  # (C, D)
            scores = (q @ k.t()) / math.sqrt(D)
            scores = scores / tau[h]

            if not self.allow_self_loop:
                eye = torch.eye(C, device=scores.device, dtype=torch.bool)
                scores = scores.masked_fill(eye, float("-inf"))

            if self.topk is not None and 0 < self.topk < C:
                scores = self._apply_topk(scores, self.topk)

            A = F.softmax(scores, dim=-1)  # row-stochastic

            # Edge dropout can, in rare cases, zero-out an entire row; guard it.
            A = self.dropout(A)
            row_sum = A.sum(dim=-1, keepdim=True)

            if self.allow_self_loop:
                # For any zero-sum row, restore a self-loop so normalization is safe & meaningful.
                zero_rows = (row_sum.squeeze(-1) < 1e-12)  # (C,)
                if zero_rows.any():
                    A = A.clone()
                    idx = torch.nonzero(zero_rows, as_tuple=False).squeeze(-1)
                    A[idx, :] = 0.0
                    A[idx, idx] = 1.0
                    row_sum = A.sum(dim=-1, keepdim=True)

            A = A / (row_sum + 1e-12)  # re-normalize

            rels.append(A)

        relation_matrices = torch.stack(rels, dim=0)  # (H, C, C)
        return relation_matrices, self.concept_embeddings

    def get_entropy_sparsity(self, relation_matrices: torch.Tensor) -> torch.Tensor:
        """
        Row entropy sparsity:
        Minimizing entropy encourages peaky distributions => practical sparsity under row-stochastic constraint.
        """
        A = relation_matrices.clamp(min=1e-12)
        entropy = -(A * A.log()).sum(dim=-1).mean()
        return entropy


class ConceptGraphConv(nn.Module):
    """
    Graph convolution supporting:
    - global adjacency: (H, C, C)
    - personal adjacency: (B, H, C, C)
    """

    def __init__(self, in_features: int, out_features: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_transforms = nn.ModuleList([
            nn.Linear(in_features, out_features, bias=False) for _ in range(self.num_heads)
        ])
        self.head_attention = nn.Parameter(torch.ones(self.num_heads) / self.num_heads)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for t in self.head_transforms:
            nn.init.xavier_normal_(t.weight)

    def forward(self, x: torch.Tensor, relation_matrices: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, Din)
        relation_matrices:
          - (H, C, C) or
          - (B, H, C, C)
        """
        B, C, _ = x.shape
        outputs = []

        for h in range(self.num_heads):
            Wh = self.head_transforms[h](x)  # (B, C, Dout)

            if relation_matrices.dim() == 3:
                # (H,C,C) -> use h-th adjacency
                A = relation_matrices[h]  # (C, C)
                # AMP safety: cast A to Wh dtype
                A = A.to(dtype=Wh.dtype)
                A = A / (A.sum(dim=-1, keepdim=True) + 1e-12)
                out = torch.matmul(A, Wh)  # (C,C) @ (B,C,D) -> (B,C,D)
            else:
                # (B,H,C,C) -> use per-batch adjacency
                A = relation_matrices[:, h, :, :]  # (B, C, C)
                A = A.to(dtype=Wh.dtype)
                A = A / (A.sum(dim=-1, keepdim=True) + 1e-12)
                out = torch.bmm(A, Wh)  # (B,C,C) @ (B,C,D) -> (B,C,D)

            outputs.append(out)

        out = torch.stack(outputs, dim=0)  # (H, B, C, Dout)
        attn = F.softmax(self.head_attention, dim=0).view(-1, 1, 1, 1)
        out = (out * attn).sum(dim=0)  # (B, C, Dout)
        out = out + self.bias
        out = self.dropout(out)
        return out


# ======================================================
# 2. Encoders
# ======================================================

class StudentKnowledgeEncoder(nn.Module):
    """
    Cognitive branch encoder:
    - student global embedding s
    - concept embedding c
    - h0 = c + s (broadcast)
    - GNN propagation on learned graph
    """

    def __init__(
        self,
        num_students: int,
        num_concepts: int,
        knowledge_dim: int,
        num_gnn_layers: int = 2,
        num_relation_heads: int = 4,
        dropout: float = 0.1,
        gnn_residual_weight: float = 0.5,
    ):
        super().__init__()
        self.num_students = int(num_students)
        self.num_concepts = int(num_concepts)
        self.knowledge_dim = int(knowledge_dim)
        self.gnn_residual_weight = float(gnn_residual_weight)

        self.student_global = nn.Embedding(num_students, knowledge_dim)
        self.concept_emb = nn.Embedding(num_concepts, knowledge_dim)

        self.gnn_layers = nn.ModuleList([
            ConceptGraphConv(knowledge_dim, knowledge_dim, num_heads=num_relation_heads, dropout=dropout)
            for _ in range(num_gnn_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(knowledge_dim) for _ in range(num_gnn_layers)])
        self.dropout = nn.Dropout(dropout)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.xavier_normal_(self.student_global.weight)
        nn.init.xavier_normal_(self.concept_emb.weight)

    def forward(self, student_ids: torch.Tensor, relation_matrices: torch.Tensor) -> torch.Tensor:
        B = student_ids.size(0)
        s = self.student_global(student_ids)  # (B, D)
        c = self.concept_emb.weight.unsqueeze(0).expand(B, -1, -1)  # (B, C, D)
        h = c + s.unsqueeze(1)  # (B, C, D)
        h = self.dropout(h)

        for gnn, ln in zip(self.gnn_layers, self.layer_norms):
            h_in = h
            h_out = gnn(h, relation_matrices)
            h = ln(h_in + self.gnn_residual_weight * h_out)
            h = F.relu(h)

        return h  # (B, C, D)


class StudentLatentEncoder(nn.Module):
    """Neural branch student latent + bias (MF)."""

    def __init__(self, num_students: int, latent_dim: int = 64):
        super().__init__()
        self.latent_emb = nn.Embedding(num_students, latent_dim)
        self.bias = nn.Embedding(num_students, 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.xavier_normal_(self.latent_emb.weight)
        nn.init.zeros_(self.bias.weight)

    def forward(self, student_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.latent_emb(student_ids)               # (B, latent)
        bias = self.bias(student_ids).squeeze(-1)           # (B,)
        return latent, bias


class ExerciseDifficultyEncoder(nn.Module):
    """
    Item parameterization:
    - IRT: item scalars b_e (difficulty) and a_e (discrimination>0)  [2PL]
    - Neural: item latent = base_item_latent + q_gate * (Q-normalized concept latent mix)
      This makes MF branch Q-conditioned.
    """

    def __init__(
        self,
        num_exercises: int,
        num_concepts: int,
        q_matrix: torch.Tensor,
        exercise_dim: int = 64,
        dropout: float = 0.1,
        use_mf_branch: bool = True,
        use_q_conditioning: bool = True,
    ):
        super().__init__()
        self.num_exercises = int(num_exercises)
        self.num_concepts = int(num_concepts)
        self.exercise_dim = int(exercise_dim)
        self.use_mf_branch = bool(use_mf_branch)
        self.use_q_conditioning = bool(use_q_conditioning and use_mf_branch)

        self.register_buffer("q_matrix", q_matrix)

        # Neural branch (optional)
        if self.use_mf_branch:
            self.exercise_latent = nn.Embedding(num_exercises, exercise_dim)
            self.exercise_bias = nn.Embedding(num_exercises, 1)

            # Concept latent for Q-conditioning
            self.concept_latent = nn.Embedding(num_concepts, exercise_dim)
            self.q_gate_raw = nn.Parameter(torch.zeros(1))  # sigmoid -> [0,1]
        else:
            self.exercise_latent = None
            self.exercise_bias = None
            self.concept_latent = None
            self.q_gate_raw = None

        # IRT 2PL item scalars
        self.b = nn.Embedding(num_exercises, 1)
        self.a_raw = nn.Embedding(num_exercises, 1)  # softplus -> >0

        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        if self.use_mf_branch:
            nn.init.xavier_normal_(self.exercise_latent.weight)
            nn.init.zeros_(self.exercise_bias.weight)
            nn.init.xavier_normal_(self.concept_latent.weight)

        nn.init.zeros_(self.b.weight)
        nn.init.normal_(self.a_raw.weight, mean=0.0, std=0.02)

    def forward(
        self,
        exercise_ids: torch.Tensor,
        concept_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Returns:
            exercise_latent: (B, De) or None
            exercise_bias:   (B,) or None
            b:               (B,) difficulty
            a:               (B,) discrimination > 0
        """
        if self.use_mf_branch:
            base = self.exercise_latent(exercise_ids)              # (B, De)
            e_bias = self.exercise_bias(exercise_ids).squeeze(-1)  # (B,)

            if self.use_q_conditioning:
                if concept_mask is None:
                    concept_mask = self.q_matrix[exercise_ids]  # (B, C)
                q = concept_mask.float()
                q_norm = q / (q.sum(dim=1, keepdim=True) + 1e-12)  # (B, C)
                c_lat = self.concept_latent.weight                 # (C, De)
                q_lat = torch.matmul(q_norm, c_lat)                # (B, De)
                gate = torch.sigmoid(self.q_gate_raw)              # scalar in [0,1]
                base = base + gate * q_lat

            base = self.dropout(base)
        else:
            base = None
            e_bias = None

        b = self.b(exercise_ids).squeeze(-1)  # (B,)
        a = F.softplus(self.a_raw(exercise_ids).squeeze(-1)) + 1e-6  # (B,)

        return base, e_bias, b, a


# ======================================================
# 3. Prediction / Prototype / Personal Graph
# ======================================================

class ResponsePredictionHead(nn.Module):
    """
    Dual-branch prediction:
    - Cognitive: theta_c = theta_proj(knowledge_state) -> (B,C)
                theta_e = masked_mean(theta_c)
                irt_logit = a * (theta_e - b)
    - Neural: cosine MF in shared space, with positive scale
    - Fusion: gated residual total_logit = irt_logit + gate * mf_logit
    """

    def __init__(
        self,
        knowledge_dim: int,
        student_latent_dim: int,
        exercise_dim: int,
        mf_dim: int = 64,
        dropout: float = 0.1,
        use_mf_branch: bool = True,
    ):
        super().__init__()
        self.use_mf_branch = bool(use_mf_branch)

        self.theta_proj = parametrizations.weight_norm(nn.Linear(knowledge_dim, 1, bias=True))

        if self.use_mf_branch:
            self.u_proj = nn.Linear(student_latent_dim, mf_dim, bias=False)
            self.v_proj = nn.Linear(exercise_dim, mf_dim, bias=False)
            nn.init.xavier_normal_(self.u_proj.weight)
            nn.init.xavier_normal_(self.v_proj.weight)

            self.mf_scale_raw = nn.Parameter(torch.tensor(1.0))
            self.mf_bias = nn.Parameter(torch.zeros(1))

            self.fusion_gate = nn.Linear(2, 1)
            nn.init.zeros_(self.fusion_gate.bias)
            nn.init.constant_(self.fusion_gate.weight, 0.0)
        else:
            self.u_proj = None
            self.v_proj = None
            self.mf_scale_raw = None
            self.mf_bias = None
            self.fusion_gate = None

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        knowledge_state: torch.Tensor,      # (B, C, Dk)
        concept_mask: torch.Tensor,         # (B, C)
        b: torch.Tensor,                    # (B,)
        a: torch.Tensor,                    # (B,)
        student_latent: Optional[torch.Tensor],       # (B, Ds) or None
        student_bias: Optional[torch.Tensor],         # (B,) or None
        exercise_latent: Optional[torch.Tensor],      # (B, De) or None
        exercise_bias: Optional[torch.Tensor],        # (B,) or None
        return_logits: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:

        # --- Cognitive / IRT ---
        theta_c = self.theta_proj(knowledge_state).squeeze(-1)  # (B, C)

        mask = concept_mask.float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        theta_e = (theta_c * mask).sum(dim=1) / denom  # (B,)

        irt_logit = a * (theta_e - b)  # (B,)

        if not self.use_mf_branch:
            total_logit = irt_logit
            if return_logits:
                return total_logit

            prob = torch.sigmoid(total_logit)
            zeros = torch.zeros_like(irt_logit)
            details = {
                "theta_c": theta_c.detach(),
                "theta_e": theta_e.detach(),
                "irt_logit": irt_logit.detach(),
                "mf_logit": zeros,
                "gate": zeros,
            }
            return prob, total_logit, details

        # --- Neural / MF residual (cosine + scale) ---
        # (These should not be None if use_mf_branch=True; keep explicit for safety)
        if student_latent is None or exercise_latent is None or student_bias is None or exercise_bias is None:
            raise RuntimeError("MF branch is enabled but MF inputs are None. Check ablation wiring.")

        u = self.u_proj(student_latent)
        v = self.v_proj(exercise_latent)
        u = F.normalize(u, dim=-1)
        v = F.normalize(v, dim=-1)

        mf_scale = F.softplus(self.mf_scale_raw) + 1e-6
        mf_logit = mf_scale * (u * v).sum(dim=-1) + self.mf_bias + student_bias + exercise_bias  # (B,)
        mf_logit = self.dropout(mf_logit)

        # --- Fusion (gated residual) ---
        stack = torch.stack([irt_logit, mf_logit], dim=1)  # (B,2)
        gate = torch.sigmoid(self.fusion_gate(stack)).squeeze(-1)  # (B,)
        total_logit = irt_logit + gate * mf_logit

        if return_logits:
            return total_logit

        prob = torch.sigmoid(total_logit)
        details = {
            "theta_c": theta_c.detach(),
            "theta_e": theta_e.detach(),
            "irt_logit": irt_logit.detach(),
            "mf_logit": mf_logit.detach(),
            "gate": gate.detach(),
        }
        return prob, total_logit, details


class SoftPrototypeModule(nn.Module):
    """Soft prototype (optional)."""

    def __init__(self, num_prototypes: int, dim: int, tau: float = 1.0):
        super().__init__()
        self.num_prototypes = int(num_prototypes)
        self.tau = float(tau)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, dim) * 0.1)
        nn.init.xavier_normal_(self.prototypes)

    def forward(self, student_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s = F.normalize(student_repr, dim=-1)
        p = F.normalize(self.prototypes, dim=-1)
        logits = torch.matmul(s, p.t()) / (self.tau + 1e-12)
        assign = F.softmax(logits, dim=-1)
        mix = torch.matmul(assign, self.prototypes)
        return mix, assign


class AdaptiveGate(nn.Module):
    """Adaptive alpha for personal graph mixing."""

    def __init__(self, student_dim: int):
        super().__init__()
        hid = max(1, student_dim // 2)
        self.gate = nn.Sequential(
            nn.Linear(student_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, 1),
            nn.Sigmoid(),
        )

    def forward(self, student_repr: torch.Tensor) -> torch.Tensor:
        return self.gate(student_repr).view(-1, 1, 1, 1)  # (B,1,1,1)


class PersonalRelationGenerator(nn.Module):
    """Generate personal adjacency (B,C,C) via low-rank factors."""

    def __init__(self, student_dim: int, num_concepts: int, rank: int = 4):
        super().__init__()
        self.num_concepts = int(num_concepts)
        self.rank = int(rank)
        self.to_u = nn.Linear(student_dim, num_concepts * rank, bias=False)
        self.to_v = nn.Linear(student_dim, num_concepts * rank, bias=False)
        nn.init.xavier_normal_(self.to_u.weight)
        nn.init.xavier_normal_(self.to_v.weight)

    def forward(self, student_repr: torch.Tensor) -> torch.Tensor:
        B = student_repr.size(0)
        u = self.to_u(student_repr).view(B, self.num_concepts, self.rank)
        v = self.to_v(student_repr).view(B, self.num_concepts, self.rank)
        scores = torch.bmm(u, v.transpose(1, 2))  # (B,C,C)
        A = F.softmax(scores, dim=-1)             # row-stochastic
        return A


# ======================================================
# 4. Main Model
# ======================================================

class CognitiveDiagnosisModel(nn.Module):
    """
    Main model:
    - relation_learning: learn global adjacency (H,C,C)
    - knowledge_encoder: concept-level knowledge state (B,C,D)
    - optional personal graph: generate per-student adjacency and re-encode
    - optional prototype: correct knowledge_state by prototype residual
    - prediction_head: produce logits/prob
    """

    def __init__(
        self,
        num_students: int,
        num_exercises: int,
        num_concepts: int,
        q_matrix: torch.Tensor,
        knowledge_dim: int = 32,
        skill_dim: int = 64,
        exercise_dim: int = 64,
        num_relation_heads: int = 4,
        num_gnn_layers: int = 2,
        dropout: float = 0.3,
        use_mf_branch: bool = True,
        use_concept_graph: bool = True,
        # graph
        graph_topk: Optional[int] = None,
        allow_self_loop: bool = True,
        # prototype
        num_prototypes: int = 3,
        proto_tau: float = 1.0,
        proto_lambda: float = 0.5,
        use_soft_prototype: bool = True,
        # personal graph
        use_personal_graph: bool = False,
        personal_rank: int = 4,
        lambda_sparse_personal: float = 0.0,
        lambda_alpha: float = 0.0,
        # regularization
        lambda_graph_entropy: float = 0.01,   # mapped from args.lambda_sparse
        mf_l2_lambda: float = 5e-5,           # mapped from args.exercise_l2_lambda
        gnn_residual_weight: float = 0.5,
        use_q_conditioning: bool = True,
    ):
        super().__init__()
        self.num_students = int(num_students)
        self.num_exercises = int(num_exercises)
        self.num_concepts = int(num_concepts)
        self.knowledge_dim = int(knowledge_dim)
        self.skill_dim = int(skill_dim)
        self.num_relation_heads = int(num_relation_heads)

        self.use_mf_branch = bool(use_mf_branch)
        self.use_concept_graph = bool(use_concept_graph)

        self.use_soft_prototype = bool(use_soft_prototype and num_prototypes > 0)
        self.proto_lambda = float(proto_lambda)

        self.use_personal_graph = bool(use_personal_graph)
        self.personal_rank = int(personal_rank)
        self.lambda_sparse_personal = float(lambda_sparse_personal)
        self.lambda_alpha = float(lambda_alpha)

        self.lambda_graph_entropy = float(lambda_graph_entropy)
        self.mf_l2_lambda = float(mf_l2_lambda)

        if self.use_concept_graph:
            self.relation_learning = MultiHeadRelationLearning(
                num_concepts=num_concepts,
                concept_dim=knowledge_dim,
                num_heads=num_relation_heads,
                dropout=dropout,
                topk=graph_topk,
                allow_self_loop=allow_self_loop,
            )
        else:
            self.relation_learning = None

        self.knowledge_encoder = StudentKnowledgeEncoder(
            num_students=num_students,
            num_concepts=num_concepts,
            knowledge_dim=knowledge_dim,
            num_gnn_layers=num_gnn_layers,
            num_relation_heads=num_relation_heads,
            dropout=dropout,
            gnn_residual_weight=gnn_residual_weight,
        )

        # ---- Optional weight tying: make graph concept embeddings == encoder concept embeddings ----
        # This resolves the “two separate concept embedding spaces” ambiguity.
        if self.use_concept_graph and self.relation_learning is not None:
            if self.knowledge_encoder.concept_emb.weight.shape == self.relation_learning.concept_embeddings.shape:
                self.knowledge_encoder.concept_emb.weight = self.relation_learning.concept_embeddings

        if self.use_mf_branch:
            self.skill_encoder = StudentLatentEncoder(num_students, latent_dim=skill_dim)
        else:
            self.skill_encoder = None

        self.exercise_encoder = ExerciseDifficultyEncoder(
            num_exercises=num_exercises,
            num_concepts=num_concepts,
            q_matrix=q_matrix,
            exercise_dim=exercise_dim,
            dropout=dropout,
            use_mf_branch=self.use_mf_branch,
            use_q_conditioning=use_q_conditioning,
        )

        self.prediction_head = ResponsePredictionHead(
            knowledge_dim=knowledge_dim,
            student_latent_dim=skill_dim,
            exercise_dim=exercise_dim,
            mf_dim=min(64, skill_dim, exercise_dim),
            dropout=dropout,
            use_mf_branch=self.use_mf_branch,
        )

        self.register_buffer("q_matrix", q_matrix)

        identity = torch.eye(num_concepts, dtype=torch.float32)
        identity = identity.unsqueeze(0).repeat(self.num_relation_heads, 1, 1)
        self.register_buffer("identity_relations", identity)

        if self.use_soft_prototype:
            self.prototype_module = SoftPrototypeModule(num_prototypes, knowledge_dim, proto_tau)
        else:
            self.prototype_module = None

        if self.use_personal_graph:
            self.adaptive_gate = AdaptiveGate(knowledge_dim)
            self.personal_generator = PersonalRelationGenerator(knowledge_dim, num_concepts, self.personal_rank)
        else:
            self.adaptive_gate = None
            self.personal_generator = None

    @staticmethod
    def _row_entropy(A: torch.Tensor) -> torch.Tensor:
        """Row entropy for row-stochastic matrices. Lower entropy => peakier => sparser in practice."""
        A = A.clamp(min=1e-12)
        return -(A * A.log()).sum(dim=-1).mean()

    def forward(
        self,
        student_ids: torch.Tensor,
        exercise_ids: torch.Tensor,
        concept_vector: Optional[torch.Tensor] = None,
        return_details: bool = False,
        return_logits: bool = False,
    ) -> Union[torch.Tensor, Tuple]:
        """
        return_logits:
          - True  => return logits (for BCEWithLogitsLoss)
          - False => return prob  (sigmoid(logits))

        return_details:
          - True  => also return details dict for regularizers / debugging
        """

        # 1) global graph
        if self.use_concept_graph and self.relation_learning is not None:
            relation_matrices, _ = self.relation_learning()  # (H,C,C)
        else:
            relation_matrices = self.identity_relations  # (H,C,C)

        # 2) concept mask (Q)
        q_vector = concept_vector if concept_vector is not None else self.q_matrix[exercise_ids]  # (B,C)

        # 3) first-pass encoding using global graph
        knowledge_state = self.knowledge_encoder(student_ids, relation_matrices)  # (B,C,D)
        student_repr = knowledge_state.mean(dim=1)  # (B,D)

        # 4) optional personal graph (used): generate, mix, re-encode
        gate_alpha = None
        personal_matrices = None
        relation_used = relation_matrices

        if self.use_personal_graph:
            gate_alpha = self.adaptive_gate(student_repr)              # (B,1,1,1)
            personal_matrices = self.personal_generator(student_repr)  # (B,C,C)

            B = student_ids.size(0)
            H = relation_matrices.size(0)

            base = relation_matrices.unsqueeze(0).expand(B, -1, -1, -1)      # (B,H,C,C)
            pers = personal_matrices.unsqueeze(1).expand(-1, H, -1, -1)      # (B,H,C,C)
            relation_used = (1.0 - gate_alpha) * base + gate_alpha * pers    # (B,H,C,C)

            knowledge_state = self.knowledge_encoder(student_ids, relation_used)
            student_repr = knowledge_state.mean(dim=1)

        # 5) optional prototype correction (used)
        proto_mix = None
        proto_assign = None
        if self.use_soft_prototype and self.prototype_module is not None:
            proto_mix, proto_assign = self.prototype_module(student_repr)  # (B,D), (B,K)
            proto_broadcast = proto_mix.unsqueeze(1).expand(-1, self.num_concepts, -1)
            knowledge_state = (1.0 - self.proto_lambda) * knowledge_state + self.proto_lambda * proto_broadcast

        # 6) MF vectors + IRT params
        if self.use_mf_branch and self.skill_encoder is not None:
            student_latent, student_bias = self.skill_encoder(student_ids)
        else:
            student_latent, student_bias = None, None

        exercise_latent, exercise_bias, b, a = self.exercise_encoder(exercise_ids, concept_mask=q_vector)

        # 7) prediction (FIX: respect return_logits)
        if return_details:
            prob, logits, head_details = self.prediction_head(
                knowledge_state=knowledge_state,
                concept_mask=q_vector,
                b=b, a=a,
                student_latent=student_latent,
                student_bias=student_bias,
                exercise_latent=exercise_latent,
                exercise_bias=exercise_bias,
                return_logits=False,
            )
        else:
            logits = self.prediction_head(
                knowledge_state=knowledge_state,
                concept_mask=q_vector,
                b=b, a=a,
                student_latent=student_latent,
                student_bias=student_bias,
                exercise_latent=exercise_latent,
                exercise_bias=exercise_bias,
                return_logits=True,
            )
            prob = torch.sigmoid(logits)
            head_details = None

        # return formatting
        if not return_details and not return_logits:
            return prob
        if not return_details and return_logits:
            return logits

        details: Dict[str, torch.Tensor] = {
            "relation_matrices": relation_matrices,  # (H,C,C)
            "relation_used": relation_used,          # (H,C,C) or (B,H,C,C)
            "knowledge_state": knowledge_state,
            "student_repr": student_repr,
            "q_vector": q_vector,
            "irt_b": b,
            "irt_a": a,
            "logits": logits,
        }
        if head_details is not None:
            details.update(head_details)

        if self.use_mf_branch:
            details["student_latent"] = student_latent
            details["exercise_latent"] = exercise_latent
        if self.use_soft_prototype:
            details["prototype_mix"] = proto_mix
            details["prototype_assign"] = proto_assign
        if self.use_personal_graph:
            details["alpha"] = gate_alpha
            details["personal_matrices"] = personal_matrices

        if return_logits:
            return logits, details
        return prob, details

    def get_regularization_loss(
        self,
        relation_matrices: torch.Tensor,
        details: Optional[Dict[str, torch.Tensor]] = None,
        lambda_proto_div: float = 0.0,
        lambda_proto_usage: float = 0.0,
    ) -> torch.Tensor:
        """
        Regularizers:
        (1) Graph entropy sparsity (weight = self.lambda_graph_entropy)
        (2) L2 on MF/IRT embeddings (weight = self.mf_l2_lambda)
        (3) Prototype diversity/usage (optional)
        (4) Personal graph sparsity + alpha penalty (optional)

        FIX: personal graph sparsity now uses row-entropy (meaningful under row-stochastic constraint).
        """
        device = relation_matrices.device
        reg = torch.tensor(0.0, device=device)

        # (1) global graph entropy sparsity
        if self.use_concept_graph and self.relation_learning is not None and self.lambda_graph_entropy > 0:
            entropy = self.relation_learning.get_entropy_sparsity(relation_matrices)
            reg = reg + self.lambda_graph_entropy * entropy

        # (2) MF/IRT L2
        if self.mf_l2_lambda > 0:
            reg_terms = [
                self.exercise_encoder.b.weight.pow(2).mean(),
                self.exercise_encoder.a_raw.weight.pow(2).mean(),
            ]
            if self.use_mf_branch and self.skill_encoder is not None and self.exercise_encoder.use_mf_branch:
                reg_terms.extend([
                    self.skill_encoder.latent_emb.weight.pow(2).mean(),
                    self.exercise_encoder.exercise_latent.weight.pow(2).mean(),
                    self.exercise_encoder.concept_latent.weight.pow(2).mean(),
                ])
            reg = reg + self.mf_l2_lambda * sum(reg_terms)

        # (3) prototype regularizers
        if self.use_soft_prototype and details is not None and "prototype_assign" in details:
            assign = details["prototype_assign"]  # (B,K)
            K = assign.size(1)

            if lambda_proto_div > 0.0 and self.prototype_module is not None:
                P = F.normalize(self.prototype_module.prototypes, dim=-1)  # (K,D)
                sim = P @ P.t()
                off = sim - torch.eye(K, device=device, dtype=sim.dtype)
                proto_div = (off.pow(2).sum() / (K * (K - 1) + 1e-12))
                reg = reg + lambda_proto_div * proto_div

            if lambda_proto_usage > 0.0:
                q_mean = assign.mean(dim=0)
                uniform = torch.full_like(q_mean, 1.0 / K)
                proto_usage = F.mse_loss(q_mean, uniform)
                reg = reg + lambda_proto_usage * proto_usage

        # (4) personal graph regularizers
        if self.use_personal_graph and details is not None:
            if "personal_matrices" in details and self.lambda_sparse_personal > 0:
                # FIX: entropy sparsity is meaningful for row-stochastic personal adjacency
                reg = reg + self.lambda_sparse_personal * self._row_entropy(details["personal_matrices"])
            if "alpha" in details and self.lambda_alpha > 0:
                reg = reg + self.lambda_alpha * details["alpha"].mean()

        return reg

    def get_student_diagnosis(self, student_id: int) -> Dict[str, torch.Tensor]:
        """
        Diagnosis output:
        - knowledge_mastery: sigmoid(theta_c) per concept
        - skill_latent: MF student latent
        """
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            sid = torch.tensor([student_id], device=device, dtype=torch.long)

            if self.use_concept_graph and self.relation_learning is not None:
                rel, _ = self.relation_learning()
            else:
                rel = self.identity_relations

            ks = self.knowledge_encoder(sid, rel).squeeze(0)  # (C,D)
            mastery = torch.sigmoid(self.prediction_head.theta_proj(ks).squeeze(-1))  # (C,)

            if self.skill_encoder is not None:
                latent, _ = self.skill_encoder(sid)
                latent = latent.squeeze(0)
            else:
                latent = torch.zeros(self.skill_dim, device=device)

            return {
                "knowledge_mastery": mastery,
                "skill_latent": latent,
                "relation_matrices": rel,
            }
