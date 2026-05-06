
"""Global graph learning and backbone graph propagation modules."""

import math
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model_ops import _gather_head_rows, _gather_head_support_features

# 模块 1（A）：全局概念图学习 - MultiHeadRelationLearning
# ======================================================

class MultiHeadRelationLearning(nn.Module):
    """
    多头概念邻接学习 A_h（row-stochastic）：
    - softmax 保证每行归一化（row-stochastic）
    - learnable temperature（softplus 保证正值）
    - 可选 top-k 硬稀疏
    - 稀疏正则使用“行熵”（row entropy）
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
        identity_residual: float = 0.0,
        edge_bias_rank: int = 8,
        prior_matrix: Optional[torch.Tensor] = None,
        prior_logit_scale: float = 0.0,
    ):
        super().__init__()
        self.num_concepts = int(num_concepts)
        self.concept_dim = int(concept_dim)
        self.num_heads = int(num_heads)
        self.topk = topk
        self.allow_self_loop = bool(allow_self_loop)
        self.identity_residual = float(max(0.0, min(1.0, identity_residual))) if self.allow_self_loop else 0.0
        rel_rank = max(4, min(16, concept_dim // 4 if concept_dim >= 4 else 4))
        self.relation_rank = int(rel_rank)
        self.edge_bias_rank = max(1, int(edge_bias_rank))
        self.prior_logit_scale = max(0.0, float(prior_logit_scale))

        if prior_matrix is None:
            prior_scores = torch.zeros(num_concepts, num_concepts, dtype=torch.float32)
        else:
            prior = prior_matrix.detach().float()
            if tuple(prior.shape) != (num_concepts, num_concepts):
                raise ValueError(
                    f"prior_matrix shape must be ({num_concepts}, {num_concepts}), got {tuple(prior.shape)}"
                )
            prior = prior.clamp(min=1e-4)
            prior = prior / prior.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            prior_scores = prior.log()
            prior_scores = prior_scores - prior_scores.mean(dim=-1, keepdim=True)
        self.register_buffer("prior_logit_scores", prior_scores, persistent=False)

        # 注意：concept_embeddings 会在主模型里“可选绑定”到 knowledge_encoder.concept_emb.weight
        self.concept_embeddings = nn.Parameter(torch.randn(num_concepts, concept_dim))  # Fix: 移除 0.02
        self.rel_query_anchor = nn.Parameter(torch.randn(num_heads, num_concepts, self.relation_rank) * 0.02)
        self.rel_key_anchor = nn.Parameter(torch.randn(num_heads, num_concepts, self.relation_rank) * 0.02)
        self.graph_edge_bias_u = nn.Parameter(torch.randn(num_heads, num_concepts, self.edge_bias_rank) * 0.02)
        self.graph_edge_bias_v = nn.Parameter(torch.randn(num_heads, num_concepts, self.edge_bias_rank) * 0.02)
        self.self_loop_bias = nn.Parameter(torch.ones(num_heads) * 0.75)

        self.Wq = nn.ModuleList([nn.Linear(concept_dim, concept_dim, bias=False) for _ in range(num_heads)])
        self.Wk = nn.ModuleList([nn.Linear(concept_dim, concept_dim, bias=False) for _ in range(num_heads)])

        # temperature > 0：用 softplus 约束
        self.tau_raw = nn.Parameter(torch.ones(num_heads) * float(tau_init))

        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.xavier_normal_(self.concept_embeddings, gain=1.0)  # Fix: gain=1.0 prevents softmax saturation
        nn.init.xavier_normal_(self.rel_query_anchor)
        nn.init.xavier_normal_(self.rel_key_anchor)
        nn.init.xavier_normal_(self.graph_edge_bias_u)
        nn.init.xavier_normal_(self.graph_edge_bias_v)
        for m in list(self.Wq) + list(self.Wk):
            nn.init.xavier_normal_(m.weight)

    @staticmethod
    def _apply_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
        """每行只保留 top-k，其余置为 -inf，使 softmax 后精确为 0。"""
        vals, idx = torch.topk(scores, k=k, dim=-1)
        masked = torch.full_like(scores, float("-inf"))
        masked.scatter_(dim=-1, index=idx, src=vals)
        return masked

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            relation_matrices: (H, C, C)  每行归一的邻接矩阵
            concept_embeddings: (C, D)    概念 embedding（可能与 encoder 绑定共享）
        """
        C, D = self.num_concepts, self.concept_dim
        x = self.concept_embeddings  # (C, D)

        # ---- Fix #3：退化情况 C==1 防 NaN ----
        if C == 1:
            A = torch.ones((self.num_heads, 1, 1), device=x.device, dtype=x.dtype)
            return A, x

        tau = F.softplus(self.tau_raw) + 1e-6  # (H,)
        rels = []

        for h in range(self.num_heads):
            q = self.Wq[h](x)  # (C, D)
            k = self.Wk[h](x)  # (C, D)
            scores = (q @ k.t()) / math.sqrt(D)  # (C, C)
            rel_bias = (self.rel_query_anchor[h] @ self.rel_key_anchor[h].t()) / math.sqrt(self.relation_rank)
            edge_bias = (self.graph_edge_bias_u[h] @ self.graph_edge_bias_v[h].t()) / math.sqrt(self.edge_bias_rank)
            scores = scores + rel_bias + edge_bias
            scores = scores / tau[h]
            scores = scores - scores.mean(dim=-1, keepdim=True)
            if self.prior_logit_scale > 0.0:
                scores = scores + self.prior_logit_scale * self.prior_logit_scores.to(
                    device=scores.device,
                    dtype=scores.dtype,
                )

            eye = torch.eye(C, device=scores.device, dtype=torch.bool)
            if self.allow_self_loop:
                scores = scores + self.self_loop_bias[h] * eye.to(dtype=scores.dtype)
            else:
                scores = scores.masked_fill(eye, float("-inf"))

            if self.topk is not None and 0 < self.topk < C:
                scores = self._apply_topk(scores, self.topk)

            A = F.softmax(scores, dim=-1)  # (C, C) row-stochastic
            A = self.dropout(A)

            # Dropout 极端情况下可能造成某一行全 0：此时需防止归一化除 0
            row_sum = A.sum(dim=-1, keepdim=True)

            if self.allow_self_loop:
                # 对于零行，强制恢复 self-loop：保证可归一且语义合理
                zero_rows = (row_sum.squeeze(-1) < 1e-12)
                if zero_rows.any():
                    A = A.clone()
                    idx = torch.nonzero(zero_rows, as_tuple=False).squeeze(-1)
                    A[idx, :] = 0.0
                    A[idx, idx] = 1.0
                    row_sum = A.sum(dim=-1, keepdim=True)

            if self.identity_residual > 0:
                eye_f = torch.eye(C, device=A.device, dtype=A.dtype)
                A = (1.0 - self.identity_residual) * A + self.identity_residual * eye_f
                row_sum = A.sum(dim=-1, keepdim=True)

            A = A / (row_sum + 1e-12)
            rels.append(A)

        relation_matrices = torch.stack(rels, dim=0)  # (H, C, C)
        return relation_matrices, self.concept_embeddings

    def get_entropy_sparsity(self, relation_matrices: torch.Tensor) -> torch.Tensor:
        """行熵（Row Entropy）：row-stochastic 下熵越小越“尖”，实践上越稀疏。"""
        A = relation_matrices.clamp(min=1e-12)
        entropy = -(A * A.log()).sum(dim=-1).mean()
        return entropy


# ======================================================
# 模块 1（A/E）内部：图卷积 - ConceptGraphConv
# ======================================================

class ConceptGraphConv(nn.Module):
    """
    图卷积支持两种邻接输入：
    - 全局邻接： (H, C, C)
    - 个性化邻接：(B, H, C, C)

    Fix #4：AMP 稳定性：matmul/bmm 前把 A cast 到 Wh.dtype
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
          - sparse/local posterior relation spec dict
        """
        outputs = []
        for h in range(self.num_heads):
            Wh = self.head_transforms[h](x)  # (B, C, Dout)

            if isinstance(relation_matrices, dict):
                global_A = relation_matrices["global_matrices"][h].to(dtype=Wh.dtype)  # (C, C)
                global_A = global_A / (global_A.sum(dim=-1, keepdim=True) + 1e-12)
                out = torch.matmul(global_A, Wh)

                active_row_index = relation_matrices.get("active_row_index")
                active_row_valid_mask = relation_matrices.get("active_row_valid_mask")
                if active_row_index is not None and active_row_valid_mask is not None and active_row_index.numel() > 0:
                    support_col_index = relation_matrices["support_col_index"][:, h]
                    support_valid_mask = relation_matrices["support_valid_mask"][:, h]
                    global_support_prob = relation_matrices["global_support_prob"][:, h].to(dtype=Wh.dtype)
                    posterior_prob = relation_matrices["posterior_prob"][:, h].to(dtype=Wh.dtype)
                    gate = relation_matrices["gate_alpha"][:, h].to(dtype=Wh.dtype).view(-1, 1, 1)

                    support_features = _gather_head_support_features(
                        Wh.unsqueeze(1),
                        support_col_index.unsqueeze(1),
                        support_valid_mask.unsqueeze(1),
                    ).squeeze(1)
                    global_local = (global_support_prob.unsqueeze(-1) * support_features).sum(dim=-2)
                    post_local = (posterior_prob.unsqueeze(-1) * support_features).sum(dim=-2)
                    mixed_local = global_local + gate * (post_local - global_local)

                    batch_idx = torch.arange(Wh.size(0), device=Wh.device, dtype=torch.long).unsqueeze(1).expand_as(active_row_index)
                    valid = active_row_valid_mask
                    out[batch_idx[valid], active_row_index.clamp(min=0)[valid]] = mixed_local[valid]
            elif relation_matrices.dim() == 3:
                # 全局邻接
                A = relation_matrices[h]  # (C, C)
                A = A.to(dtype=Wh.dtype)  # Fix #4
                A = A / (A.sum(dim=-1, keepdim=True) + 1e-12)
                out = torch.matmul(A, Wh)  # (C,C) @ (B,C,D) -> (B,C,D)
            else:
                # 4D 个性化邻接（兼容旧代码）
                A = relation_matrices[:, h, :, :]  # (B, C, C)
                A = A.to(dtype=Wh.dtype)  # Fix #4
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
# 模块 1（A/E）：学生知识状态编码 - StudentKnowledgeEncoder
# ======================================================

class StudentKnowledgeEncoder(nn.Module):
    """
    认知分支编码器（Cognitive Branch Encoder）：
    - student_global embedding：s
    - concept embedding：c
    - h0 = c + s（广播）
    - 多层 GNN（用 relation_matrices 做传播）
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
        propagation_alpha: float = 0.20,
    ):
        super().__init__()
        self.num_students = int(num_students)
        self.num_concepts = int(num_concepts)
        self.knowledge_dim = int(knowledge_dim)
        self.gnn_residual_weight = float(gnn_residual_weight)
        self.propagation_alpha = max(0.0, min(1.0, float(propagation_alpha)))

        self.student_global = nn.Embedding(num_students, knowledge_dim)
        self.concept_emb = nn.Embedding(num_concepts, knowledge_dim)

        self.gnn_layers = nn.ModuleList([
            ConceptGraphConv(knowledge_dim, knowledge_dim, num_heads=num_relation_heads, dropout=dropout)
            for _ in range(num_gnn_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(knowledge_dim) for _ in range(num_gnn_layers)])
        self.hop_mix_logits = nn.Parameter(torch.zeros(num_gnn_layers + 1))
        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.xavier_normal_(self.student_global.weight)
        nn.init.xavier_normal_(self.concept_emb.weight)

    def compose_initial_state(self, student_ids: torch.Tensor) -> torch.Tensor:
        B = student_ids.size(0)
        s = self.student_global(student_ids)
        c = self.concept_emb.weight.unsqueeze(0).expand(B, -1, -1)
        return self.dropout(c + s.unsqueeze(1))

    def forward(self, student_ids: torch.Tensor, relation_matrices: torch.Tensor) -> torch.Tensor:
        h0 = self.compose_initial_state(student_ids)
        h = h0
        hop_states = [h0]

        for gnn, ln in zip(self.gnn_layers, self.layer_norms):
            propagated = gnn(h, relation_matrices)
            residual = h + self.gnn_residual_weight * propagated
            h = ln((1.0 - self.propagation_alpha) * residual + self.propagation_alpha * h0)
            h = F.relu(h)
            hop_states.append(h)

        hop_weights = F.softmax(self.hop_mix_logits[: len(hop_states)], dim=0)
        mixed = sum(weight * state for weight, state in zip(hop_weights, hop_states))
        return mixed  # (B, C, D)
