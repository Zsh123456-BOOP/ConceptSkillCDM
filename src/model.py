# src/model.py
"""
Cognitive Diagnosis Model

模块 1：概念结构建模（A + E）
  A) 全局概念图学习（Multi-head adjacency, row-stochastic）
  E) 个性化概念图（Personal graph）生成与混合（可选）
  输出：relation_matrices（全局）/ relation_used（全局或个性化混合）与 knowledge_state

预测头：固定 2PL-IRT
  theta_c -> Q-masked pooling -> theta_e -> irt_logit = a*(theta_e - b)
  输出：irt_logit 及可解释中间量（theta_c, theta_e）

当前代码只保留 A/E + 固定预测头：
- ablate_module1=True：模块1完全消融（A/E/knowledge_encoder 全部不实例化、forward 只返回全0 knowledge_state）
- 不再支持删除预测头或残差分支；B 已物理移除，D 固定启用

已包含并显式标注的修复点（Fixes）：
1) 个性化图稀疏正则：abs(mean) -> 行熵（Row-Entropy）
2) return_logits 语义：端到端尊重
3) C==1 且禁 self-loop 的 NaN 防护：退化情况直接返回单位图
4) AMP 稳定性：邻接矩阵 matmul 前 cast 到 Wh.dtype
5) 可选权重共享：启用概念图时，relation_learning 与 knowledge_encoder 的 concept embedding 绑定为同一参数
"""

import math
from typing import Tuple, Optional, Dict, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.prediction_head import CognitiveDiagnosisHead, ExerciseDifficultyEncoder


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


def _masked_support_softmax(logits: torch.Tensor, support_valid_mask: torch.Tensor) -> torch.Tensor:
    masked_logits = torch.where(
        support_valid_mask,
        logits,
        torch.full_like(logits, -30.0),
    )
    probs = F.softmax(masked_logits, dim=-1)
    probs = probs * support_valid_mask.to(dtype=probs.dtype)
    denom = probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    fallback = support_valid_mask.to(dtype=probs.dtype)
    fallback = fallback / fallback.sum(dim=-1, keepdim=True).clamp(min=1.0)
    return torch.where(support_valid_mask.any(dim=-1, keepdim=True), probs / denom, fallback)


def _masked_sparse_row_entropy(probs: torch.Tensor, support_valid_mask: torch.Tensor) -> torch.Tensor:
    mask = support_valid_mask.to(dtype=probs.dtype)
    probs = probs.clamp(min=1e-12) * mask
    denom = mask.sum(dim=-1).clamp(min=1.0)
    row_entropy = -(probs * probs.clamp(min=1e-12).log()).sum(dim=-1)
    return (row_entropy * (denom > 0).to(dtype=probs.dtype)).sum() / (denom > 0).to(dtype=probs.dtype).sum().clamp(min=1.0)


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

    active_row_index = relation_spec.get("active_row_index")
    active_row_valid_mask = relation_spec.get("active_row_valid_mask")
    if active_row_index is None or active_row_valid_mask is None or active_row_index.numel() == 0:
        return global_out.mean(dim=1) if reduce_heads else global_out

    support_col_index = relation_spec["support_col_index"]
    support_valid_mask = relation_spec["support_valid_mask"]
    global_support_prob = relation_spec["global_support_prob"]
    posterior_prob = relation_spec["posterior_prob"]
    gate_alpha = relation_spec["gate_alpha"]

    row_features = _gather_head_rows(expanded_states, active_row_index, active_row_valid_mask)
    support_features = _gather_head_support_features(expanded_states, support_col_index, support_valid_mask)
    del row_features  # row features are not needed in the operator itself; keep gather path symmetrical.

    global_local = (global_support_prob.unsqueeze(-1) * support_features).sum(dim=-2)
    post_local = (posterior_prob.unsqueeze(-1) * support_features).sum(dim=-2)
    mixed_local = global_local + gate_alpha.unsqueeze(-1).unsqueeze(-1) * (post_local - global_local)

    out = global_out.clone()
    batch_idx = torch.arange(B, device=states.device, dtype=torch.long).unsqueeze(1).expand_as(active_row_index)
    valid = active_row_valid_mask
    for h in range(H):
        out[:, h][batch_idx[valid], active_row_index.clamp(min=0)[valid]] = mixed_local[:, h][valid]
    return out.mean(dim=1) if reduce_heads else out


# ======================================================
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

        # 注意：concept_embeddings 会在主模型里“可选绑定”到 knowledge_encoder.concept_emb.weight
        self.concept_embeddings = nn.Parameter(torch.randn(num_concepts, concept_dim))  # Fix: 移除 0.02
        self.rel_query_anchor = nn.Parameter(torch.randn(num_heads, num_concepts, self.relation_rank) * 0.02)
        self.rel_key_anchor = nn.Parameter(torch.randn(num_heads, num_concepts, self.relation_rank) * 0.02)
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
            scores = scores + rel_bias
            scores = scores / tau[h]
            scores = scores - scores.mean(dim=-1, keepdim=True)

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


# ======================================================
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
            fallback_cache = _build_support_cache(
                torch.eye(self.num_concepts, device=knowledge_state.device, dtype=knowledge_state.dtype)
                .unsqueeze(0)
                .expand(self.num_heads, -1, -1),
                allow_full_support=True,
            )
            support_row_cache = _gather_row_support(fallback_cache, active_row_index, active_row_valid_mask)

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


# ======================================================
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
        personal_disable_student_global_context: bool,
        personal_local_hops: int,
        personal_query_row_budget: float,
        personal_neighbor_row_budget: float,
        personal_support_only: bool,
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
        self.personal_disable_student_global_context = bool(personal_disable_student_global_context)
        self.personal_local_hops = max(0, int(personal_local_hops))
        self.personal_query_row_budget = max(0.0, float(personal_query_row_budget))
        self.personal_neighbor_row_budget = max(0.0, float(personal_neighbor_row_budget))
        self.personal_support_only = bool(personal_support_only)
        self._current_epoch = 1

        # -------- 完全消融：不创建任何可训练参数 --------
        if not self.enable_module:
            self.use_concept_graph = False
            self.use_personal_graph = False
            self.relation_learning = None
            self.knowledge_encoder = None
            self.adaptive_gate = None
            self.personal_generator = None
            self.personal_gate_embedding = None
            self.personal_generator_embedding = None
            self.personal_alpha_bias = None
            self.personal_gate_from_state = None
            self.personal_generator_from_state = None
            self.personal_gate_id_logit = None
            self.personal_generator_id_logit = None
            return

        # -------- 正常启用：A/E 可选 --------
        self.use_concept_graph = bool(use_concept_graph)
        self.use_personal_graph = bool(use_personal_graph)

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
        )

        # E) 个性化图（可选）
        if self.use_personal_graph:
            self.personal_gate_embedding = nn.Embedding(num_students, self.personal_student_dim)
            self.personal_generator_embedding = nn.Embedding(num_students, self.personal_student_dim)
            self.personal_alpha_bias = (
                nn.Embedding(num_students, num_relation_heads)
                if self.personal_alpha_bias_scale > 0
                else None
            )
            self.personal_gate_from_state = nn.Linear(knowledge_dim, self.personal_student_dim, bias=False)
            self.personal_generator_from_state = nn.Linear(knowledge_dim, self.personal_student_dim, bias=False)
            self.personal_gate_id_logit = nn.Parameter(torch.tensor(-2.9444390))
            self.personal_generator_id_logit = nn.Parameter(torch.tensor(-2.9444390))
            nn.init.normal_(self.personal_gate_embedding.weight, mean=0.0, std=0.05)
            nn.init.normal_(self.personal_generator_embedding.weight, mean=0.0, std=0.05)
            if self.personal_alpha_bias is not None:
                nn.init.normal_(self.personal_alpha_bias.weight, mean=0.0, std=0.05)
            nn.init.xavier_normal_(self.personal_gate_from_state.weight)
            nn.init.xavier_normal_(self.personal_generator_from_state.weight)
            context_dim = knowledge_dim * (6 if self.personal_disable_student_global_context else 7)
            personal_hidden_dim = max(self.personal_student_dim, knowledge_dim)
            self.adaptive_gate = AdaptiveGate(
                self.personal_student_dim,
                context_dim,
                num_heads=num_relation_heads,
                max_alpha=self.personal_max_alpha,
                hidden_dim=personal_hidden_dim,
                alpha_temperature=self.personal_alpha_temperature,
                alpha_budget=self.personal_alpha_budget,
                alpha_base_init=self.personal_alpha_base_init,
            )
            self.personal_generator = PersonalRelationGenerator(
                self.personal_student_dim,
                context_dim,
                knowledge_dim,
                num_concepts,
                num_relation_heads,
                personal_rank,
                hidden_dim=personal_hidden_dim,
            )
        else:
            self.adaptive_gate = None
            self.personal_generator = None
            self.personal_gate_embedding = None
            self.personal_generator_embedding = None
            self.personal_alpha_bias = None
            self.personal_gate_from_state = None
            self.personal_generator_from_state = None
            self.personal_gate_id_logit = None
            self.personal_generator_id_logit = None

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
        row_budget_mask = (
            self.personal_query_row_budget * query_rows
            + self.personal_neighbor_row_budget * neighbor_rows
        )
        return row_budget_mask, query_rows, neighbor_rows

    def _build_personal_prior_logits(
        self,
        relation_matrices: torch.Tensor,
        identity_relations: torch.Tensor,
        local_row_mask: Optional[torch.Tensor],
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = identity_relations.device
        dtype = identity_relations.dtype
        row_mask = None if local_row_mask is None else local_row_mask.unsqueeze(1).unsqueeze(-1).bool()

        if self.use_concept_graph and self.relation_learning is not None:
            support_mask = relation_matrices > 0
            prior_logits = torch.full(
                (batch_size, self.num_relation_heads, self.num_concepts, self.num_concepts),
                -30.0,
                device=device,
                dtype=dtype,
            )
            support_logits = relation_matrices.clamp(min=1e-8).log().unsqueeze(0).expand_as(prior_logits)
            support_mask_4d = support_mask.unsqueeze(0).expand_as(prior_logits)
            prior_logits = torch.where(support_mask_4d, support_logits, prior_logits)
            return prior_logits, support_mask_4d

        neutral_logits = torch.zeros(
            (batch_size, self.num_relation_heads, self.num_concepts, self.num_concepts),
            device=device,
            dtype=dtype,
        )
        support_mask = torch.ones_like(neutral_logits, dtype=torch.bool)
        if row_mask is not None:
            identity_logits = torch.full_like(neutral_logits, -30.0)
            eye = torch.eye(self.num_concepts, device=device, dtype=torch.bool).view(1, 1, self.num_concepts, self.num_concepts)
            identity_logits = torch.where(eye, torch.zeros_like(identity_logits), identity_logits)
            neutral_logits = torch.where(row_mask.expand_as(neutral_logits), neutral_logits, identity_logits)
        return neutral_logits, support_mask

    def forward(
        self,
        student_ids: torch.Tensor,
        identity_relations: torch.Tensor,   # (H,C,C)
        concept_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        输出统一字典，便于主模型组装 details 与正则项。
        """
        device = student_ids.device
        dtype = identity_relations.dtype
        B = student_ids.size(0)

        # -------- 完全消融：直接返回全0，不走任何结构路径 --------
        if not self.enable_module:
            knowledge_state = torch.zeros((B, self.num_concepts, self.knowledge_dim), device=device, dtype=dtype)
            student_repr = torch.zeros((B, self.knowledge_dim), device=device, dtype=dtype)
            return {
                "relation_matrices": identity_relations,
                "relation_used": identity_relations,
                "knowledge_state": knowledge_state,
                "student_repr": student_repr,
                "alpha": None,
                "personal_matrices": None,
            }

        # 1) 全局图（A）
        if self.use_concept_graph and self.relation_learning is not None:
            relation_matrices, _ = self.relation_learning()     # (H,C,C)
        else:
            relation_matrices = identity_relations              # (H,C,C)

        # 2) 基于全局图编码（第一遍）
        knowledge_state = self.knowledge_encoder(student_ids, relation_matrices)  # (B,C,D)
        student_repr = knowledge_state.mean(dim=1)                                # (B,D)

        # 3) 个性化图（E，可选）：生成并混合，再编码第二遍
        #    优化：避免创建 (B,H,C,C) 4D 张量，改用逐 head 计算
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
        personal_student_adapter_scale = None
        personal_context_adapter_scale = None
        personal_delta_nonfinite_count = None
        personal_logits_nonfinite_count = None
        personal_matrix_nonfinite_count = None
        personal_bad_row_count = None
        personal_fallback_row_count = None
        personal_logits_absmax = None
        personal_delta_absmax = None
        local_row_ratio = None
        personal_support_density = None
        query_state_norm = None
        alpha_base_mean = None
        alpha_delta_absmean = None
        alpha_saturation_ratio = None
        query_row_personal_delta = None
        neighbor_row_personal_delta = None
        personal_row_budget_mean = None
        personal_query_row_std = None
        active_row_index = None
        active_row_valid_mask = None
        support_col_index = None
        support_valid_mask = None
        global_support_prob = None
        posterior_prob = None
        relation_used = relation_matrices
        initial_state = self.knowledge_encoder.compose_initial_state(student_ids)
        knowledge_state_pre_personal = knowledge_state
        relation_identity_delta = (relation_matrices - identity_relations).pow(2).mean().sqrt()
        knowledge_state_graph_delta = (knowledge_state_pre_personal - initial_state).pow(2).mean().sqrt()
        knowledge_state_personal_delta = relation_matrices.new_tensor(0.0)
        local_row_mask = self._build_local_row_mask(
            concept_mask,
            relation_matrices if (self.use_concept_graph and self.relation_learning is not None) else None,
        )
        if local_row_mask is not None:
            local_row_ratio = local_row_mask.mean()

        if self.use_personal_graph and self.adaptive_gate is not None and self.personal_generator is not None:
            student_global_repr = self.knowledge_encoder.student_global(student_ids)  # (B,D)
            query_mask = concept_mask.float() if concept_mask is not None else None
            local_mask = local_row_mask if local_row_mask is not None else query_mask
            query_state = self._masked_state_pool(knowledge_state, query_mask)
            local_state = self._masked_state_pool(knowledge_state, local_mask)
            local_dispersion = self._masked_state_std(knowledge_state, local_mask, local_state)
            global_state = knowledge_state.mean(dim=1)
            query_contrast = query_state - global_state
            local_contrast = local_state - global_state
            query_state_norm = query_state.norm(dim=-1).mean()
            if self.personal_disable_student_global_context:
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
            gate_state_input = query_state + 0.5 * query_contrast
            generator_state_input = query_state + 0.5 * local_contrast
            gate_state_repr = torch.tanh(self.personal_gate_from_state(gate_state_input))
            generator_state_repr = torch.tanh(
                self.personal_generator_from_state(generator_state_input)
            )
            gate_id_scale = torch.sigmoid(self.personal_gate_id_logit)
            generator_id_scale = torch.sigmoid(self.personal_generator_id_logit)
            gate_id_embedding = self.personal_gate_embedding(student_ids)
            generator_id_embedding = self.personal_generator_embedding(student_ids)
            row_budget_mask, query_row_mask, neighbor_row_mask = self._build_personal_row_budget_mask(
                concept_mask,
                local_row_mask,
            )
            active_row_index, active_row_valid_mask = _pack_active_row_index(local_row_mask.bool() if local_row_mask is not None else None)
            if row_budget_mask is not None and active_row_index is not None:
                row_budget_values = torch.gather(row_budget_mask, 1, active_row_index.clamp(min=0))
                row_budget_values = row_budget_values * active_row_valid_mask.to(dtype=row_budget_values.dtype)
                personal_row_budget_mean = row_budget_values.sum() / active_row_valid_mask.float().sum().clamp(min=1.0)
            else:
                row_budget_values = None
            allow_full_support = (not self.use_concept_graph) or (not self.personal_support_only)
            support_cache = _build_support_cache(relation_matrices, allow_full_support=allow_full_support)
            if active_row_index is not None and active_row_valid_mask is not None:
                support_row_cache = _gather_row_support(support_cache, active_row_index, active_row_valid_mask)
                support_col_index = support_row_cache["support_col_index"]
                support_valid_mask = support_row_cache["support_valid_mask"]
                global_support_prob = support_row_cache["global_support_prob"]
            else:
                support_row_cache = None
            gate_alpha_bias = None
            if self.personal_alpha_bias is not None:
                gate_alpha_bias = self.personal_alpha_bias_scale * torch.tanh(
                    self.personal_alpha_bias(student_ids)
                )
            personal_warmup_scale = self._get_personal_warmup_scale()
            gate_out = self.adaptive_gate(
                gate_state_repr,
                context_repr,
                gate_id_embedding,
                id_adapter_scale=gate_id_scale,
                extra_bias=gate_alpha_bias,
                warmup_scale=personal_warmup_scale,
                return_diagnostics=True,
            )
            gate_alpha, gate_alpha_logit, gate_diag = gate_out
            alpha_state_path_absmean = gate_diag["state_path_absmean"]
            alpha_id_path_absmean = gate_diag["id_path_absmean"]
            alpha_bias_path_absmean = gate_diag["bias_path_absmean"]
            head_bias_path_absmean = gate_diag["head_bias_path_absmean"]
            alpha_base_mean = gate_diag["alpha_base_mean"]
            alpha_delta_absmean = gate_diag["alpha_delta_absmean"]
            alpha_saturation_ratio = gate_diag["alpha_saturation_ratio"]
            alpha_id_adapter_scale = gate_diag["effective_id_scale"]
            gate_alpha_effective = gate_alpha
            personal_out = self.personal_generator(
                generator_state_repr,
                context_repr,
                knowledge_state,
                student_id_embedding=generator_id_embedding,
                id_adapter_scale=generator_id_scale,
                active_row_index=active_row_index,
                active_row_valid_mask=active_row_valid_mask,
                support_row_cache=support_row_cache,
                row_budget_values=row_budget_values,
                return_diagnostics=True,
            )  # (B,H,R,K)
            personal_delta, personal_diag = personal_out
            personal_state_mix = personal_diag["state_mix"]
            personal_student_mix = personal_diag["student_mix"]
            personal_student_adapter_scale = personal_diag["student_adapter_scale"]
            personal_context_adapter_scale = personal_diag["context_adapter_scale"]
            personal_delta_nonfinite_count = personal_diag["scores_nonfinite_count"]
            personal_delta_absmax = personal_diag["scores_absmax"]
            personal_delta = torch.nan_to_num(personal_delta, nan=0.0, posinf=1.0, neginf=-1.0)
            personal_support_density = support_valid_mask.float().mean() if support_valid_mask is not None else relation_matrices.new_tensor(0.0)
            posterior_delta = self.personal_delta_scale * personal_delta
            global_logprob = support_row_cache["global_support_logprob"] if support_row_cache is not None else None
            if global_logprob is not None:
                posterior_logits = torch.where(
                    support_valid_mask,
                    global_logprob + posterior_delta,
                    torch.full_like(posterior_delta, -30.0),
                )
                posterior_prob = _masked_support_softmax(posterior_logits, support_valid_mask)
                personal_logits_nonfinite_count = posterior_logits.new_tensor(
                    int((~torch.isfinite(posterior_logits)).sum().item()), dtype=torch.long
                )
                personal_logits_absmax = torch.nan_to_num(
                    posterior_logits, nan=0.0, posinf=20.0, neginf=-20.0
                ).abs().max()
                personal_matrix_nonfinite_count = posterior_prob.new_tensor(
                    int((~torch.isfinite(posterior_prob)).sum().item()), dtype=torch.long
                )
                bad_rows = ~support_valid_mask.any(dim=-1)
                personal_bad_row_count = posterior_prob.new_tensor(int(bad_rows.sum().item()), dtype=torch.long)
                personal_fallback_row_count = personal_bad_row_count.clone()

                valid_float = support_valid_mask.float()
                delta_denom = valid_float.sum(dim=(1, 2, 3)).clamp(min=1.0)
                personal_matrix_delta = (
                    (posterior_prob - global_support_prob).abs() * valid_float
                ).sum(dim=(1, 2, 3)) / delta_denom
                personal_matrix_student_std = posterior_prob.std(dim=0, unbiased=False).mean()
                personal_delta_pre_softmax_norm = (
                    (posterior_delta.pow(2) * valid_float).sum() / valid_float.sum().clamp(min=1.0)
                ).sqrt()
                personal_delta_student_std = (posterior_delta * valid_float).std(dim=0, unbiased=False).mean()
                alpha_head_std = gate_alpha_effective.squeeze(-1).squeeze(-1).std(dim=1, unbiased=False).mean()

                if query_row_mask is not None and active_row_index is not None:
                    query_active = torch.gather(query_row_mask.float(), 1, active_row_index.clamp(min=0))
                    query_active = query_active * active_row_valid_mask.float()
                    query_mask_sparse = query_active.unsqueeze(1).unsqueeze(-1) * valid_float
                    query_count = query_mask_sparse.sum(dim=(1, 2, 3)).clamp(min=1.0)
                    query_delta_per_sample = (
                        posterior_delta.abs() * query_mask_sparse
                    ).sum(dim=(1, 2, 3)) / query_count
                    query_row_personal_delta = query_delta_per_sample.mean()
                    personal_query_row_std = query_delta_per_sample.std(unbiased=False)
                if neighbor_row_mask is not None and active_row_index is not None:
                    neighbor_active = torch.gather(neighbor_row_mask.float(), 1, active_row_index.clamp(min=0))
                    neighbor_active = neighbor_active * active_row_valid_mask.float()
                    neighbor_mask_sparse = neighbor_active.unsqueeze(1).unsqueeze(-1) * valid_float
                    neighbor_count = neighbor_mask_sparse.sum(dim=(1, 2, 3)).clamp(min=1.0)
                    neighbor_delta_per_sample = (
                        posterior_delta.abs() * neighbor_mask_sparse
                    ).sum(dim=(1, 2, 3)) / neighbor_count
                    neighbor_row_personal_delta = neighbor_delta_per_sample.mean()

                relation_used = {
                    "global_matrices": relation_matrices,
                    "active_row_index": active_row_index,
                    "active_row_valid_mask": active_row_valid_mask,
                    "support_col_index": support_col_index,
                    "support_valid_mask": support_valid_mask,
                    "global_support_prob": global_support_prob,
                    "posterior_prob": posterior_prob,
                    "gate_alpha": gate_alpha_effective.squeeze(-1).squeeze(-1),
                }
                personal_matrices = None
            knowledge_state = self.knowledge_encoder(student_ids, relation_used)
            student_repr = knowledge_state.mean(dim=1)
            knowledge_state_personal_delta = (knowledge_state - knowledge_state_pre_personal).pow(2).mean().sqrt()

        return {
            "relation_matrices": relation_matrices,
            "relation_used": relation_used,
            "knowledge_state": knowledge_state,
            "student_repr": student_repr,
            "relation_identity_delta": relation_identity_delta,
            "knowledge_state_graph_delta": knowledge_state_graph_delta,
            "knowledge_state_personal_delta": knowledge_state_personal_delta,
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
            "personal_gate_id_scale": None
            if self.personal_gate_id_logit is None
            else torch.sigmoid(self.personal_gate_id_logit).to(device=device, dtype=dtype),
            "personal_generator_id_scale": None
            if self.personal_generator_id_logit is None
            else torch.sigmoid(self.personal_generator_id_logit).to(device=device, dtype=dtype),
            "personal_matrices": personal_matrices,
            "personal_matrix_delta": personal_matrix_delta,
            "personal_matrix_student_std": personal_matrix_student_std,
            "personal_delta_pre_softmax_norm": personal_delta_pre_softmax_norm,
            "personal_delta_student_std": personal_delta_student_std,
            "alpha_head_std": alpha_head_std,
            "query_row_personal_delta": query_row_personal_delta,
            "neighbor_row_personal_delta": neighbor_row_personal_delta,
            "personal_row_budget_mean": personal_row_budget_mean,
            "personal_query_row_std": personal_query_row_std,
            "active_row_index": active_row_index,
            "active_row_valid_mask": active_row_valid_mask,
            "support_col_index": support_col_index,
            "support_valid_mask": support_valid_mask,
            "global_support_prob": global_support_prob,
            "posterior_prob": posterior_prob,
            "personal_state_mix": personal_state_mix,
            "personal_student_mix": personal_student_mix,
            "personal_student_adapter_scale": personal_student_adapter_scale,
            "personal_context_adapter_scale": personal_context_adapter_scale,
            "personal_delta_nonfinite_count": personal_delta_nonfinite_count,
            "personal_logits_nonfinite_count": personal_logits_nonfinite_count,
            "personal_matrix_nonfinite_count": personal_matrix_nonfinite_count,
            "personal_bad_row_count": personal_bad_row_count,
            "personal_fallback_row_count": personal_fallback_row_count,
            "personal_logits_absmax": personal_logits_absmax,
            "personal_delta_absmax": personal_delta_absmax,
            "local_row_ratio": local_row_ratio,
            "personal_support_density": personal_support_density,
            "query_state_norm": query_state_norm,
            "local_row_mask": local_row_mask,
            "personal_warmup_scale": torch.tensor(
                self._get_personal_warmup_scale(), device=device, dtype=dtype
            ),
            "personal_reg_warmup_scale": torch.tensor(
                self._get_personal_reg_warmup_scale(), device=device, dtype=dtype
            ),
            "personal_alpha_bias_scale": torch.tensor(
                self.personal_alpha_bias_scale, device=device, dtype=dtype
            ),
        }


# ======================================================
# 主模型：组合模块 1/2/3，提供统一 forward 与正则/诊断接口
# ======================================================

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
        graph_readout_1hop_scale: float = 0.35,
        graph_readout_2hop_scale: float = 0.15,
        graph_query_writeback_scale: Optional[float] = None,
        graph_query_writeback_2hop_scale: Optional[float] = None,
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
        personal_query_row_budget: float = 1.0,
        personal_neighbor_row_budget: float = 0.30,
        personal_support_only: bool = True,
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
        self.graph_readout_1hop_scale = max(0.0, float(graph_readout_1hop_scale))
        self.graph_readout_2hop_scale = max(0.0, float(graph_readout_2hop_scale))
        self.graph_query_writeback_scale = max(
            0.0,
            float(self.graph_readout_1hop_scale if graph_query_writeback_scale is None else graph_query_writeback_scale),
        )
        self.graph_query_writeback_2hop_scale = max(
            0.0,
            float(self.graph_readout_2hop_scale if graph_query_writeback_2hop_scale is None else graph_query_writeback_2hop_scale),
        )
        self.graph_readout_1hop_scale = self.graph_query_writeback_scale
        self.graph_readout_2hop_scale = self.graph_query_writeback_2hop_scale
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
        self.personal_query_row_budget = max(0.0, float(personal_query_row_budget))
        self.personal_neighbor_row_budget = max(0.0, float(personal_neighbor_row_budget))
        self.personal_support_only = bool(personal_support_only)
        self.share_concept_embeddings = bool(share_concept_embeddings)

        self.register_buffer("q_matrix", q_matrix)

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
            personal_query_row_budget=self.personal_query_row_budget,
            personal_neighbor_row_budget=self.personal_neighbor_row_budget,
            personal_support_only=self.personal_support_only,
            enable_module=self.enable_module1,
        )

        if self.share_concept_embeddings:
            self._tie_concept_embeddings()

        self.diagnosis_head = CognitiveDiagnosisHead(
            knowledge_dim=knowledge_dim,
            use_weight_norm=self.enable_module1,
        )
        self.exercise_encoder = ExerciseDifficultyEncoder(num_exercises=num_exercises)

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

    def _build_query_enhanced_state(
        self,
        knowledge_state: torch.Tensor,
        relation_used: torch.Tensor,
        concept_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            (self.graph_query_writeback_scale <= 0 and self.graph_query_writeback_2hop_scale <= 0)
            or not self.enable_module1
        ):
            zero = knowledge_state.new_tensor(0.0)
            return knowledge_state, zero, zero, zero

        hop1 = self._aggregate_with_relation(knowledge_state, relation_used)
        hop2 = self._aggregate_with_relation(hop1, relation_used)
        query_mask = concept_mask.float()
        query_rows = query_mask.unsqueeze(-1).bool()
        graph_delta = (
            self.graph_query_writeback_scale * (hop1 - knowledge_state)
            + self.graph_query_writeback_2hop_scale * (hop2 - hop1)
        )
        enhanced = torch.where(query_rows, knowledge_state + graph_delta, knowledge_state)
        delta = (enhanced - knowledge_state).pow(2).mean().sqrt()

        query_weight = query_mask.unsqueeze(-1)
        query_denom = (query_weight.sum() * float(knowledge_state.size(-1))).clamp(min=1.0)
        query_row_graph_delta = (
            ((enhanced - knowledge_state).pow(2) * query_weight).sum() / query_denom
        ).sqrt()

        query_seed = query_mask / query_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        query_support = self._aggregate_with_relation(query_seed.unsqueeze(-1), relation_used).squeeze(-1)
        readout_query_support_mass = (query_support * (1.0 - query_mask)).sum(dim=1).mean()
        return enhanced, delta, query_row_graph_delta, readout_query_support_mass

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
        device = student_ids.device

        # ========== 0) Q mask ==========
        q_vector = concept_vector if concept_vector is not None else self.q_matrix[exercise_ids]  # (B,C)

        # ========== 1) Module 1：结构输出 ==========
        s_out = self.structure_module(
            student_ids,
            identity_relations=self.identity_relations,
            concept_mask=q_vector,
        )
        relation_matrices = s_out["relation_matrices"]
        relation_used = s_out["relation_used"]
        knowledge_state = s_out["knowledge_state"]
        student_repr = s_out["student_repr"]
        gate_alpha = s_out["alpha"]
        personal_matrices = s_out["personal_matrices"]
        prediction_state, readout_query_delta, query_row_graph_delta, readout_query_support_mass = self._build_query_enhanced_state(
            knowledge_state=knowledge_state,
            relation_used=relation_used,
            concept_mask=q_vector,
        )

        # ========== 2) 固定预测头 D ==========
        b, a = self.exercise_encoder(exercise_ids)
        if return_details:
            irt_logit, diag_details = self.diagnosis_head(
                knowledge_state=prediction_state,
                concept_mask=q_vector,
                b=b,
                a=a,
                return_details=True,
            )
        else:
            irt_logit = self.diagnosis_head(
                knowledge_state=prediction_state,
                concept_mask=q_vector,
                b=b,
                a=a,
                return_details=False,
            )
            diag_details = None

        total_logit = irt_logit

        # ========== 3) 返回 logits 或 prob ==========
        out_main = total_logit if return_logits else torch.sigmoid(total_logit)

        if not return_details:
            return out_main

        # ========== 4) details（用于正则、可解释输出、排查消融是否生效） ==========
        details: Dict[str, torch.Tensor] = {
            "enable_module1": torch.tensor(int(self.enable_module1), device=device),
            "use_concept_graph": torch.tensor(int(self.use_concept_graph), device=device),
            "use_personal_graph": torch.tensor(int(self.use_personal_graph), device=device),
            "share_concept_embeddings": torch.tensor(int(self.share_concept_embeddings), device=device),
            "relation_matrices": relation_matrices,
            "relation_used": relation_used,
            "knowledge_state": knowledge_state,
            "prediction_state": prediction_state,
            "student_repr": student_repr,
            "q_vector": q_vector,
            "irt_b": b.detach(),
            "irt_a": a.detach(),
            "irt_logit": irt_logit.detach(),
            "logits": total_logit.detach(),
            "relation_identity_delta": s_out["relation_identity_delta"].detach(),
            "knowledge_state_graph_delta": s_out["knowledge_state_graph_delta"].detach(),
            "knowledge_state_personal_delta": s_out["knowledge_state_personal_delta"].detach(),
            "readout_query_delta": readout_query_delta.detach(),
            "query_row_graph_delta": query_row_graph_delta.detach(),
            "readout_query_support_mass": readout_query_support_mass.detach(),
        }
        details["irt_logit_for_reg"] = irt_logit
        details["readout_local_row_ratio"] = q_vector.float().mean().detach()

        if diag_details is not None:
            details.update(diag_details)

        if self.use_personal_graph:
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
                "personal_state_mix",
                "personal_student_mix",
                "personal_student_adapter_scale",
                "personal_context_adapter_scale",
                "personal_delta_nonfinite_count",
                "personal_logits_nonfinite_count",
                "personal_matrix_nonfinite_count",
                "personal_bad_row_count",
                "personal_fallback_row_count",
                "personal_logits_absmax",
                "personal_delta_absmax",
                "local_row_ratio",
                "personal_support_density",
                "query_state_norm",
                "query_row_personal_delta",
                "neighbor_row_personal_delta",
                "personal_row_budget_mean",
                "personal_query_row_std",
                "local_row_mask",
                "active_row_index",
                "active_row_valid_mask",
                "support_col_index",
                "support_valid_mask",
                "global_support_prob",
                "posterior_prob",
            ):
                if s_out.get(key) is not None:
                    details[key] = s_out[key].detach()
            if s_out.get("personal_gate_id_scale") is not None:
                details["personal_gate_id_scale"] = s_out["personal_gate_id_scale"]
            if s_out.get("personal_generator_id_scale") is not None:
                details["personal_generator_id_scale"] = s_out["personal_generator_id_scale"]
            if s_out.get("personal_alpha_bias_scale") is not None:
                details["personal_alpha_bias_scale"] = s_out["personal_alpha_bias_scale"]
            details["personal_matrices"] = None
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
        device = relation_matrices.device
        terms: Dict[str, torch.Tensor] = {
            "graph_entropy": torch.tensor(0.0, device=device),
            "graph_diag": torch.tensor(0.0, device=device),
            "graph_uniform": torch.tensor(0.0, device=device),
            "graph_reg_scale": torch.tensor(1.0, device=device),
            "prediction_l2": torch.tensor(0.0, device=device),
            "personal_sparse": torch.tensor(0.0, device=device),
            "alpha_var": torch.tensor(0.0, device=device),
            "alpha_collapse": torch.tensor(0.0, device=device),
        }
        graph_reg_ramp_t = relation_matrices.new_tensor(self._get_graph_reg_ramp())
        personal_reg_ramp_t = relation_matrices.new_tensor(self._get_linear_warmup(self.personal_reg_warmup_epochs))
        if details is not None:
            details["graph_reg_ramp"] = graph_reg_ramp_t.detach()
            details["personal_reg_ramp"] = personal_reg_ramp_t.detach()

        # (1) Global graph entropy band penalty
        if self.enable_module1 and self.use_concept_graph and self.lambda_graph_entropy > 0:
            if self.structure_module.relation_learning is not None:
                # Keep raw entropy computation for diagnostics/logging.
                entropy = self.structure_module.relation_learning.get_entropy_sparsity(relation_matrices)
                num_nodes = max(2, int(relation_matrices.size(-1)))
                h_norm = entropy / (math.log(float(num_nodes)) + 1e-8)

                h_min = torch.tensor(self.graph_entropy_min, device=device, dtype=entropy.dtype)
                h_max = torch.tensor(self.graph_entropy_max, device=device, dtype=entropy.dtype)
                pen = F.relu(h_min - h_norm) + F.relu(h_norm - h_max)
                terms["graph_entropy"] = self.lambda_graph_entropy * pen * graph_reg_ramp_t

                if details is not None:
                    details["graph_entropy_raw"] = entropy.detach()
                    details["graph_entropy_norm"] = h_norm.detach()
                    details["graph_entropy_pen"] = pen.detach()

                if self.lambda_graph_diag > 0:
                    diag_mass = torch.diagonal(relation_matrices, dim1=-2, dim2=-1).mean()
                    terms["graph_diag"] = self.lambda_graph_diag * diag_mass * graph_reg_ramp_t
                    if details is not None:
                        details["graph_diag_mass"] = diag_mass.detach()

                num_nodes = max(2, int(relation_matrices.size(-1)))
                uniform_val = 1.0 / float(num_nodes)
                uniform_dist = torch.sqrt(
                    torch.clamp((relation_matrices - uniform_val).pow(2).mean(), min=1e-12)
                )
                identity = torch.eye(
                    num_nodes, device=device, dtype=relation_matrices.dtype
                ).unsqueeze(0).expand_as(relation_matrices)
                identity_dist = torch.sqrt(
                    torch.clamp((relation_matrices - identity).pow(2).mean(), min=1e-12)
                )

                if self.lambda_graph_uniform > 0:
                    uniform_margin = torch.tensor(
                        self.graph_uniform_margin, device=device, dtype=uniform_dist.dtype
                    )
                    uniform_pen = F.relu(uniform_margin - uniform_dist)
                    terms["graph_uniform"] = (
                        self.lambda_graph_uniform * uniform_pen * graph_reg_ramp_t
                    )
                    if details is not None:
                        details["graph_uniform_pen"] = uniform_pen.detach()

                if details is not None:
                    details["graph_to_uniform_l2"] = uniform_dist.detach()
                    details["graph_to_identity_l2"] = identity_dist.detach()

                tau = F.softplus(self.structure_module.relation_learning.tau_raw) + 1e-6
                if details is not None:
                    details["graph_tau_mean"] = tau.mean().detach()
                    details["graph_tau_std"] = tau.std(unbiased=False).detach()

        # Graph regularization cap relative to base loss.
        # Only scales graph-specific terms; non-graph regularizers remain unchanged.
        if base_loss is not None and self.graph_reg_cap_ratio > 0:
            graph_reg_raw = terms["graph_entropy"] + terms["graph_diag"] + terms["graph_uniform"]
            cap = self.graph_reg_cap_ratio * base_loss.detach().abs()
            denom = graph_reg_raw.detach().abs() + 1e-8
            scale = torch.clamp(cap / denom, max=1.0)
            scale = torch.where(torch.isfinite(scale), scale, torch.ones_like(scale))
            terms["graph_reg_scale"] = scale.detach()
            terms["graph_entropy"] = terms["graph_entropy"] * scale
            terms["graph_diag"] = terms["graph_diag"] * scale
            terms["graph_uniform"] = terms["graph_uniform"] * scale
            if details is not None:
                details["graph_reg_raw"] = graph_reg_raw.detach()
                details["graph_reg_cap"] = cap.detach()
                details["graph_reg_scale"] = scale.detach()

        # (2) Prediction-head L2
        if self.prediction_l2_lambda > 0:
            reg_terms = []
            if self.exercise_encoder.b is not None and self.exercise_encoder.a_raw is not None:
                reg_terms.extend(
                    [
                        self.exercise_encoder.b.weight.pow(2).mean(),
                        self.exercise_encoder.a_raw.weight.pow(2).mean(),
                    ]
                )
            if len(reg_terms) > 0:
                terms["prediction_l2"] = self.prediction_l2_lambda * sum(reg_terms)

        # (3) Personal graph regularizers
        if self.enable_module1 and self.use_personal_graph and details is not None:
            posterior_prob = details.get("posterior_prob")
            support_valid_mask = details.get("support_valid_mask")
            if posterior_prob is not None and support_valid_mask is not None and self.lambda_sparse_personal > 0:
                terms["personal_sparse"] = (
                    self.lambda_sparse_personal
                    * _masked_sparse_row_entropy(posterior_prob, support_valid_mask.bool())
                    * personal_reg_ramp_t
                )
            elif (
                "personal_matrices" in details
                and details["personal_matrices"] is not None
                and self.lambda_sparse_personal > 0
            ):
                pm = details["personal_matrices"]
                terms["personal_sparse"] = (
                    self.lambda_sparse_personal * self._row_entropy(pm) * personal_reg_ramp_t
                )

            alpha_for_reg = details.get("alpha_effective", details.get("alpha"))
            if alpha_for_reg is not None and self.lambda_alpha > 0:
                alpha_flat = alpha_for_reg.view(-1)
                alpha_var = alpha_flat.var() + 1e-6
                if "alpha_student_bias" in details and details["alpha_student_bias"] is not None:
                    alpha_bias_flat = details["alpha_student_bias"].view(-1)
                    alpha_var = alpha_var + 0.5 * (alpha_bias_flat.var() + 1e-6)
                    details["alpha_bias_std_runtime"] = alpha_bias_flat.std(unbiased=False).detach()
                terms["alpha_var"] = -self.lambda_alpha * alpha_var * personal_reg_ramp_t

            if alpha_for_reg is not None and self.lambda_alpha_min > 0:
                alpha_flat = alpha_for_reg.view(-1)
                alpha_std = alpha_flat.std(unbiased=False)
                alpha_target = torch.tensor(self.alpha_min_target, device=device, dtype=alpha_std.dtype)
                alpha_pen = F.relu(alpha_target - alpha_std)
                if "alpha_student_bias" in details and details["alpha_student_bias"] is not None:
                    alpha_bias_flat = details["alpha_student_bias"].view(-1)
                    alpha_bias_std = alpha_bias_flat.std(unbiased=False)
                    alpha_pen = alpha_pen + 0.5 * F.relu(alpha_target - alpha_bias_std)
                    details["alpha_bias_std_runtime"] = alpha_bias_std.detach()
                if details.get("personal_delta_pre_softmax_norm") is not None:
                    delta_norm = details["personal_delta_pre_softmax_norm"]
                    delta_target = alpha_target
                    alpha_pen = alpha_pen + 0.5 * F.relu(delta_target - delta_norm)
                    details["personal_delta_pre_softmax_norm_runtime"] = delta_norm.detach()
                if details.get("personal_delta_student_std") is not None:
                    delta_student_std = details["personal_delta_student_std"]
                    delta_student_target = 0.5 * alpha_target
                    alpha_pen = alpha_pen + 0.5 * F.relu(delta_student_target - delta_student_std)
                    details["personal_delta_student_std_runtime"] = delta_student_std.detach()
                terms["alpha_collapse"] = self.lambda_alpha_min * alpha_pen * personal_reg_ramp_t
                details["alpha_std_runtime"] = alpha_std.detach()
                details["alpha_collapse_pen"] = alpha_pen.detach()

        total = (
            terms["graph_entropy"]
            + terms["graph_diag"]
            + terms["graph_uniform"]
            + terms["prediction_l2"]
            + terms["personal_sparse"]
            + terms["alpha_var"]
            + terms["alpha_collapse"]
        )
        terms["total"] = total
        return terms

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
