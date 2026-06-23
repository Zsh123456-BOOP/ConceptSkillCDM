
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
    可解释概念邻接 A_h（row-stochastic）。

    A starts from train-only evidence and a small set of calibrated corrections:
    - item cooccurrence prior：同一习题内概念共同出现的强弱；
    - sequence transition prior：同一学生训练序列中概念接续的强弱；
    - receiver_bias：某个概念作为依赖来源的全局倾向；
    - self_loop_logit：保留本概念状态的强度。
    - learned low-rank edge correction：零门控初始化，由训练数据决定是否启用。
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
        sequence_prior_matrix: Optional[torch.Tensor] = None,
        prior_logit_scale: float = 0.0,
        learnable_edges: bool = True,
    ):
        super().__init__()
        self.num_concepts = int(num_concepts)
        self.concept_dim = int(concept_dim)
        self.num_heads = int(num_heads)
        self.topk = topk
        self.allow_self_loop = bool(allow_self_loop)
        self.identity_residual = float(max(0.0, min(1.0, identity_residual))) if self.allow_self_loop else 0.0
        self.edge_bias_rank = max(1, int(edge_bias_rank))
        self.prior_logit_scale = max(0.0, float(prior_logit_scale))
        # 学习型边权：在 train-only 先验之上叠加一个由概念坐标决定的低秩双线性项。
        # 这样边权不再是被冻结的共现/转移统计量，而是端到端训练得到的可达强度。
        # 初始化为 0，使图在训练开始时与纯先验图完全一致（零回归风险）。
        self.learnable_edges = bool(learnable_edges) and self.num_concepts > 1

        item_scores, item_support_scores = self._build_prior_buffers(prior_matrix, "prior_matrix")
        seq_scores, seq_support_scores = self._build_prior_buffers(sequence_prior_matrix, "sequence_prior_matrix")
        self.has_item_prior = prior_matrix is not None
        self.has_sequence_prior = sequence_prior_matrix is not None
        self.register_buffer("prior_logit_scores", item_scores, persistent=False)
        self.register_buffer("sequence_prior_logit_scores", seq_scores, persistent=False)
        self.register_buffer("item_support_mask", self._build_topk_support_mask(item_support_scores), persistent=False)
        self.register_buffer("sequence_support_mask", self._build_topk_support_mask(seq_support_scores), persistent=False)

        # 保留 concept_embeddings 只是为了和 knowledge encoder 共享同一概念坐标系；A 的边权不再由它生成。
        self.concept_embeddings = nn.Parameter(torch.zeros(num_concepts, concept_dim))
        init_strength = max(1e-4, self.prior_logit_scale if self.prior_logit_scale > 0 else 1.0)
        item_init, seq_init = self._build_source_specialized_strength_init(init_strength)
        self.prior_strength_raw = nn.Parameter(self._inverse_softplus(item_init))
        self.sequence_prior_strength_raw = nn.Parameter(self._inverse_softplus(seq_init))
        self.receiver_bias = nn.Parameter(torch.zeros(num_heads, num_concepts))
        self.self_loop_logit = nn.Parameter(torch.full((num_heads,), 0.75))
        self.temperature_raw = nn.Parameter(torch.full((num_heads,), float(tau_init)))

        if self.learnable_edges:
            r = self.edge_bias_rank
            self.edge_query_proj = nn.Parameter(torch.empty(num_heads, concept_dim, r))
            self.edge_key_proj = nn.Parameter(torch.empty(num_heads, concept_dim, r))
            nn.init.normal_(self.edge_query_proj, std=0.02)
            nn.init.normal_(self.edge_key_proj, std=0.02)
            # 零初始化的逐头门控：训练开始时学习边项贡献为 0。
            self.learned_edge_scale = nn.Parameter(torch.zeros(num_heads))
        else:
            self.register_parameter("edge_query_proj", None)
            self.register_parameter("edge_key_proj", None)
            self.register_parameter("learned_edge_scale", None)

        self.dropout = nn.Dropout(dropout)
        self._last_support_diagnostics: Dict[str, torch.Tensor] = {}
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.normal_(self.concept_embeddings, mean=0.0, std=0.02)

    @staticmethod
    def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
        value = value.detach().float().clamp(min=1e-4)
        return torch.log(torch.expm1(value).clamp(min=1e-12))

    def _build_source_specialized_strength_init(self, base_strength: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize heads with interpretable item-vs-sequence evidence roles."""
        base = max(1e-4, float(base_strength))
        item = torch.full((self.num_heads,), base, dtype=torch.float32)
        seq_base = base if self.has_sequence_prior else 1e-4
        seq = torch.full((self.num_heads,), seq_base, dtype=torch.float32)
        if self.has_item_prior and self.has_sequence_prior and self.num_heads >= 2:
            high = max(1e-4, base * 1.45)
            low = max(1e-4, base * 0.65)
            for h in range(self.num_heads):
                if h % 2 == 0:
                    item[h] = high
                    seq[h] = low
                else:
                    item[h] = low
                    seq[h] = high
        return item, seq

    def _build_prior_buffers(self, prior_matrix: Optional[torch.Tensor], name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if prior_matrix is None:
            zeros = torch.zeros(self.num_concepts, self.num_concepts, dtype=torch.float32)
            return zeros, zeros
        prior = prior_matrix.detach().float()
        if tuple(prior.shape) != (self.num_concepts, self.num_concepts):
            raise ValueError(
                f"{name} shape must be ({self.num_concepts}, {self.num_concepts}), got {tuple(prior.shape)}"
            )
        if self.num_concepts == 1:
            ones = torch.ones(1, 1, dtype=torch.float32)
            return ones, ones
        eye = torch.eye(self.num_concepts, dtype=torch.float32, device=prior.device)
        support_scores = prior.clamp(min=0.0) * (1.0 - eye)
        prior = prior.clamp(min=1e-4)
        prior = prior / prior.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        prior_scores = prior.log()
        prior_scores = prior_scores - prior_scores.mean(dim=-1, keepdim=True)
        prior_scores = prior_scores.clamp(min=-4.0, max=4.0)
        return prior_scores.cpu(), support_scores.cpu()

    def _build_topk_support_mask(self, support_scores: torch.Tensor) -> torch.Tensor:
        C = self.num_concepts
        mask = torch.zeros(C, C, dtype=torch.bool)
        if self.topk is None or self.topk <= 0 or C <= 1:
            return mask
        k = min(int(self.topk), C - 1)
        eye = torch.eye(C, dtype=torch.bool)
        for row_idx in range(C):
            row = support_scores[row_idx].detach().float().clone()
            positive = (row > 0.0) & (~eye[row_idx])
            positive_count = int(positive.sum().item())
            if positive_count <= 0:
                continue
            take = min(k, positive_count)
            row = row.masked_fill(~positive, float("-inf"))
            _, idx = torch.topk(row, k=take)
            mask[row_idx, idx] = True
        return mask

    @staticmethod
    def _apply_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
        """每行只保留 top-k，其余置为 -inf，使 softmax 后精确为 0。"""
        vals, idx = torch.topk(scores, k=k, dim=-1)
        masked = torch.full_like(scores, float("-inf"))
        masked.scatter_(dim=-1, index=idx, src=vals)
        return masked

    def _apply_source_balanced_topk(
        self,
        scores: torch.Tensor,
        *,
        item_mask: torch.Tensor,
        sequence_mask: torch.Tensor,
        self_mask: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Keep observed item evidence from being crowded out by dense sequence evidence."""
        C = scores.size(0)
        total_budget = min(max(1, int(k)), C)
        offdiag_budget = total_budget - 1 if self.allow_self_loop else total_budget
        selected = torch.zeros((C, C), device=scores.device, dtype=torch.bool)
        offdiag = ~self_mask
        if self.allow_self_loop:
            selected = selected | self_mask
        if offdiag_budget <= 0:
            return scores.masked_fill(~selected, float("-inf"))

        item_quota_default = max(1, int(math.ceil(offdiag_budget * 0.25)))
        sequence_quota_default = max(1, int(math.ceil(offdiag_budget * 0.50)))

        def _offdiag_selected_count(mask: torch.Tensor) -> torch.Tensor:
            return (mask & offdiag).sum(dim=-1)

        def _pick(
            source_mask: torch.Tensor,
            quota: int,
            current: torch.Tensor,
            row_limit: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            take = min(max(0, int(quota)), C)
            if take <= 0:
                return current
            available = source_mask & offdiag & (~current)
            vals, idx = torch.topk(scores.masked_fill(~available, float("-inf")), k=take, dim=-1)
            keep = torch.isfinite(vals)
            if row_limit is not None:
                rank = torch.arange(take, device=scores.device).view(1, take)
                keep = keep & (rank < row_limit.clamp(min=0).view(-1, 1))
            picked = torch.zeros_like(current)
            picked.scatter_(dim=-1, index=idx, src=keep)
            return current | (picked & available)

        selected = _pick(item_mask, item_quota_default, selected)
        remaining = offdiag_budget - _offdiag_selected_count(selected)
        selected = _pick(sequence_mask, sequence_quota_default, selected, remaining)
        remaining = offdiag_budget - _offdiag_selected_count(selected)
        selected = _pick(item_mask | sequence_mask, offdiag_budget, selected, remaining)

        if not self.allow_self_loop:
            row_has_support = selected.any(dim=-1)
            fallback_take = min(max(1, offdiag_budget), C)
            vals, idx = torch.topk(scores.masked_fill(~offdiag, float("-inf")), k=fallback_take, dim=-1)
            fallback = torch.zeros_like(selected)
            fallback.scatter_(dim=-1, index=idx, src=torch.isfinite(vals))
            selected = torch.where(row_has_support.view(-1, 1), selected, fallback)

        return scores.masked_fill(~selected, float("-inf"))

    def _update_support_diagnostics(self, relation_matrices: torch.Tensor) -> None:
        """Record how much item/sequence evidence survives final support pruning."""
        H, C, _ = relation_matrices.shape
        device = relation_matrices.device
        dtype = relation_matrices.dtype
        eye = torch.eye(C, device=device, dtype=torch.bool)
        item_mask = self.item_support_mask.to(device=device)
        seq_mask = self.sequence_support_mask.to(device=device)
        candidate_mask = item_mask | seq_mask
        if self.allow_self_loop:
            candidate_mask = candidate_mask | eye
        final_mask = relation_matrices.detach() > 1e-12

        def _survival_rate(source_mask: torch.Tensor) -> torch.Tensor:
            denom = (source_mask.to(dtype=dtype).sum() * float(max(1, H))).clamp(min=1.0)
            return (
                final_mask
                & source_mask.view(1, C, C)
            ).to(dtype=dtype).sum() / denom

        overlap_union = (item_mask | seq_mask).to(dtype=dtype).sum().clamp(min=1.0)
        self._last_support_diagnostics = {
            "support_candidate_size_mean": candidate_mask.to(dtype=dtype).sum(dim=-1).mean().detach(),
            "support_final_size_mean": final_mask.to(dtype=dtype).sum(dim=-1).mean().detach(),
            "support_item_survival_rate": _survival_rate(item_mask).detach(),
            "support_seq_survival_rate": _survival_rate(seq_mask).detach(),
            "support_item_seq_overlap": ((item_mask & seq_mask).to(dtype=dtype).sum() / overlap_union).detach(),
            "support_self_retention_rate": _survival_rate(eye).detach(),
        }

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            relation_matrices: (H, C, C)  每行归一的邻接矩阵
            concept_embeddings: (C, D)    概念 embedding（可能与 encoder 绑定共享）
        """
        C = self.num_concepts
        x = self.concept_embeddings  # (C, D)

        # ---- Fix #3：退化情况 C==1 防 NaN ----
        if C == 1:
            A = torch.ones((self.num_heads, 1, 1), device=x.device, dtype=x.dtype)
            self._update_support_diagnostics(A)
            return A, x

        tau = F.softplus(self.temperature_raw) + 1e-6  # (H,)
        item_prior_scores = self.prior_logit_scores.to(device=x.device, dtype=x.dtype)
        seq_prior_scores = self.sequence_prior_logit_scores.to(device=x.device, dtype=x.dtype)
        item_strength = F.softplus(self.prior_strength_raw).to(device=x.device, dtype=x.dtype)
        seq_strength = F.softplus(self.sequence_prior_strength_raw).to(device=x.device, dtype=x.dtype)
        rels = []

        for h in range(self.num_heads):
            scores = item_strength[h] * item_prior_scores + seq_strength[h] * seq_prior_scores
            scores = scores + self.receiver_bias[h].to(device=x.device, dtype=x.dtype).view(1, C)
            if self.learnable_edges:
                # 由学习到的概念坐标决定的可达强度（低秩双线性）。
                # 行 c 为 query 概念，列 k 为支持概念，方向与先验一致。
                q_proj = x @ self.edge_query_proj[h].to(device=x.device, dtype=x.dtype)  # (C, r)
                k_proj = x @ self.edge_key_proj[h].to(device=x.device, dtype=x.dtype)    # (C, r)
                learned_scores = (q_proj @ k_proj.t()) / math.sqrt(max(1, q_proj.size(-1)))
                scores = scores + self.learned_edge_scale[h].to(dtype=scores.dtype) * learned_scores
            scores = scores / tau[h]
            scores = scores - scores.mean(dim=-1, keepdim=True)

            eye = torch.eye(C, device=scores.device, dtype=torch.bool)
            if self.allow_self_loop:
                scores = scores + self.self_loop_logit[h].to(dtype=scores.dtype) * eye.to(dtype=scores.dtype)
            else:
                scores = scores.masked_fill(eye, float("-inf"))

            if self.topk is not None and 0 < self.topk < C:
                item_mask = self.item_support_mask.to(device=scores.device)
                sequence_mask = self.sequence_support_mask.to(device=scores.device)
                support_mask = item_mask | sequence_mask
                if self.allow_self_loop:
                    support_mask = support_mask | eye
                row_has_support = support_mask.any(dim=-1, keepdim=True)
                if bool(row_has_support.all()):
                    scores = self._apply_source_balanced_topk(
                        scores.masked_fill(~support_mask, float("-inf")),
                        item_mask=item_mask,
                        sequence_mask=sequence_mask,
                        self_mask=eye,
                        k=self.topk,
                    )
                else:
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
        self._update_support_diagnostics(relation_matrices)
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
                # Dense per-sample adjacency path.
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
        graph_evidence_scale: float = 1.0,
        graph_evidence_reliability_smoothing: float = 8.0,
        evidence_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_students = int(num_students)
        self.num_concepts = int(num_concepts)
        self.knowledge_dim = int(knowledge_dim)
        self.gnn_residual_weight = float(gnn_residual_weight)
        self.propagation_alpha = max(0.0, min(1.0, float(propagation_alpha)))
        self.graph_evidence_scale = max(0.0, float(graph_evidence_scale))
        self.graph_evidence_reliability_smoothing = max(0.0, float(graph_evidence_reliability_smoothing))

        self.student_global = nn.Embedding(num_students, knowledge_dim)
        self.concept_emb = nn.Embedding(num_concepts, knowledge_dim)

        # Evidence encoder：把该学生在训练集中对每个概念的观测证据
        # （掌握 logit、近期表现、观测可靠度）编码为节点初始状态的一部分。
        # 这是机制的核心：图传播的是“这名学生已观测概念上的真实证据”，
        # 由此把证据沿概念关系路由到未观测的目标概念。
        # 最后一层零初始化 → 训练开始时初始状态严格等于 concept_emb + student_global。
        # Features: [mastery, recency, reliability, observed-indicator].
        # The vector is gated once by reliability at the end of encode_evidence,
        # which makes the effective evidence ~linear in reliability and keeps the
        # held-out (count==0) node exactly blank — so we do NOT pre-multiply the
        # mastery/recency channels by reliability here (avoids redundant rel**2).
        self.evidence_feature_dim = 4
        if self.graph_evidence_scale > 0.0:
            hid = int(evidence_hidden_dim) if evidence_hidden_dim else max(8, knowledge_dim // 2)
            self.evidence_proj = nn.Sequential(
                nn.Linear(self.evidence_feature_dim, hid),
                nn.GELU(),
                nn.Linear(hid, knowledge_dim),
            )
            nn.init.zeros_(self.evidence_proj[-1].weight)
            nn.init.zeros_(self.evidence_proj[-1].bias)
        else:
            self.evidence_proj = None

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

    def encode_evidence(
        self,
        mastery: Optional[torch.Tensor] = None,
        recent: Optional[torch.Tensor] = None,
        count: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Build a (B, C, D) per-(student, concept) evidence state.

        - ``mastery`` / ``recent``: (B, C) train-only mastery/recency logits.
        - ``count``: (B, C) train-only observation counts (reliability source).

        The evidence vector is multiplied by reliability so that concepts the
        student never observed (e.g. the held-out target concept) contribute
        exactly zero and can only be estimated through graph propagation.
        Returns ``None`` when evidence is disabled or unavailable.
        """
        if self.evidence_proj is None or self.graph_evidence_scale <= 0.0 or mastery is None:
            return None
        weight = self.concept_emb.weight
        device, dtype = weight.device, weight.dtype
        mastery = mastery.to(device=device, dtype=dtype)
        recent = recent.to(device=device, dtype=dtype) if recent is not None else torch.zeros_like(mastery)
        if count is not None:
            count = count.to(device=device, dtype=dtype).clamp(min=0.0)
            smooth = self.graph_evidence_reliability_smoothing
            reliability = count / (count + smooth) if smooth > 0.0 else (count > 0).to(dtype=dtype)
            observed = (count > 0).to(dtype=dtype)
        else:
            reliability = torch.ones_like(mastery)
            observed = torch.ones_like(mastery)
        features = torch.stack(
            [mastery, recent, reliability, observed],
            dim=-1,
        )  # (B, C, 4)
        evidence = self.evidence_proj(features)  # (B, C, D)
        # Final reliability gate: count==0 -> reliability==0 -> evidence==0 exactly,
        # so the held-out target node carries no direct evidence and must be filled
        # by graph propagation from the student's observed concepts.
        evidence = evidence * reliability.unsqueeze(-1) * self.graph_evidence_scale
        return evidence

    def compose_initial_state(
        self,
        student_ids: torch.Tensor,
        evidence_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = student_ids.size(0)
        s = self.student_global(student_ids)
        c = self.concept_emb.weight.unsqueeze(0).expand(B, -1, -1)
        h0 = self.dropout(c + s.unsqueeze(1))
        if evidence_state is not None:
            h0 = h0 + evidence_state.to(device=h0.device, dtype=h0.dtype)
        return h0

    def forward(
        self,
        student_ids: torch.Tensor,
        relation_matrices: torch.Tensor,
        evidence_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h0 = self.compose_initial_state(student_ids, evidence_state=evidence_state)
        h = h0
        hop_states = [h0]

        for gnn, ln in zip(self.gnn_layers, self.layer_norms):
            propagated = gnn(h, relation_matrices)
            candidate = F.relu(ln(h + self.gnn_residual_weight * propagated))
            h = (1.0 - self.propagation_alpha) * h + self.propagation_alpha * candidate
            hop_states.append(h)

        hop_weights = F.softmax(self.hop_mix_logits[: len(hop_states)], dim=0)
        mixed = sum(weight * state for weight, state in zip(hop_weights, hop_states))
        return mixed  # (B, C, D)
