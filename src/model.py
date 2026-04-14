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
        """
        outputs = []
        for h in range(self.num_heads):
            Wh = self.head_transforms[h](x)  # (B, C, Dout)

            if isinstance(relation_matrices, dict):
                # 优化路径：个性化图混合，逐 head 计算避免 4D 张量
                global_A = relation_matrices["global_matrices"][h]  # (C, C)
                personal_all = relation_matrices["personal_matrices"]
                gate_all = relation_matrices["gate_alpha"]
                if personal_all.dim() == 4:
                    personal_A = personal_all[:, h, :, :]  # (B, C, C)
                else:
                    personal_A = personal_all  # (B, C, C)
                if gate_all.dim() == 2:
                    gate = gate_all[:, h].view(-1, 1, 1)  # (B, 1, 1)
                else:
                    gate = gate_all.view(-1, 1, 1)  # (B, 1, 1)
                 
                global_A = global_A.to(dtype=Wh.dtype)
                personal_A = personal_A.to(dtype=Wh.dtype)
                
                # 混合：A = (1-gate)*global + gate*personal，逐样本计算
                global_A_expanded = global_A.unsqueeze(0)  # (1, C, C)
                A = (1.0 - gate) * global_A_expanded + gate * personal_A  # (B, C, C)
                A = A / (A.sum(dim=-1, keepdim=True) + 1e-12)
                out = torch.bmm(A, Wh)  # (B,C,C) @ (B,C,D) -> (B,C,D)
                
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


# ======================================================
# 模块 1（E）：个性化图相关组件 - AdaptiveGate / PersonalRelationGenerator
# ======================================================

class AdaptiveGate(nn.Module):
    """个性化图混合系数 alpha（B,H,1,1）。

    设计原则：
    - 保留一条显式 student bias -> alpha 的短路径，避免个体信号被上下文支路淹没；
    - student-id 分支与 context 分支分开建模，避免个体信号被 LayerNorm 洗掉；
    - context 只做修正项，不取代 student-specific 路径。
    """

    def __init__(
        self,
        student_dim: int,
        context_dim: int,
        num_heads: int,
        max_alpha: float = 0.35,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        hid = max(1, max(student_dim, context_dim) // 2 if hidden_dim is None else int(hidden_dim))
        self.context_norm = nn.LayerNorm(context_dim)
        self.student_proj = nn.Linear(student_dim, hid, bias=False)
        self.context_proj = nn.Linear(context_dim, hid)
        self.hidden_proj = nn.Linear(hid, hid)
        self.student_to_logit = nn.Linear(student_dim, self.num_heads, bias=False)
        self.context_to_logit = nn.Linear(context_dim, self.num_heads)
        self.out = nn.Linear(hid, self.num_heads)
        self.max_alpha = float(max_alpha)

        nn.init.xavier_normal_(self.student_proj.weight)
        nn.init.xavier_normal_(self.context_proj.weight)
        nn.init.zeros_(self.context_proj.bias)
        nn.init.xavier_normal_(self.hidden_proj.weight)
        nn.init.zeros_(self.hidden_proj.bias)
        nn.init.xavier_normal_(self.student_to_logit.weight, gain=0.5)
        nn.init.xavier_normal_(self.context_to_logit.weight, gain=0.25)
        nn.init.zeros_(self.context_to_logit.bias)
        nn.init.xavier_normal_(self.out.weight, gain=0.5)
        nn.init.constant_(self.out.bias, -1.0)

    def forward(
        self,
        student_embedding: torch.Tensor,
        context_repr: torch.Tensor,
        student_bias: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        context_norm = self.context_norm(context_repr)
        context_hidden = self.context_proj(context_norm)
        student_hidden = self.student_proj(student_embedding)
        hidden = F.silu(context_hidden + student_hidden)
        hidden = F.silu(self.hidden_proj(hidden))
        alpha_logit = (
            student_bias
            + self.student_to_logit(student_embedding)
            + self.context_to_logit(context_norm)
            + self.out(hidden)
        )
        alpha = self.max_alpha * torch.sigmoid(alpha_logit)
        return alpha.unsqueeze(-1).unsqueeze(-1), alpha_logit.unsqueeze(-1).unsqueeze(-1)


class PersonalRelationGenerator(nn.Module):
    """显式 student code + context residual 生成 per-head 个性化邻接 logits（B,H,C,C）。"""

    def __init__(
        self,
        student_dim: int,
        context_dim: int,
        num_concepts: int,
        num_heads: int,
        rank: int = 4,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.student_norm = nn.LayerNorm(student_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.num_concepts = int(num_concepts)
        self.num_heads = int(num_heads)
        self.rank = int(rank)
        hidden = max(1, max(student_dim, context_dim) // 2 if hidden_dim is None else int(hidden_dim))

        self.context_proj = nn.Linear(context_dim, hidden)
        self.hidden_proj = nn.Linear(hidden, hidden)
        self.base_u = nn.Parameter(torch.randn(self.num_heads, num_concepts, rank) * 0.02)
        self.base_v = nn.Parameter(torch.randn(self.num_heads, num_concepts, rank) * 0.02)
        self.student_basis_u = nn.Parameter(torch.randn(self.num_heads, num_concepts, rank, student_dim) * 0.02)
        self.student_basis_v = nn.Parameter(torch.randn(self.num_heads, num_concepts, rank, student_dim) * 0.02)
        self.context_to_u = nn.Linear(hidden, self.num_heads * num_concepts * rank, bias=False)
        self.context_to_v = nn.Linear(hidden, self.num_heads * num_concepts * rank, bias=False)
        self.student_row_proj = nn.Linear(student_dim, self.num_heads * num_concepts, bias=False)
        self.student_col_proj = nn.Linear(student_dim, self.num_heads * num_concepts, bias=False)
        self.student_scale = nn.Parameter(torch.tensor(1.25))
        self.context_scale = nn.Parameter(torch.tensor(0.25))
        self.low_rank_scale = nn.Parameter(torch.tensor(2.0))
        self.direct_scale = nn.Parameter(torch.tensor(0.50))

        nn.init.xavier_normal_(self.context_proj.weight)
        nn.init.zeros_(self.context_proj.bias)
        nn.init.xavier_normal_(self.hidden_proj.weight)
        nn.init.zeros_(self.hidden_proj.bias)
        nn.init.xavier_normal_(self.context_to_u.weight)
        nn.init.xavier_normal_(self.context_to_v.weight)
        nn.init.xavier_normal_(self.student_row_proj.weight, gain=0.8)
        nn.init.xavier_normal_(self.student_col_proj.weight, gain=0.8)
        nn.init.xavier_normal_(self.base_u)
        nn.init.xavier_normal_(self.base_v)
        nn.init.xavier_normal_(self.student_basis_u)
        nn.init.xavier_normal_(self.student_basis_v)

    def forward(self, student_embedding: torch.Tensor, context_repr: torch.Tensor) -> torch.Tensor:
        student_code = torch.tanh(1.5 * self.student_norm(student_embedding))
        hidden = self.context_proj(self.context_norm(context_repr))
        hidden = F.silu(hidden)
        hidden = F.silu(self.hidden_proj(hidden))
        B = hidden.size(0)
        student_u = torch.einsum("bd,hcrd->bhcr", student_code, self.student_basis_u)
        student_v = torch.einsum("bd,hcrd->bhcr", student_code, self.student_basis_v)
        context_u = self.context_to_u(hidden).view(B, self.num_heads, self.num_concepts, self.rank)
        context_v = self.context_to_v(hidden).view(B, self.num_heads, self.num_concepts, self.rank)
        u = self.base_u.unsqueeze(0) + self.student_scale * student_u + self.context_scale * torch.tanh(context_u)
        v = self.base_v.unsqueeze(0) + self.student_scale * student_v + self.context_scale * torch.tanh(context_v)
        low_rank_scores = torch.einsum("bhcr,bhkr->bhck", u, v) / math.sqrt(self.rank)
        row_bias = torch.tanh(self.student_row_proj(student_code)).view(B, self.num_heads, self.num_concepts, 1)
        col_bias = torch.tanh(self.student_col_proj(student_code)).view(B, self.num_heads, 1, self.num_concepts)
        direct_scores = row_bias + col_bias
        scores = self.low_rank_scale * low_rank_scores + self.direct_scale * direct_scores
        scores = scores - scores.mean(dim=-1, keepdim=True)
        return scores


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
        # personal graph
        use_personal_graph: bool,
        personal_rank: int,
        personal_max_alpha: float,
        personal_delta_scale: float,
        personal_warmup_epochs: int,
        personal_student_dim: int,
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
        self.personal_student_dim = max(1, int(personal_student_dim))
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
        )

        # E) 个性化图（可选）
        if self.use_personal_graph:
            self.personal_gate_embedding = nn.Embedding(num_students, self.personal_student_dim)
            self.personal_generator_embedding = nn.Embedding(num_students, self.personal_student_dim)
            self.personal_alpha_bias = nn.Embedding(num_students, num_relation_heads)
            self.personal_gate_from_state = nn.Linear(knowledge_dim, self.personal_student_dim, bias=False)
            self.personal_generator_from_state = nn.Linear(knowledge_dim, self.personal_student_dim, bias=False)
            self.personal_gate_id_logit = nn.Parameter(torch.tensor(-2.1972246))
            self.personal_generator_id_logit = nn.Parameter(torch.tensor(-2.1972246))
            nn.init.normal_(self.personal_gate_embedding.weight, mean=0.0, std=0.05)
            nn.init.normal_(self.personal_generator_embedding.weight, mean=0.0, std=0.05)
            nn.init.zeros_(self.personal_alpha_bias.weight)
            nn.init.xavier_normal_(self.personal_gate_from_state.weight)
            nn.init.xavier_normal_(self.personal_generator_from_state.weight)
            context_dim = knowledge_dim * 3
            personal_hidden_dim = max(self.personal_student_dim, knowledge_dim)
            self.adaptive_gate = AdaptiveGate(
                self.personal_student_dim,
                context_dim,
                num_heads=num_relation_heads,
                max_alpha=self.personal_max_alpha,
                hidden_dim=personal_hidden_dim,
            )
            self.personal_generator = PersonalRelationGenerator(
                self.personal_student_dim,
                context_dim,
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

    def forward(
        self,
        student_ids: torch.Tensor,
        identity_relations: torch.Tensor,   # (H,C,C)
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
        relation_used = relation_matrices

        if self.use_personal_graph and self.adaptive_gate is not None and self.personal_generator is not None:
            student_global_repr = self.knowledge_encoder.student_global(student_ids)  # (B,D)
            context_repr = torch.cat(
                [
                    student_global_repr,
                    student_repr,
                    student_repr - student_global_repr,
                ],
                dim=-1,
            )
            gate_state_repr = torch.tanh(self.personal_gate_from_state(student_repr))
            generator_state_repr = torch.tanh(self.personal_generator_from_state(student_repr))
            gate_id_scale = torch.sigmoid(self.personal_gate_id_logit)
            generator_id_scale = torch.sigmoid(self.personal_generator_id_logit)
            gate_student_repr = gate_state_repr + gate_id_scale * self.personal_gate_embedding(student_ids)
            generator_student_repr = (
                generator_state_repr + generator_id_scale * self.personal_generator_embedding(student_ids)
            )
            gate_alpha_bias = self.personal_alpha_bias(student_ids)
            gate_alpha, gate_alpha_logit = self.adaptive_gate(
                gate_student_repr,
                context_repr,
                gate_alpha_bias,
            )
            personal_warmup_scale = self._get_personal_warmup_scale()
            gate_alpha_effective = gate_alpha * personal_warmup_scale
            personal_delta = self.personal_generator(generator_student_repr, context_repr)  # (B,H,C,C)
            if self.use_concept_graph and self.relation_learning is not None:
                global_prior = relation_matrices.clamp(min=1e-8).log().unsqueeze(0)  # (1,H,C,C)
            else:
                # no_A 时不能继续使用 identity 的 log prior，
                # 否则 off-diagonal 会被 -inf 级别的先验压死，E 实际上无法生成 personalized graph。
                # 这里改为中性 prior，让 E 在没有 A 的条件下独立生成 row-stochastic 邻接。
                global_prior = torch.zeros_like(relation_matrices).unsqueeze(0)  # (1,H,C,C)
            personal_logits = global_prior + gate_alpha_effective * (
                self.personal_delta_scale * personal_delta
            )
            personal_matrices = F.softmax(personal_logits, dim=-1)            # (B,H,C,C)
            global_matrix = relation_matrices.unsqueeze(0)
            personal_matrix_delta = (personal_matrices - global_matrix).abs().mean(dim=(-1, -2, -3))
            personal_matrix_student_std = personal_matrices.std(dim=0, unbiased=False).mean()
            personal_delta_pre_softmax_norm = personal_delta.pow(2).mean().sqrt()
            personal_delta_student_std = personal_delta.std(dim=0, unbiased=False).mean()
            alpha_head_std = gate_alpha_effective.squeeze(-1).squeeze(-1).std(dim=1, unbiased=False).mean()

            # personal_matrices 已经是最终 row-stochastic 个性化邻接，
            # 其中全局 prior 与 alpha * delta 的混合已经体现在 softmax logits 中。
            # 这里必须直接把它作为第二遍编码所使用的 relation_used，
            # 否则在图卷积里再次按 alpha 与 global 混合，会把 E 的有效扰动压成近似 alpha^2。
            relation_used = personal_matrices
            knowledge_state = self.knowledge_encoder(student_ids, relation_used)
            student_repr = knowledge_state.mean(dim=1)

        return {
            "relation_matrices": relation_matrices,
            "relation_used": relation_used,
            "knowledge_state": knowledge_state,
            "student_repr": student_repr,
            "alpha": gate_alpha,
            "alpha_logit": gate_alpha_logit,
            "alpha_student_bias": gate_alpha_bias,
            "alpha_effective": gate_alpha_effective,
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
            "personal_warmup_scale": torch.tensor(
                self._get_personal_warmup_scale(), device=device, dtype=dtype
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
        prediction_l2_lambda: float = 5e-5,
        gnn_residual_weight: float = 0.5,
        personal_max_alpha: float = 0.35,
        personal_delta_scale: float = 1.0,
        personal_warmup_epochs: int = 0,
        personal_student_dim: Optional[int] = None,
        lambda_alpha_min: float = 0.0,
        alpha_min_target: float = 0.0,
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
        self._current_epoch = 1
        self.lambda_sparse_personal = float(lambda_sparse_personal)
        self.lambda_alpha = float(lambda_alpha)
        self.prediction_l2_lambda = float(prediction_l2_lambda)
        self.personal_max_alpha = max(0.0, float(personal_max_alpha))
        self.personal_delta_scale = max(0.0, float(personal_delta_scale))
        self.personal_warmup_epochs = max(0, int(personal_warmup_epochs))
        self.personal_student_dim = int(knowledge_dim if personal_student_dim is None else personal_student_dim)
        self.lambda_alpha_min = max(0.0, float(lambda_alpha_min))
        self.alpha_min_target = max(0.0, float(alpha_min_target))

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
            use_personal_graph=self.use_personal_graph,
            personal_rank=personal_rank,
            personal_max_alpha=self.personal_max_alpha,
            personal_delta_scale=self.personal_delta_scale,
            personal_warmup_epochs=self.personal_warmup_epochs,
            personal_student_dim=self.personal_student_dim,
            enable_module=self.enable_module1,
        )

        self.diagnosis_head = CognitiveDiagnosisHead(
            knowledge_dim=knowledge_dim,
            use_weight_norm=self.enable_module1,
        )
        self.exercise_encoder = ExerciseDifficultyEncoder(num_exercises=num_exercises)

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
        s_out = self.structure_module(student_ids, identity_relations=self.identity_relations)
        relation_matrices = s_out["relation_matrices"]
        relation_used = s_out["relation_used"]
        knowledge_state = s_out["knowledge_state"]
        student_repr = s_out["student_repr"]
        gate_alpha = s_out["alpha"]
        personal_matrices = s_out["personal_matrices"]

        # ========== 2) 固定预测头 D ==========
        b, a = self.exercise_encoder(exercise_ids)
        if return_details:
            irt_logit, diag_details = self.diagnosis_head(
                knowledge_state=knowledge_state,
                concept_mask=q_vector,
                b=b,
                a=a,
                return_details=True,
            )
        else:
            irt_logit = self.diagnosis_head(
                knowledge_state=knowledge_state,
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
            "relation_matrices": relation_matrices,
            "relation_used": relation_used,
            "knowledge_state": knowledge_state,
            "student_repr": student_repr,
            "q_vector": q_vector,
            "irt_b": b.detach(),
            "irt_a": a.detach(),
            "irt_logit": irt_logit.detach(),
            "logits": total_logit.detach(),
        }
        details["irt_logit_for_reg"] = irt_logit

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
            if s_out.get("personal_gate_id_scale") is not None:
                details["personal_gate_id_scale"] = s_out["personal_gate_id_scale"]
            if s_out.get("personal_generator_id_scale") is not None:
                details["personal_generator_id_scale"] = s_out["personal_generator_id_scale"]
            if personal_matrices is not None:
                details["personal_matrices"] = personal_matrices
                details["personal_matrices_detached"] = personal_matrices.detach()
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
        if details is not None:
            details["graph_reg_ramp"] = graph_reg_ramp_t.detach()

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
            if (
                "personal_matrices" in details
                and details["personal_matrices"] is not None
                and self.lambda_sparse_personal > 0
            ):
                pm = details["personal_matrices"]
                terms["personal_sparse"] = (
                    self.lambda_sparse_personal * self._row_entropy(pm) * graph_reg_ramp_t
                )

            if "alpha" in details and details["alpha"] is not None and self.lambda_alpha > 0:
                alpha_flat = details["alpha"].view(-1)
                alpha_var = alpha_flat.var() + 1e-6
                if "alpha_student_bias" in details and details["alpha_student_bias"] is not None:
                    alpha_bias_flat = details["alpha_student_bias"].view(-1)
                    alpha_var = alpha_var + 0.5 * (alpha_bias_flat.var() + 1e-6)
                    details["alpha_bias_std_runtime"] = alpha_bias_flat.std(unbiased=False).detach()
                terms["alpha_var"] = -self.lambda_alpha * alpha_var

            if "alpha" in details and details["alpha"] is not None and self.lambda_alpha_min > 0:
                alpha_flat = details["alpha"].view(-1)
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
                terms["alpha_collapse"] = self.lambda_alpha_min * alpha_pen
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
