# src/model.py
"""
Cognitive Diagnosis Model - Reviewer-friendly Dual-Branch Version (模块化/可消融)

目标：保证“三个组合模块”都能被“完全消融”，不存在“路径还在跑/参数还在训”的不彻底情况。

------------------------------------------------------------
模块 1：概念结构建模（Concept Structure Modeling）= A + E
  A) 全局概念图学习（Multi-head adjacency, row-stochastic）
  E) 个性化概念图（Personal graph）生成与混合（可选）
  输出：relation_matrices（全局）/ relation_used（全局或个性化混合）与 knowledge_state

模块 2：认知诊断头（Cognitive Diagnosis Head）= D
  2PL-IRT：theta_c -> Q-masked pooling -> theta_e -> irt_logit = a*(theta_e - b)
  输出：irt_logit 及可解释中间量（theta_c, theta_e）

模块 3：神经增强与校正（Neural Residual）= B
  B) MF/Q-conditioning 残差分支（mf_logit）
  Fusion：门控残差融合 total_logit = irt_logit + gate * mf_logit

------------------------------------------------------------
关键：提供三个“模块级完全消融”开关（ablate_module1/2/3）
- ablate_module1=True：模块1完全消融（A/E/knowledge_encoder 全部不实例化、forward 只返回全0 knowledge_state）
- ablate_module2=True：模块2完全消融（diagnosis_head 不实例化，ExerciseDifficultyEncoder 不创建 b/a 参数；IRT 路径完全不存在）
- ablate_module3=True：模块3完全消融（MF 分支/融合门控全部不实例化；forward 不计算任何 residual）

注意：
- FusionGate 仅在 module2 与 module3 同时启用时存在；若 module2 被消融，则直接使用 mf_logit 作为 total_logit（无 gate 参数/无 gate 计算）。

------------------------------------------------------------
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
from torch.nn.utils import parametrizations  # 新 API：不触发 weight_norm 弃用警告


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
    ):
        super().__init__()
        self.num_concepts = int(num_concepts)
        self.concept_dim = int(concept_dim)
        self.num_heads = int(num_heads)
        self.topk = topk
        self.allow_self_loop = bool(allow_self_loop)
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
                personal_A = relation_matrices["personal_matrices"]  # (B, C, C)
                gate = relation_matrices["gate_alpha"].view(-1, 1, 1)  # (B, 1, 1)
                
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
# 模块 3（B）：学生 latent（MF）编码 - StudentLatentEncoder
# ======================================================

class StudentLatentEncoder(nn.Module):
    """神经分支学生 latent + bias（MF 残差）。"""

    def __init__(self, num_students: int, latent_dim: int = 64):
        super().__init__()
        self.latent_emb = nn.Embedding(num_students, latent_dim)
        self.bias = nn.Embedding(num_students, 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.xavier_normal_(self.latent_emb.weight)
        nn.init.zeros_(self.bias.weight)

    def forward(self, student_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.latent_emb(student_ids)             # (B, latent_dim)
        bias = self.bias(student_ids).squeeze(-1)         # (B,)
        return latent, bias


# ======================================================
# 共享：题目参数编码（IRT + 可选 MF/Q-conditioning）- ExerciseDifficultyEncoder
# ======================================================

class ExerciseDifficultyEncoder(nn.Module):
    """
    Shared item encoder:
    - IRT 2PL params (b, a) controlled by use_irt
    - Module3 item branch: Q-conditioned item_q_repr as the main path
      plus a small item_id_adapter as the auxiliary path
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
        use_irt: bool = True,
        use_id_adapter: bool = True,
        use_bias: bool = True,
    ):
        super().__init__()
        self.num_exercises = int(num_exercises)
        self.num_concepts = int(num_concepts)
        self.exercise_dim = int(exercise_dim)

        self.use_mf_branch = bool(use_mf_branch)
        self.use_q_conditioning = bool(use_q_conditioning and use_mf_branch)
        self.use_irt = bool(use_irt)
        self.use_id_adapter = bool(use_id_adapter and use_mf_branch)
        self.use_bias = bool(use_bias and use_mf_branch)

        self.register_buffer("q_matrix", q_matrix)

        if self.use_mf_branch:
            self.concept_latent = nn.Embedding(num_concepts, exercise_dim)
            self.item_id_adapter = nn.Embedding(num_exercises, exercise_dim) if self.use_id_adapter else None
            self.exercise_bias = nn.Embedding(num_exercises, 1) if self.use_bias else None
            self.q_gate_raw = nn.Parameter(torch.zeros(1))
        else:
            self.concept_latent = None
            self.item_id_adapter = None
            self.exercise_bias = None
            self.q_gate_raw = None

        if self.use_irt:
            self.b = nn.Embedding(num_exercises, 1)
            self.a_raw = nn.Embedding(num_exercises, 1)
        else:
            self.b = None
            self.a_raw = None

        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        if self.use_mf_branch:
            nn.init.xavier_normal_(self.concept_latent.weight)
            if self.item_id_adapter is not None:
                nn.init.xavier_normal_(self.item_id_adapter.weight, gain=0.25)
            if self.exercise_bias is not None:
                nn.init.zeros_(self.exercise_bias.weight)

        if self.use_irt:
            nn.init.zeros_(self.b.weight)
            nn.init.normal_(self.a_raw.weight, mean=0.0, std=0.02)

    def _knowledge_scores(self, knowledge_state: torch.Tensor) -> torch.Tensor:
        """
        知识打分函数，保证对 knowledge_state 单调。
        支持两种形状：
        - (B, C, D)
        - (C, D)  （诊断时单个学生）
        返回：
        - (B, C) 或 (C,)
        """
        orig_dim = knowledge_state.dim()
        if orig_dim == 2:
            # (C, D) -> (1, C, D)
            knowledge_state = knowledge_state.unsqueeze(0)

        B, C, D = knowledge_state.size()

        # softplus(W) >= 0，保证单调性
        W1_pos = F.softplus(self.kn_w1_raw)   # (H, D)
        W2_pos = F.softplus(self.kn_w2_raw)   # (1, H)

        x = knowledge_state.view(B * C, D)    # (B*C, D)

        # 第一层：线性 + 单调激活
        h = torch.matmul(x, W1_pos.t()) + self.kn_b1  # (B*C, H)
        h = F.softplus(h)                             # 单调非减激活

        # 第二层：线性输出
        out = torch.matmul(h, W2_pos.t()) + self.kn_b2  # (B*C, 1)
        scores = out.view(B, C)                         # (B, C)

        if orig_dim == 2:
            scores = scores.squeeze(0)                  # (C,)
        return scores

    def forward(
        self,
        exercise_ids: torch.Tensor,
        concept_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Returns:
            item_components: dict or None
            exercise_bias: (B,) or None
            b:             (B,)
            a:             (B,)
        """
        device = exercise_ids.device
        B = exercise_ids.size(0)

        if self.use_mf_branch:
            e_bias = (
                self.exercise_bias(exercise_ids).squeeze(-1)
                if self.exercise_bias is not None
                else torch.zeros(B, device=device)
            )
            if concept_mask is None:
                concept_mask = self.q_matrix[exercise_ids]
            q = concept_mask.float()
            q_norm = q / (q.sum(dim=1, keepdim=True) + 1e-12)
            c_lat = self.concept_latent.weight
            q_latent = torch.matmul(q_norm, c_lat)

            if self.use_q_conditioning and self.q_gate_raw is not None:
                q_gate = torch.sigmoid(self.q_gate_raw)
                item_q_repr = q_gate * q_latent
            else:
                q_gate = q_latent.new_tensor(0.0)
                item_q_repr = torch.zeros_like(q_latent)

            if self.item_id_adapter is not None:
                item_id_adapter = self.dropout(self.item_id_adapter(exercise_ids))
            else:
                item_id_adapter = torch.zeros_like(item_q_repr)

            item_components = {
                "item_q_repr": self.dropout(item_q_repr),
                "item_id_adapter": item_id_adapter,
                "item_q_gate": q_gate.reshape(1).detach(),
            }
        else:
            item_components = None
            e_bias = None

        if self.use_irt:
            b = self.b(exercise_ids).squeeze(-1)
            a = F.softplus(self.a_raw(exercise_ids).squeeze(-1)) + 1e-6
        else:
            b = torch.zeros(B, device=device)
            a = torch.ones(B, device=device)

        return item_components, e_bias, b, a


# ======================================================
# 模块 2（D）：认知诊断头（2PL-IRT）- CognitiveDiagnosisHead
# ======================================================

class CognitiveDiagnosisHead(nn.Module):
    """
    模块 2：2PL-IRT 诊断头
    - theta_c：对每个概念的能力（由 knowledge_state 投影得到）
    - theta_e：按 Q-mask 聚合到题目层
    - irt_logit = a * (theta_e - b)
    """

    def __init__(self, knowledge_dim: int, use_weight_norm: bool = True):
        super().__init__()
        base = nn.Linear(knowledge_dim, 1, bias=True)
        # Module1 完全消融时 knowledge_state 恒为 0；此时 theta_proj.weight 仅受优化器衰减，
        # 使用 weight_norm 会在极小范数下带来数值不稳定风险，因此改用普通线性层。
        self.theta_proj = parametrizations.weight_norm(base) if use_weight_norm else base

    def forward(
        self,
        knowledge_state: torch.Tensor,   # (B, C, Dk)
        concept_mask: torch.Tensor,      # (B, C)
        b: torch.Tensor,                 # (B,)
        a: torch.Tensor,                 # (B,)
        return_details: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        theta_c = self.theta_proj(knowledge_state).squeeze(-1)  # (B, C)

        mask = concept_mask.float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        theta_e = (theta_c * mask).sum(dim=1) / denom  # (B,)

        irt_logit = a * (theta_e - b)  # (B,)

        if not return_details:
            return irt_logit

        details = {
            "theta_c": theta_c.detach(),
            "theta_e": theta_e.detach(),
            "irt_logit": irt_logit.detach(),
        }
        return irt_logit, details


# ======================================================
# 模块 3（B）：Q-aware residual adapter
# ======================================================

class QAwareStudentResidualEncoder(nn.Module):
    """Build B's student residual representation from A/E knowledge_state + Q mask."""

    def __init__(
        self,
        num_students: int,
        knowledge_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        use_id_adapter: bool = True,
        use_bias: bool = True,
    ):
        super().__init__()
        self.use_id_adapter = bool(use_id_adapter)
        self.use_bias = bool(use_bias)
        self.q_proj = nn.Linear(knowledge_dim, out_dim, bias=False)
        self.student_id_adapter = nn.Embedding(num_students, out_dim) if self.use_id_adapter else None
        self.student_bias = nn.Embedding(num_students, 1) if self.use_bias else None
        self.dropout = nn.Dropout(min(0.2, float(dropout)))

        nn.init.xavier_normal_(self.q_proj.weight)
        if self.student_id_adapter is not None:
            nn.init.xavier_normal_(self.student_id_adapter.weight, gain=0.25)
        if self.student_bias is not None:
            nn.init.zeros_(self.student_bias.weight)

    def forward(
        self,
        student_ids: torch.Tensor,
        knowledge_state: torch.Tensor,
        concept_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        q = concept_mask.float()
        q_norm = q / (q.sum(dim=1, keepdim=True) + 1e-12)
        student_q_state = torch.bmm(q_norm.unsqueeze(1), knowledge_state).squeeze(1)
        student_q_repr = self.dropout(self.q_proj(student_q_state))

        if self.student_id_adapter is not None:
            student_id_adapter = self.dropout(self.student_id_adapter(student_ids))
        else:
            student_id_adapter = torch.zeros_like(student_q_repr)

        if self.student_bias is not None:
            student_bias = self.student_bias(student_ids).squeeze(-1)
        else:
            student_bias = torch.zeros(student_ids.size(0), device=student_ids.device, dtype=student_q_repr.dtype)

        return {
            "student_q_repr": student_q_repr,
            "student_id_adapter": student_id_adapter,
            "student_bias": student_bias,
            "student_q_norm": student_q_repr.norm(dim=-1).detach(),
            "student_id_adapter_norm": student_id_adapter.norm(dim=-1).detach(),
        }


class QAwareResidualAdapterHead(nn.Module):
    """Residual adapter whose main contribution must pass through Q-aware representations."""

    def __init__(
        self,
        q_dim: int,
        adapter_dim: int,
        residual_dim: int = 32,
        dropout: float = 0.1,
        residual_clip_t: float = 2.0,
        residual_scale_init: float = 0.1,
        use_q_path: bool = True,
        use_id_adapter: bool = True,
        use_bias: bool = True,
    ):
        super().__init__()
        self.use_q_path = bool(use_q_path)
        self.use_id_adapter = bool(use_id_adapter)
        self.use_bias = bool(use_bias)

        self.q_student_proj = nn.Linear(q_dim, residual_dim, bias=False)
        self.q_item_proj = nn.Linear(q_dim, residual_dim, bias=False)
        self.id_student_proj = nn.Linear(adapter_dim, residual_dim, bias=False)
        self.id_item_proj = nn.Linear(adapter_dim, residual_dim, bias=False)
        nn.init.xavier_normal_(self.q_student_proj.weight)
        nn.init.xavier_normal_(self.q_item_proj.weight)
        nn.init.xavier_normal_(self.id_student_proj.weight, gain=0.35)
        nn.init.xavier_normal_(self.id_item_proj.weight, gain=0.35)

        init_scale = max(1e-4, float(residual_scale_init))
        raw_init = math.log(math.expm1(init_scale))
        self.q_scale_raw = nn.Parameter(torch.tensor(raw_init))
        self.id_scale_raw = nn.Parameter(torch.tensor(raw_init * 0.5))
        self.bias_scale_raw = nn.Parameter(torch.tensor(raw_init * 0.5))
        self.residual_bias = nn.Parameter(torch.zeros(1))
        self.residual_clip_t = float(residual_clip_t)
        self.dropout = nn.Dropout(min(0.2, float(dropout)))

    @staticmethod
    def _cosine_interaction(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_raw = left
        right_raw = right
        left = F.normalize(left_raw, dim=-1, eps=1e-12)
        right = F.normalize(right_raw, dim=-1, eps=1e-12)
        cosine_term = (left * right).sum(dim=-1)
        magnitude_term = torch.tanh((left_raw * right_raw).mean(dim=-1))
        return 0.7 * cosine_term + 0.3 * magnitude_term

    def forward(
        self,
        student_q_repr: torch.Tensor,
        item_q_repr: torch.Tensor,
        student_id_adapter: torch.Tensor,
        item_id_adapter: torch.Tensor,
        student_bias: torch.Tensor,
        exercise_bias: torch.Tensor,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        q_scale = F.softplus(self.q_scale_raw) + 1e-6
        id_scale = F.softplus(self.id_scale_raw) + 1e-6
        bias_scale = F.softplus(self.bias_scale_raw) + 1e-6

        if self.use_q_path:
            q_student = self.q_student_proj(student_q_repr)
            q_item = self.q_item_proj(item_q_repr)
            q_interaction_logit = q_scale * self._cosine_interaction(q_student, q_item)
        else:
            q_interaction_logit = torch.zeros(student_q_repr.size(0), device=student_q_repr.device, dtype=student_q_repr.dtype)

        if self.use_id_adapter:
            id_student = self.id_student_proj(student_id_adapter)
            id_item = self.id_item_proj(item_id_adapter)
            id_adapter_logit = id_scale * self._cosine_interaction(id_student, id_item)
        else:
            id_adapter_logit = torch.zeros_like(q_interaction_logit)

        if self.use_bias:
            bias_logit = bias_scale * (student_bias + exercise_bias) + self.residual_bias
        else:
            bias_logit = torch.zeros_like(q_interaction_logit)

        residual = q_interaction_logit + id_adapter_logit + bias_logit

        if self.residual_clip_t > 0:
            t = self.residual_clip_t
            residual = t * torch.tanh(residual / t)

        residual = self.dropout(residual)

        if not return_details:
            return residual

        residual_abs = residual.detach().abs().mean() + 1e-8
        residual_abs_for_reg = residual.abs().mean() + 1e-8
        details = {
            "mf_logit": residual.detach(),
            "residual_logit": residual.detach(),
            "q_interaction_logit": q_interaction_logit.detach(),
            "id_adapter_logit": id_adapter_logit.detach(),
            "bias_logit": bias_logit.detach(),
            "interaction_residual": q_interaction_logit.detach(),
            "bias_residual": bias_logit.detach(),
            "mf_scale": q_scale.detach(),
            "q_scale": q_scale.detach(),
            "id_scale": id_scale.detach(),
            "bias_scale": bias_scale.detach(),
            "b_q_share": q_interaction_logit.detach().abs().mean() / residual_abs,
            "b_id_share": id_adapter_logit.detach().abs().mean() / residual_abs,
            "b_bias_share": bias_logit.detach().abs().mean() / residual_abs,
            "b_id_share_for_reg": id_adapter_logit.abs().mean() / residual_abs_for_reg,
            "student_q_norm": student_q_repr.detach().norm(dim=-1),
            "student_id_adapter_norm": student_id_adapter.detach().norm(dim=-1),
            "item_q_norm": item_q_repr.detach().norm(dim=-1),
            "item_id_adapter_norm": item_id_adapter.detach().norm(dim=-1),
        }
        return residual, details


QAlignedResidualHead = QAwareResidualAdapterHead


class ConservativeFusionGate(nn.Module):
    """Conservative residual fusion: gate_max * sigmoid(linear)."""

    def __init__(self, gate_max: float = 1.0, gate_bias_init: float = -1.1):
        super().__init__()
        self.fusion_gate = nn.Linear(5, 1)
        self.gate_max = float(gate_max)
        nn.init.constant_(self.fusion_gate.bias, float(gate_bias_init))
        nn.init.constant_(self.fusion_gate.weight, 0.0)
        with torch.no_grad():
            self.fusion_gate.weight[0, 3] = 0.35
            self.fusion_gate.weight[0, 4] = 0.20

    def forward(
        self,
        irt_logit: torch.Tensor,
        mf_logit: torch.Tensor,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        stack = torch.stack(
            [
                irt_logit,
                mf_logit,
                torch.abs(irt_logit - mf_logit),
                torch.abs(mf_logit),
                torch.abs(irt_logit),
            ],
            dim=1,
        )
        gate_raw = self.fusion_gate(stack).squeeze(-1)
        gate = self.gate_max * torch.sigmoid(gate_raw)
        delta = gate * mf_logit
        total_logit = irt_logit + delta

        if not return_details:
            return total_logit

        details = {
            "gate": gate.detach(),
            "gate_raw": gate_raw.detach(),
            "delta_logit": delta.detach(),
            "total_logit": total_logit.detach(),
        }
        return total_logit, details


# ======================================================
# 模块 1（E）：个性化图相关组件 - AdaptiveGate / PersonalRelationGenerator
# ======================================================

class AdaptiveGate(nn.Module):
    """个性化图混合系数 alpha（B,1,1,1）。

    设计原则：
    - 保留一条显式 student bias -> alpha 的短路径，避免个体信号被上下文支路淹没；
    - student-id 分支与 context 分支分开建模，避免个体信号被 LayerNorm 洗掉；
    - context 只做修正项，不取代 student-specific 路径。
    """

    def __init__(
        self,
        student_dim: int,
        context_dim: int,
        max_alpha: float = 0.35,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        hid = max(1, max(student_dim, context_dim) // 2 if hidden_dim is None else int(hidden_dim))
        self.context_norm = nn.LayerNorm(context_dim)
        self.student_proj = nn.Linear(student_dim, hid, bias=False)
        self.context_proj = nn.Linear(context_dim, hid)
        self.hidden_proj = nn.Linear(hid, hid)
        self.student_to_logit = nn.Linear(student_dim, 1, bias=False)
        self.context_to_logit = nn.Linear(context_dim, 1)
        self.out = nn.Linear(hid, 1)
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
        return alpha.view(-1, 1, 1, 1), alpha_logit.view(-1, 1, 1, 1)


class PersonalRelationGenerator(nn.Module):
    """显式 student code + context residual 生成个性化邻接 logits（B,C,C）。"""

    def __init__(
        self,
        student_dim: int,
        context_dim: int,
        num_concepts: int,
        rank: int = 4,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.context_norm = nn.LayerNorm(context_dim)
        self.num_concepts = int(num_concepts)
        self.rank = int(rank)
        hidden = max(1, max(student_dim, context_dim) // 2 if hidden_dim is None else int(hidden_dim))

        self.context_proj = nn.Linear(context_dim, hidden)
        self.hidden_proj = nn.Linear(hidden, hidden)
        self.base_u = nn.Parameter(torch.randn(num_concepts, rank) * 0.02)
        self.base_v = nn.Parameter(torch.randn(num_concepts, rank) * 0.02)
        self.student_basis_u = nn.Parameter(torch.randn(num_concepts, rank, student_dim) * 0.02)
        self.student_basis_v = nn.Parameter(torch.randn(num_concepts, rank, student_dim) * 0.02)
        self.context_to_u = nn.Linear(hidden, num_concepts * rank, bias=False)
        self.context_to_v = nn.Linear(hidden, num_concepts * rank, bias=False)
        self.student_scale = nn.Parameter(torch.tensor(0.5))
        self.context_scale = nn.Parameter(torch.tensor(0.1))

        nn.init.xavier_normal_(self.context_proj.weight)
        nn.init.zeros_(self.context_proj.bias)
        nn.init.xavier_normal_(self.hidden_proj.weight)
        nn.init.zeros_(self.hidden_proj.bias)
        nn.init.xavier_normal_(self.context_to_u.weight)
        nn.init.xavier_normal_(self.context_to_v.weight)
        nn.init.xavier_normal_(self.base_u)
        nn.init.xavier_normal_(self.base_v)
        nn.init.xavier_normal_(self.student_basis_u)
        nn.init.xavier_normal_(self.student_basis_v)

    def forward(self, student_embedding: torch.Tensor, context_repr: torch.Tensor) -> torch.Tensor:
        student_code = torch.tanh(student_embedding)
        hidden = self.context_proj(self.context_norm(context_repr))
        hidden = F.silu(hidden)
        hidden = F.silu(self.hidden_proj(hidden))
        B = hidden.size(0)
        student_u = torch.einsum("bd,crd->bcr", student_code, self.student_basis_u)
        student_v = torch.einsum("bd,crd->bcr", student_code, self.student_basis_v)
        context_u = self.context_to_u(hidden).view(B, self.num_concepts, self.rank)
        context_v = self.context_to_v(hidden).view(B, self.num_concepts, self.rank)
        u = self.base_u.unsqueeze(0) + self.student_scale * student_u + self.context_scale * torch.tanh(context_u)
        v = self.base_v.unsqueeze(0) + self.student_scale * student_v + self.context_scale * torch.tanh(context_v)
        scores = torch.bmm(u, v.transpose(1, 2)) / math.sqrt(self.rank)
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
            self.personal_alpha_bias = nn.Embedding(num_students, 1)
            nn.init.normal_(self.personal_gate_embedding.weight, mean=0.0, std=0.05)
            nn.init.normal_(self.personal_generator_embedding.weight, mean=0.0, std=0.05)
            nn.init.zeros_(self.personal_alpha_bias.weight)
            context_dim = knowledge_dim * 3
            personal_hidden_dim = max(self.personal_student_dim, knowledge_dim)
            self.adaptive_gate = AdaptiveGate(
                self.personal_student_dim,
                context_dim,
                max_alpha=self.personal_max_alpha,
                hidden_dim=personal_hidden_dim,
            )
            self.personal_generator = PersonalRelationGenerator(
                self.personal_student_dim,
                context_dim,
                num_concepts,
                personal_rank,
                hidden_dim=personal_hidden_dim,
            )
        else:
            self.adaptive_gate = None
            self.personal_generator = None
            self.personal_gate_embedding = None
            self.personal_generator_embedding = None
            self.personal_alpha_bias = None

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
            gate_student_repr = self.personal_gate_embedding(student_ids)
            generator_student_repr = self.personal_generator_embedding(student_ids)
            gate_alpha_bias = self.personal_alpha_bias(student_ids)
            gate_alpha, gate_alpha_logit = self.adaptive_gate(
                gate_student_repr,
                context_repr,
                gate_alpha_bias,
            )
            personal_warmup_scale = self._get_personal_warmup_scale()
            gate_alpha_effective = gate_alpha * personal_warmup_scale
            personal_delta = self.personal_generator(generator_student_repr, context_repr)  # (B,C,C)
            global_prior = relation_matrices.mean(dim=0).clamp(min=1e-8).log().unsqueeze(0)
            personal_logits = global_prior + gate_alpha_effective.squeeze(-1) * (
                self.personal_delta_scale * personal_delta
            )
            personal_matrices = F.softmax(personal_logits, dim=-1)            # (B,C,C)
            global_matrix = relation_matrices.mean(dim=0, keepdim=True)
            personal_matrix_delta = (personal_matrices - global_matrix).abs().mean(dim=(-1, -2))
            personal_matrix_student_std = personal_matrices.std(dim=0, unbiased=False).mean()

            # 优化：不展开为 (B,H,C,C)，而是保存 gate_alpha 和 personal_matrices
            # 让 GNN 层在需要时逐 head 混合，减少显存占用
            # relation_used 改为字典传递必要信息
            relation_used = {
                "global_matrices": relation_matrices,        # (H,C,C)
                "personal_matrices": personal_matrices,      # (B,C,C)
                "gate_alpha": gate_alpha_effective.squeeze(-1).squeeze(-1).squeeze(-1),  # (B,)
            }

            knowledge_state = self.knowledge_encoder(student_ids, relation_used)
            student_repr = knowledge_state.mean(dim=1)

        return {
            "relation_matrices": relation_matrices,
            "relation_used": relation_used if isinstance(relation_used, torch.Tensor) else relation_matrices,
            "knowledge_state": knowledge_state,
            "student_repr": student_repr,
            "alpha": gate_alpha,
            "alpha_logit": gate_alpha_logit,
            "alpha_student_bias": gate_alpha_bias,
            "alpha_effective": gate_alpha_effective,
            "personal_matrices": personal_matrices,
            "personal_matrix_delta": personal_matrix_delta,
            "personal_matrix_student_std": personal_matrix_student_std,
            "personal_warmup_scale": torch.tensor(
                self._get_personal_warmup_scale(), device=device, dtype=dtype
            ),
        }


# ======================================================
# 主模型：组合模块 1/2/3，提供统一 forward 与正则/诊断接口
# ======================================================

class CognitiveDiagnosisModel(nn.Module):
    """
    主模型（组合 3 个模块）：

    - Module 1: ConceptStructureModeling（A + E）
    - Module 2: CognitiveDiagnosisHead（D）
    - Module 3: Neural Residual（B）

    关键：支持三个“模块级完全消融”开关：
      - ablate_module1：完全移除模块1（A/E/knowledge_encoder 都不存在）
      - ablate_module2：完全移除模块2（theta_proj 不存在；IRT b/a 参数不存在）
      - ablate_module3：完全移除模块3（MF/Fusion 都不存在）

    同时保留旧的“子模块消融”开关：
      - use_concept_graph / use_personal_graph
      - use_mf_branch
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
        # ===== 子模块级开关（兼容旧 ablation）=====
        use_mf_branch: bool = True,
        use_concept_graph: bool = True,
        graph_topk: Optional[int] = None,
        allow_self_loop: bool = True,
        use_personal_graph: bool = False,
        personal_rank: int = 4,
        # ===== 模块级“完全消融”开关 =====
        ablate_module1: bool = False,
        ablate_module2: bool = False,
        ablate_module3: bool = False,
        # ===== Module3 (Q-aligned residual + conservative fusion) =====
        use_q_aligned_residual: bool = True,
        fusion_gate_max: float = 1.0,
        fusion_gate_bias_init: float = -1.1,
        residual_clip_t: float = 2.0,
        residual_scale_init: float = 0.1,
        # ===== 正则权重 =====
        lambda_sparse_personal: float = 0.0,
        lambda_alpha: float = 0.0,
        lambda_graph_entropy: float = 0.01,  # mapped from args.lambda_sparse
        graph_entropy_min: float = 0.15,
        graph_entropy_max: float = 0.85,
        lambda_graph_diag: float = 0.10,
        lambda_graph_uniform: float = 0.04,
        graph_uniform_margin: float = 0.10,
        graph_reg_warmup_epochs: int = 1,
        graph_reg_cap_ratio: float = 6.0,
        graph_dropout: Optional[float] = None,
        graph_tau_init: float = 1.0,
        mf_l2_lambda: float = 5e-5,          # mapped from args.exercise_l2_lambda
        gnn_residual_weight: float = 0.5,
        use_q_conditioning: bool = True,
        use_b_id_adapter: bool = True,
        use_b_bias: bool = True,
        lambda_b_id_budget: float = 0.0,
        b_id_budget_target: float = 0.25,
        # ===== Rescue knobs (default off for baseline compatibility) =====
        mf_warmup_epochs: int = 0,
        lambda_delta_ratio: float = 0.0,
        delta_ratio_target: float = 0.15,
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
        self.skill_dim = int(skill_dim)
        self.num_relation_heads = int(num_relation_heads)

        # ===== 模块级开关（保存用于日志与严格控制）=====
        self.enable_module1 = not bool(ablate_module1)
        self.enable_module2 = not bool(ablate_module2)
        self.enable_module3 = not bool(ablate_module3)

        # 如果 module2 和 module3 都被消融，则模型没有任何预测路径：直接报错（避免“跑了但无意义”）
        if (not self.enable_module2) and (not self.enable_module3):
            raise ValueError("Invalid ablation: both Module 2 (IRT head) and Module 3 (MF) are disabled; no prediction path.")

        # ===== 子模块开关：根据“模块级消融”强制覆盖，保证彻底 =====
        # 模块1完全消融 => A/E 全部关闭（且结构模块不创建参数）
        if not self.enable_module1:
            use_concept_graph = False
            use_personal_graph = False

        # 模块3完全消融 => MF 关闭
        if not self.enable_module3:
            use_mf_branch = False

        # 最终保存子模块开关（用于 details）
        self.use_concept_graph = bool(use_concept_graph)
        self.use_personal_graph = bool(use_personal_graph)
        self.use_mf_branch = bool(use_mf_branch and use_q_aligned_residual)
        self.use_q_aligned_residual = bool(use_q_aligned_residual)
        self.fusion_gate_max = float(fusion_gate_max)
        self.fusion_gate_bias_init = float(fusion_gate_bias_init)
        self.residual_clip_t = float(residual_clip_t)
        self.residual_scale_init = float(residual_scale_init)

        # ===== 正则权重 =====
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
        self._current_epoch = 1
        self.lambda_sparse_personal = float(lambda_sparse_personal)
        self.lambda_alpha = float(lambda_alpha)
        self.mf_l2_lambda = float(mf_l2_lambda)
        self.mf_warmup_epochs = max(0, int(mf_warmup_epochs))
        self.lambda_delta_ratio = max(0.0, float(lambda_delta_ratio))
        self.delta_ratio_target = max(0.0, float(delta_ratio_target))
        self.personal_max_alpha = max(0.0, float(personal_max_alpha))
        self.personal_delta_scale = max(0.0, float(personal_delta_scale))
        self.personal_warmup_epochs = max(0, int(personal_warmup_epochs))
        self.personal_student_dim = int(knowledge_dim if personal_student_dim is None else personal_student_dim)
        self.lambda_alpha_min = max(0.0, float(lambda_alpha_min))
        self.alpha_min_target = max(0.0, float(alpha_min_target))
        self.use_b_id_adapter = bool(use_b_id_adapter)
        self.use_b_bias = bool(use_b_bias)
        self.lambda_b_id_budget = max(0.0, float(lambda_b_id_budget))
        self.b_id_budget_target = max(0.0, float(b_id_budget_target))

        # 固定 Q 矩阵
        self.register_buffer("q_matrix", q_matrix)

        # identity_relations：当不学习概念图时使用
        identity = torch.eye(num_concepts, dtype=torch.float32).unsqueeze(0).repeat(self.num_relation_heads, 1, 1)
        self.register_buffer("identity_relations", identity)

        # ------------------------------
        # Module 1：概念结构建模（A+E）
        # ------------------------------
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
            use_personal_graph=self.use_personal_graph,
            personal_rank=personal_rank,
            personal_max_alpha=self.personal_max_alpha,
            personal_delta_scale=self.personal_delta_scale,
            personal_warmup_epochs=self.personal_warmup_epochs,
            personal_student_dim=self.personal_student_dim,
            enable_module=self.enable_module1,  # 关键：模块级完全消融
        )

        # ------------------------------
        # Module 2：认知诊断头（D）
        # ------------------------------
        if self.enable_module2:
            self.diagnosis_head = CognitiveDiagnosisHead(
                knowledge_dim=knowledge_dim,
                use_weight_norm=self.enable_module1,
            )
        else:
            self.diagnosis_head = None

        # ------------------------------
        # 共享：题目参数编码（IRT + 可选 MF）
        # - module2 消融 => use_irt=False：b/a 参数不创建
        # - module3 消融 => use_mf_branch=False：MF 参数不创建
        # ------------------------------
        self.exercise_encoder = ExerciseDifficultyEncoder(
            num_exercises=num_exercises,
            num_concepts=num_concepts,
            q_matrix=q_matrix,
            exercise_dim=exercise_dim,
            dropout=dropout,
            use_mf_branch=self.use_mf_branch,
            use_q_conditioning=use_q_conditioning,
            use_irt=self.enable_module2,  # 关键：模块2完全消融 => 不创建 IRT 参数
            use_id_adapter=self.use_b_id_adapter,
            use_bias=self.use_b_bias,
        )

        # ------------------------------
        # Module 3：神经增强（B）
        # ------------------------------
        if self.enable_module3 and self.use_mf_branch:
            self.skill_encoder = QAwareStudentResidualEncoder(
                num_students=num_students,
                knowledge_dim=knowledge_dim,
                out_dim=exercise_dim,
                dropout=dropout,
                use_id_adapter=self.use_b_id_adapter,
                use_bias=self.use_b_bias,
            )
            self.mf_head = QAwareResidualAdapterHead(
                q_dim=exercise_dim,
                adapter_dim=exercise_dim,
                residual_dim=min(64, skill_dim, exercise_dim),
                dropout=dropout,
                residual_clip_t=self.residual_clip_t,
                residual_scale_init=self.residual_scale_init,
                use_q_path=use_q_conditioning,
                use_id_adapter=self.use_b_id_adapter,
                use_bias=self.use_b_bias,
            )
        else:
            self.skill_encoder = None
            self.mf_head = None

        # Fusion exists only when module2 + module3 are both enabled
        if self.enable_module2 and self.enable_module3 and self.use_mf_branch:
            self.fusion = ConservativeFusionGate(
                gate_max=self.fusion_gate_max,
                gate_bias_init=self.fusion_gate_bias_init,
            )
        else:
            self.fusion = None

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

        # ========== 2) 共享：题目参数（b,a）与 module3 item components ==========
        item_components, exercise_bias, b, a = self.exercise_encoder(exercise_ids, concept_mask=q_vector)

        # ========== 3) Module 2：IRT head（可完全消融） ==========
        if self.enable_module2:
            if self.diagnosis_head is None:
                raise RuntimeError("enable_module2=True but diagnosis_head is None. Check init wiring.")
            if return_details:
                irt_logit, diag_details = self.diagnosis_head(
                    knowledge_state=knowledge_state,
                    concept_mask=q_vector,
                    b=b, a=a,
                    return_details=True,
                )
            else:
                irt_logit = self.diagnosis_head(
                    knowledge_state=knowledge_state,
                    concept_mask=q_vector,
                    b=b, a=a,
                    return_details=False,
                )
                diag_details = None
        else:
            # module2 完全消融：IRT logit 直接为 0，占位用于统一接口
            irt_logit = torch.zeros(exercise_ids.size(0), device=device)
            diag_details = None

        # ========== 4) Module 3：MF residual（可完全消融） ==========
        if self.enable_module3 and self.use_mf_branch:
            if self.skill_encoder is None or self.mf_head is None:
                raise RuntimeError("enable_module3 & use_mf_branch=True but MF modules are None. Check init wiring.")

            student_components = self.skill_encoder(
                student_ids=student_ids,
                knowledge_state=knowledge_state,
                concept_mask=q_vector,
            )

            if item_components is None or exercise_bias is None:
                raise RuntimeError("Module3 is enabled but item components are None. Check exercise_encoder wiring.")

            if return_details:
                mf_logit, mf_details = self.mf_head(
                    student_q_repr=student_components["student_q_repr"],
                    item_q_repr=item_components["item_q_repr"],
                    student_id_adapter=student_components["student_id_adapter"],
                    item_id_adapter=item_components["item_id_adapter"],
                    student_bias=student_components["student_bias"],
                    exercise_bias=exercise_bias,
                    return_details=True,
                )
            else:
                mf_logit = self.mf_head(
                    student_q_repr=student_components["student_q_repr"],
                    item_q_repr=item_components["item_q_repr"],
                    student_id_adapter=student_components["student_id_adapter"],
                    item_id_adapter=item_components["item_id_adapter"],
                    student_bias=student_components["student_bias"],
                    exercise_bias=exercise_bias,
                    return_details=False,
                )
                mf_details = None
            mf_warmup_scale = self._get_linear_warmup(self.mf_warmup_epochs)
            mf_logit = mf_logit * mf_warmup_scale
            if mf_details is not None:
                mf_details["mf_logit_raw"] = mf_details["mf_logit"]
                mf_details["residual_logit_raw"] = mf_details["residual_logit"]
                mf_details["mf_logit"] = mf_logit.detach()
                mf_details["residual_logit"] = mf_logit.detach()
                mf_details["mf_warmup_scale"] = mf_logit.new_tensor(mf_warmup_scale).detach()
                mf_details["student_q_norm"] = student_components["student_q_norm"]
                mf_details["student_id_adapter_norm"] = student_components["student_id_adapter_norm"]
                mf_details["item_q_norm"] = item_components["item_q_repr"].detach().norm(dim=-1)
                mf_details["item_id_adapter_norm"] = item_components["item_id_adapter"].detach().norm(dim=-1)
                mf_details["item_q_gate"] = item_components["item_q_gate"]
        else:
            student_components = None
            mf_logit = torch.zeros_like(irt_logit)
            mf_details = None

        # ========== 5) 组合输出：严格避免“半消融” ==========
        # 情况1：module2+module3 均启用且 MF 存在 => 使用 fusion gate
        if self.enable_module2 and (self.enable_module3 and self.use_mf_branch):
            if self.fusion is None:
                raise RuntimeError("Fusion should exist when module2 & module3 are enabled with MF. Check init wiring.")
            if return_details:
                total_logit, fuse_details = self.fusion(
                    irt_logit=irt_logit,
                    mf_logit=mf_logit,
                    return_details=True,
                )
            else:
                total_logit = self.fusion(
                    irt_logit=irt_logit,
                    mf_logit=mf_logit,
                    return_details=False,
                )
                fuse_details = None

        # 情况2：仅 module2 启用（或 module3 被完全消融/或 MF 被消融）=> 纯 IRT
        elif self.enable_module2:
            total_logit = irt_logit
            if return_details:
                zero = torch.zeros_like(irt_logit).detach()
                fuse_details = {"gate": zero, "gate_raw": zero, "delta_logit": zero}
            else:
                fuse_details = None

        # 情况3：module2 完全消融，但 module3 启用 => 纯 residual（无 gate 计算）
        elif self.enable_module3 and self.use_mf_branch:
            total_logit = mf_logit
            if return_details:
                zero = torch.zeros_like(mf_logit).detach()
                fuse_details = {"gate": zero, "gate_raw": zero, "delta_logit": zero}
            else:
                fuse_details = None

        # 情况4：无预测路径（已在 init 报错，forward 再兜底）
        else:
            raise RuntimeError("No valid prediction path. Check ablation flags.")

        # ========== 6) 返回 logits 或 prob ==========
        out_main = total_logit if return_logits else torch.sigmoid(total_logit)

        if not return_details:
            return out_main

        # ========== 7) details（用于正则、可解释输出、排查消融是否生效） ==========
        details: Dict[str, torch.Tensor] = {
            # 模块级开关（用于日志对齐）
            "enable_module1": torch.tensor(int(self.enable_module1), device=device),
            "enable_module2": torch.tensor(int(self.enable_module2), device=device),
            "enable_module3": torch.tensor(int(self.enable_module3), device=device),

            # 子模块开关
            "use_concept_graph": torch.tensor(int(self.use_concept_graph), device=device),
            "use_personal_graph": torch.tensor(int(self.use_personal_graph), device=device),
            "use_mf_branch": torch.tensor(int(self.use_mf_branch), device=device),

            # 模块 1 输出
            "relation_matrices": relation_matrices,
            "relation_used": relation_used,
            "knowledge_state": knowledge_state,
            "student_repr": student_repr,

            # 共享输入
            "q_vector": q_vector,

            # IRT 参数（若 module2 消融，这里是占位常量）
            "irt_b": b.detach(),
            "irt_a": a.detach(),

            # logits
            "irt_logit": irt_logit.detach(),
            "mf_logit": mf_logit.detach(),
            "residual_logit": mf_logit.detach(),
            "logits": total_logit.detach(),
        }

        details["irt_logit_for_reg"] = irt_logit
        if self.enable_module2 and (self.enable_module3 and self.use_mf_branch):
            details["delta_logit_for_reg"] = total_logit - irt_logit

        if diag_details is not None:
            details.update(diag_details)
        if mf_details is not None:
            details.update(mf_details)
        if fuse_details is not None:
            details.update(fuse_details)

        if self.use_personal_graph:
            if gate_alpha is not None:
                # Keep gradient path for personal regularizers in training.
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
            if personal_matrices is not None:
                # Keep gradient path for personal regularizers in training.
                details["personal_matrices"] = personal_matrices
                details["personal_matrices_detached"] = personal_matrices.detach()
            if s_out.get("personal_matrix_delta") is not None:
                details["personal_matrix_delta"] = s_out["personal_matrix_delta"]
                details["personal_matrix_delta_detached"] = s_out["personal_matrix_delta"].detach()
            if s_out.get("personal_matrix_student_std") is not None:
                details["personal_matrix_student_std"] = s_out["personal_matrix_student_std"]
                details["personal_matrix_student_std_detached"] = s_out["personal_matrix_student_std"].detach()
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
            "mf_l2": torch.tensor(0.0, device=device),
            "delta_ratio": torch.tensor(0.0, device=device),
            "b_id_budget": torch.tensor(0.0, device=device),
            "personal_sparse": torch.tensor(0.0, device=device),
            "alpha_var": torch.tensor(0.0, device=device),  # signed term (negative when maximizing variance)
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

        # (2) MF/IRT L2
        if self.mf_l2_lambda > 0:
            reg_terms = []

            if (
                self.enable_module2
                and self.exercise_encoder.use_irt
                and self.exercise_encoder.b is not None
                and self.exercise_encoder.a_raw is not None
            ):
                reg_terms.extend(
                    [
                        self.exercise_encoder.b.weight.pow(2).mean(),
                        self.exercise_encoder.a_raw.weight.pow(2).mean(),
                    ]
                )

            if self.enable_module3 and self.use_mf_branch:
                if self.skill_encoder is not None:
                    student_adapter = getattr(self.skill_encoder, "student_id_adapter", None)
                    student_bias = getattr(self.skill_encoder, "student_bias", None)
                    if student_adapter is not None:
                        reg_terms.append(student_adapter.weight.pow(2).mean())
                    if student_bias is not None:
                        reg_terms.append(student_bias.weight.pow(2).mean())
                    q_proj = getattr(self.skill_encoder, "q_proj", None)
                    if q_proj is not None:
                        reg_terms.append(q_proj.weight.pow(2).mean())
                if self.exercise_encoder.use_mf_branch and self.exercise_encoder.item_id_adapter is not None:
                    reg_terms.append(self.exercise_encoder.item_id_adapter.weight.pow(2).mean())
                if self.exercise_encoder.use_mf_branch and self.exercise_encoder.concept_latent is not None:
                    reg_terms.append(self.exercise_encoder.concept_latent.weight.pow(2).mean())
                if self.exercise_encoder.use_mf_branch and self.exercise_encoder.exercise_bias is not None:
                    reg_terms.append(self.exercise_encoder.exercise_bias.weight.pow(2).mean())
                if self.exercise_encoder.use_mf_branch and self.exercise_encoder.q_gate_raw is not None:
                    reg_terms.append(self.exercise_encoder.q_gate_raw.pow(2).mean())

            if len(reg_terms) > 0:
                terms["mf_l2"] = self.mf_l2_lambda * sum(reg_terms)

        if (
            self.enable_module3
            and self.use_mf_branch
            and self.lambda_b_id_budget > 0
            and details is not None
            and (details.get("b_id_share_for_reg") is not None or details.get("b_id_share") is not None)
        ):
            id_share = details.get("b_id_share_for_reg", details.get("b_id_share"))
            if not torch.is_tensor(id_share):
                id_share = torch.tensor(float(id_share), device=device)
            id_target = torch.tensor(self.b_id_budget_target, device=device, dtype=id_share.dtype)
            id_pen = F.relu(id_share - id_target)
            terms["b_id_budget"] = self.lambda_b_id_budget * id_pen
            details["b_id_budget_pen"] = id_pen.detach()

        if (
            self.enable_module2
            and self.enable_module3
            and self.use_mf_branch
            and self.lambda_delta_ratio > 0
            and details is not None
            and details.get("delta_logit_for_reg") is not None
            and details.get("irt_logit_for_reg") is not None
        ):
            delta = details["delta_logit_for_reg"]
            irt_logit = details["irt_logit_for_reg"]
            delta_ratio = delta.abs().mean() / (irt_logit.abs().mean() + 1e-6)
            delta_target = torch.tensor(self.delta_ratio_target, device=device, dtype=delta_ratio.dtype)
            delta_pen = F.relu(delta_ratio - delta_target)
            terms["delta_ratio"] = self.lambda_delta_ratio * delta_pen
            details["delta_ratio_value"] = delta_ratio.detach()
            details["delta_ratio_pen"] = delta_pen.detach()

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
                terms["alpha_collapse"] = self.lambda_alpha_min * alpha_pen
                details["alpha_std_runtime"] = alpha_std.detach()
                details["alpha_collapse_pen"] = alpha_pen.detach()

        total = (
            terms["graph_entropy"]
            + terms["graph_diag"]
            + terms["graph_uniform"]
            + terms["mf_l2"]
            + terms["delta_ratio"]
            + terms["b_id_budget"]
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
        (2) MF/IRT 参数 L2（mf_l2_lambda）—— 仅对应模块启用时才计入（避免不彻底消融）
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

        - 若 module2 启用：knowledge_mastery = sigmoid(theta_c)
        - 若 module2 消融：knowledge_mastery 返回全0（因为 theta_proj 不存在，严格意义上也无“认知诊断头”）
        - 若 module3(MF) 启用：skill_latent 返回 MF latent，否则返回全0
        """
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            sid = torch.tensor([student_id], device=device, dtype=torch.long)

            s_out = self.structure_module(sid, identity_relations=self.identity_relations)
            ks = s_out["knowledge_state"].squeeze(0)  # (C,D)

            if self.enable_module2 and self.diagnosis_head is not None:
                mastery = torch.sigmoid(self.diagnosis_head.theta_proj(ks).squeeze(-1))  # (C,)
            else:
                mastery = torch.zeros(self.num_concepts, device=device)

            if self.enable_module3 and self.use_mf_branch and self.skill_encoder is not None:
                latent, _ = self.skill_encoder(sid)
                latent = latent.squeeze(0)
            else:
                latent = torch.zeros(self.skill_dim, device=device)

            return {
                "knowledge_mastery": mastery,
                "skill_latent": latent,
                "relation_matrices": s_out["relation_matrices"],
            }
