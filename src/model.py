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

模块 3：神经增强与校正（Neural Residual + Prototype Calibration）= B + C
  B) MF/Q-conditioning 残差分支（mf_logit）
  C) Soft Prototype 表示校正（可选）
  Fusion：门控残差融合 total_logit = irt_logit + gate * mf_logit

------------------------------------------------------------
关键：提供三个“模块级完全消融”开关（ablate_module1/2/3）
- ablate_module1=True：模块1完全消融（A/E/knowledge_encoder 全部不实例化、forward 只返回全0 knowledge_state）
- ablate_module2=True：模块2完全消融（diagnosis_head 不实例化，ExerciseDifficultyEncoder 不创建 b/a 参数；IRT 路径完全不存在）
- ablate_module3=True：模块3完全消融（MF 分支/融合门控/Prototype 全部不实例化；forward 不计算任何 MF/Proto）

注意：
- 当 module2 被消融（ablate_module2=True）时，Prototype（C）在结构上对输出无贡献且容易引入“伪路径”，因此强制关闭（完全消融语义更干净）。
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

        # 注意：concept_embeddings 会在主模型里“可选绑定”到 knowledge_encoder.concept_emb.weight
        self.concept_embeddings = nn.Parameter(torch.randn(num_concepts, concept_dim))  # Fix: 移除 0.02

        self.Wq = nn.ModuleList([nn.Linear(concept_dim, concept_dim, bias=False) for _ in range(num_heads)])
        self.Wk = nn.ModuleList([nn.Linear(concept_dim, concept_dim, bias=False) for _ in range(num_heads)])

        # temperature > 0：用 softplus 约束
        self.tau_raw = nn.Parameter(torch.ones(num_heads) * float(tau_init))

        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.xavier_normal_(self.concept_embeddings, gain=1.0)  # Fix: gain=1.0 prevents softmax saturation
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
            scores = scores / tau[h]

            if not self.allow_self_loop:
                eye = torch.eye(C, device=scores.device, dtype=torch.bool)
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
    - Module3 item branch: exercise_latent + Q-conditioned concept_latent
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
    ):
        super().__init__()
        self.num_exercises = int(num_exercises)
        self.num_concepts = int(num_concepts)
        self.exercise_dim = int(exercise_dim)

        self.use_mf_branch = bool(use_mf_branch)
        self.use_q_conditioning = bool(use_q_conditioning and use_mf_branch)
        self.use_irt = bool(use_irt)

        self.register_buffer("q_matrix", q_matrix)

        if self.use_mf_branch:
            self.exercise_latent = nn.Embedding(num_exercises, exercise_dim)
            self.exercise_bias = nn.Embedding(num_exercises, 1)
            self.concept_latent = nn.Embedding(num_concepts, exercise_dim)
            self.q_gate_raw = nn.Parameter(torch.zeros(1))
        else:
            self.exercise_latent = None
            self.exercise_bias = None
            self.concept_latent = None
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
            nn.init.xavier_normal_(self.exercise_latent.weight)
            nn.init.zeros_(self.exercise_bias.weight)
            nn.init.xavier_normal_(self.concept_latent.weight)

        if self.use_irt:
            nn.init.zeros_(self.b.weight)
            nn.init.normal_(self.a_raw.weight, mean=0.0, std=0.02)

    def forward(
        self,
        exercise_ids: torch.Tensor,
        concept_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Returns:
            item_latent:   (B, De) or None
            exercise_bias: (B,) or None
            b:             (B,)
            a:             (B,)
        """
        device = exercise_ids.device
        B = exercise_ids.size(0)

        if self.use_mf_branch:
            e_latent = self.exercise_latent(exercise_ids)
            e_bias = self.exercise_bias(exercise_ids).squeeze(-1)
            if concept_mask is None:
                concept_mask = self.q_matrix[exercise_ids]
            q = concept_mask.float()
            q_norm = q / (q.sum(dim=1, keepdim=True) + 1e-12)
            c_lat = self.concept_latent.weight
            q_latent = torch.matmul(q_norm, c_lat)

            if self.use_q_conditioning and self.q_gate_raw is not None:
                q_gate = torch.sigmoid(self.q_gate_raw)
                item_latent = e_latent + q_gate * q_latent
            else:
                item_latent = e_latent

            item_latent = self.dropout(item_latent)
        else:
            item_latent = None
            e_bias = None

        if self.use_irt:
            b = self.b(exercise_ids).squeeze(-1)
            a = F.softplus(self.a_raw(exercise_ids).squeeze(-1)) + 1e-6
        else:
            b = torch.zeros(B, device=device)
            a = torch.ones(B, device=device)

        return item_latent, e_bias, b, a


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

    def __init__(self, knowledge_dim: int):
        super().__init__()
        self.theta_proj = parametrizations.weight_norm(nn.Linear(knowledge_dim, 1, bias=True))

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
# 模块 3（B）：MF 残差头 - MFResidualHead
# ======================================================

class QAlignedResidualHead(nn.Module):
    """Low-capacity Q-aligned residual head for Module3."""

    def __init__(
        self,
        student_latent_dim: int,
        concept_latent_dim: int,
        residual_dim: int = 32,
        dropout: float = 0.1,
        residual_clip_t: float = 2.0,
    ):
        super().__init__()
        self.u_proj = nn.Linear(student_latent_dim, residual_dim, bias=False)
        self.v_proj = nn.Linear(concept_latent_dim, residual_dim, bias=False)
        nn.init.xavier_normal_(self.u_proj.weight)
        nn.init.xavier_normal_(self.v_proj.weight)

        self.mf_scale_raw = nn.Parameter(torch.tensor(0.0))
        self.bias_scale_raw = nn.Parameter(torch.tensor(0.0))
        self.mf_bias = nn.Parameter(torch.zeros(1))
        self.residual_clip_t = float(residual_clip_t)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        student_latent: torch.Tensor,
        student_bias: torch.Tensor,
        item_latent: torch.Tensor,
        exercise_bias: torch.Tensor,
        q_gate: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        u = self.u_proj(student_latent)
        v = self.v_proj(item_latent)
        u = F.normalize(u, dim=-1, eps=1e-12)
        v = F.normalize(v, dim=-1, eps=1e-12)

        interaction_scale = F.softplus(self.mf_scale_raw) + 1e-6
        bias_scale = F.softplus(self.bias_scale_raw) + 1e-6

        if q_gate is None:
            q_gate_value = interaction_scale.new_tensor(1.0)
        else:
            q_gate_value = q_gate.to(dtype=interaction_scale.dtype, device=interaction_scale.device)

        interaction_residual = interaction_scale * q_gate_value * (u * v).sum(dim=-1)
        bias_residual = bias_scale * (student_bias + exercise_bias)
        residual = interaction_residual + bias_residual + self.mf_bias

        if self.residual_clip_t > 0:
            t = self.residual_clip_t
            residual = t * torch.tanh(residual / t)

        residual = self.dropout(residual)

        if not return_details:
            return residual

        details = {
            "mf_logit": residual.detach(),
            "residual_logit": residual.detach(),
            "interaction_residual": interaction_residual.detach(),
            "bias_residual": bias_residual.detach(),
            "mf_scale": interaction_scale.detach(),
            "bias_scale": bias_scale.detach(),
            "q_gate": q_gate_value.detach(),
        }
        return residual, details


class ConservativeFusionGate(nn.Module):
    """Conservative residual fusion: gate_max * sigmoid(linear)."""

    def __init__(self, gate_max: float = 0.4, gate_bias_init: float = -2.5):
        super().__init__()
        self.fusion_gate = nn.Linear(2, 1)
        self.gate_max = float(gate_max)
        nn.init.constant_(self.fusion_gate.bias, float(gate_bias_init))
        nn.init.constant_(self.fusion_gate.weight, 0.0)

    def forward(
        self,
        irt_logit: torch.Tensor,
        mf_logit: torch.Tensor,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        stack = torch.stack([irt_logit, mf_logit], dim=1)
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
# 模块 3（C）：Soft Prototype 校正 - SoftPrototypeModule
# ======================================================

class SoftPrototypeModule(nn.Module):
    """Soft prototype（可消融）：用于对 student_repr/knowledge_state 做稳定校正。"""

    def __init__(self, num_prototypes: int, dim: int, tau: float = 1.0):
        super().__init__()
        self.num_prototypes = int(num_prototypes)
        self.tau = float(tau)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, dim) * 0.1)
        nn.init.xavier_normal_(self.prototypes)

    def forward(self, student_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # eps 防止 student_repr 为零向量时 normalize NaN
        s = F.normalize(student_repr, dim=-1, eps=1e-12)
        p = F.normalize(self.prototypes, dim=-1, eps=1e-12)
        logits = torch.matmul(s, p.t()) / (self.tau + 1e-12)
        assign = F.softmax(logits, dim=-1)
        mix = torch.matmul(assign, self.prototypes)
        return mix, assign


# ======================================================
# 模块 1（E）：个性化图相关组件 - AdaptiveGate / PersonalRelationGenerator
# ======================================================

class AdaptiveGate(nn.Module):
    """个性化图混合系数 alpha（B,1,1,1）。
    
    Fix: 添加 LayerNorm 标准化输入，解决 student embedding variance 过低的问题。
    """

    def __init__(self, student_dim: int):
        super().__init__()
        self.input_norm = nn.LayerNorm(student_dim)  # Fix: 标准化输入
        hid = max(1, student_dim // 2)
        self.gate = nn.Sequential(
            nn.Linear(student_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, 1),
            nn.Sigmoid(),
        )

    def forward(self, student_repr: torch.Tensor) -> torch.Tensor:
        student_repr = self.input_norm(student_repr)  # Fix: 标准化输入
        return self.gate(student_repr).view(-1, 1, 1, 1)


class PersonalRelationGenerator(nn.Module):
    """低秩分解生成个性化邻接（B,C,C），softmax 保证 row-stochastic。
    
    Fix: 添加 LayerNorm 标准化输入，解决 student embedding variance 过低的问题。
    """

    def __init__(self, student_dim: int, num_concepts: int, rank: int = 4):
        super().__init__()
        self.input_norm = nn.LayerNorm(student_dim)  # Fix: 标准化输入
        self.num_concepts = int(num_concepts)
        self.rank = int(rank)
        self.to_u = nn.Linear(student_dim, num_concepts * rank, bias=False)
        self.to_v = nn.Linear(student_dim, num_concepts * rank, bias=False)
        nn.init.xavier_normal_(self.to_u.weight)
        nn.init.xavier_normal_(self.to_v.weight)

    def forward(self, student_repr: torch.Tensor) -> torch.Tensor:
        student_repr = self.input_norm(student_repr)  # Fix: 标准化输入
        B = student_repr.size(0)
        u = self.to_u(student_repr).view(B, self.num_concepts, self.rank)
        v = self.to_v(student_repr).view(B, self.num_concepts, self.rank)
        scores = torch.bmm(u, v.transpose(1, 2))  # (B,C,C)
        A = F.softmax(scores, dim=-1)             # row-stochastic
        return A


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
        gnn_residual_weight: float,
        use_concept_graph: bool,
        graph_topk: Optional[int],
        allow_self_loop: bool,
        # personal graph
        use_personal_graph: bool,
        personal_rank: int,
        # 完全消融开关
        enable_module: bool = True,
    ):
        super().__init__()
        self.enable_module = bool(enable_module)

        # 保存形状信息：用于完全消融时构造全0张量
        self.num_concepts = int(num_concepts)
        self.knowledge_dim = int(knowledge_dim)
        self.num_relation_heads = int(num_relation_heads)

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
        self.use_personal_graph = bool(use_personal_graph)

        # A) 全局概念图学习
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

        # Fix #5：可选权重共享（只要启用概念图就绑定）
        if self.use_concept_graph and self.relation_learning is not None:
            if self.knowledge_encoder.concept_emb.weight.shape == self.relation_learning.concept_embeddings.shape:
                self.knowledge_encoder.concept_emb.weight = self.relation_learning.concept_embeddings

        # E) 个性化图（可选）
        if self.use_personal_graph:
            self.adaptive_gate = AdaptiveGate(knowledge_dim)
            self.personal_generator = PersonalRelationGenerator(knowledge_dim, num_concepts, personal_rank)
        else:
            self.adaptive_gate = None
            self.personal_generator = None

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
        personal_matrices = None
        relation_used = relation_matrices

        if self.use_personal_graph and self.adaptive_gate is not None and self.personal_generator is not None:
            # Fix: 使用学生的 global embedding 作为个性化图生成的输入
            # 这样每个学生有独特的表示，而不是经过概念平均后几乎相同的 student_repr
            student_global_repr = self.knowledge_encoder.student_global(student_ids)  # (B,D)
            gate_alpha = self.adaptive_gate(student_global_repr)              # (B,1,1,1)
            personal_matrices = self.personal_generator(student_global_repr)  # (B,C,C)

            # 优化：不展开为 (B,H,C,C)，而是保存 gate_alpha 和 personal_matrices
            # 让 GNN 层在需要时逐 head 混合，减少显存占用
            # relation_used 改为字典传递必要信息
            relation_used = {
                "global_matrices": relation_matrices,        # (H,C,C)
                "personal_matrices": personal_matrices,      # (B,C,C)
                "gate_alpha": gate_alpha.squeeze(-1).squeeze(-1).squeeze(-1),  # (B,)
            }

            knowledge_state = self.knowledge_encoder(student_ids, relation_used)
            student_repr = knowledge_state.mean(dim=1)

        return {
            "relation_matrices": relation_matrices,
            "relation_used": relation_used if isinstance(relation_used, torch.Tensor) else relation_matrices,
            "knowledge_state": knowledge_state,
            "student_repr": student_repr,
            "alpha": gate_alpha,
            "personal_matrices": personal_matrices,
        }


# ======================================================
# 主模型：组合模块 1/2/3，提供统一 forward 与正则/诊断接口
# ======================================================

class CognitiveDiagnosisModel(nn.Module):
    """
    主模型（组合 3 个模块）：

    - Module 1: ConceptStructureModeling（A + E）
    - Module 2: CognitiveDiagnosisHead（D）
    - Module 3: Neural Residual + Prototype Calibration（B + C）

    关键：支持三个“模块级完全消融”开关：
      - ablate_module1：完全移除模块1（A/E/knowledge_encoder 都不存在）
      - ablate_module2：完全移除模块2（theta_proj 不存在；IRT b/a 参数不存在）
      - ablate_module3：完全移除模块3（MF/Proto/Fusion 都不存在）

    同时保留旧的“子模块消融”开关：
      - use_concept_graph / use_personal_graph
      - use_mf_branch
      - use_soft_prototype
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
        num_prototypes: int = 3,
        proto_tau: float = 1.0,
        proto_lambda: float = 0.5,
        use_soft_prototype: bool = True,
        use_soft_prototype_main_path: bool = False,
        use_personal_graph: bool = False,
        personal_rank: int = 4,
        # ===== 模块级“完全消融”开关 =====
        ablate_module1: bool = False,
        ablate_module2: bool = False,
        ablate_module3: bool = False,
        # ===== Module3 (Q-aligned residual + conservative fusion) =====
        use_q_aligned_residual: bool = True,
        fusion_gate_max: float = 0.4,
        fusion_gate_bias_init: float = -2.5,
        residual_clip_t: float = 2.0,
        # ===== 正则权重 =====
        lambda_sparse_personal: float = 0.0,
        lambda_alpha: float = 0.0,
        lambda_graph_entropy: float = 0.01,  # mapped from args.lambda_sparse
        mf_l2_lambda: float = 5e-5,          # mapped from args.exercise_l2_lambda
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

        # ===== 模块级开关（保存用于日志与严格控制）=====
        self.enable_module1 = not bool(ablate_module1)
        self.enable_module2 = not bool(ablate_module2)
        self.enable_module3 = not bool(ablate_module3)

        # 如果 module2 和 module3 都被消融，则模型没有任何预测路径：直接报错（避免“跑了但无意义”）
        if (not self.enable_module2) and (not self.enable_module3):
            raise ValueError("Invalid ablation: both Module 2 (IRT head) and Module 3 (MF/Proto) are disabled; no prediction path.")

        # ===== 子模块开关：根据“模块级消融”强制覆盖，保证彻底 =====
        # 模块1完全消融 => A/E 全部关闭（且结构模块不创建参数）
        if not self.enable_module1:
            use_concept_graph = False
            use_personal_graph = False

        # 模块3完全消融 => MF/Proto 都关闭
        if not self.enable_module3:
            use_mf_branch = False
            use_soft_prototype = False

        # 模块2完全消融 => 强制关闭 Prototype（C），避免“无输出贡献但仍计算/训练”的不彻底
        if not self.enable_module2:
            use_soft_prototype = False

        # 最终保存子模块开关（用于 details）
        self.use_concept_graph = bool(use_concept_graph)
        self.use_personal_graph = bool(use_personal_graph)
        self.use_mf_branch = bool(use_mf_branch and use_q_aligned_residual)
        self.use_soft_prototype = bool(use_soft_prototype and num_prototypes > 0 and self.enable_module3 and self.enable_module2)
        self.proto_lambda = float(proto_lambda)
        self.use_soft_prototype_main_path = bool(use_soft_prototype_main_path and self.use_soft_prototype)
        self.use_q_aligned_residual = bool(use_q_aligned_residual)
        self.fusion_gate_max = float(fusion_gate_max)
        self.fusion_gate_bias_init = float(fusion_gate_bias_init)
        self.residual_clip_t = float(residual_clip_t)

        # ===== 正则权重 =====
        self.lambda_graph_entropy = float(lambda_graph_entropy)
        self.lambda_sparse_personal = float(lambda_sparse_personal)
        self.lambda_alpha = float(lambda_alpha)
        self.mf_l2_lambda = float(mf_l2_lambda)

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
            gnn_residual_weight=gnn_residual_weight,
            use_concept_graph=self.use_concept_graph,
            graph_topk=graph_topk,
            allow_self_loop=allow_self_loop,
            use_personal_graph=self.use_personal_graph,
            personal_rank=personal_rank,
            enable_module=self.enable_module1,  # 关键：模块级完全消融
        )

        # ------------------------------
        # Module 2：认知诊断头（D）
        # ------------------------------
        if self.enable_module2:
            self.diagnosis_head = CognitiveDiagnosisHead(knowledge_dim=knowledge_dim)
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
        )

        # ------------------------------
        # Module 3：神经增强与校正（B + C）
        # ------------------------------
        if self.enable_module3 and self.use_mf_branch:
            self.skill_encoder = StudentLatentEncoder(num_students, latent_dim=skill_dim)
            self.mf_head = QAlignedResidualHead(
                student_latent_dim=skill_dim,
                concept_latent_dim=exercise_dim,
                residual_dim=min(64, skill_dim, exercise_dim),
                dropout=dropout,
                residual_clip_t=self.residual_clip_t,
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

        # Prototype 仅在 module2 启用（否则无输出贡献）且 module3 启用时存在
        if self.use_soft_prototype:
            self.prototype_module = SoftPrototypeModule(num_prototypes, knowledge_dim, proto_tau)
        else:
            self.prototype_module = None

    # ------------------------------
    # Fix #1：行熵稀疏度（用于 personal graph 正则）
    # ------------------------------
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

        # ========== 2) Module 3（C）：Prototype 校正（只在 module2+module3 启用时存在） ==========
        proto_mix = None
        proto_assign = None
        if self.prototype_module is not None:
            proto_mix, proto_assign = self.prototype_module(student_repr)  # (B,D), (B,K)
            if self.use_soft_prototype_main_path:
                proto_broadcast = proto_mix.unsqueeze(1).expand(-1, self.num_concepts, -1)
                knowledge_state = (1.0 - self.proto_lambda) * knowledge_state + self.proto_lambda * proto_broadcast

        # ========== 3) 共享：题目参数（b,a）与 module3 item-latent ==========
        item_latent, exercise_bias, b, a = self.exercise_encoder(exercise_ids, concept_mask=q_vector)

        # ========== 4) Module 2：IRT head（可完全消融） ==========
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

        # ========== 5) Module 3：MF residual（可完全消融） ==========
        if self.enable_module3 and self.use_mf_branch:
            if self.skill_encoder is None or self.mf_head is None:
                raise RuntimeError("enable_module3 & use_mf_branch=True but MF modules are None. Check init wiring.")

            student_latent, student_bias = self.skill_encoder(student_ids)

            if item_latent is None or exercise_bias is None:
                raise RuntimeError("Module3 is enabled but item_latent/exercise_bias is None. Check exercise_encoder wiring.")

            if return_details:
                mf_logit, mf_details = self.mf_head(
                    student_latent=student_latent,
                    student_bias=student_bias,
                    item_latent=item_latent,
                    exercise_bias=exercise_bias,
                    return_details=True,
                )
            else:
                mf_logit = self.mf_head(
                    student_latent=student_latent,
                    student_bias=student_bias,
                    item_latent=item_latent,
                    exercise_bias=exercise_bias,
                    return_details=False,
                )
                mf_details = None
        else:
            student_latent, student_bias = None, None
            mf_logit = torch.zeros_like(irt_logit)
            mf_details = None

        # ========== 6) 组合输出：严格避免“半消融” ==========
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

        # ========== 7) 返回 logits 或 prob ==========
        out_main = total_logit if return_logits else torch.sigmoid(total_logit)

        if not return_details:
            return out_main

        # ========== 8) details（用于正则、可解释输出、排查消融是否生效） ==========
        details: Dict[str, torch.Tensor] = {
            # 模块级开关（用于日志对齐）
            "enable_module1": torch.tensor(int(self.enable_module1), device=device),
            "enable_module2": torch.tensor(int(self.enable_module2), device=device),
            "enable_module3": torch.tensor(int(self.enable_module3), device=device),

            # 子模块开关
            "use_concept_graph": torch.tensor(int(self.use_concept_graph), device=device),
            "use_personal_graph": torch.tensor(int(self.use_personal_graph), device=device),
            "use_mf_branch": torch.tensor(int(self.use_mf_branch), device=device),
            "use_soft_prototype": torch.tensor(int(self.prototype_module is not None), device=device),
            "use_soft_prototype_main_path": torch.tensor(int(self.use_soft_prototype_main_path), device=device),

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

        if diag_details is not None:
            details.update(diag_details)
        if mf_details is not None:
            details.update(mf_details)
        if fuse_details is not None:
            details.update(fuse_details)

        if self.prototype_module is not None and proto_assign is not None:
            details["prototype_mix"] = proto_mix.detach()
            details["prototype_assign"] = proto_assign.detach()

        if self.use_personal_graph:
            if gate_alpha is not None:
                # Keep gradient path for personal regularizers in training.
                details["alpha"] = gate_alpha
                details["alpha_detached"] = gate_alpha.detach()
            if personal_matrices is not None:
                # Keep gradient path for personal regularizers in training.
                details["personal_matrices"] = personal_matrices
                details["personal_matrices_detached"] = personal_matrices.detach()

        return out_main, details

    def get_regularization_components(
        self,
        relation_matrices: torch.Tensor,
        details: Optional[Dict[str, torch.Tensor]] = None,
        lambda_proto_div: float = 0.0,
        lambda_proto_usage: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Return decomposed regularization terms.
        This does not change optimization objective; it is used for logging/diagnostics.
        """
        device = relation_matrices.device
        terms: Dict[str, torch.Tensor] = {
            "graph_entropy": torch.tensor(0.0, device=device),
            "mf_l2": torch.tensor(0.0, device=device),
            "proto_div": torch.tensor(0.0, device=device),
            "proto_usage": torch.tensor(0.0, device=device),
            "personal_sparse": torch.tensor(0.0, device=device),
            "alpha_var": torch.tensor(0.0, device=device),  # signed term (negative when maximizing variance)
        }

        # (1) Global graph entropy
        if self.enable_module1 and self.use_concept_graph and self.lambda_graph_entropy > 0:
            if self.structure_module.relation_learning is not None:
                entropy = self.structure_module.relation_learning.get_entropy_sparsity(relation_matrices)
                terms["graph_entropy"] = self.lambda_graph_entropy * entropy

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
                    reg_terms.append(self.skill_encoder.latent_emb.weight.pow(2).mean())
                if self.exercise_encoder.use_mf_branch and self.exercise_encoder.exercise_latent is not None:
                    reg_terms.append(self.exercise_encoder.exercise_latent.weight.pow(2).mean())
                if self.exercise_encoder.use_mf_branch and self.exercise_encoder.concept_latent is not None:
                    reg_terms.append(self.exercise_encoder.concept_latent.weight.pow(2).mean())
                if self.exercise_encoder.use_mf_branch and self.exercise_encoder.exercise_bias is not None:
                    reg_terms.append(self.exercise_encoder.exercise_bias.weight.pow(2).mean())
                if self.exercise_encoder.use_mf_branch and self.exercise_encoder.q_gate_raw is not None:
                    reg_terms.append(self.exercise_encoder.q_gate_raw.pow(2).mean())

            if len(reg_terms) > 0:
                terms["mf_l2"] = self.mf_l2_lambda * sum(reg_terms)

        # (3) Prototype regularizers
        if self.prototype_module is not None and details is not None and "prototype_assign" in details:
            assign = details["prototype_assign"]  # (B,K)
            K = assign.size(1)

            if lambda_proto_div > 0.0:
                P = F.normalize(self.prototype_module.prototypes, dim=-1, eps=1e-12)  # (K,D)
                sim = P @ P.t()
                off = sim - torch.eye(K, device=device, dtype=sim.dtype)
                proto_div = (off.pow(2).sum() / (K * (K - 1) + 1e-12))
                terms["proto_div"] = lambda_proto_div * proto_div

            if lambda_proto_usage > 0.0:
                q_mean = assign.mean(dim=0)
                uniform = torch.full_like(q_mean, 1.0 / K)
                proto_usage = F.mse_loss(q_mean, uniform)
                terms["proto_usage"] = lambda_proto_usage * proto_usage

        # (4) Personal graph regularizers
        if self.enable_module1 and self.use_personal_graph and details is not None:
            if (
                "personal_matrices" in details
                and details["personal_matrices"] is not None
                and self.lambda_sparse_personal > 0
            ):
                pm = details["personal_matrices"]
                terms["personal_sparse"] = self.lambda_sparse_personal * self._row_entropy(pm)

            if "alpha" in details and details["alpha"] is not None and self.lambda_alpha > 0:
                alpha_flat = details["alpha"].view(-1)
                alpha_var = alpha_flat.var() + 1e-6
                terms["alpha_var"] = -self.lambda_alpha * alpha_var

        total = (
            terms["graph_entropy"]
            + terms["mf_l2"]
            + terms["proto_div"]
            + terms["proto_usage"]
            + terms["personal_sparse"]
            + terms["alpha_var"]
        )
        terms["total"] = total
        return terms

    def get_regularization_loss(
        self,
        relation_matrices: torch.Tensor,
        details: Optional[Dict[str, torch.Tensor]] = None,
        lambda_proto_div: float = 0.0,
        lambda_proto_usage: float = 0.0,
    ) -> torch.Tensor:
        """
        正则项汇总：
        (1) 全局概念图行熵（lambda_graph_entropy）—— 仅 module1 启用且 use_concept_graph=True 时有效
        (2) MF/IRT 参数 L2（mf_l2_lambda）—— 仅对应模块启用时才计入（避免不彻底消融）
        (3) Prototype 正则（div / usage）—— 仅 prototype 存在时计入
        (4) 个性化图稀疏 + alpha 惩罚 —— 仅 personal graph 存在时计入
        """
        terms = self.get_regularization_components(
            relation_matrices=relation_matrices,
            details=details,
            lambda_proto_div=lambda_proto_div,
            lambda_proto_usage=lambda_proto_usage,
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
