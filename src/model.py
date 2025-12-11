"""
认知诊断模型（单文件实现 - 神经/认知双路融合版）：
- [架构升级] Dual-Branch Fusion:
    1. Cognitive Branch (IRT): 使用 Masked Mean 聚合（取其稳定性）。
    2. Neural Branch (MF): 使用 Student Latent * Exercise Latent（取其高拟合上限）。
- [简化] 移除 Exercise GNN：减少参数量，防止在习题侧过拟合。
- [增强] Skill Encoder 升级：维度提升至 64，作为 Neural Branch 的学生输入。
- [正则] Dropout 提升至 0.3，加强泛化。

适用数据集：Assist09, Junyi
目标：利用 Neural Branch 捕捉 IRT 无法解释的残差，冲击 0.7790+。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Union
import math

# ======================================================
# 1. 图与关系模块
# ======================================================


class MultiHeadRelationLearning(nn.Module):
    """多头概念关系学习，输出稀疏邻接矩阵。"""

    def __init__(
            self,
            num_concepts: int,
            concept_dim: int,
            num_heads: int = 4,
            dropout: float = 0.1
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.concept_dim = concept_dim
        self.num_heads = num_heads

        self.concept_embeddings = nn.Parameter(
            torch.randn(num_concepts, concept_dim)
        )

        self.attention_heads = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=concept_dim,
                num_heads=1,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_heads)
        ])

        # 固定缩放
        self.scale = 1.0 / math.sqrt(concept_dim)
        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.concept_embeddings)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        concepts = self.concept_embeddings.unsqueeze(0)
        relation_matrices = []

        for i, attn_head in enumerate(self.attention_heads):
            _, attn_weights = attn_head(
                concepts, concepts, concepts,
                need_weights=True,
                average_attn_weights=True
            )
            attn_weights = attn_weights.squeeze(0)
            relation_matrices.append(attn_weights)

        relation_matrices = torch.stack(relation_matrices, dim=0)
        return relation_matrices, self.concept_embeddings

    def get_sparsity_loss(self, relation_matrices: torch.Tensor) -> torch.Tensor:
        # L2 Loss 稳定性更好
        return torch.mean(relation_matrices ** 2)


class ConceptGraphConv(nn.Module):
    """图卷积层"""

    def __init__(
            self,
            in_features: int,
            out_features: int,
            num_heads: int = 4,
            dropout: float = 0.1
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_transforms = nn.ModuleList([
            nn.Linear(in_features, out_features, bias=False)
            for _ in range(num_heads)
        ])
        self.head_attention = nn.Parameter(torch.ones(num_heads) / num_heads)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()

    def _initialize_weights(self):
        for transform in self.head_transforms:
            nn.init.xavier_normal_(transform.weight)

    def forward(self, x: torch.Tensor, relation_matrices: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i in range(self.num_heads):
            adj = relation_matrices[i]
            # 归一化
            adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-6)
            
            h = self.head_transforms[i](x)
            h = torch.matmul(adj, h)
            outputs.append(h)

        output = torch.stack(outputs, dim=0)
        attn_weights = F.softmax(self.head_attention, dim=0).view(-1, 1, 1, 1)
        output = (output * attn_weights).sum(dim=0)
        output = output + self.bias
        output = self.dropout(output)
        return output


# ======================================================
# 2. 编码器
# ======================================================


class StudentKnowledgeEncoder(nn.Module):
    """学生知识状态编码器 (Cognitive Branch)"""

    def __init__(
            self,
            num_students: int,
            num_concepts: int,
            knowledge_dim: int,
            num_gnn_layers: int = 2,
            num_relation_heads: int = 4,
            dropout: float = 0.1,
            gnn_residual_weight: float = 0.5 
    ):
        super().__init__()
        self.num_students = num_students
        self.num_concepts = num_concepts
        self.knowledge_dim = knowledge_dim
        self.gnn_residual_weight = gnn_residual_weight

        self.student_emb = nn.Embedding(num_students, knowledge_dim)
        self.concept_emb = nn.Embedding(num_concepts, knowledge_dim)

        self.gnn_layers = nn.ModuleList([
            ConceptGraphConv(
                knowledge_dim,
                knowledge_dim,
                num_heads=num_relation_heads,
                dropout=dropout
            ) for _ in range(num_gnn_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(knowledge_dim)
            for _ in range(num_gnn_layers)
        ])

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.student_emb.weight)
        nn.init.xavier_normal_(self.concept_emb.weight)

    def forward(self, student_ids: torch.Tensor, relation_matrices: torch.Tensor) -> torch.Tensor:
        batch_size = student_ids.size(0)
        student_vec = self.student_emb(student_ids)
        concept_vec = self.concept_emb.weight.unsqueeze(0).expand(batch_size, -1, -1)
        student_vec_expanded = student_vec.unsqueeze(1).expand(-1, self.num_concepts, -1)
        
        h = student_vec_expanded + concept_vec

        for gnn, norm in zip(self.gnn_layers, self.layer_norms):
            h_in = h
            h_out = gnn(h, relation_matrices)
            h = norm(h_in + self.gnn_residual_weight * h_out)
            h = F.relu(h)

        return h


class StudentLatentEncoder(nn.Module):
    """
    [升级] 学生隐向量编码器 (Neural Branch)
    替代之前的 TestTakingSkillEncoder，维度提升，用于捕捉 IRT 无法解释的潜在特征。
    """
    def __init__(self, num_students: int, latent_dim: int = 64):
        super().__init__()
        self.latent_emb = nn.Embedding(num_students, latent_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.latent_emb.weight)

    def forward(self, student_ids: torch.Tensor) -> torch.Tensor:
        return self.latent_emb(student_ids)


class ExerciseDifficultyEncoder(nn.Module):
    """
    [简化] 习题编码器
    移除 GNN，直接学习 Embedding，减少噪声和过拟合。
    """
    def __init__(
            self,
            num_exercises: int,
            num_concepts: int,
            q_matrix: torch.Tensor,
            exercise_dim: int = 64,
            knowledge_dim: int = 32, # 仅占位，保持接口兼容
            num_heads: int = 4,      # 仅占位
            num_gnn_layers: int = 2, # 仅占位
            dropout: float = 0.1,    # 仅占位
            use_graph: bool = False, # 强制 False
    ):
        super().__init__()
        self.exercise_dim = exercise_dim
        self.register_buffer('q_matrix', q_matrix)

        # 习题隐向量 (用于 Neural Branch)
        self.exercise_emb = nn.Embedding(num_exercises, exercise_dim)
        
        # 习题 IRT 参数 (用于 Cognitive Branch)
        self.difficulty = nn.Embedding(num_exercises, num_concepts)
        self.discrimination = nn.Embedding(num_exercises, num_concepts)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.exercise_emb.weight)
        nn.init.zeros_(self.difficulty.weight)
        nn.init.ones_(self.discrimination.weight)

    def forward(self, exercise_ids: torch.Tensor, relation_matrices: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 直接查表，不进行 GNN 传播，减少过拟合风险
        exercise_emb = self.exercise_emb(exercise_ids)
        difficulty = self.difficulty(exercise_ids)
        discrimination = self.discrimination(exercise_ids)

        # 区分度保证非负
        discrimination = F.softplus(discrimination)
        
        return exercise_emb, difficulty, discrimination


# ======================================================
# 3. 预测与原型模块
# ======================================================


class ResponsePredictionHead(nn.Module):
    """
    [重构] 双路预测头 (Dual-Branch Prediction Head)
    Branch 1: IRT (Mean Aggregation) -> 负责显性知识推理 (稳定)
    Branch 2: Neural (Dot Product) -> 负责隐性特征拟合 (高上限)
    """

    def __init__(
            self,
            knowledge_dim: int,
            skill_dim: int,   # 现在这是 latent_dim
            exercise_dim: int,
            hidden_dim: int = 128
    ):
        super().__init__()
        # IRT 参数
        self.knowledge_weight_raw = nn.Parameter(torch.randn(knowledge_dim))
        self.knowledge_bias = nn.Parameter(torch.zeros(1))
        
        # Neural 参数 (简单的 MLP 用于融合或调整)
        # 这里我们使用直接的点积作为 Neural Branch 的核心，再加一个 Bias
        self.neural_bias = nn.Parameter(torch.zeros(1))

        # 融合层：学习两路的权重
        self.fusion_gate = nn.Linear(2, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.knowledge_weight_raw, mean=0.0, std=0.1)
        nn.init.zeros_(self.knowledge_bias)
        nn.init.zeros_(self.neural_bias)
        # 初始化 fusion gate，使其均衡
        nn.init.constant_(self.fusion_gate.weight, 0.5)
        nn.init.zeros_(self.fusion_gate.bias)

    def forward(
            self,
            knowledge_state: torch.Tensor,   # (B, C, D)
            student_latent: torch.Tensor,    # (B, Latent) - 原 skill_vector
            exercise_emb: torch.Tensor,      # (B, Latent)
            difficulty: torch.Tensor,        # (B, C)
            discrimination: torch.Tensor,    # (B, C)
            concept_mask: torch.Tensor,      # (B, C)
    ) -> torch.Tensor:
        
        # === Branch 1: Cognitive IRT (Masked Mean) ===
        # 这一路保证模型的可解释性和基本稳定性 (0.775 的基础)
        w_pos = F.softplus(self.knowledge_weight_raw)
        knowledge_scores = torch.matmul(knowledge_state, w_pos.view(-1, 1)).squeeze(-1) + self.knowledge_bias
        
        irt_logits = discrimination * (knowledge_scores - difficulty)
        masked_irt = irt_logits * concept_mask
        num_concepts = concept_mask.sum(dim=1) + 1e-9
        # 使用 Mean 聚合，因为它已被验证最稳定
        irt_score = masked_irt.sum(dim=1) / num_concepts # (B,)

        # === Branch 2: Neural Matrix Factorization ===
        # 这一路负责拟合残差，提升上限
        # Dot Product Interaction
        neural_score = (student_latent * exercise_emb).sum(dim=-1) + self.neural_bias # (B,)

        # === Fusion ===
        # 简单的加和通常最有效，或者加权和
        # total_logit = irt_score + neural_score
        
        # 尝试加权融合
        # stack: (B, 2)
        scores_stack = torch.stack([irt_score, neural_score], dim=1)
        total_logit = scores_stack.sum(dim=1) # 直接相加，让梯度自由流动

        final_prob = torch.sigmoid(total_logit).clamp(min=1e-6, max=1 - 1e-6)
        return final_prob


class SoftPrototypeModule(nn.Module):
    """软原型模块"""
    def __init__(self, num_prototypes: int, dim: int, tau: float = 1.0):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.tau = tau
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, dim) * 0.1)
        nn.init.xavier_normal_(self.prototypes)

    def forward(self, student_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s = F.normalize(student_repr, dim=-1)
        p = F.normalize(self.prototypes, dim=-1)
        logits = torch.matmul(s, p.t()) / self.tau
        assign_q = F.softmax(logits, dim=-1)
        proto_mix = torch.matmul(assign_q, self.prototypes)
        return proto_mix, assign_q


class AdaptiveGate(nn.Module):
    """自适应门控"""
    def __init__(self, student_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(student_dim, student_dim // 2),
            nn.ReLU(),
            nn.Linear(student_dim // 2, 1),
            nn.Sigmoid()
        )
    def forward(self, student_repr: torch.Tensor) -> torch.Tensor:
        return self.gate(student_repr).view(-1, 1, 1)


class PersonalRelationGenerator(nn.Module):
    """个性化关系生成"""
    def __init__(self, student_dim: int, num_concepts: int, rank: int = 4):
        super().__init__()
        self.num_concepts = num_concepts
        self.rank = rank
        self.to_u = nn.Linear(student_dim, num_concepts * rank, bias=False)
        self.to_v = nn.Linear(student_dim, num_concepts * rank, bias=False)
        nn.init.xavier_normal_(self.to_u.weight)
        nn.init.xavier_normal_(self.to_v.weight)

    def forward(self, student_repr: torch.Tensor) -> torch.Tensor:
        batch_size = student_repr.size(0)
        u = self.to_u(student_repr).view(batch_size, self.num_concepts, self.rank)
        v = self.to_v(student_repr).view(batch_size, self.num_concepts, self.rank)
        return torch.matmul(u, v.transpose(-1, -2))


# ======================================================
# 4. 主模型
# ======================================================


class CognitiveDiagnosisModel(nn.Module):
    def __init__(
            self,
            num_students: int,
            num_exercises: int,
            num_concepts: int,
            q_matrix: torch.Tensor,
            knowledge_dim: int = 32,
            skill_dim: int = 2,
            exercise_dim: int = 64,
            num_relation_heads: int = 4,
            num_gnn_layers: int = 2,
            dropout: float = 0.1,
            num_prototypes: int = 3,
            proto_tau: float = 1.0,
            proto_lambda: float = 0.5,
            use_soft_prototype: bool = True,
            use_skill_encoder: bool = True,
            use_exercise_graph: bool = True,
            use_personal_graph: bool = False,
            personal_rank: int = 4,
            lambda_sparse_personal: float = 0.0,
            lambda_alpha: float = 0.0,
            exercise_l2_lambda: float = 5e-5,
            gnn_residual_weight: float = 0.5,
    ):
        super().__init__()
        self.num_students = num_students
        self.num_exercises = num_exercises
        self.num_concepts = num_concepts
        self.knowledge_dim = knowledge_dim
        
        # [调整] skill_dim 现在代表 neural branch 的 latent dim，强制与 exercise_dim 一致
        # 如果传入的 skill_dim 很小(如2)，这里强制覆盖为 exercise_dim 以保证 Neural Branch 能力
        self.skill_dim = exercise_dim 
        self.exercise_dim = exercise_dim
        
        self.use_skill_encoder = bool(use_skill_encoder)
        self.use_exercise_graph = bool(use_exercise_graph)
        self.use_personal_graph = bool(use_personal_graph)
        self.personal_rank = int(personal_rank)
        self.lambda_sparse_personal = float(lambda_sparse_personal)
        self.lambda_alpha = float(lambda_alpha)
        self.exercise_l2_lambda = float(exercise_l2_lambda)
        
        self.use_soft_prototype = bool(use_soft_prototype and num_prototypes > 0)
        self.proto_lambda = float(proto_lambda)

        self.relation_learning = MultiHeadRelationLearning(
            num_concepts=num_concepts,
            concept_dim=knowledge_dim,
            num_heads=num_relation_heads,
            dropout=dropout,
        )

        self.knowledge_encoder = StudentKnowledgeEncoder(
            num_students=num_students,
            num_concepts=num_concepts,
            knowledge_dim=knowledge_dim,
            num_gnn_layers=num_gnn_layers,
            num_relation_heads=num_relation_heads,
            dropout=dropout,
            gnn_residual_weight=gnn_residual_weight 
        )

        # [升级] 使用 StudentLatentEncoder 替代原 Skill Encoder
        self.skill_encoder = StudentLatentEncoder(num_students, self.skill_dim)

        self.exercise_encoder = ExerciseDifficultyEncoder(
            num_exercises=num_exercises,
            num_concepts=num_concepts,
            q_matrix=q_matrix,
            exercise_dim=exercise_dim,
            knowledge_dim=knowledge_dim,
            num_heads=num_relation_heads,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout,
            use_graph=False, # 强制关闭习题图，简化模型
        )

        self.prediction_head = ResponsePredictionHead(
            knowledge_dim=knowledge_dim,
            skill_dim=self.skill_dim,
            exercise_dim=exercise_dim,
        )

        self.register_buffer("q_matrix", q_matrix)

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

    def forward(
            self,
            student_ids: torch.Tensor,
            exercise_ids: torch.Tensor,
            concept_vector: torch.Tensor = None,
            return_details: bool = False
    ) -> Union[torch.Tensor, Tuple]:
        
        relation_matrices, concept_emb = self.relation_learning()
        knowledge_state = self.knowledge_encoder(student_ids, relation_matrices)

        proto_mix = None
        proto_assign = None
        student_repr = knowledge_state.mean(dim=1)

        if self.use_soft_prototype:
            proto_mix, proto_assign = self.prototype_module(student_repr)
            proto_broadcast = proto_mix.unsqueeze(1).expand(-1, self.num_concepts, -1)
            knowledge_state = (1.0 - self.proto_lambda) * knowledge_state + self.proto_lambda * proto_broadcast

        gate_alpha = None
        personal_matrices = None
        if self.use_personal_graph:
            gate_alpha = self.adaptive_gate(student_repr)
            personal_matrices = self.personal_generator(student_repr)

        # 获取 Student Latent Vector
        skill_vector = self.skill_encoder(student_ids)

        # 获取 Exercise Embeddings
        exercise_emb, difficulty, discrimination = self.exercise_encoder(
            exercise_ids, relation_matrices
        )

        q_vector = self.q_matrix[exercise_ids]
        
        pred_prob = self.prediction_head(
            knowledge_state=knowledge_state,
            student_latent=skill_vector, # 传入 Neural Branch
            exercise_emb=exercise_emb,   # 传入 Neural Branch
            difficulty=difficulty,
            discrimination=discrimination,
            concept_mask=q_vector, 
        )

        if return_details:
            details = {
                "relation_matrices": relation_matrices,
                "knowledge_state": knowledge_state,
                "skill_vector": skill_vector,
                "difficulty": difficulty,
                "discrimination": discrimination,
                "q_vector": q_vector,
                "student_repr": student_repr,
            }
            if self.use_soft_prototype:
                details["prototype_assign"] = proto_assign
                details["prototype_mix"] = proto_mix
            if self.use_personal_graph:
                details["alpha"] = gate_alpha
                details["personal_matrices"] = personal_matrices
            return pred_prob, details
        else:
            return pred_prob

    def get_regularization_loss(
            self,
            relation_matrices: torch.Tensor,
            skill_vector: torch.Tensor,
            knowledge_state: torch.Tensor,
            prototype_assign: Optional[torch.Tensor] = None,
            lambda_sparse: float = 0.01,
            lambda_proto_div: float = 0.0,
            lambda_proto_usage: float = 0.0,
            personal_matrices: Optional[torch.Tensor] = None,
            alpha: Optional[torch.Tensor] = None,
            lambda_sparse_personal: Optional[float] = None,
            lambda_alpha: Optional[float] = None,
    ) -> torch.Tensor:
        if lambda_sparse_personal is None: lambda_sparse_personal = self.lambda_sparse_personal
        if lambda_alpha is None: lambda_alpha = self.lambda_alpha
        device = knowledge_state.device

        # L2 Sparse Loss
        sparse_loss = self.relation_learning.get_sparsity_loss(relation_matrices)
        reg_loss = lambda_sparse * sparse_loss

        # Exercise L2
        if hasattr(self, "exercise_encoder") and self.exercise_l2_lambda > 0:
            ex_emb = self.exercise_encoder.exercise_emb.weight
            diff_w = self.exercise_encoder.difficulty.weight
            disc_w = self.exercise_encoder.discrimination.weight
            # 增加对 skill (student latent) 的正则化，防止 Neural Branch 过拟合
            skill_emb = self.skill_encoder.latent_emb.weight
            
            exercise_l2 = (ex_emb.pow(2).mean() + diff_w.pow(2).mean() + 
                           disc_w.pow(2).mean() + skill_emb.pow(2).mean())
            reg_loss = reg_loss + self.exercise_l2_lambda * exercise_l2

        # Proto Reg
        proto_div_loss = torch.tensor(0.0, device=device)
        proto_usage_loss = torch.tensor(0.0, device=device)

        if self.use_soft_prototype and prototype_assign is not None:
            K = prototype_assign.size(1)
            if lambda_proto_div > 0.0 and self.prototype_module is not None:
                P = self.prototype_module.prototypes
                P_norm = F.normalize(P, dim=-1)
                sim = torch.matmul(P_norm, P_norm.t())
                eye = torch.eye(K, device=sim.device, dtype=sim.dtype)
                off_diag = sim - eye
                proto_div_loss = (off_diag ** 2).sum() / (K * (K - 1) + 1e-12)
            if lambda_proto_usage > 0.0:
                q_mean = prototype_assign.mean(dim=0)
                uniform = torch.full_like(q_mean, 1.0 / K)
                proto_usage_loss = F.mse_loss(q_mean, uniform)
            reg_loss = reg_loss + lambda_proto_div * proto_div_loss + lambda_proto_usage * proto_usage_loss

        # Personal Graph Reg
        if personal_matrices is not None and lambda_sparse_personal > 0:
            sparse_personal = personal_matrices.abs().mean()
            reg_loss = reg_loss + lambda_sparse_personal * sparse_personal

        if alpha is not None and lambda_alpha > 0:
            reg_loss = reg_loss + lambda_alpha * alpha.mean()

        return reg_loss

    def get_student_diagnosis(self, student_id: int) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            student_ids = torch.tensor([student_id], device=device)
            relation_matrices, _ = self.relation_learning()
            knowledge_state = self.knowledge_encoder(student_ids, relation_matrices)

            if self.use_soft_prototype:
                student_repr = knowledge_state.mean(dim=1)
                proto_mix, _ = self.prototype_module(student_repr)
                proto_broadcast = proto_mix.unsqueeze(1).expand(-1, self.num_concepts, -1)
                knowledge_state = (1.0 - self.proto_lambda) * knowledge_state + self.proto_lambda * proto_broadcast

            knowledge_state = knowledge_state.squeeze(0)
            
            # Neural Branch 的 Latent Vector
            skill_vector = self.skill_encoder(student_ids).squeeze(0)

            # 诊断主要看 Cognitive Branch
            w_pos = F.softplus(self.prediction_head.knowledge_weight_raw)
            scores = torch.matmul(knowledge_state, w_pos.view(-1, 1)).squeeze(-1) + self.prediction_head.knowledge_bias
            knowledge_mastery = torch.sigmoid(scores)

            diagnosis = {
                "knowledge_mastery": knowledge_mastery,
                "skill_level": skill_vector, # 现在这是 latent vector
                "relation_matrices": relation_matrices,
            }
        return diagnosis