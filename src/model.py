"""
认知诊断模型（单文件实现）：
- 多头概念关系学习（生成概念图）
- 学生知识状态编码（GNN 传播）
- 学生应试技巧编码（猜测/失误）
- 习题难度/区分度编码
- IRT 风格作答预测头
- 可选 soft prototype 对学生表示做残差校正
主类：CognitiveDiagnosisModel，整合上述组件完成训练与推理。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Union
import math

# ======================================================
# 1. 图与关系模块
#    - MultiHeadRelationLearning
#    - ConceptGraphConv
# ======================================================


class MultiHeadRelationLearning(nn.Module):
    """多头概念关系学习，输出稀疏邻接矩阵。输入：无显式输入；输出：relation_matrices (H, C, C)、concept_embeddings (C, D)。"""

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

        # 知识点的可学习嵌入
        self.concept_embeddings = nn.Parameter(
            torch.randn(num_concepts, concept_dim)
        )

        # 多头注意力
        self.attention_heads = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=concept_dim,
                num_heads=1,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_heads)
        ])

        # 用于生成稀疏邻接矩阵的可学习温度参数
        self.temperature = nn.Parameter(torch.ones(num_heads))

        # Dropout
        self.dropout = nn.Dropout(dropout)

        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        nn.init.xavier_normal_(self.concept_embeddings)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向计算多头注意力，得到关系矩阵 (H, C, C) 与概念嵌入 (C, D)。"""
        # 扩展维度用于批处理
        concepts = self.concept_embeddings.unsqueeze(0)  # (1, num_concepts, concept_dim)

        relation_matrices = []

        for i, attn_head in enumerate(self.attention_heads):
            # 自注意力计算
            _, attn_weights = attn_head(
                concepts, concepts, concepts,
                need_weights=True,
                average_attn_weights=True
            )
            # attn_weights: (1, num_concepts, num_concepts)

            attn_weights = attn_weights.squeeze(0)  # (num_concepts, num_concepts)

            # 温度缩放（简化版稀疏化，可配合 L1）
            attn_weights = attn_weights / self.temperature[i]

            relation_matrices.append(attn_weights)

        # 堆叠所有头的关系矩阵
        relation_matrices = torch.stack(relation_matrices, dim=0)
        # (num_heads, num_concepts, num_concepts)

        return relation_matrices, self.concept_embeddings

    def get_sparsity_loss(self, relation_matrices: torch.Tensor) -> torch.Tensor:
        """对关系矩阵做 L1 稀疏正则，relation_matrices 形状 (H, C, C)。"""
        return torch.mean(torch.abs(relation_matrices))


class ConceptGraphConv(nn.Module):
    """基于学习到的关系矩阵的图卷积。输入：x (B, C, Din)，relation_matrices (H, C, C)；输出：节点特征 (B, C, Dout)。"""

    def __init__(
            self,
            in_features: int,
            out_features: int,
            num_heads: int = 4,
            dropout: float = 0.1
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads

        # 每个关系头的变换矩阵
        self.head_transforms = nn.ModuleList([
            nn.Linear(in_features, out_features, bias=False)
            for _ in range(num_heads)
        ])

        # 聚合多头信息的注意力权重
        self.head_attention = nn.Parameter(torch.ones(num_heads) / num_heads)

        self.bias = nn.Parameter(torch.zeros(out_features))
        self.dropout = nn.Dropout(dropout)

        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        for transform in self.head_transforms:
            nn.init.xavier_normal_(transform.weight)

    def forward(
            self,
            x: torch.Tensor,
            relation_matrices: torch.Tensor
    ) -> torch.Tensor:
        """执行图卷积，输出形状 (B, C, Dout)。"""
        outputs = []

        for i in range(self.num_heads):
            # 获取该头的关系矩阵
            adj = relation_matrices[i]  # (num_concepts, num_concepts)

            # 简化版行归一化
            degree = adj.sum(dim=1, keepdim=True).clamp(min=1e-12)
            adj_norm = adj / degree  # (num_concepts, num_concepts)

            # 特征变换
            h = self.head_transforms[i](x)  # (batch_size, num_concepts, out_features)

            # 图传播
            h = torch.matmul(adj_norm, h)  # (batch_size, num_concepts, out_features)

            outputs.append(h)

        # 加权聚合多头输出
        output = torch.stack(outputs, dim=0)  # (num_heads, batch_size, num_concepts, out_features)

        # 使用softmax归一化的注意力权重
        attn_weights = F.softmax(self.head_attention, dim=0).view(-1, 1, 1, 1)
        output = (output * attn_weights).sum(dim=0)  # (batch_size, num_concepts, out_features)

        # 添加偏置
        output = output + self.bias

        # Dropout
        output = self.dropout(output)

        return output


# ======================================================
# 2. 编码器
#    - StudentKnowledgeEncoder
#    - TestTakingSkillEncoder
#    - ExerciseDifficultyEncoder
# ======================================================


class StudentKnowledgeEncoder(nn.Module):
    """学生知识状态编码器。输入 student_ids (B,) 与 relation_matrices (H, C, C)，输出 knowledge_state (B, C, D)。"""

    def __init__(
            self,
            num_students: int,
            num_concepts: int,
            knowledge_dim: int,
            num_gnn_layers: int = 2,
            num_relation_heads: int = 4,
            dropout: float = 0.1
    ):
        super().__init__()
        self.num_students = num_students
        self.num_concepts = num_concepts
        self.knowledge_dim = knowledge_dim

        # 学生级别的知识向量
        self.student_emb = nn.Embedding(num_students, knowledge_dim)

        # 概念级别的知识偏置向量（所有学生共享的概念特征）
        self.concept_emb = nn.Embedding(num_concepts, knowledge_dim)

        # GNN层：在学生 × 概念的初始状态上做图传播
        self.gnn_layers = nn.ModuleList([
            ConceptGraphConv(
                knowledge_dim,
                knowledge_dim,
                num_heads=num_relation_heads,
                dropout=dropout
            ) for _ in range(num_gnn_layers)
        ])

        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(knowledge_dim)
            for _ in range(num_gnn_layers)
        ])

        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        nn.init.xavier_normal_(self.student_emb.weight)
        nn.init.xavier_normal_(self.concept_emb.weight)

    def forward(
            self,
            student_ids: torch.Tensor,
            relation_matrices: torch.Tensor
    ) -> torch.Tensor:
        """生成知识状态张量 (B, C, D)。"""
        batch_size = student_ids.size(0)

        # 学生向量: (batch_size, knowledge_dim)
        student_vec = self.student_emb(student_ids)  # 每个学生一个向量

        # 概念向量: (1, num_concepts, knowledge_dim) -> (batch_size, num_concepts, knowledge_dim)
        concept_vec = self.concept_emb.weight.unsqueeze(0).expand(batch_size, -1, -1)

        # 扩展学生向量到每个概念: (batch_size, 1, knowledge_dim) -> (batch_size, num_concepts, knowledge_dim)
        student_vec_expanded = student_vec.unsqueeze(1).expand(-1, self.num_concepts, -1)

        # 学生 × 概念 的初始知识状态
        h = student_vec_expanded + concept_vec  # (batch_size, num_concepts, knowledge_dim)

        # 通过GNN层传播
        for gnn, norm in zip(self.gnn_layers, self.layer_norms):
            h_new = gnn(h, relation_matrices)
            h = norm(h + h_new)
            h = F.relu(h)

        return h


class TestTakingSkillEncoder(nn.Module):
    """学生应试技巧编码，输入 student_ids (B,)，输出 skill_vector (B, skill_dim)。"""

    def __init__(
            self,
            num_students: int,
            skill_dim: int = 2
    ):
        super().__init__()
        self.num_students = num_students
        self.skill_dim = skill_dim

        # 学生的应试技巧嵌入（猜测能力、失误倾向等）
        self.skill_emb = nn.Embedding(num_students, skill_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        # 初始化为接近0
        nn.init.normal_(self.skill_emb.weight, mean=0, std=0.01)

    def forward(self, student_ids: torch.Tensor) -> torch.Tensor:
        """返回技巧向量 (B, skill_dim)。"""
        return self.skill_emb(student_ids)


class ExerciseDifficultyEncoder(nn.Module):
    """习题难度/区分度编码，通过概念图传播。输入 exercise_ids (B,) 与 relation_matrices (H, C, C)，输出 exercise_emb (B, E)、difficulty (B, C)、discrimination (B, C)。"""

    def __init__(
            self,
            num_exercises: int,
            num_concepts: int,
            q_matrix: torch.Tensor,
            exercise_dim: int = 64,
            knowledge_dim: int = 32,
            num_heads: int = 4,
            num_gnn_layers: int = 2,
            dropout: float = 0.1,
            use_graph: bool = True,
    ):
        super().__init__()
        self.num_exercises = num_exercises
        self.num_concepts = num_concepts
        self.exercise_dim = exercise_dim
        self.knowledge_dim = knowledge_dim
        self.use_graph = use_graph

        # 注册Q矩阵（不参与训练）
        self.register_buffer('q_matrix', q_matrix)

        # 习题的可学习嵌入
        self.exercise_emb = nn.Embedding(num_exercises, exercise_dim)

        # 难度和区分度的可学习嵌入
        self.difficulty = nn.Embedding(num_exercises, num_concepts)
        self.discrimination = nn.Embedding(num_exercises, num_concepts)

        # 使用图卷积传播难度和区分度
        self.gnn_layers = nn.ModuleList([
            ConceptGraphConv(knowledge_dim, knowledge_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_gnn_layers)
        ])

        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        nn.init.xavier_normal_(self.exercise_emb.weight)

        # 难度初始化
        nn.init.zeros_(self.difficulty.weight)

        # 区分度初始化
        nn.init.ones_(self.discrimination.weight)

    def forward(self, exercise_ids: torch.Tensor, relation_matrices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """输出 exercise_emb (B, E)、difficulty (B, C)、discrimination (B, C)。"""
        # 获取习题的原始嵌入
        exercise_emb = self.exercise_emb(exercise_ids)

        # 获取习题的难度和区分度
        difficulty = self.difficulty(exercise_ids)  # (batch_size, num_concepts)
        discrimination = torch.sigmoid(self.discrimination(exercise_ids))  # (batch_size, num_concepts)

        if self.use_graph:
            # 转换为适合图卷积的形状
            difficulty_g = difficulty.unsqueeze(-1).expand(-1, -1, self.knowledge_dim)  # (B, C, D)
            discrimination_g = discrimination.unsqueeze(-1).expand(-1, -1, self.knowledge_dim)  # (B, C, D)

            h_difficulty = difficulty_g
            h_discrimination = discrimination_g

            for gnn in self.gnn_layers:
                h_difficulty = gnn(h_difficulty, relation_matrices)
                h_discrimination = gnn(h_discrimination, relation_matrices)

            # 平均池化回到 (batch_size, num_concepts)
            difficulty = h_difficulty.mean(dim=-1)
            discrimination = h_discrimination.mean(dim=-1)
        else:
            # 消融：跳过图传播，仅使用习题-概念的可学习标量
            difficulty = difficulty
            discrimination = discrimination

        return exercise_emb, difficulty, discrimination


# ======================================================
# 3. 预测与原型模块
#    - ResponsePredictionHead
#    - SoftPrototypeModule
#    - （后续 hook）个性化关系图相关模块
# ======================================================


class ResponsePredictionHead(nn.Module):
    """IRT 风格预测头：融合知识状态 (B, C, D)、技巧向量 (B, S) 与习题嵌入 (B, E) 计算作答概率。"""

    def __init__(
            self,
            knowledge_dim: int,
            skill_dim: int,
            exercise_dim: int,
            hidden_dim: int = 128
    ):
        super().__init__()

        self.knowledge_net = nn.Sequential(
            nn.Linear(knowledge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

        self.skill_net = nn.Sequential(
            nn.Linear(skill_dim + exercise_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in [self.knowledge_net, self.skill_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(
            self,
            knowledge_state: torch.Tensor,
            skill_vector: torch.Tensor,
            exercise_emb: torch.Tensor,
            difficulty: torch.Tensor,
            discrimination: torch.Tensor,
            concept_mask: torch.Tensor,
    ) -> torch.Tensor:
        """返回作答概率 (B,)。concept_mask 为 Q 行或样本级概念掩码或两者融合。"""
        batch_size, num_concepts, knowledge_dim = knowledge_state.size()

        # ✅ 向量化：一次性对所有 (学生, 知识点) 计算掌握得分
        # 原来是对每个概念单独跑 MLP，现在把 (B, C, D) 展平成 (B*C, D) 一次过
        h_flat = knowledge_state.view(batch_size * num_concepts, knowledge_dim)  # (B*C, D)
        scores_flat = self.knowledge_net(h_flat).view(batch_size, num_concepts) # (B, C)
        knowledge_scores = scores_flat

        # IRT logits
        irt_logits = discrimination * (knowledge_scores - difficulty)

        # 归一化 concept_mask，防止全 0
        mask_sum = concept_mask.sum(dim=1, keepdim=True) + 1e-12
        concept_norm = concept_mask / mask_sum

        knowledge_prob = torch.sigmoid((irt_logits * concept_norm).sum(dim=1))

        # 2. 技巧影响
        skill_input = torch.cat([skill_vector, exercise_emb], dim=1)
        skill_adjustment = self.skill_net(skill_input).squeeze(-1)
        skill_adjustment = torch.tanh(skill_adjustment) * 0.2

        final_prob = torch.clamp(
            knowledge_prob + skill_adjustment,
            min=1e-6,
            max=1 - 1e-6
        )

        return final_prob


class SoftPrototypeModule(nn.Module):
    """软原型模块：维护 K 个原型，对学生表示做 soft assignment 并生成混合原型。"""

    def __init__(
            self,
            num_prototypes: int,
            dim: int,
            tau: float = 1.0
    ):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.dim = dim
        self.tau = tau

        self.prototypes = nn.Parameter(torch.randn(num_prototypes, dim) * 0.1)
        nn.init.xavier_normal_(self.prototypes)

    def forward(self, student_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """输入 student_repr (B, D)，输出 proto_mix (B, D) 与 assign_q (B, K)。"""
        # 归一化后做相似度更稳定
        s = F.normalize(student_repr, dim=-1)  # (B, D)
        p = F.normalize(self.prototypes, dim=-1)  # (K, D)

        logits = torch.matmul(s, p.t()) / self.tau  # (B, K)
        assign_q = F.softmax(logits, dim=-1)  # (B, K)

        proto_mix = torch.matmul(assign_q, self.prototypes)  # (B, D)

        return proto_mix, assign_q


# ======================================================
# 3.5 个性化关系图模块（G-PDS hook）
#    - AdaptiveGate：自适应门控，控制全局图与个性化图的权重
#    - PersonalRelationGenerator：LoRA 风格生成个性化关系矩阵
# ======================================================


class AdaptiveGate(nn.Module):
    """根据学生表征输出门控系数 α（0-1），偏向全局或个性化图。"""

    def __init__(self, student_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(student_dim, student_dim // 2),
            nn.ReLU(),
            nn.Linear(student_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, student_repr: torch.Tensor) -> torch.Tensor:
        alpha = self.gate(student_repr).view(-1, 1, 1)  # (B,1,1)
        return alpha


class PersonalRelationGenerator(nn.Module):
    """LoRA 风格生成个性化关系矩阵，低秩近似学生特定的概念关联。"""

    def __init__(self, student_dim: int, num_concepts: int, rank: int = 4):
        super().__init__()
        self.num_concepts = num_concepts
        self.rank = rank
        self.to_u = nn.Linear(student_dim, num_concepts * rank, bias=False)
        self.to_v = nn.Linear(student_dim, num_concepts * rank, bias=False)

        nn.init.xavier_normal_(self.to_u.weight)
        nn.init.xavier_normal_(self.to_v.weight)

    def forward(self, student_repr: torch.Tensor) -> torch.Tensor:
        """
        输入: student_repr (B, D)
        输出: personal_matrices (B, C, C)，通过 U V^T 构造的低秩个性化关系
        """
        batch_size = student_repr.size(0)
        u = self.to_u(student_repr).view(batch_size, self.num_concepts, self.rank)
        v = self.to_v(student_repr).view(batch_size, self.num_concepts, self.rank)
        personal = torch.matmul(u, v.transpose(-1, -2))  # (B, C, C)
        return personal


# ======================================================
# 4. CognitiveDiagnosisModel (主组装)
# ======================================================


class CognitiveDiagnosisModel(nn.Module):
    """主模型：组装关系学习、各类编码器、预测头与可选软原型。"""

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
            # ====== soft prototype 相关参数（新增） ======
            num_prototypes: int = 3,
            proto_tau: float = 1.0,
            proto_lambda: float = 0.5,
            use_soft_prototype: bool = True,
            use_skill_encoder: bool = True,
            use_exercise_graph: bool = True,
            # ====== 个性化关系图（G-PDS hook） ======
            use_personal_graph: bool = False,
            personal_rank: int = 4,
            lambda_sparse_personal: float = 0.0,
            lambda_alpha: float = 0.0,
    ):
        super().__init__()

        self.num_students = num_students
        self.num_exercises = num_exercises
        self.num_concepts = num_concepts
        self.knowledge_dim = knowledge_dim
        self.use_skill_encoder = bool(use_skill_encoder)
        self.use_exercise_graph = bool(use_exercise_graph)
        self.use_personal_graph = bool(use_personal_graph)
        self.personal_rank = int(personal_rank)
        self.lambda_sparse_personal = float(lambda_sparse_personal)
        self.lambda_alpha = float(lambda_alpha)

        # ===== 概念关系学习 =====
        self.relation_learning = MultiHeadRelationLearning(
            num_concepts=num_concepts,
            concept_dim=knowledge_dim,
            num_heads=num_relation_heads,
            dropout=dropout
        )

        # ===== 学生知识状态编码器 =====
        self.knowledge_encoder = StudentKnowledgeEncoder(
            num_students=num_students,
            num_concepts=num_concepts,
            knowledge_dim=knowledge_dim,
            num_gnn_layers=num_gnn_layers,
            num_relation_heads=num_relation_heads,
            dropout=dropout
        )

        # ===== 学生应试技巧编码器 =====
        self.skill_encoder = TestTakingSkillEncoder(
            num_students=num_students,
            skill_dim=skill_dim
        )

        # ===== 习题编码器 =====
        self.exercise_encoder = ExerciseDifficultyEncoder(
            num_exercises=num_exercises,
            num_concepts=num_concepts,
            q_matrix=q_matrix,
            exercise_dim=exercise_dim,
            knowledge_dim=knowledge_dim,
            num_heads=num_relation_heads,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout,
            use_graph=self.use_exercise_graph
        )

        # ===== 诊断预测头 =====
        self.prediction_head = ResponsePredictionHead(
            knowledge_dim=knowledge_dim,
            skill_dim=skill_dim,
            exercise_dim=exercise_dim
        )

        # 注册Q矩阵
        self.register_buffer('q_matrix', q_matrix)

        # ===== soft prototype 层（新增） =====
        self.use_soft_prototype = bool(use_soft_prototype and num_prototypes > 0)
        self.proto_lambda = float(proto_lambda)

        if self.use_soft_prototype:
            self.prototype_module = SoftPrototypeModule(
                num_prototypes=num_prototypes,
                dim=knowledge_dim,
                tau=proto_tau
            )
        else:
            self.prototype_module = None

        # ===== 个性化关系图模块（默认关闭） =====
        if self.use_personal_graph:
            self.adaptive_gate = AdaptiveGate(student_dim=knowledge_dim)
            self.personal_generator = PersonalRelationGenerator(
                student_dim=knowledge_dim,
                num_concepts=num_concepts,
                rank=self.personal_rank,
            )
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
        """
        前向计算作答概率。
        - student_ids: (B,) 学生索引
        - exercise_ids: (B,) 习题索引
        - concept_vector: 为兼容保留（当前版本直接使用结构性 Q，不再融合）
        - return_details: 若为 True，返回中间张量诊断信息
        返回：pred_prob (B,)；如 return_details=True，同时返回 details 字典，包含关系矩阵、知识状态、技巧向量、习题难度/区分度、融合概念向量与原型分配等。

        流程：
        1) 关系学习得到 relation_matrices；
        2) 学生知识编码 (GNN) 得到 knowledge_state；
        3) soft prototype（如启用）对 knowledge_state 做残差校正；
        4) 学生技巧与习题难度/区分度编码；
        5) 直接使用 Q 矩阵；通过预测头计算 IRT 风格概率（concept_vector 仅保留接口兼容，当前不参与融合）。
        """
        # 1. 学习概念关系图
        relation_matrices, concept_emb = self.relation_learning()

        # 2. 编码学生知识状态（通过GNN传播）
        knowledge_state = self.knowledge_encoder(student_ids, relation_matrices)
        # knowledge_state: (B, num_concepts, knowledge_dim)

        # === soft prototype 混合：基于学生级表示进行原型校正 ===
        proto_mix = None
        proto_assign = None
        student_repr = knowledge_state.mean(dim=1)  # (B, D)，作为学生级表示

        if self.use_soft_prototype:
            proto_mix, proto_assign = self.prototype_module(student_repr)  # (B, D), (B, K)

            # 将混合原型广播到所有概念上，做残差矫正
            proto_broadcast = proto_mix.unsqueeze(1).expand(-1, self.num_concepts, -1)
            knowledge_state = (1.0 - self.proto_lambda) * knowledge_state + self.proto_lambda * proto_broadcast

        # === G-PDS hook：个性化关系图（暂不替换主路径，仅返回诊断信息） ===
        gate_alpha = None
        personal_matrices = None
        if self.use_personal_graph:
            gate_alpha = self.adaptive_gate(student_repr)  # (B,1,1)
            personal_matrices = self.personal_generator(student_repr)  # (B, C, C)

        # 3. 编码学生应试技巧
        if self.use_skill_encoder:
            skill_vector = self.skill_encoder(student_ids)  # (batch_size, skill_dim)
        else:
            # 消融：技巧向量置零
            skill_vector = torch.zeros_like(self.skill_encoder(student_ids))

        # 4. 编码习题特征
        exercise_emb, difficulty, discrimination = self.exercise_encoder(exercise_ids, relation_matrices)

        # 5. 获取Q矩阵向量（当前版本直接使用结构性 Q，不再融合 concept_vector）
        q_vector = self.q_matrix[exercise_ids]  # (batch_size, num_concepts)
        effective_concept = q_vector

        # 7. 预测
        pred_prob = self.prediction_head(
            knowledge_state,
            skill_vector,
            exercise_emb,
            difficulty,
            discrimination,
            effective_concept
        )

        if return_details:
            details = {
                'relation_matrices': relation_matrices,
                'knowledge_state': knowledge_state,
                'skill_vector': skill_vector,
                'difficulty': difficulty,
                'discrimination': discrimination,
                'q_vector': q_vector,
                'effective_concept': effective_concept,
                'student_repr': student_repr,
            }
            if self.use_soft_prototype:
                details['prototype_assign'] = proto_assign  # (B, K)
                details['prototype_mix'] = proto_mix        # (B, D)
            if self.use_personal_graph:
                details['alpha'] = gate_alpha
                details['personal_matrices'] = personal_matrices
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
        """
        计算正则项：
        - 稀疏 (lambda_sparse)：关系矩阵 L1
        - 原型多样性 (lambda_proto_div)：原型间相似度去相关
        - 原型使用均衡 (lambda_proto_usage)：平均分配接近均匀
        - 个性化稀疏 (lambda_sparse_personal)：个性化图的稀疏惩罚
        - 门控约束 (lambda_alpha)：鼓励 α 更小偏向全局图
        输入形状：relation_matrices (H, C, C)、skill_vector (B, S)、knowledge_state (B, C, D)、prototype_assign (B, K, 可选)。
        """
        if lambda_sparse_personal is None:
            lambda_sparse_personal = self.lambda_sparse_personal
        if lambda_alpha is None:
            lambda_alpha = self.lambda_alpha

        # 1) 稀疏性：关系矩阵 L1 约束
        sparse_loss = self.relation_learning.get_sparsity_loss(relation_matrices)

        reg_loss = lambda_sparse * sparse_loss

        # 3) 原型相关正则（可选）
        proto_div_loss = torch.tensor(0.0, device=knowledge_state.device)
        proto_usage_loss = torch.tensor(0.0, device=knowledge_state.device)

        if self.use_soft_prototype and prototype_assign is not None:
            K = prototype_assign.size(1)

            if lambda_proto_div > 0.0 and hasattr(self, "prototype_module") and self.prototype_module is not None:
                P = self.prototype_module.prototypes  # (K, D)
                P_norm = F.normalize(P, dim=-1)
                sim = torch.matmul(P_norm, P_norm.t())  # (K, K)

                eye = torch.eye(K, device=sim.device, dtype=sim.dtype)
                off_diag = (sim - eye)
                proto_div_loss = (off_diag ** 2).sum() / (K * (K - 1) + 1e-12)  # 原型多样性

            if lambda_proto_usage > 0.0:
                # 平衡使用：平均分配接近均匀，避免坍缩到单个原型
                q_mean = prototype_assign.mean(dim=0)  # (K,)
                uniform = torch.full_like(q_mean, 1.0 / K)
                proto_usage_loss = F.mse_loss(q_mean, uniform)

            reg_loss = reg_loss + lambda_proto_div * proto_div_loss + lambda_proto_usage * proto_usage_loss

        # 4) 个性化关系图稀疏正则（仅当提供 personal_matrices 且权重大于 0 时生效）
        if personal_matrices is not None and lambda_sparse_personal > 0:
            sparse_personal = personal_matrices.abs().mean()
            reg_loss = reg_loss + lambda_sparse_personal * sparse_personal

        # 5) 门控 α 惩罚，鼓励更多依赖全局图
        if alpha is not None and lambda_alpha > 0:
            alpha_mean = alpha.mean()
            reg_loss = reg_loss + lambda_alpha * alpha_mean

        return reg_loss

    def get_student_diagnosis(
            self,
            student_id: int
    ) -> Dict[str, torch.Tensor]:
        """
        生成单个学生的诊断结果（内部索引）。
        返回包含知识掌握度 (C,)、技巧向量 (S,) 以及关系矩阵的字典。
        """
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            student_ids = torch.tensor([student_id], device=device)

            # 学习关系图
            relation_matrices, _ = self.relation_learning()

            # 获取知识状态
            knowledge_state = self.knowledge_encoder(student_ids, relation_matrices)
            # soft prototype 校正（与 forward 保持一致）
            if self.use_soft_prototype:
                student_repr = knowledge_state.mean(dim=1)  # (1, D)
                proto_mix, _ = self.prototype_module(student_repr)
                proto_broadcast = proto_mix.unsqueeze(1).expand(-1, self.num_concepts, -1)
                knowledge_state = (1.0 - self.proto_lambda) * knowledge_state + self.proto_lambda * proto_broadcast

            knowledge_state = knowledge_state.squeeze(0)  # (num_concepts, knowledge_dim)

            # 获取技巧向量
            if self.use_skill_encoder:
                skill_vector = self.skill_encoder(student_ids).squeeze(0)  # (skill_dim,)
            else:
                skill_vector = torch.zeros_like(self.skill_encoder(student_ids).squeeze(0))

            # 计算每个知识点的掌握程度
            knowledge_scores = []
            for i in range(self.num_concepts):
                k = knowledge_state[i:i + 1, :]
                score = self.prediction_head.knowledge_net(k).squeeze()
                knowledge_scores.append(torch.sigmoid(score).item())

            diagnosis = {
                'knowledge_mastery': torch.tensor(knowledge_scores, device=device),  # (num_concepts,)
                'skill_level': skill_vector,  # (skill_dim,)
                'relation_matrices': relation_matrices  # (num_heads, num_concepts, num_concepts)
            }

        return diagnosis
