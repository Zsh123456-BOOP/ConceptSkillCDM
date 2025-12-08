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
    """学生知识状态编码器。输入 student_ids (B,) 与 relation_matrices (H, C, C)，输出 knowledge_state (B, C, D)。

    """

    def __init__(
            self,
            num_students: int,
            num_concepts: int,
            knowledge_dim: int,
            num_gnn_layers: int = 2,
            num_relation_heads: int = 4,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.num_students = num_students
        self.num_concepts = num_concepts
        self.knowledge_dim = knowledge_dim

        # ===== 基础学生/概念嵌入 =====
        self.student_emb = nn.Embedding(num_students, knowledge_dim)
        self.concept_emb = nn.Embedding(num_concepts, knowledge_dim)

        # ===== GNN 层：在学生×概念初始状态上做图传播 =====
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

        # 学生向量: (B, D)
        student_vec = self.student_emb(student_ids)

        # 概念向量: (1, C, D) -> (B, C, D)
        concept_vec = self.concept_emb.weight.unsqueeze(0).expand(batch_size, -1, -1)

        # 扩展学生向量到每个概念: (B, 1, D) -> (B, C, D)
        student_vec_expanded = student_vec.unsqueeze(1).expand(-1, self.num_concepts, -1)

        # ===== 学生×概念 初始知识状态 =====
        h = student_vec_expanded + concept_vec  # (B, C, D)

        # ===== 通过 GNN 层传播 =====
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
    """习题难度/区分度编码（纯 embedding 形式）。输入 exercise_ids (B,)，输出 exercise_emb (B, E)、difficulty (B, C)、discrimination (B, C)。"""

    def __init__(
            self,
            num_exercises: int,
            num_concepts: int,
            q_matrix: torch.Tensor,
            exercise_dim: int = 64,
    ):
        super().__init__()
        self.num_exercises = num_exercises
        self.num_concepts = num_concepts
        self.exercise_dim = exercise_dim

        # 注册Q矩阵（不参与训练）
        self.register_buffer('q_matrix', q_matrix)

        # 习题的可学习嵌入
        self.exercise_emb = nn.Embedding(num_exercises, exercise_dim)

        # 难度和区分度的可学习嵌入
        self.difficulty = nn.Embedding(num_exercises, num_concepts)
        self.discrimination = nn.Embedding(num_exercises, num_concepts)

        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        nn.init.xavier_normal_(self.exercise_emb.weight)

        # 难度初始化
        nn.init.zeros_(self.difficulty.weight)

        # 区分度初始化
        nn.init.ones_(self.discrimination.weight)

    def forward(self, exercise_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """输出 exercise_emb (B, E)、difficulty (B, C)、discrimination (B, C)。"""
        # 获取习题的原始嵌入
        exercise_emb = self.exercise_emb(exercise_ids)

        # 获取习题的难度和区分度
        difficulty = self.difficulty(exercise_ids)  # (batch_size, num_concepts)
        discrimination = torch.sigmoid(self.discrimination(exercise_ids))  # (batch_size, num_concepts)

        return exercise_emb, difficulty, discrimination


# ======================================================
# 3. 预测与原型模块
#    - ResponsePredictionHead
#    - SoftPrototypeModule
#    - （后续 hook）个性化关系图相关模块
# ======================================================

class ResponsePredictionHead(nn.Module):
    """IRT 风格预测头：融合知识状态 (B, C, D)、技巧向量 (B, S) 与习题嵌入 (B, E) 计算作答概率。
    知识部分使用 2 层单调 MLP（权重非负 + 单调激活），保证对 knowledge_state 单调。
    """

    def __init__(
            self,
            knowledge_dim: int,
            skill_dim: int,
            exercise_dim: int,
            hidden_dim: int = 128,
            mono_hidden_dim: int = 32,   # 单调 MLP 隐层维
    ):
        super().__init__()

        self.knowledge_dim = knowledge_dim
        self.mono_hidden_dim = mono_hidden_dim

        # ===== 知识部分：2 层单调 MLP =====
        # raw 权重，通过 softplus 变成非负，保证单调性
        self.kn_w1_raw = nn.Parameter(torch.randn(mono_hidden_dim, knowledge_dim))
        self.kn_b1 = nn.Parameter(torch.zeros(mono_hidden_dim))

        self.kn_w2_raw = nn.Parameter(torch.randn(1, mono_hidden_dim))
        self.kn_b2 = nn.Parameter(torch.zeros(1))

        # ===== 技巧部分：保留你原来的结构 =====
        self.skill_net = nn.Sequential(
            nn.Linear(skill_dim + exercise_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # 知识 MLP 的 raw 权重
        nn.init.normal_(self.kn_w1_raw, mean=0.0, std=0.1)
        nn.init.normal_(self.kn_w2_raw, mean=0.0, std=0.1)
        nn.init.zeros_(self.kn_b1)
        nn.init.zeros_(self.kn_b2)

        # 技巧网络
        for layer in self.skill_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

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
            knowledge_state: torch.Tensor,   # (B, C, D)
            skill_vector: torch.Tensor,      # (B, S)
            exercise_emb: torch.Tensor,      # (B, E)
            difficulty: torch.Tensor,        # (B, C)
            discrimination: torch.Tensor,    # (B, C)
            concept_mask: torch.Tensor,      # (B, C)
    ) -> torch.Tensor:
        """返回作答概率 (B,)。concept_mask 为 Q 行或样本级概念掩码或两者融合。"""
        batch_size, num_concepts, knowledge_dim = knowledge_state.size()
        assert knowledge_dim == self.knowledge_dim

        # 1) 知识打分：单调 2 层 MLP
        knowledge_scores = self._knowledge_scores(knowledge_state)  # (B, C)

        # 2) IRT logits
        irt_logits = discrimination * (knowledge_scores - difficulty)  # (B, C)

        # 归一化 concept_mask，防止全 0
        mask_sum = concept_mask.sum(dim=1, keepdim=True) + 1e-12
        concept_norm = concept_mask / mask_sum  # (B, C)

        # 聚合到题目层
        knowledge_prob = torch.sigmoid((irt_logits * concept_norm).sum(dim=1))  # (B,)

        # 3) 技巧影响
        skill_input = torch.cat([skill_vector, exercise_emb], dim=1)  # (B, S+E)
        skill_adjustment = self.skill_net(skill_input).squeeze(-1)    # (B,)
        skill_adjustment = torch.tanh(skill_adjustment) * 0.2         # 控制幅度

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
    ):
        super().__init__()

        # ===== 这里三行一定要有 =====
        self.num_students = num_students
        self.num_exercises = num_exercises
        self.num_concepts = num_concepts

        self.knowledge_dim = knowledge_dim
        self.use_skill_encoder = bool(use_skill_encoder)

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
            dropout=dropout,
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
        )

        # ===== 诊断预测头 =====
        self.prediction_head = ResponsePredictionHead(
            knowledge_dim=knowledge_dim,
            skill_dim=skill_dim,
            exercise_dim=exercise_dim,
            mono_hidden_dim=32,   # 可以调，比如 32 / 64
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

    def forward(
            self,
            student_ids: torch.Tensor,
            exercise_ids: torch.Tensor,
            return_details: bool = False
    ) -> Union[torch.Tensor, Tuple]:
        """
        前向计算作答概率。
        - student_ids: (B,) 学生索引
        - exercise_ids: (B,) 习题索引
        - return_details: 若为 True，返回中间张量诊断信息
        返回：pred_prob (B,)；如 return_details=True，同时返回 details 字典，包含关系矩阵、知识状态、技巧向量、习题难度/区分度、融合概念向量与原型分配等。

        流程：
        1) 关系学习得到 relation_matrices；
        2) 学生知识编码 (GNN) 得到 knowledge_state；
        3) soft prototype（如启用）对 knowledge_state 做残差校正；
        4) 学生技巧与习题难度/区分度编码；
        5) 直接使用 Q 矩阵；通过预测头计算 IRT 风格概率。
        """
        # 1. 学习概念关系图
        relation_matrices, _ = self.relation_learning()

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

        # 3. 编码学生应试技巧
        if self.use_skill_encoder:
            skill_vector = self.skill_encoder(student_ids)  # (batch_size, skill_dim)
        else:
            # 消融：技巧向量置零
            skill_vector = torch.zeros_like(self.skill_encoder(student_ids))

        # 4. 编码习题特征
        exercise_emb, difficulty, discrimination = self.exercise_encoder(exercise_ids)

        # 5. 获取Q矩阵向量（当前版本直接使用结构性 Q）
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
    ) -> torch.Tensor:
        """
        计算正则项：
        - 稀疏 (lambda_sparse)：关系矩阵 L1
        - 原型多样性 (lambda_proto_div)：原型间相似度去相关
        - 原型使用均衡 (lambda_proto_usage)：平均分配接近均匀
        输入形状：relation_matrices (H, C, C)、skill_vector (B, S)、knowledge_state (B, C, D)、prototype_assign (B, K, 可选)。
        """
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

            # 计算每个知识点的掌握程度（与预测头保持一致的单调 2 层 MLP）
            # knowledge_state: (C, D)
            scores = self.prediction_head._knowledge_scores(knowledge_state)  # (C,)
            knowledge_mastery = torch.sigmoid(scores)  # (C,)


            diagnosis = {
                'knowledge_mastery': knowledge_mastery,  # (num_concepts,)
                'skill_level': skill_vector,             # (skill_dim,)
                'relation_matrices': relation_matrices   # (num_heads, num_concepts, num_concepts)
            }

        return diagnosis
