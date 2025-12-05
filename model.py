import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Union
import math


class MultiHeadRelationLearning(nn.Module):
    """多头关系学习模块 - 学习知识点之间的多种关系"""

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
        """
        前向传播，学习多种关系的邻接矩阵

        Returns:
            relation_matrices: (num_heads, num_concepts, num_concepts)
            concept_embeddings: (num_concepts, concept_dim)
        """
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

            # 使用Gumbel-Softmax实现稀疏化
            attn_weights = attn_weights.squeeze(0)  # (num_concepts, num_concepts)

            # 温度缩放 + 稀疏化
            attn_weights = attn_weights / self.temperature[i]   # (num_concepts, num_concepts)

            relation_matrices.append(attn_weights)

        # 堆叠所有头的关系矩阵
        relation_matrices = torch.stack(relation_matrices, dim=0)
        # (num_heads, num_concepts, num_concepts)

        return relation_matrices, self.concept_embeddings

    def get_sparsity_loss(self, relation_matrices: torch.Tensor) -> torch.Tensor:
        """
        计算稀疏性损失（L1正则）

        Args:
            relation_matrices: (num_heads, num_concepts, num_concepts)

        Returns:
            稀疏性损失
        """
        return torch.mean(torch.abs(relation_matrices))


class ConceptGraphConv(nn.Module):
    """基于学习到的关系图的图卷积层"""

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
        """
        图卷积前向传播

        Args:
            x: 节点特征 (batch_size, num_concepts, in_features)
            relation_matrices: 关系矩阵 (num_heads, num_concepts, num_concepts)

        Returns:
            更新后的节点特征 (batch_size, num_concepts, out_features)
        """
        outputs = []

        for i in range(self.num_heads):
            # 获取该头的关系矩阵
            adj = relation_matrices[i]  # (num_concepts, num_concepts)

            # 图卷积: D^{-1/2} A D^{-1/2} X W
            # 简化版本: 归一化邻接矩阵
            degree = adj.sum(dim=1, keepdim=True).clamp(min=1e-12)
            adj_norm = adj / degree  # 行归一化

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


class StudentKnowledgeEncoder(nn.Module):
    """学生知识状态编码器 - 使用GNN进行知识传播（轻量版：学生向量 + 概念向量 因子分解）"""

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

        # 学生级别的知识向量（不再为每个概念单独存一串巨大的向量）
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
        """
        前向传播

        Args:
            student_ids: (batch_size,)
            relation_matrices: (num_heads, num_concepts, num_concepts)

        Returns:
            知识状态向量 (batch_size, num_concepts, knowledge_dim)
        """
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
    """应试技巧编码器 - 建模猜测和失误"""

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
        """
        前向传播

        Args:
            student_ids: (batch_size,)

        Returns:
            技巧向量 (batch_size, skill_dim)
        """
        return self.skill_emb(student_ids)


class ExerciseDifficultyEncoder(nn.Module):
    """习题难度和区分度编码器（利用知识图传播）"""

    def __init__(
            self,
            num_exercises: int,
            num_concepts: int,
            q_matrix: torch.Tensor,
            exercise_dim: int = 64,
            knowledge_dim: int = 32,
            num_heads: int = 4,
            num_gnn_layers: int = 2,
            dropout: float = 0.1
    ):
        super().__init__()
        self.num_exercises = num_exercises
        self.num_concepts = num_concepts
        self.exercise_dim = exercise_dim
        self.knowledge_dim = knowledge_dim

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
        """
        前向传播，利用图传播更新难度和区分度

        Args:
            exercise_ids: (batch_size,)
            relation_matrices: (num_heads, num_concepts, num_concepts)

        Returns:
            exercise_emb: (batch_size, exercise_dim)
            difficulty: (batch_size, num_concepts)
            discrimination: (batch_size, num_concepts)
        """
        # 获取习题的原始嵌入
        exercise_emb = self.exercise_emb(exercise_ids)

        # 获取习题的难度和区分度
        difficulty = self.difficulty(exercise_ids)  # (batch_size, num_concepts)
        discrimination = torch.sigmoid(self.discrimination(exercise_ids))  # (batch_size, num_concepts)

        # 转换为适合图卷积的形状
        difficulty = difficulty.unsqueeze(-1).expand(-1, -1, self.knowledge_dim)  # (batch_size, num_concepts, knowledge_dim)
        discrimination = discrimination.unsqueeze(-1).expand(-1, -1, self.knowledge_dim)  # (batch_size, num_concepts, knowledge_dim)

        # 通过图卷积传播难度和区分度
        h_difficulty = difficulty  # (batch_size, num_concepts, knowledge_dim)
        h_discrimination = discrimination  # (batch_size, num_concepts, knowledge_dim)

        # 图卷积传播
        for gnn in self.gnn_layers:
            h_difficulty = gnn(h_difficulty, relation_matrices)
            h_discrimination = gnn(h_discrimination, relation_matrices)

        # 更新难度和区分度 - 使用平均池化将 (batch_size, num_concepts, knowledge_dim) 聚合为 (batch_size, num_concepts)
        difficulty = h_difficulty.mean(dim=-1)  # (batch_size, num_concepts)
        discrimination = h_discrimination.mean(dim=-1)  # (batch_size, num_concepts)

        return exercise_emb, difficulty, discrimination



class ResponsePredictionHead(nn.Module):
    """作答预测头 - 结合知识状态、技巧和习题特征"""

    def __init__(
            self,
            knowledge_dim: int,
            skill_dim: int,
            exercise_dim: int,
            hidden_dim: int = 128
    ):
        super().__init__()

        # 知识匹配网络
        self.knowledge_net = nn.Sequential(
            nn.Linear(knowledge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

        # 技巧影响网络
        self.skill_net = nn.Sequential(
            nn.Linear(skill_dim + exercise_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
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
            q_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            knowledge_state: (batch_size, num_concepts, knowledge_dim)
            skill_vector: (batch_size, skill_dim)
            exercise_emb: (batch_size, exercise_dim)
            difficulty: (batch_size, num_concepts)
            discrimination: (batch_size, num_concepts)
            q_vector: (batch_size, num_concepts) - Q矩阵的行向量

        Returns:
            预测概率 (batch_size,)
        """
        batch_size = knowledge_state.size(0)

        # 1. 计算知识掌握程度（基于IRT）
        # 对每个知识点计算掌握得分
        knowledge_scores = []
        for i in range(knowledge_state.size(1)):  # 遍历每个知识点
            k = knowledge_state[:, i, :]  # (batch_size, knowledge_dim)
            score = self.knowledge_net(k).squeeze(-1)  # (batch_size,)
            knowledge_scores.append(score)

        knowledge_scores = torch.stack(knowledge_scores, dim=1)  # (batch_size, num_concepts)


        irt_logits = discrimination * (knowledge_scores - difficulty)  # (batch_size, num_concepts)

        # 使用Q矩阵加权聚合相关知识点
        # 归一化Q矩阵的行
        q_norm = q_vector / (q_vector.sum(dim=1, keepdim=True) + 1e-12)

        # 加权求和
        knowledge_prob = torch.sigmoid((irt_logits * q_norm).sum(dim=1))  # (batch_size,)

        # 2. 计算技巧影响（猜测和失误）
        skill_input = torch.cat([skill_vector, exercise_emb], dim=1)
        skill_adjustment = self.skill_net(skill_input).squeeze(-1)  # (batch_size,)
        skill_adjustment = torch.tanh(skill_adjustment) * 0.2  # 限制影响范围在[-0.2, 0.2]

        # 3. 最终预测
        final_prob = torch.clamp(knowledge_prob + skill_adjustment, min=1e-6, max=1 - 1e-6)

        return final_prob


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
            dropout: float = 0.1
    ):
        super().__init__()

        self.num_students = num_students
        self.num_exercises = num_exercises
        self.num_concepts = num_concepts
        self.knowledge_dim = knowledge_dim

        self.knowledge_projector = nn.Linear(num_concepts * knowledge_dim, skill_dim)


        self.relation_learning = MultiHeadRelationLearning(
            num_concepts=num_concepts,
            concept_dim=knowledge_dim,
            num_heads=num_relation_heads,
            dropout=dropout
        )



        self.knowledge_encoder = StudentKnowledgeEncoder(
            num_students=num_students,
            num_concepts=num_concepts,
            knowledge_dim=knowledge_dim,
            num_gnn_layers=num_gnn_layers,
            num_relation_heads=num_relation_heads,
            dropout=dropout
        )


        self.skill_encoder = TestTakingSkillEncoder(
            num_students=num_students,
            skill_dim=skill_dim
        )

        # 习题编码器
        # self.exercise_encoder = ExerciseDifficultyEncoder(
        #     num_exercises=num_exercises,
        #     num_concepts=num_concepts,
        #     q_matrix=q_matrix,
        #     exercise_dim=exercise_dim
        # )

        self.exercise_encoder = ExerciseDifficultyEncoder(
            num_exercises=num_exercises,
            num_concepts=num_concepts,
            q_matrix=q_matrix,
            exercise_dim=exercise_dim,
            knowledge_dim=knowledge_dim,
            num_heads=num_relation_heads,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout
        )

        # 诊断预测头
        self.prediction_head = ResponsePredictionHead(
            knowledge_dim=knowledge_dim,
            skill_dim=skill_dim,
            exercise_dim=exercise_dim
        )

        # 注册Q矩阵
        self.register_buffer('q_matrix', q_matrix)

    def forward(
            self,
            student_ids: torch.Tensor,
            exercise_ids: torch.Tensor,
            return_details: bool = False
    ) -> Union[torch.Tensor, Tuple]:
        """
        前向传播

        Args:
            student_ids: (batch_size,)
            exercise_ids: (batch_size,)
            return_details: 是否返回详细信息

        Returns:
            如果return_details=False: 预测概率 (batch_size,)
            如果return_details=True: (预测概率, 详细信息字典)
        """
        # 1. 学习概念关系图
        relation_matrices, concept_emb = self.relation_learning()

        # 2. 编码学生知识状态（通过GNN传播）
        knowledge_state = self.knowledge_encoder(student_ids, relation_matrices)

        # 3. 编码学生应试技巧
        skill_vector = self.skill_encoder(student_ids)  # (batch_size, skill_dim)

        # 4. 编码习题特征
        exercise_emb, difficulty, discrimination = self.exercise_encoder(exercise_ids, relation_matrices)

        # 5. 获取Q矩阵向量
        q_vector = self.q_matrix[exercise_ids]

        # 6. 预测作答概率
        pred_prob = self.prediction_head(
            knowledge_state,
            skill_vector,
            exercise_emb,
            difficulty,
            discrimination,
            q_vector
        )

        if return_details:
            details = {
                'relation_matrices': relation_matrices,
                'knowledge_state': knowledge_state,
                'skill_vector': skill_vector,
                'difficulty': difficulty,
                'discrimination': discrimination
            }
            return pred_prob, details
        else:
            return pred_prob

    def get_regularization_loss(
            self,
            relation_matrices: torch.Tensor,
            skill_vector: torch.Tensor,
            knowledge_state: torch.Tensor,
            lambda_sparse: float = 0.01,
            lambda_independence: float = 0.01
    ) -> torch.Tensor:
        """
        计算正则化损失

        Args:
            relation_matrices: (num_heads, num_concepts, num_concepts)
            skill_vector: (batch_size, skill_dim)
            knowledge_state: (batch_size, num_concepts, knowledge_dim)
            lambda_sparse: 稀疏性正则化系数
            lambda_independence: 独立性正则化系数

        Returns:
            总正则化损失
        """
        # 1. 稀疏性损失（L1正则）
        sparse_loss = self.relation_learning.get_sparsity_loss(relation_matrices)

        # 2. 知识-技巧独立性损失
        batch_size = knowledge_state.size(0)

        # 将知识状态展平
        knowledge_flat = knowledge_state.view(batch_size, -1)  # (batch_size, num_concepts * knowledge_dim)

        # 投影到相同维度
        knowledge_proj = self.knowledge_projector(knowledge_flat)  # (batch_size, skill_dim)

        # 归一化
        knowledge_norm = F.normalize(knowledge_proj, dim=1)
        skill_norm = F.normalize(skill_vector, dim=1)

        # 计算余弦相似度并最小化
        independence_loss = torch.abs((knowledge_norm * skill_norm).sum(dim=1)).mean()

        # 总正则化损失
        reg_loss = lambda_sparse * sparse_loss + lambda_independence * independence_loss

        return reg_loss

    def get_student_diagnosis(
            self,
            student_id: int
    ) -> Dict[str, torch.Tensor]:
        """
        获取单个学生的诊断结果

        Args:
            student_id: 学生ID

        Returns:
            诊断结果字典
        """
        self.eval()
        with torch.no_grad():
            student_ids = torch.tensor([student_id], device=next(self.parameters()).device)

            # 学习关系图
            relation_matrices, _ = self.relation_learning()

            # 获取知识状态
            knowledge_state = self.knowledge_encoder(student_ids, relation_matrices)
            knowledge_state = knowledge_state.squeeze(0)  # (num_concepts, knowledge_dim)

            # 获取技巧向量
            skill_vector = self.skill_encoder(student_ids).squeeze(0)  # (skill_dim,)

            # 计算每个知识点的掌握程度
            knowledge_scores = []
            for i in range(self.num_concepts):
                k = knowledge_state[i:i + 1, :]
                score = self.prediction_head.knowledge_net(k).squeeze()
                knowledge_scores.append(torch.sigmoid(score).item())

            diagnosis = {
                'knowledge_mastery': torch.tensor(knowledge_scores),  # (num_concepts,)
                'skill_level': skill_vector,  # (skill_dim,)
                'relation_matrices': relation_matrices  # (num_heads, num_concepts, num_concepts)
            }

        return diagnosis


# # 使用示例
# if __name__ == "__main__":
#     # 模拟数据
#     num_students = 25
#     num_exercises = 50
#     num_concepts = 32
#     batch_size = 32
#
#     # 模拟Q矩阵
#     q_matrix = torch.zeros(num_exercises, num_concepts)
#     for i in range(num_exercises):
#         # 每个习题关联2-4个知识点
#         num_related = torch.randint(2, 5, (1,)).item()
#         related_concepts = torch.randperm(num_concepts)[:num_related]
#         q_matrix[i, related_concepts] = 1
#
#     # 创建模型
#     model = CognitiveDiagnosisModel(
#         num_students=num_students,
#         num_exercises=num_exercises,
#         num_concepts=num_concepts,
#         q_matrix=q_matrix,
#         knowledge_dim=32,
#         skill_dim=2,
#         exercise_dim=64,
#         num_relation_heads=4,
#         num_gnn_layers=2,
#         dropout=0.1
#     )
#
#     print("模型结构:")
#     print(model)
#     print(f"\n总参数量: {sum(p.numel() for p in model.parameters()):,}")
#
#     # 测试前向传播
#     student_ids = torch.randint(0, num_students, (batch_size,))
#     exercise_ids = torch.randint(0, num_exercises, (batch_size,))
#
#     # 不返回详细信息
#     pred_prob = model(student_ids, exercise_ids, return_details=False)
#     print(f"\n预测概率形状: {pred_prob.shape}")
#     print(f"预测概率示例: {pred_prob[:5]}")
#
#     # 返回详细信息
#     pred_prob, details = model(student_ids, exercise_ids, return_details=True)
#     print(f"\n详细信息键: {details.keys()}")
#     print(f"关系矩阵形状: {details['relation_matrices'].shape}")
#     print(f"知识状态形状: {details['knowledge_state'].shape}")
#
#     # 计算正则化损失
#     reg_loss = model.get_regularization_loss(
#         details['relation_matrices'],
#         details['skill_vector'],
#         details['knowledge_state']
#     )
#     print(f"\n正则化损失: {reg_loss.item():.4f}")
#
#     # 获取单个学生的诊断
#     diagnosis = model.get_student_diagnosis(student_id=0)
#     print(f"\n学生诊断:")
#     print(f"知识掌握度: {diagnosis['knowledge_mastery'][:10]}")
#     print(f"技巧水平: {diagnosis['skill_level']}")
