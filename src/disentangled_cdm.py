import torch
import torch.nn as nn
from typing import Tuple

from .graph_modules import GraphStructureLearner, MonotonicPropagator


class DisentangledCDM(nn.Module):
    """
    主模型架构：
    - 学生嵌入 -> 知识状态 (K 维概率)
    - 学生嵌入 -> 技巧向量 (skill)
    - 概念嵌入 -> 图结构 (前置图 / 相似图)
    - 单调传播：利用前置图对知识状态进行增强
    - DINA-like 预测：结合 guess / slip 与 non-compensatory 题目结构
    """
    def __init__(
        self,
        num_students: int,
        num_exercises: int,
        num_concepts: int,
        dim_emb: int = 64,
        dim_skill: int = 4,
        q_matrix: torch.Tensor = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.dim_emb = dim_emb
        self.num_exercises = num_exercises
        self.q_matrix = q_matrix  # 目前在模型内没用到，由 dataset 提供 q_mask

        # --- Embeddings ---
        self.student_emb = nn.Embedding(num_students, dim_emb)

        # 概念嵌入用于图结构学习
        self.concept_emb = nn.Parameter(torch.randn(num_concepts, dim_emb))
        nn.init.xavier_normal_(self.concept_emb)

        # --- Module 1: Graph Learner ---
        self.graph_learner = GraphStructureLearner(
            num_concepts=num_concepts,
            dim=dim_emb,
            num_heads=2,
            tau=0.05,
        )

        # --- Module 2: Disentangled Representation ---
        # 2.1 知识状态映射
        self.knowledge_head = nn.Sequential(
            nn.Linear(dim_emb, dim_emb),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(dim_emb, num_concepts),
            nn.Sigmoid(),
        )

        # 2.2 技巧向量
        self.skill_head = nn.Sequential(
            nn.Linear(dim_emb, dim_skill),
            nn.Tanh(),
        )

        # 2.3 Guess / Slip
        self.guess_slip_generator = nn.Linear(dim_skill, 2)

        # 2.4 图传播模块
        self.propagator = MonotonicPropagator(impact_factor=0.5)

    def forward(
        self,
        student_ids: torch.Tensor,
        exercise_ids: torch.Tensor,
        q_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input:
            student_ids: (B,)
            exercise_ids: (B,)   # 当前版本暂时未使用，但保留以兼容 Trainer 接口
            q_mask:      (B, K) 当前 batch 题目的 Q 矩阵行
        Output:
            pred_prob:   (B,)
            adj_dag:     (K, K)
            h_knowledge: (B, K)
            z_skill:     (B, D_skill)
        """
        # 1. 学生嵌入
        stu_vec = self.student_emb(student_ids)  # (B, dim)

        # 2. Graph Structure
        graphs = self.graph_learner(self.concept_emb)
        adj_dag = graphs[0]  # 有向前置图 (K, K)

        # 3. 解耦表征
        h_init = self.knowledge_head(stu_vec)           # (B, K)
        h_knowledge = self.propagator(h_init, adj_dag)  # (B, K)

        z_skill = self.skill_head(stu_vec)              # (B, D_skill)

        # Guess / Slip
        gs_params = torch.sigmoid(self.guess_slip_generator(z_skill)) * 0.3  # (B, 2)
        guess_prob = gs_params[:, 0].unsqueeze(1)  # (B, 1)
        slip_prob = gs_params[:, 1].unsqueeze(1)   # (B, 1)

        # 4. Non-compensatory aggregation (Softmin)
        infinity_mask = (1.0 - q_mask) * 1e9
        masked_knowledge = h_knowledge + infinity_mask

        alpha = 10.0
        neg_log_sum_exp = -torch.logsumexp(-alpha * masked_knowledge, dim=1, keepdim=True) / alpha

        knowledge_score = torch.clamp(neg_log_sum_exp, 0.0, 1.0)  # (B, 1)

        # 5. DINA 组合
        pred_prob = (1.0 - slip_prob) * knowledge_score + guess_prob * (1.0 - knowledge_score)

        return pred_prob.squeeze(-1), adj_dag, h_knowledge, z_skill
