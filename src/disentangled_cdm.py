import torch
import torch.nn as nn

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
        q_matrix: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.dim_emb = dim_emb
        self.q_matrix = q_matrix  # (num_exercises, num_concepts) 常量张量，可由 dataset 提供

        # Embeddings
        self.student_emb = nn.Embedding(num_students, dim_emb)
        # 用于生成图的概念语义向量 (K, dim_emb)
        self.concept_emb = nn.Parameter(torch.randn(num_concepts, dim_emb))

        # 模块一：图学习器
        self.graph_learner = GraphStructureLearner(
            num_concepts=num_concepts,
            dim=dim_emb,
            num_heads=2,
            tau=0.1,
        )

        # 模块二：解耦表征
        # 1. 知识状态映射头 (Student Emb -> K 维概率)
        self.knowledge_head = nn.Sequential(
            nn.Linear(dim_emb, dim_emb),
            nn.ReLU(),
            nn.Linear(dim_emb, num_concepts),
            nn.Sigmoid(),  # 输出为掌握概率
        )

        # 2. 技巧映射头 (Student Emb -> skill 向量)
        self.skill_head = nn.Sequential(
            nn.Linear(dim_emb, dim_skill),
            nn.Tanh(),  # 技巧特征归一化到 [-1, 1]
        )

        # 技巧 -> Guess / Slip 的生成器
        # 输出 2 维，[0] = guess, [1] = slip
        self.guess_slip_generator = nn.Linear(dim_skill, 2)

        # 图传播模块
        self.propagator = MonotonicPropagator()

    def forward(
        self,
        student_ids: torch.Tensor,
        exercise_ids: torch.Tensor,
        q_mask: torch.Tensor,
    ):
        """
        Input:
            student_ids: (batch_size,)  学生索引
            exercise_ids: (batch_size,) 题目索引（当前实现中只用于外部扩展，可选）
            q_mask: (batch_size, num_concepts)
                    当前 batch 中每道题对应的 Q 矩阵行（0/1）
        Output:
            pred_prob:  (batch_size,)  预测答对概率
            adj_dag:    (num_concepts, num_concepts) 学到的前置关系图
            h_knowledge:(batch_size, num_concepts)   传播后的知识状态
            z_skill:    (batch_size, dim_skill)      技巧向量
        """
        batch_size = student_ids.size(0)
        assert q_mask.shape == (batch_size, self.num_concepts), \
            f"q_mask shape {q_mask.shape} 与 (B, K) 不匹配"

        # 1. 学生向量 (B, dim_emb)
        stu_vec = self.student_emb(student_ids)

        # 2. 模块一：生成图结构
        graphs = self.graph_learner(self.concept_emb)
        # graphs[0] = A_dag，有向前置图
        adj_dag = graphs[0]

        # 3. 模块二：解耦表示
        # 3.1 初始知识状态 (B, K)
        h_init = self.knowledge_head(stu_vec)

        # 3.2 单调图传播，得到增强后的知识状态 (B, K)
        h_knowledge = self.propagator(h_init, adj_dag)

        # 3.3 技巧向量 (B, dim_skill)
        z_skill = self.skill_head(stu_vec)

        # 生成 Guess 和 Slip 概率 (B, 2)
        gs_params = torch.sigmoid(self.guess_slip_generator(z_skill)) * 0.3
        guess_prob = gs_params[:, 0].unsqueeze(1)  # (B, 1)
        slip_prob = gs_params[:, 1].unsqueeze(1)   # (B, 1)

        # 4. 模块三：诊断预测
        # 4.1 non-compensatory 知识掌握得分
        # 对每道题，只关注 q_mask=1 的知识点
        # 对 q_mask=0 的位置加上一个大数，使其在 softmin 中失效
        infinity_mask = (1.0 - q_mask) * 1e9
        masked_knowledge = h_knowledge + infinity_mask

        # softmin 近似 min pooling:
        # min(x) ≈ -1/alpha * log(sum(exp(-alpha * x)))
        alpha = 10.0
        neg_log_sum_exp = -torch.logsumexp(-alpha * masked_knowledge, dim=1, keepdim=True) / alpha

        # 限制在 [0, 1] 范围内，保持“掌握度”的概率解释
        knowledge_score = torch.clamp(neg_log_sum_exp, 0.0, 1.0)  # (B, 1)

        # 4.2 结合技巧进行最终预测 (DINA-like)
        # P(correct) = (1 - slip) * P(know) + guess * (1 - P(know))
        pred_prob = (1.0 - slip_prob) * knowledge_score + guess_prob * (1.0 - knowledge_score)

        return pred_prob.squeeze(-1), adj_dag, h_knowledge, z_skill
