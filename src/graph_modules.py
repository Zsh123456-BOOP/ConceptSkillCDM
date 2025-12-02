import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphStructureLearner(nn.Module):
    """
    模块一：无监督概念关系发现
    使用多头注意力机制学习概念间的潜在关系图。
    Head 1: 前置关系 (DAG约束, 有向图)
    Head 2: 相似关系 (对称约束, 无向图)
    """
    def __init__(self, num_concepts: int, dim: int, num_heads: int = 2, tau: float = 0.1):
        super().__init__()
        self.num_concepts = num_concepts
        self.dim = dim
        self.num_heads = num_heads
        self.tau = tau  # 稀疏化阈值

        # 用于生成 Query 和 Key 的线性层（按 head 存在 ModuleList 中）
        self.W_Q = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_heads)])
        self.W_K = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_heads)])

    def forward(self, concept_emb: torch.Tensor):
        """
        Input:
            concept_emb: (num_concepts, dim)
        Output:
            graphs: List[Tensor]，每个元素形状为 (num_concepts, num_concepts)
                    graphs[0] = A_dag (前置关系图，有向)
                    graphs[1] = A_sim (相似图，对称)（如果 num_heads > 1）
        """
        graphs = []

        # 公用的自环 mask
        mask = torch.eye(self.num_concepts, device=concept_emb.device).bool()

        # --- Head 1: 前置/进阶关系 (Directed) ---
        # 逻辑：A -> B 意味着掌握 A 有助于掌握 B
        Q1 = self.W_Q[0](concept_emb)  # (K, dim)
        K1 = self.W_K[0](concept_emb)  # (K, dim)

        # 注意力打分：(K, dim) @ (dim, K) -> (K, K)
        attn_scores1 = torch.matmul(Q1, K1.transpose(0, 1)) / np.sqrt(self.dim)

        # 稀疏性与非负性约束 (ReLU + Threshold)
        # 这里的 A_dag 是加权邻接矩阵
        A_dag = F.relu(attn_scores1 - self.tau)

        # 移除自环 (Self-loop)
        A_dag = A_dag.masked_fill(mask, 0.0)
        graphs.append(A_dag)

        # --- Head 2: 相似/混淆关系 (Undirected/Symmetric) ---
        if self.num_heads > 1:
            Q2 = self.W_Q[1](concept_emb)  # (K, dim)
            # 使用欧氏距离的平方作为相似度的反向度量
            # dist_sq[i, j] = ||qi - qj||^2
            dist_sq = torch.cdist(Q2, Q2, p=2).pow(2)

            # 距离越小，相似度越高，使用高斯核转换
            sigma = 1.0
            A_sim = torch.exp(-dist_sq / (2 * sigma ** 2))

            # 阈值稀疏化，并移除自环
            A_sim = F.relu(A_sim - self.tau)
            A_sim = A_sim.masked_fill(mask, 0.0)

            # A_sim 本身即为对称矩阵（由 pairwise 距离构造）
            graphs.append(A_sim)

        return graphs


class MonotonicPropagator(nn.Module):
    """
    模块二：学生能力的解耦表征 - 图传播部分
    实现单调性 GNN：前置概念的掌握只能“正向”增强进阶概念的掌握概率。
    """
    def __init__(self):
        super().__init__()
        # 此模块没有可学习参数，完全依赖学习到的图结构 A_dag 进行传播

    def forward(self, h_init: torch.Tensor, adj_dag: torch.Tensor) -> torch.Tensor:
        """
        Input:
            h_init:  (batch_size, num_concepts) 初始知识状态 (Sigmoid 后的概率)
            adj_dag: (num_concepts, num_concepts) 学习到的前置关系图
                     adj_dag[i, j] > 0 表示 i 是 j 的前置概念
        Output:
            h_prop:  (batch_size, num_concepts) 传播后的知识状态
        """
        # 邻居贡献: (B, K) @ (K, K) -> (B, K)
        neighbor_contribution = torch.matmul(h_init, adj_dag)

        # 归一化：按列求度，防止数值爆炸
        degree = adj_dag.sum(dim=0, keepdim=True) + 1e-6  # (1, K)
        neighbor_contribution = neighbor_contribution / degree

        # 单调性融合：只加正向增量，不引入线性层
        h_prop = h_init + 0.5 * torch.tanh(neighbor_contribution)

        # 再次截断到 (0, 1) 范围，保持概率意义
        h_prop = torch.clamp(h_prop, 0.0, 1.0)

        return h_prop
