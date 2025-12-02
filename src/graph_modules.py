import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List


class GraphStructureLearner(nn.Module):
    """
    模块一：无监督概念关系发现
    使用多头注意力机制学习概念间的潜在关系图。
    Head 1: 前置关系 (DAG 约束, 有向图)
    Head 2: 相似关系 (对称约束, 无向图)
    """
    def __init__(self, num_concepts: int, dim: int, num_heads: int = 2, tau: float = 0.1):
        super().__init__()
        self.num_concepts = num_concepts
        self.dim = dim
        self.num_heads = num_heads
        self.tau = tau  # 稀疏化阈值

        # 不同 head 学不同子空间
        self.W_Q = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_heads)])
        self.W_K = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_heads)])

    def forward(self, concept_emb: torch.Tensor) -> List[torch.Tensor]:
        """
        Input:
            concept_emb: (num_concepts, dim)
        Output:
            graphs: List[Tensor], graphs[0] = A_dag, graphs[1] = A_sim
        """
        graphs: List[torch.Tensor] = []
        mask = torch.eye(self.num_concepts, device=concept_emb.device).bool()

        # --- Head 1: 前置/进阶关系 (Directed, Asymmetric) ---
        Q1 = self.W_Q[0](concept_emb)  # (K, dim)
        K1 = self.W_K[0](concept_emb)  # (K, dim)

        attn_scores1 = torch.matmul(Q1, K1.transpose(0, 1)) / np.sqrt(self.dim)

        # 非负 + 稀疏
        A_dag = F.relu(attn_scores1 - self.tau)
        A_dag = A_dag.masked_fill(mask, 0.0)
        graphs.append(A_dag)

        # --- Head 2: 相似/混淆关系 (Undirected, Symmetric) ---
        if self.num_heads > 1:
            Q2 = self.W_Q[1](concept_emb)  # (K, dim)

            # 对称距离
            dist_sq = torch.cdist(Q2, Q2, p=2).pow(2)
            sigma = 1.0
            A_sim = torch.exp(-dist_sq / (2 * sigma ** 2))

            A_sim = F.relu(A_sim - self.tau)
            A_sim = A_sim.masked_fill(mask, 0.0)

            graphs.append(A_sim)

        return graphs


class MonotonicPropagator(nn.Module):
    """
    模块二：学生能力的解耦表征 - 图传播部分
    实现单调性 GNN：前置概念的掌握只能“正向”增强进阶概念的掌握概率。
    """
    def __init__(self, impact_factor: float = 0.5):
        super().__init__()
        # 如果想让模型自动学，可以改成：
        # self.impact_factor = nn.Parameter(torch.tensor(impact_factor))
        self.impact_factor = impact_factor

    def forward(self, h_init: torch.Tensor, adj_dag: torch.Tensor) -> torch.Tensor:
        """
        Input:
            h_init:  (batch_size, num_concepts) 初始知识状态 (Sigmoid 概率)
            adj_dag:(num_concepts, num_concepts) 学习到的前置关系图
        Output:
            h_prop:  (batch_size, num_concepts) 传播后的知识状态
        """
        # 邻居贡献聚合: (B, K) @ (K, K) -> (B, K)
        neighbor_contribution = torch.matmul(h_init, adj_dag)

        # 归一化按列 in-degree
        in_degree = adj_dag.sum(dim=0, keepdim=True) + 1e-6  # (1, K)
        normalized_contribution = neighbor_contribution / in_degree

        # 单调性融合
        h_prop = h_init + self.impact_factor * torch.tanh(normalized_contribution)
        h_prop = torch.clamp(h_prop, 0.0, 1.0)

        return h_prop
