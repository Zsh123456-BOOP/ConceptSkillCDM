import math
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadConceptGraphLearner(nn.Module):
    def __init__(
        self,
        num_concepts: int,
        dim_concept: int,
        num_heads: int,
        graph_dropout: float = 0.0,
        graph_topk: int = 0,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_heads = num_heads
        self.graph_topk = graph_topk

        self.q_linears = nn.ModuleList([nn.Linear(dim_concept, dim_concept, bias=False) for _ in range(num_heads)])
        self.k_linears = nn.ModuleList([nn.Linear(dim_concept, dim_concept, bias=False) for _ in range(num_heads)])
        self.dropout = nn.Dropout(graph_dropout)

    def forward(self, concept_emb: torch.Tensor) -> List[torch.Tensor]:
        """
        参数:
            concept_emb: [num_concepts, dim_concept]
        返回:
            A_list: 每个头对应的邻接矩阵列表，尺寸 [num_concepts, num_concepts]
        """
        A_list: List[torch.Tensor] = []
        for q_linear, k_linear in zip(self.q_linears, self.k_linears):
            Q = q_linear(concept_emb)  # [N, d]
            K = k_linear(concept_emb)  # [N, d]
            scores = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(K.size(-1))
            A = F.softmax(scores, dim=-1)
            A = self.dropout(A)
            A = F.relu(A)
            if self.graph_topk and self.graph_topk > 0 and self.graph_topk < A.size(-1):
                # 按行保留 top-k 边以进一步稀疏化
                topk_val, topk_idx = torch.topk(A, k=self.graph_topk, dim=-1)
                mask = torch.zeros_like(A)
                mask.scatter_(dim=-1, index=topk_idx, src=torch.ones_like(topk_val))
                A = A * mask
            A_list.append(A)
        return A_list

    def graph_regularization(self, A_list: List[torch.Tensor], head_types: List[str]) -> Dict[str, torch.Tensor]:
        L_sparse = torch.tensor(0.0, device=A_list[0].device)
        L_sym = torch.tensor(0.0, device=A_list[0].device)
        L_dag = torch.tensor(0.0, device=A_list[0].device)
        L_trans = torch.tensor(0.0, device=A_list[0].device)
        L_contain = torch.tensor(0.0, device=A_list[0].device)
        L_confuse = torch.tensor(0.0, device=A_list[0].device)
        N = self.num_concepts

        for idx, A in enumerate(A_list):
            L_sparse = L_sparse + torch.mean(torch.abs(A))
            h_type = head_types[idx] if idx < len(head_types) else "other"
            if h_type == "similarity":
                L_sym = L_sym + torch.mean((A - A.transpose(0, 1)) ** 2)
            if h_type == "precedence":
                B = A * A
                h_val = torch.trace(torch.matrix_exp(B)) - self.num_concepts
                L_dag = L_dag + h_val * h_val
                # 传递性/层次性软约束：A@A 不应显著强于 A
                trans_gap = torch.relu(torch.matmul(A, A) - A)
                L_trans = L_trans + torch.mean(trans_gap * trans_gap)
            if h_type == "contain":
                # 近似包含：鼓励 A_ij 与 A_ik*A_kj 的一致性（传递闭包收缩）
                closure = torch.matmul(A, A)
                L_contain = L_contain + torch.mean((closure - A) * (closure - A))
            if h_type == "confuse":
                # 混淆关系：鼓励对称且行/列接近均值，避免过于集中（类似均衡混淆）
                sym_part = torch.mean((A - A.transpose(0, 1)) ** 2)
                row_mean = A.mean(dim=1, keepdim=True)
                col_mean = A.mean(dim=0, keepdim=True)
                balance = torch.mean((A - row_mean) ** 2) + torch.mean((A - col_mean) ** 2)
                L_confuse = L_confuse + sym_part + balance
        # 对 DAG 项做轻微缩放，避免大图时数值过大
        L_dag = L_dag / max(N, 1)

        return {
            "sparse": L_sparse,
            "sym": L_sym,
            "dag": L_dag,
            "trans": L_trans,
            "contain": L_contain,
            "confuse": L_confuse,
        }


__all__ = ["MultiHeadConceptGraphLearner"]
