import math
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadConceptGraphLearner(nn.Module):
    def __init__(self, num_concepts: int, dim_concept: int, num_heads: int, graph_dropout: float = 0.0):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_heads = num_heads

        self.q_linears = nn.ModuleList([nn.Linear(dim_concept, dim_concept, bias=False) for _ in range(num_heads)])
        self.k_linears = nn.ModuleList([nn.Linear(dim_concept, dim_concept, bias=False) for _ in range(num_heads)])
        self.dropout = nn.Dropout(graph_dropout)

    def forward(self, concept_emb: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            concept_emb: [num_concepts, dim_concept]
        Returns:
            A_list: list of adjacency matrices per head, each [num_concepts, num_concepts]
        """
        A_list: List[torch.Tensor] = []
        for q_linear, k_linear in zip(self.q_linears, self.k_linears):
            Q = q_linear(concept_emb)  # [N, d]
            K = k_linear(concept_emb)  # [N, d]
            scores = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(K.size(-1))
            A = F.softmax(scores, dim=-1)
            A = self.dropout(A)
            A = F.relu(A)
            A_list.append(A)
        return A_list

    def graph_regularization(self, A_list: List[torch.Tensor], head_types: List[str]) -> Dict[str, torch.Tensor]:
        L_sparse = torch.tensor(0.0, device=A_list[0].device)
        L_sym = torch.tensor(0.0, device=A_list[0].device)
        L_dag = torch.tensor(0.0, device=A_list[0].device)
        N = self.num_concepts
        norm_factor = max(N * N, 1)

        for idx, A in enumerate(A_list):
            L_sparse = L_sparse + torch.sum(torch.abs(A))
            h_type = head_types[idx] if idx < len(head_types) else "other"
            if h_type == "similarity":
                L_sym = L_sym + torch.norm(A - A.transpose(0, 1), p="fro") ** 2
            if h_type == "precedence":
                B = A * A
                h_val = torch.trace(torch.matrix_exp(B)) - self.num_concepts
                L_dag = L_dag + h_val * h_val
        # Normalize scale to prevent explosion on large concept graphs
        L_sparse = L_sparse / norm_factor
        L_sym = L_sym / norm_factor
        # Optional mild scaling for DAG term to avoid overly large values on big graphs
        L_dag = L_dag / max(N, 1)

        return {"sparse": L_sparse, "sym": L_sym, "dag": L_dag}


__all__ = ["MultiHeadConceptGraphLearner"]
