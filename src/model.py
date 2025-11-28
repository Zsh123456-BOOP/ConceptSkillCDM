import math
from typing import List, Tuple

import torch
import torch.nn as nn

from .graph_learner import MultiHeadConceptGraphLearner
from .gnn_layers import ConceptGNN


class ConceptSkillCDM(nn.Module):
    def __init__(self, config, num_students: int, num_items: int, num_concepts: int):
        super().__init__()
        self.config = config
        self.num_concepts = num_concepts

        # 各类嵌入
        self.student_emb = nn.Embedding(num_students, config.dim_student)
        self.item_emb = nn.Embedding(num_items, config.dim_item)
        self.concept_emb = nn.Embedding(num_concepts, config.dim_concept)
        if config.dim_student != config.dim_concept:
            self.student_to_concept = nn.Linear(config.dim_student, config.dim_concept, bias=False)
        else:
            self.student_to_concept = nn.Identity()

        # 概念图学习器
        self.graph_learner = MultiHeadConceptGraphLearner(
            num_concepts=num_concepts,
            dim_concept=config.dim_concept,
            num_heads=config.num_heads,
            graph_dropout=config.graph_dropout,
            graph_topk=config.graph_topk,
        )
        self.head_types = self._init_head_types(config.num_heads)
        self.head_weights = nn.Parameter(torch.ones(config.num_heads))

        # GNN 传播层
        self.gnn = ConceptGNN(num_layers=config.gnn_layers)

        # 解耦的技能分支
        self.skill_mlp = nn.Sequential(
            nn.Linear(config.dim_student, config.gnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.gnn_hidden_dim, config.skill_dim),
        )
        self.know_proj = nn.Linear(num_concepts, config.skill_dim)

        # 猜测/失误偏置
        self.guess_slip_mlp = nn.Sequential(
            nn.Linear(config.skill_dim + config.dim_item, config.gnn_hidden_dim),
            nn.ReLU(),
        )
        self.guess_head = nn.Linear(config.gnn_hidden_dim, 1)
        self.slip_head = nn.Linear(config.gnn_hidden_dim, 1)

        self.tau = config.tau
        self.agg_type = getattr(config, "agg_type", "softmin")

    @staticmethod
    def _init_head_types(num_heads: int) -> List[str]:
        head_types = []
        for h in range(num_heads):
            if h == 0:
                head_types.append("precedence")
            elif h == 1:
                head_types.append("similarity")
            else:
                head_types.append("other")
        return head_types

    def forward(self, batch) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        student_ids = batch["student_ids"]
        item_ids = batch["item_ids"]
        batch_concept_ids = batch["batch_concept_ids"]
        concept_ptr = batch["concept_ptr"]

        e_student = self.student_emb(student_ids)
        e_item = self.item_emb(item_ids)

        E_c = self.concept_emb.weight  # [num_concepts, dim_concept]
        A_list = self.graph_learner(E_c)

        head_w = torch.softmax(self.head_weights, dim=0)
        A_agg = sum(w * A for w, A in zip(head_w, A_list))
        A_agg = self._normalize_adj(A_agg)

        e_student_proj = self.student_to_concept(e_student)
        S_init = torch.matmul(e_student_proj, E_c.transpose(0, 1)) / math.sqrt(e_student_proj.size(-1))
        S_prop = self.gnn(A_agg, S_init)

        z_skill = self.skill_mlp(e_student)
        z_know = self.know_proj(S_prop)

        score_knowledge = self._aggregate_knowledge(S_prop, batch_concept_ids, concept_ptr)
        feat_gs = self.guess_slip_mlp(torch.cat([z_skill, e_item], dim=-1))
        guess = torch.sigmoid(self.guess_head(feat_gs)).squeeze(-1)  # 猜对概率上浮
        slip = torch.sigmoid(self.slip_head(feat_gs)).squeeze(-1)    # 失误概率下压
        delta = guess - slip

        logits = score_knowledge + delta
        return logits, A_list, S_prop, z_skill, z_know

    def _normalize_adj(self, A: torch.Tensor) -> torch.Tensor:
        A = A + torch.eye(self.num_concepts, device=A.device)
        row_sum = A.sum(dim=-1, keepdim=True) + 1e-8
        return A / row_sum

    def _aggregate_knowledge(
        self, S_prop: torch.Tensor, batch_concept_ids: torch.Tensor, concept_ptr: torch.Tensor
    ) -> torch.Tensor:
        # 概念层面做聚合；若样本无概念则知识得分为 0.0（此时预测仅依赖技能偏置）
        scores = []
        B = S_prop.size(0)
        for i in range(B):
            start = concept_ptr[i].item()
            end = concept_ptr[i + 1].item()
            idx = batch_concept_ids[start:end]
            if idx.numel() == 0:
                scores.append(torch.tensor(0.0, device=S_prop.device))
                continue
            s_sub = S_prop[i, idx]
            if self.agg_type == "min":
                score = torch.min(s_sub)
            elif self.agg_type == "mean":
                score = torch.mean(s_sub)
            else:
                # softmin，tau 控制非补偿程度
                score = -self.tau * torch.logsumexp(-s_sub / self.tau, dim=0)
            scores.append(score)
        return torch.stack(scores)


__all__ = ["ConceptSkillCDM"]
