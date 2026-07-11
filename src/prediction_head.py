import math
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ITEM_MATCHING_RANK


class ExerciseDifficultyEncoder(nn.Module):
    """固定的 IRT 题目参数编码器，只负责输出 b/a。"""

    def __init__(self, num_exercises: int):
        super().__init__()
        self.b = nn.Embedding(num_exercises, 1)
        self.a_raw = nn.Embedding(num_exercises, 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.zeros_(self.b.weight)
        nn.init.normal_(self.a_raw.weight, mean=0.0, std=0.02)

    def forward(self, exercise_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b = self.b(exercise_ids).squeeze(-1)
        a = F.softplus(self.a_raw(exercise_ids).squeeze(-1)) + 1e-6
        return b, a


class CognitiveDiagnosisHead(nn.Module):
    """
    Q-aware low-rank MIRT readout followed by a 2PL item response function.

    The shared projection keeps a stable one-dimensional ability backbone.  A
    low-rank direction shared by each Q concept then selects a second view of
    the Q-pooled student state.  Items share statistical strength through their
    concepts instead of owning a free per-item vector, and no second
    prediction/logit branch is introduced.
    """

    def __init__(
        self,
        knowledge_dim: int,
        num_concepts: int,
        *,
        enable_item_matching: bool = True,
        item_matching_rank: int = ITEM_MATCHING_RANK,
    ):
        super().__init__()
        self.knowledge_dim = int(knowledge_dim)
        self.num_concepts = int(num_concepts)
        self.enable_item_matching = bool(enable_item_matching)
        self.item_matching_rank = min(
            self.knowledge_dim,
            max(1, int(item_matching_rank)),
        )

        self.theta_proj = nn.Linear(self.knowledge_dim, 1, bias=True)
        nn.init.normal_(self.theta_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.theta_proj.bias)

        # Keep these parameters in both full and no-item-matching checkpoints.
        # That makes common initialization/state keys identical across the
        # ablation; disabling the path also freezes the unused parameters.
        self.item_matching_projection = nn.Linear(
            self.knowledge_dim,
            self.item_matching_rank,
            bias=False,
        )
        self.concept_matching_direction = nn.Embedding(
            self.num_concepts,
            self.item_matching_rank,
        )
        nn.init.xavier_normal_(self.item_matching_projection.weight)
        # A zero concept factor makes the new architecture start from the
        # shared 2PL solution while still giving the factor a non-zero
        # first-step gradient through the seeded projection.
        nn.init.zeros_(self.concept_matching_direction.weight)
        if not self.enable_item_matching:
            self.item_matching_projection.requires_grad_(False)
            self.concept_matching_direction.requires_grad_(False)

    def forward(
        self,
        knowledge_state: torch.Tensor,
        concept_mask: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        theta_c = self.theta_proj(knowledge_state).squeeze(-1)

        mask = concept_mask.float()
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled_state = (knowledge_state * mask.unsqueeze(-1)).sum(dim=1) / denom
        theta_base = self.theta_proj(pooled_state).squeeze(-1)

        if self.enable_item_matching:
            normalized_state = F.layer_norm(
                pooled_state,
                normalized_shape=(self.knowledge_dim,),
            )
            student_factor = self.item_matching_projection(normalized_state)
            item_factor = mask.matmul(self.concept_matching_direction.weight) / denom
            raw_item_matching = (student_factor * item_factor).sum(dim=-1) / math.sqrt(
                float(self.item_matching_rank)
            )
            # The Q-conditioned direction is a bounded correction to the
            # shared ability, never a competing predictor.  The 1/rank bound
            # tightens naturally as the factorization gains dimensions.
            item_matching = torch.tanh(raw_item_matching) / float(
                self.item_matching_rank
            )
        else:
            raw_item_matching = theta_base.new_zeros(theta_base.shape)
            item_matching = raw_item_matching

        theta_e = theta_base + item_matching

        irt_logit = a * (theta_e - b)

        if not return_details:
            return irt_logit

        details = {
            "theta_c": theta_c.detach(),
            "theta_e_base": theta_base.detach(),
            "item_matching_raw": raw_item_matching.detach(),
            "item_matching": item_matching.detach(),
            "theta_e": theta_e.detach(),
            "irt_logit": irt_logit.detach(),
        }
        return irt_logit, details

    def item_matching_l2(self) -> torch.Tensor:
        """Return the small prediction-side L2 term used by the main loss."""
        if not self.enable_item_matching:
            return self.theta_proj.weight.new_tensor(0.0)
        return (
            self.item_matching_projection.weight.pow(2).mean()
            + self.concept_matching_direction.weight.pow(2).mean()
        )
