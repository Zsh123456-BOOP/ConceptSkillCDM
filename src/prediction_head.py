"""Single-path, per-concept 2PL prediction components."""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExerciseDifficultyEncoder(nn.Module):
    """Learn the scalar difficulty and positive discrimination of each item."""

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
    """Apply one auditable 2PL path before aggregating an item's Q concepts.

    The previous head first collapsed the student state to one scalar and then
    added a bounded Q-conditioned matching correction.  That correction
    saturated and overfit.  Here the student keeps one ability per concept and
    the collaborative item state supplies a factorized per-concept difficulty.
    Both meet only inside the same 2PL equation; no side prediction branch is
    introduced.
    """

    def __init__(self, knowledge_dim: int, num_concepts: int):
        super().__init__()
        self.knowledge_dim = int(knowledge_dim)
        self.num_concepts = int(num_concepts)
        self.theta_proj = nn.Linear(self.knowledge_dim, 1, bias=True)
        self.item_difficulty_projection = nn.Linear(
            self.knowledge_dim,
            self.knowledge_dim,
            bias=False,
        )
        nn.init.normal_(self.theta_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.theta_proj.bias)
        # Start exactly from scalar 2PL. The projection receives a non-zero
        # first-step gradient, after which item/concept coordinates can adapt.
        nn.init.zeros_(self.item_difficulty_projection.weight)

    def forward(
        self,
        knowledge_state: torch.Tensor,
        item_state: torch.Tensor,
        concept_basis: torch.Tensor,
        concept_mask: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if knowledge_state.dim() != 3:
            raise ValueError(
                f"knowledge_state must have shape (batch, concepts, dim), got {tuple(knowledge_state.shape)}"
            )
        batch_size, num_concepts, knowledge_dim = knowledge_state.shape
        if (num_concepts, knowledge_dim) != (self.num_concepts, self.knowledge_dim):
            raise ValueError(
                "knowledge_state trailing shape must be "
                f"{(self.num_concepts, self.knowledge_dim)}, got {(num_concepts, knowledge_dim)}"
            )
        if tuple(item_state.shape) != (batch_size, self.knowledge_dim):
            raise ValueError(
                f"item_state must have shape {(batch_size, self.knowledge_dim)}, got {tuple(item_state.shape)}"
            )
        if tuple(concept_basis.shape) != (self.num_concepts, self.knowledge_dim):
            raise ValueError(
                "concept_basis must have shape "
                f"{(self.num_concepts, self.knowledge_dim)}, got {tuple(concept_basis.shape)}"
            )
        if tuple(concept_mask.shape) != (batch_size, self.num_concepts):
            raise ValueError(
                f"concept_mask must have shape {(batch_size, self.num_concepts)}, got {tuple(concept_mask.shape)}"
            )

        theta_c = self.theta_proj(knowledge_state).squeeze(-1)
        normalized_items = F.layer_norm(item_state, (self.knowledge_dim,))
        normalized_concepts = F.layer_norm(concept_basis, (self.knowledge_dim,))
        item_coordinate = self.item_difficulty_projection(normalized_items)
        item_concept_delta = item_coordinate.matmul(normalized_concepts.t()) / (
            float(self.knowledge_dim) ** 0.5
        )
        difficulty_c = b.unsqueeze(-1) + item_concept_delta
        concept_irt_logit = a.unsqueeze(-1) * (theta_c - difficulty_c)

        mask = concept_mask.float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        irt_logit = (concept_irt_logit * mask).sum(dim=1) / denom

        if not return_details:
            return irt_logit

        theta_e = (theta_c * mask).sum(dim=1) / denom
        difficulty_e = (difficulty_c * mask).sum(dim=1) / denom
        details = {
            "theta_c": theta_c.detach(),
            "theta_e": theta_e.detach(),
            "item_difficulty_delta": item_concept_delta.detach(),
            "concept_difficulty": difficulty_c.detach(),
            "difficulty_e": difficulty_e.detach(),
            "concept_irt_logit": concept_irt_logit.detach(),
            "irt_logit": irt_logit.detach(),
        }
        return irt_logit, details

    def item_difficulty_l2(self) -> torch.Tensor:
        """Regularize only the factorized item-to-concept difficulty map."""
        return self.item_difficulty_projection.weight.pow(2).mean()
