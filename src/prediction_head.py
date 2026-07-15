"""Single-path, Q-masked scalar-difficulty 2PL components."""

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
    """Aggregate Q-masked concept abilities, then apply one scalar 2PL.

    ``theta_c`` remains available for diagnosis, but prediction has exactly one
    item logit: ``a * (mean_Q(theta_c) - b)``. There is no item-concept
    difficulty factor, matching term, residual, or calibration branch.
    """

    def __init__(self, knowledge_dim: int, num_concepts: int):
        super().__init__()
        self.knowledge_dim = int(knowledge_dim)
        self.num_concepts = int(num_concepts)
        self.theta_proj = nn.Linear(self.knowledge_dim, 1, bias=True)
        nn.init.normal_(self.theta_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.theta_proj.bias)

    def forward(
        self,
        knowledge_state: torch.Tensor,
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
        if tuple(concept_mask.shape) != (batch_size, self.num_concepts):
            raise ValueError(
                f"concept_mask must have shape {(batch_size, self.num_concepts)}, got {tuple(concept_mask.shape)}"
            )
        for name, value in (("b", b), ("a", a)):
            if tuple(value.shape) != (batch_size,):
                raise ValueError(
                    f"{name} must have shape {(batch_size,)}, got {tuple(value.shape)}"
                )

        theta_c = self.theta_proj(knowledge_state).squeeze(-1)
        mask = concept_mask.to(dtype=theta_c.dtype)
        denom = mask.sum(dim=1).clamp(min=1.0)
        theta_e = (theta_c * mask).sum(dim=1) / denom
        irt_logit = a * (theta_e - b)

        if not return_details:
            return irt_logit

        details = {
            "theta_c": theta_c.detach(),
            "theta_e": theta_e.detach(),
            "difficulty_e": b.detach(),
            "irt_logit": irt_logit.detach(),
        }
        return irt_logit, details
