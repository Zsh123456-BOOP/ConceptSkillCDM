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

    When ``evidence_anchor_channels > 0`` the concept ability is anchored at
    the train-only response-evidence logits: ``theta_c`` becomes the state
    readout plus a non-negative per-channel weighting of the evidence anchor.
    The anchor enters *before* the single 2PL readout, so the architectural
    invariant ``logits == irt_logit`` is unchanged, and the non-negative
    weights keep the readout monotonic in the evidence.
    """

    def __init__(
        self,
        knowledge_dim: int,
        num_concepts: int,
        evidence_anchor_channels: int = 0,
    ):
        super().__init__()
        self.knowledge_dim = int(knowledge_dim)
        self.num_concepts = int(num_concepts)
        self.evidence_anchor_channels = max(0, int(evidence_anchor_channels))
        self.theta_proj = nn.Linear(self.knowledge_dim, 1, bias=True)
        nn.init.normal_(self.theta_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.theta_proj.bias)
        if self.evidence_anchor_channels > 0:
            # softplus(0.5413) == 1.0: every evidence channel starts at unit
            # weight and stays non-negative while remaining learnable.
            self.evidence_anchor_raw = nn.Parameter(
                torch.full((self.evidence_anchor_channels,), 0.5413)
            )

    def evidence_anchor_weights(self) -> torch.Tensor:
        if self.evidence_anchor_channels <= 0:
            raise RuntimeError("evidence anchoring is disabled for this head")
        return F.softplus(self.evidence_anchor_raw)

    def forward(
        self,
        knowledge_state: torch.Tensor,
        concept_mask: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        evidence_anchor: torch.Tensor = None,
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
        if evidence_anchor is not None:
            if self.evidence_anchor_channels <= 0:
                raise ValueError("evidence_anchor supplied to a head without anchor channels")
            expected = (batch_size, self.num_concepts, self.evidence_anchor_channels)
            if tuple(evidence_anchor.shape) != expected:
                raise ValueError(
                    f"evidence_anchor must have shape {expected}, got {tuple(evidence_anchor.shape)}"
                )
            anchor_weights = self.evidence_anchor_weights().to(dtype=theta_c.dtype)
            theta_c = theta_c + (
                evidence_anchor.to(dtype=theta_c.dtype) * anchor_weights
            ).sum(dim=-1)
        elif self.evidence_anchor_channels > 0:
            raise ValueError("head expects an evidence_anchor but none was supplied")
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
        if self.evidence_anchor_channels > 0:
            details["evidence_anchor_weights"] = self.evidence_anchor_weights().detach()
        return irt_logit, details
