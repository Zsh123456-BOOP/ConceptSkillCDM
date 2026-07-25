"""A bounded sparse-evidence adjustment for concept-level ability.

The module is deliberately independent from the production model so it can be
attached to an already-trained v1 checkpoint without changing the checkpoint's
prediction path at initialization.  It consumes sufficient statistics whose
current training outcome has already been excluded by the caller.
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


class SparseEvidenceThetaResidual(nn.Module):
    """Compute a zero-initialized, bounded concept-ability adjustment.

    ``relation_matrices`` follow the ``A[target, source]`` convention.  The
    relation graph is detached, averaged over heads, and stripped of self
    loops.  Importantly, the remaining off-diagonal mass is *not*
    re-normalized: a graph row that assigned almost all mass to its self loop
    must not suddenly become a strong cross-concept relation.

    ``count``, ``correct``, and ``concept_rate`` must all have shape
    ``(batch, concepts)``.  During training they must be the caller's
    leave-one-out statistics; this module never accepts the current label.
    """

    SPARSE_CONCEPT_THRESHOLD = 192

    def __init__(
        self,
        max_abs_adjustment: float = 0.20,
        max_logit_delta: float = 4.0,
        probability_eps: float = 1e-4,
    ):
        super().__init__()
        if not 0.0 < float(max_abs_adjustment) <= 1.0:
            raise ValueError("max_abs_adjustment must lie in (0, 1]")
        if float(max_logit_delta) <= 0.0:
            raise ValueError("max_logit_delta must be positive")
        if not 0.0 < float(probability_eps) < 0.5:
            raise ValueError("probability_eps must lie in (0, 0.5)")

        self.max_abs_adjustment = float(max_abs_adjustment)
        self.max_logit_delta = float(max_logit_delta)
        self.probability_eps = float(probability_eps)
        # ReZero-style scalar.  Constant initialization consumes no RNG and
        # makes a newly attached module exactly equal to the v1 model.
        self.rho = nn.Parameter(torch.zeros(()))

    @staticmethod
    def reliability_gate(
        target_count: torch.Tensor,
        support: torch.Tensor,
        conflict: torch.Tensor,
    ) -> torch.Tensor:
        """Return an analytic gate with the required monotonic directions."""
        target_count = target_count.clamp(min=0.0)
        support = support.clamp(min=0.0)
        conflict = conflict.clamp(min=0.0)
        need = 1.0 / (target_count + 1.0)
        backed = support / (support + 1.0)
        agreement = 1.0 / (conflict + 1.0)
        return (need * backed * agreement).clamp(min=0.0, max=1.0)

    @staticmethod
    def _off_diagonal_head_mean(
        relation_matrices: torch.Tensor,
    ) -> torch.Tensor:
        """Detach, average heads, and remove self loops without re-normalizing."""
        if relation_matrices.dim() != 3:
            raise ValueError(
                "relation_matrices must have shape (heads, concepts, concepts); "
                f"got {tuple(relation_matrices.shape)}"
            )
        heads, rows, columns = relation_matrices.shape
        if heads <= 0 or rows <= 0 or rows != columns:
            raise ValueError(
                "relation_matrices must contain at least one square graph; "
                f"got {tuple(relation_matrices.shape)}"
            )
        relation = relation_matrices.detach().clamp(min=0.0).mean(dim=0)
        if rows == 1:
            return torch.zeros_like(relation)
        eye = torch.eye(rows, device=relation.device, dtype=torch.bool)
        return relation.masked_fill(eye, 0.0)

    @classmethod
    def _propagate(
        cls,
        source: torch.Tensor,
        relation: torch.Tensor,
    ) -> torch.Tensor:
        """Apply ``A[target, source]`` without constructing BxCxC tensors."""
        if source.dim() != 2 or relation.dim() != 2:
            raise ValueError("source and relation must both be two-dimensional")
        if relation.size(0) != relation.size(1) or source.size(1) != relation.size(0):
            raise ValueError(
                f"incompatible source/relation shapes: {tuple(source.shape)} and "
                f"{tuple(relation.shape)}"
            )
        if int(relation.size(0)) >= cls.SPARSE_CONCEPT_THRESHOLD:
            propagated = torch.sparse.mm(
                relation.to_sparse_coo().coalesce(),
                source.transpose(0, 1),
            )
            return propagated.transpose(0, 1)
        return torch.matmul(source, relation.transpose(0, 1))

    def forward(
        self,
        relation_matrices: torch.Tensor,
        *,
        count: torch.Tensor,
        correct: torch.Tensor,
        concept_rate: torch.Tensor,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Return a concept-level theta adjustment and optional diagnostics."""
        relation = self._off_diagonal_head_mean(relation_matrices)
        expected = tuple(count.shape)
        if count.dim() != 2:
            raise ValueError(
                f"count must have shape (batch, concepts), got {expected}"
            )
        if expected[1] != relation.size(0):
            raise ValueError(
                f"count has {expected[1]} concepts but relation has "
                f"{relation.size(0)}"
            )
        for name, value in (("correct", correct), ("concept_rate", concept_rate)):
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, got {tuple(value.shape)}"
                )

        device = relation.device
        dtype = relation.dtype
        count = count.to(device=device, dtype=dtype).clamp(min=0.0)
        correct = correct.to(device=device, dtype=dtype)
        eps = self.probability_eps
        concept_rate = concept_rate.to(device=device, dtype=dtype).clamp(
            min=eps,
            max=1.0 - eps,
        )

        centered = correct - count * concept_rate
        source_residual = centered / count.clamp(min=1.0)
        second_moment_mass = count * source_residual.square()
        combined = torch.cat((count, centered, second_moment_mass), dim=0)
        propagated = self._propagate(combined, relation)
        batch_size = int(count.size(0))
        support, centered_sum, second_sum = propagated.split(batch_size, dim=0)
        support = support.clamp(min=0.0)

        has_support = support > 1e-12
        neighbor_mean = torch.where(
            has_support,
            centered_sum / support.clamp(min=1e-12),
            torch.zeros_like(centered_sum),
        )
        probability_delta = centered_sum / (support + 1.0)
        variance = torch.where(
            has_support,
            second_sum / support.clamp(min=1e-12) - neighbor_mean.square(),
            torch.zeros_like(second_sum),
        ).clamp(min=0.0)
        direct_delta = centered / (count + 1.0)
        target_reliability = count / (count + 1.0)
        conflict = (
            variance
            + target_reliability * (neighbor_mean - direct_delta).square()
        ).clamp(min=0.0)
        gate = self.reliability_gate(count, support, conflict)

        neighbor_rate = (concept_rate + probability_delta).clamp(
            min=eps,
            max=1.0 - eps,
        )
        evidence_logit_delta = (
            torch.logit(neighbor_rate) - torch.logit(concept_rate)
        ).clamp(
            min=-self.max_logit_delta,
            max=self.max_logit_delta,
        )
        evidence_logit_delta = torch.nan_to_num(
            evidence_logit_delta,
            nan=0.0,
            posinf=self.max_logit_delta,
            neginf=-self.max_logit_delta,
        )

        alpha = (
            evidence_logit_delta.new_tensor(self.max_abs_adjustment)
            * torch.tanh(self.rho.to(device=device, dtype=dtype))
        )
        theta_adjustment = alpha * torch.tanh(gate * evidence_logit_delta)
        theta_adjustment = torch.where(
            has_support,
            theta_adjustment,
            torch.zeros_like(theta_adjustment),
        )

        if not return_details:
            return theta_adjustment
        details = {
            "offdiag_relation": relation,
            "support": support.detach(),
            "neighbor_centered_sum": centered_sum.detach(),
            "neighbor_mean": neighbor_mean.detach(),
            "variance": variance.detach(),
            "conflict": conflict.detach(),
            "gate": gate.detach(),
            "probability_delta": probability_delta.detach(),
            "evidence_logit_delta": evidence_logit_delta.detach(),
            "alpha": alpha.detach(),
        }
        return theta_adjustment, details
